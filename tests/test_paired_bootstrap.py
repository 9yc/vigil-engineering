"""Unit tests for paired cluster bootstrap and calibration-subset sensitivity."""

import csv
import json
import math
import os
import pathlib
import random
import tempfile
import unittest

from tools.paired_bootstrap import (
    ItemMeasurement,
    build_paired_dataset,
    linear_quantile,
    load_paired_dataset,
    main,
    run_calibration_sensitivity,
    run_paired_bootstrap,
)


def _make_synthetic_dataset(num_items: int = 30) -> list[ItemMeasurement]:
    records: list[ItemMeasurement] = []
    for idx in range(num_items):
        item_id = f"item_{idx:03d}"
        base_scale = 1.0 + 0.04 * idx
        # Baseline config: high quality (1.0), higher energy and latency.
        records.append(
            ItemMeasurement(
                item_id=item_id,
                config_id="cfg_base",
                energy_j=10.0 * base_scale,
                latency_s=2.0 * base_scale,
                quality=1.0,
                calibration_time_s=2.2 * base_scale,
                calibration_energy_j=11.0 * base_scale,
            )
        )
        # Balanced candidate: meets 0.90 quality (28/30 = 0.9333), 35% lower energy.
        records.append(
            ItemMeasurement(
                item_id=item_id,
                config_id="cfg_opt",
                energy_j=6.5 * base_scale,
                latency_s=1.4 * base_scale,
                quality=1.0 if idx >= 2 else 0.0,
                calibration_time_s=1.5 * base_scale,
                calibration_energy_j=7.0 * base_scale,
            )
        )
        # Aggressive config: cheap energy, low quality (0.50), infeasible at 0.90 target.
        records.append(
            ItemMeasurement(
                item_id=item_id,
                config_id="cfg_aggressive",
                energy_j=3.5 * base_scale,
                latency_s=0.8 * base_scale,
                quality=1.0 if (idx % 2 == 0) else 0.0,
                calibration_time_s=0.9 * base_scale,
                calibration_energy_j=4.0 * base_scale,
            )
        )
    return records


class TestPairedBootstrap(unittest.TestCase):
    """Verify ratio-of-means parity, bootstrap CI invariance/monotonicity, and subset draws."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.tmpdir = pathlib.Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_linear_quantile_exact_interpolation(self) -> None:
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertAlmostEqual(linear_quantile(vals, 0.0), 10.0)
        self.assertAlmostEqual(linear_quantile(vals, 1.0), 50.0)
        self.assertAlmostEqual(linear_quantile(vals, 0.5), 30.0)
        self.assertAlmostEqual(linear_quantile(vals, 0.25), 20.0)
        self.assertAlmostEqual(linear_quantile(vals, 0.125), 15.0)
        self.assertAlmostEqual(linear_quantile([7.5], 0.8), 7.5)
        with self.assertRaises(ValueError):
            linear_quantile([], 0.5)
        with self.assertRaises(ValueError):
            linear_quantile(vals, -0.01)
        with self.assertRaises(ValueError):
            linear_quantile(vals, 1.01)

    def test_exact_ratio_of_means_vs_mean_of_ratios_distinction(self) -> None:
        # Construct 2 items where ratio-of-means != mean-of-ratios:
        # Item 0: E_a = 5, E_b = 10 (item ratio = 50% reduction)
        # Item 1: E_a = 25, E_b = 30 (item ratio = 16.67% reduction)
        # Mean E_a = 15, Mean E_b = 20 -> Ratio-of-means reduction = 100 * (1 - 15/20) = 25.0%
        # (whereas mean-of-ratios would be 33.33%!)
        records = [
            ItemMeasurement("i0", "a", energy_j=5.0, latency_s=1.0, quality=1.0),
            ItemMeasurement("i0", "b", energy_j=10.0, latency_s=2.0, quality=1.0),
            ItemMeasurement("i1", "a", energy_j=25.0, latency_s=3.0, quality=1.0),
            ItemMeasurement("i1", "b", energy_j=30.0, latency_s=6.0, quality=1.0),
        ]
        ds = build_paired_dataset(records)
        res = run_paired_bootstrap(ds, "a", "b", num_resamples=500, confidence_level=0.95, seed=7)

        self.assertAlmostEqual(res.mean_energy_a_j.point, 15.0, places=12)
        self.assertAlmostEqual(res.mean_energy_b_j.point, 20.0, places=12)
        self.assertAlmostEqual(res.abs_energy_savings_j.point, 5.0, places=12)
        self.assertAlmostEqual(res.rel_energy_reduction_pct.point, 25.0, places=12)
        self.assertAlmostEqual(res.mean_latency_a_s.point, 2.0, places=12)
        self.assertAlmostEqual(res.mean_latency_b_s.point, 4.0, places=12)
        self.assertAlmostEqual(res.abs_latency_savings_s.point, 2.0, places=12)
        self.assertAlmostEqual(res.rel_latency_reduction_pct.point, 50.0, places=12)

    def test_bootstrap_seed_determinism_proportional_invariance_and_ci_monotonicity(self) -> None:
        ds = build_paired_dataset(_make_synthetic_dataset(30))
        res_90 = run_paired_bootstrap(
            ds, "cfg_opt", "cfg_base", num_resamples=1000, confidence_level=0.90, seed=123
        )
        res_90_repeat = run_paired_bootstrap(
            ds, "cfg_opt", "cfg_base", num_resamples=1000, confidence_level=0.90, seed=123
        )
        res_99 = run_paired_bootstrap(
            ds, "cfg_opt", "cfg_base", num_resamples=1000, confidence_level=0.99, seed=123
        )

        # Exact seed reproducibility.
        self.assertEqual(res_90.to_dict(), res_90_repeat.to_dict())

        # In _make_synthetic_dataset, E_opt(i) = 0.65 * E_base(i) for EVERY item i!
        # Therefore, for EVERY bootstrap resample, ratio-of-means relative energy reduction
        # is identically 35.0% with zero bootstrap variance!
        self.assertAlmostEqual(res_90.rel_energy_reduction_pct.point, 35.0, places=10)
        self.assertAlmostEqual(res_90.rel_energy_reduction_pct.lower, 35.0, places=10)
        self.assertAlmostEqual(res_90.rel_energy_reduction_pct.upper, 35.0, places=10)

        # Absolute energy savings varies across items (since base_scale varies), so 99% CI
        # must monotonically enclose 90% CI and enclose the point estimate.
        self.assertLessEqual(res_99.abs_energy_savings_j.lower, res_90.abs_energy_savings_j.lower)
        self.assertGreaterEqual(res_99.abs_energy_savings_j.upper, res_90.abs_energy_savings_j.upper)
        self.assertLess(res_90.abs_energy_savings_j.lower, res_90.abs_energy_savings_j.point)
        self.assertGreater(res_90.abs_energy_savings_j.upper, res_90.abs_energy_savings_j.point)

    def test_without_replacement_subset_uniqueness_agreement_and_cost_accounting(self) -> None:
        n_items = 30
        ds = build_paired_dataset(_make_synthetic_dataset(n_items))
        # Verify RNG without-replacement k=N gives zero variance and 100% agreement!
        pts = run_calibration_sensitivity(
            dataset=ds,
            subset_sizes=[5, 15, 30],
            target_quality=0.90,
            num_draws=300,
            seed=99,
        )
        self.assertEqual(len(pts), 3)
        pt_5, pt_15, pt_30 = pts

        self.assertEqual(pt_30.reference_config_id, "cfg_opt")
        self.assertAlmostEqual(pt_30.selection_agreement_rate, 1.0, places=12)
        self.assertAlmostEqual(pt_30.target_attainment_rate, 1.0, places=12)
        self.assertIsNone(pt_30.complement_attainment_rate)
        self.assertAlmostEqual(pt_30.calibration_time_s.stddev, 0.0, places=10)
        self.assertAlmostEqual(pt_30.calibration_energy_j.stddev, 0.0, places=10)

        # Total full-grid calibration energy must equal sum over all 30 items and all 3 configs.
        expected_total_cal_energy = sum(
            m.effective_cal_energy_j()
            for cfg in ds.config_ids
            for m in ds.column(cfg)
        )
        expected_total_cal_time = sum(
            m.effective_cal_time_s()
            for cfg in ds.config_ids
            for m in ds.column(cfg)
        )
        self.assertAlmostEqual(
            pt_30.calibration_energy_j.mean, expected_total_cal_energy, places=9
        )
        self.assertAlmostEqual(
            pt_30.calibration_time_s.mean, expected_total_cal_time, places=9
        )

        # Expected calibration cost of without-replacement k-of-N draw scales linearly ~ (k / N).
        self.assertLess(pt_5.calibration_energy_j.mean, pt_15.calibration_energy_j.mean)
        self.assertLess(pt_15.calibration_energy_j.mean, pt_30.calibration_energy_j.mean)
        self.assertAlmostEqual(
            pt_15.calibration_energy_j.mean / expected_total_cal_energy, 0.5, delta=0.03
        )
        self.assertIsNotNone(pt_15.complement_attainment_rate)

        # Confirm without-replacement uniqueness: if a single item has spike energy 10000 J,
        # no draw of size k=3 can ever exceed 1 * 10000 + 2 * 1 = 10002 J (whereas with-replacement could hit 30000 J).
        spike_records: list[ItemMeasurement] = []
        for idx in range(10):
            cal_e = 10000.0 if idx == 0 else 1.0
            spike_records.append(
                ItemMeasurement(
                    item_id=f"s_{idx}",
                    config_id="only_cfg",
                    energy_j=1.0,
                    latency_s=1.0,
                    quality=1.0,
                    calibration_energy_j=cal_e,
                    calibration_time_s=1.0,
                )
            )
        spike_ds = build_paired_dataset(spike_records)
        spike_pt = run_calibration_sensitivity(
            spike_ds, subset_sizes=[3], target_quality=0.9, num_draws=400, seed=11
        )[0]
        self.assertLessEqual(spike_pt.calibration_energy_j.max_value, 10002.0 + 1e-9)

        # Confirm invalid subset sizes rejected.
        with self.assertRaises(ValueError):
            run_calibration_sensitivity(ds, subset_sizes=[0], target_quality=0.9)
        with self.assertRaises(ValueError):
            run_calibration_sensitivity(ds, subset_sizes=[31], target_quality=0.9)

    def test_unpaired_duplicate_and_invalid_inputs_rejected(self) -> None:
        # Missing ("i1", "b") -> unpaired matrix rejected.
        unpaired = [
            ItemMeasurement("i0", "a", energy_j=1.0, latency_s=1.0),
            ItemMeasurement("i0", "b", energy_j=2.0, latency_s=1.0),
            ItemMeasurement("i1", "a", energy_j=1.5, latency_s=1.0),
        ]
        with self.assertRaises(ValueError):
            build_paired_dataset(unpaired)

        # Duplicate ("i0", "a") rejected.
        duplicate = [
            ItemMeasurement("i0", "a", energy_j=1.0, latency_s=1.0),
            ItemMeasurement("i0", "a", energy_j=1.2, latency_s=1.0),
        ]
        with self.assertRaises(ValueError):
            build_paired_dataset(duplicate)

        # Negative energy or non-finite latency rejected.
        with self.assertRaises(ValueError):
            build_paired_dataset([ItemMeasurement("i0", "a", energy_j=-0.1, latency_s=1.0)])
        with self.assertRaises(ValueError):
            build_paired_dataset([ItemMeasurement("i0", "a", energy_j=1.0, latency_s=math.inf)])

        # Zero baseline energy rejected when computing ratio-of-means relative reduction.
        zero_base = build_paired_dataset(
            [
                ItemMeasurement("i0", "a", energy_j=1.0, latency_s=1.0),
                ItemMeasurement("i0", "b", energy_j=0.0, latency_s=1.0),
            ]
        )
        with self.assertRaises(ValueError):
            run_paired_bootstrap(zero_base, "a", "b", num_resamples=50)

    def test_loaders_and_cli_execution(self) -> None:
        records = _make_synthetic_dataset(12)
        jsonl_path = self.tmpdir / "paired.jsonl"
        csv_path = self.tmpdir / "paired.csv"
        out_json = self.tmpdir / "boot_out.json"
        out_md = self.tmpdir / "boot_out.md"

        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for rec in records:
                handle.write(
                    json.dumps(
                        {
                            "item_id": rec.item_id,
                            "config_id": rec.config_id,
                            "energy_j": rec.energy_j,
                            "latency_s": rec.latency_s,
                            "quality": rec.quality,
                            "calibration_time_s": rec.calibration_time_s,
                            "calibration_energy_j": rec.calibration_energy_j,
                        }
                    )
                    + "\n"
                )

        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["item_id", "config_id", "energy_j", "latency_s", "quality"],
            )
            writer.writeheader()
            for rec in records:
                writer.writerow(
                    {
                        "item_id": rec.item_id,
                        "config_id": rec.config_id,
                        "energy_j": rec.energy_j,
                        "latency_s": rec.latency_s,
                        "quality": rec.quality,
                    }
                )

        ds_csv = load_paired_dataset(csv_path)
        self.assertEqual(ds_csv.num_items(), 12)
        self.assertEqual(ds_csv.num_configs(), 3)

        rc = main(
            [
                "--input",
                str(jsonl_path),
                "--candidate",
                "cfg_opt",
                "--baseline",
                "cfg_base",
                "--resamples",
                "300",
                "--confidence",
                "0.95",
                "--seed",
                "21",
                "--subset-sizes",
                "4,8,12",
                "--subset-draws",
                "100",
                "--target-quality",
                "0.80",
                "--output",
                str(out_json),
                "--markdown-output",
                str(out_md),
            ]
        )
        self.assertEqual(rc, 0)
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            payload["bootstrap"]["rel_energy_reduction_pct"]["point"], 35.0, places=8
        )
        self.assertEqual(len(payload["calibration_sensitivity"]), 3)
        self.assertTrue(out_md.exists())

        # Exclusive open collision must return exit code 1.
        rc_dup = main(
            [
                "--input",
                str(jsonl_path),
                "--candidate",
                "cfg_opt",
                "--baseline",
                "cfg_base",
                "--output",
                str(out_json),
            ]
        )
        self.assertEqual(rc_dup, 1)


if __name__ == "__main__":
    unittest.main()
