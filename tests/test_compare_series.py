"""Unit tests for tools.compare_series."""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.compare_series import (
    SeriesComparisonError,
    bootstrap_comparison,
    cliffs_delta,
    compare_series,
    describe_series,
    detect_outliers,
    hodges_lehmann_shift,
    kolmogorov_smirnov_2sample,
    load_series_file,
    main,
    mann_whitney_u,
    quantile,
    wilcoxon_signed_rank,
)


class CompareSeriesTest(unittest.TestCase):
    """Analytical, invariance, tie-handling, bootstrap, outlier, and CLI tests."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.tmp_dir = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_exact_quantiles_and_hodges_lehmann_shift_properties(self) -> None:
        xs = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertAlmostEqual(quantile(xs, 0.0), 10.0)
        self.assertAlmostEqual(quantile(xs, 0.25), 20.0)
        self.assertAlmostEqual(quantile(xs, 0.50), 30.0)
        self.assertAlmostEqual(quantile(xs, 0.75), 40.0)
        self.assertAlmostEqual(quantile(xs, 1.0), 50.0)
        self.assertAlmostEqual(quantile([1.0, 4.0], 0.5), 2.5)
        self.assertAlmostEqual(quantile([42.0], 0.99), 42.0)

        base = [12.0, 15.0, 19.0, 24.0, 31.0, 40.0]
        cand_shifted = [b + 7.5 for b in base]
        self.assertAlmostEqual(hodges_lehmann_shift(base, cand_shifted, paired=True), 7.5)

        self.assertAlmostEqual(
            hodges_lehmann_shift([10.0, 20.0, 30.0], [12.0, 24.0, 40.0], paired=True), 5.0
        )
        self.assertAlmostEqual(
            hodges_lehmann_shift([1.0, 2.0, 4.0], [5.0, 7.0], paired=False), 3.5
        )
        self.assertAlmostEqual(
            hodges_lehmann_shift([5.0, 7.0], [1.0, 2.0, 4.0], paired=False), -3.5
        )

    def test_wilcoxon_signed_rank_analytical_ties_and_zeros(self) -> None:
        base = [10.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        cand = [10.0, 12.0, 22.0, 25.0, 48.0, 60.0]
        res = wilcoxon_signed_rank(base, cand)
        self.assertEqual(res["n_pairs"], 6)
        self.assertEqual(res["n_zero_diffs"], 1)
        self.assertEqual(res["n_effective"], 5)
        self.assertAlmostEqual(res["w_plus"], 12.0)
        self.assertAlmostEqual(res["w_minus"], 3.0)
        self.assertAlmostEqual(res["statistic"], 3.0)
        expected_z = (12.0 - 7.5) / math.sqrt(13.625)
        self.assertAlmostEqual(res["z_score"], expected_z, places=9)

        identical = wilcoxon_signed_rank(base, base)
        self.assertEqual(identical["n_effective"], 0)
        self.assertAlmostEqual(identical["p_value"], 1.0)

        b16 = [100.0 + i for i in range(16)]
        c16 = [80.0 + i for i in range(16)]
        sig = wilcoxon_signed_rank(b16, c16)
        self.assertAlmostEqual(sig["w_plus"], 0.0)
        self.assertAlmostEqual(sig["w_minus"], 16 * 17 / 2.0)
        self.assertLess(sig["z_score"], -3.0)
        self.assertLess(sig["p_value"], 0.001)

    def test_mann_whitney_u_cliffs_delta_and_kolmogorov_smirnov(self) -> None:
        base = [10.0, 20.0, 30.0, 40.0]
        cand = [1.0, 2.0, 3.0, 4.0, 5.0]

        mwu = mann_whitney_u(base, cand)
        self.assertAlmostEqual(mwu["u_baseline"], 20.0)
        self.assertAlmostEqual(mwu["u_candidate"], 0.0)
        self.assertAlmostEqual(mwu["statistic"], 0.0)
        self.assertLess(mwu["p_value"], 0.05)

        cd = cliffs_delta(base, cand)
        self.assertAlmostEqual(cd["delta"], -1.0)
        self.assertEqual(cd["magnitude"], "large")
        self.assertAlmostEqual(cliffs_delta(cand, base)["delta"], 1.0)

        cd_neg = cliffs_delta([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(cd_neg["delta"], 0.0)
        self.assertEqual(cd_neg["magnitude"], "negligible")
        self.assertEqual(cd_neg["pairs_tied"], 4)

        mwu_exp = mann_whitney_u(
            [math.exp(x / 10.0) for x in base],
            [math.exp(y / 10.0) for y in cand],
        )
        self.assertAlmostEqual(mwu["u_candidate"], mwu_exp["u_candidate"])
        self.assertAlmostEqual(mwu["p_value"], mwu_exp["p_value"])

        mwu_all_tied = mann_whitney_u([5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0])
        self.assertAlmostEqual(mwu_all_tied["p_value"], 1.0)
        self.assertAlmostEqual(mwu_all_tied["z_score"], 0.0)

        ks_disjoint = kolmogorov_smirnov_2sample(base, cand)
        self.assertAlmostEqual(ks_disjoint["d_statistic"], 1.0)
        self.assertLess(ks_disjoint["p_value"], 0.02)

        ks_same = kolmogorov_smirnov_2sample([1.0, 2.0, 2.0, 5.0], [1.0, 2.0, 2.0, 5.0])
        self.assertAlmostEqual(ks_same["d_statistic"], 0.0)
        self.assertAlmostEqual(ks_same["p_value"], 1.0)

        ks_half = kolmogorov_smirnov_2sample([1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0])
        self.assertAlmostEqual(ks_half["d_statistic"], 0.5)
        self.assertAlmostEqual(ks_half["d_plus"], 0.5)
        self.assertAlmostEqual(ks_half["d_minus"], 0.0)

    def test_bootstrap_ci_seed_determinism_unpaired_and_speedup_ratio(self) -> None:
        base = [100.0, 104.0, 98.0, 102.0, 106.0, 99.0, 101.0, 103.0, 97.0, 105.0]
        cand = [50.0, 52.0, 49.0, 51.0, 53.0, 49.5, 50.5, 51.5, 48.5, 52.5]

        b1 = bootstrap_comparison(
            base, cand, paired=True, confidence=0.95, resamples=1000, seed=12345
        )
        b2 = bootstrap_comparison(
            base, cand, paired=True, confidence=0.95, resamples=1000, seed=12345
        )
        self.assertEqual(b1, b2)

        speedup = b1["mean_speedup_ratio"]
        self.assertIsNotNone(speedup)
        self.assertAlmostEqual(speedup["estimate"], 2.0, places=9)
        self.assertAlmostEqual(speedup["ci_lower"], 2.0, places=9)
        self.assertAlmostEqual(speedup["ci_upper"], 2.0, places=9)

        b_unpaired = bootstrap_comparison(
            base, cand, paired=False, confidence=0.95, resamples=1000, seed=12345
        )
        up_sp = b_unpaired["mean_speedup_ratio"]
        self.assertLess(up_sp["ci_lower"], 2.0)
        self.assertGreater(up_sp["ci_upper"], 2.0)

        mean_diff = b1["mean_diff"]
        self.assertLess(mean_diff["ci_upper"], -49.0)
        self.assertLessEqual(mean_diff["ci_lower"], mean_diff["estimate"])
        self.assertGreaterEqual(mean_diff["ci_upper"], mean_diff["estimate"])

        b_signed = bootstrap_comparison(
            [-5.0, -2.0, 3.0], [-4.0, -1.0, 2.0], resamples=200, seed=7
        )
        self.assertIsNone(b_signed["mean_speedup_ratio"])
        self.assertIsNone(b_signed["median_speedup_ratio"])

    def test_tukey_iqr_and_mad_outlier_detection_and_validation(self) -> None:
        clean_plus_spikes = [10.0, 10.2, 9.9, 10.1, 10.0, 9.8, 10.3, 10.0, 45.0, -25.0]
        out = detect_outliers(clean_plus_spikes)
        self.assertEqual(out["tukey_outlier_indices"], [8, 9])
        self.assertEqual(out["tukey_outlier_values"], [45.0, -25.0])
        self.assertEqual(out["mad_outlier_indices"], [8, 9])

        zero_mad_series = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 100.0]
        desc = describe_series(zero_mad_series)
        serialized = json.dumps(desc, allow_nan=False)
        self.assertTrue(len(serialized) > 0)
        self.assertIn(7, desc["outliers"]["tukey_outlier_indices"])
        self.assertIn(7, desc["outliers"]["mad_outlier_indices"])

        with self.assertRaises(SeriesComparisonError):
            describe_series([])
        with self.assertRaises(SeriesComparisonError):
            describe_series([1.0, float("nan"), 2.0])
        with self.assertRaises(SeriesComparisonError):
            describe_series([1.0, True, 2.0])  # type: ignore[list-item]
        with self.assertRaises(SeriesComparisonError):
            compare_series([1.0, 2.0, 3.0], [1.0, 2.0], paired=True)

    def test_cli_csv_grouped_json_strict_serialization_and_nondestructive_output(self) -> None:
        base_csv = self.tmp_dir / "baseline.csv"
        cand_csv = self.tmp_dir / "candidate.csv"
        base_csv.write_text(
            "workload,latency_ms\nio_bound,40.0\nio_bound,42.0\nio_bound,41.0\nio_bound,43.0\n"
            "compute,100.0\ncompute,102.0\ncompute,98.0\ncompute,101.0\n",
            encoding="utf-8",
        )
        cand_csv.write_text(
            "workload,latency_ms\nio_bound,20.0\nio_bound,21.0\nio_bound,20.5\nio_bound,21.5\n"
            "compute,80.0\ncompute,81.0\ncompute,79.0\ncompute,80.5\n",
            encoding="utf-8",
        )
        out_json = self.tmp_dir / "report.json"

        rc = main([
            "--baseline", str(base_csv), "--candidate", str(cand_csv),
            "--paired", "--column", "latency_ms", "--group-by", "workload",
            "--confidence", "0.90", "--resamples", "400", "--seed", "99",
            "--output", str(out_json),
        ])
        self.assertEqual(rc, 0)
        self.assertTrue(out_json.is_file())
        report = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(report["group_by"], "workload")
        self.assertEqual(sorted(report["groups"].keys()), ["compute", "io_bound"])
        self.assertAlmostEqual(
            report["groups"]["io_bound"]["bootstrap"]["mean_speedup_ratio"]["estimate"],
            2.0,
            places=6,
        )
        self.assertIsNotNone(report["groups"]["compute"]["wilcoxon_signed_rank"])

        stderr_buf = io.StringIO()
        with mock.patch("sys.stderr", stderr_buf):
            rc_dup = main([
                "--baseline", str(base_csv), "--candidate", str(cand_csv),
                "--column", "latency_ms", "--output", str(out_json),
            ])
        self.assertEqual(rc_dup, 2)
        self.assertIn("Refusing to overwrite", stderr_buf.getvalue())

        obj_json = self.tmp_dir / "records.json"
        obj_json.write_text(
            json.dumps([{"grp": "g1", "ms": 12.5}, {"grp": "g1", "ms": 13.5}]),
            encoding="utf-8",
        )
        self.assertEqual(load_series_file(obj_json, column="ms", group_by="grp"), {"g1": [12.5, 13.5]})

        base_j = self.tmp_dir / "b.json"
        cand_j = self.tmp_dir / "c.json"
        base_j.write_text('{"values": [15.0, 16.0, 14.5, 15.5, 16.5]}', encoding="utf-8")
        cand_j.write_text("[10.0, 11.0, 9.5, 10.5, 11.5]", encoding="utf-8")
        stdout_buf = io.StringIO()
        with mock.patch("sys.stdout", stdout_buf):
            rc_stdout = main([
                "--baseline", str(base_j), "--candidate", str(cand_j),
                "--resamples", "250", "--seed", "7",
            ])
        self.assertEqual(rc_stdout, 0)
        parsed_stdout = json.loads(stdout_buf.getvalue())
        self.assertAlmostEqual(parsed_stdout["hodges_lehmann_shift"], -5.0)
        self.assertIsNone(parsed_stdout["wilcoxon_signed_rank"])

    def test_mann_whitney_tie_variance_correction_and_unmatched_groups_cli(self) -> None:
        # Exact tie-correction formula check: base=[2, 2, 5], cand=[2, 5, 8] -> N=6
        # Pooled=[2, 2, 2, 5, 5, 8] -> tie groups t=3 (27-3=24) and t=2 (8-2=6), tie_sum=30
        # Ranks: 2->2.0, 5->4.5, 8->6.0; R_base = 2+2+4.5 = 8.5 -> u_base = 8.5 - 6 = 2.5, u_cand = 6.5
        # Variance = (9/12) * (7 - 30 / 30) = 0.75 * 6 = 4.5 -> z_score = (6.5 - 4.5) / sqrt(4.5)
        mwu = mann_whitney_u([2.0, 2.0, 5.0], [2.0, 5.0, 8.0])
        self.assertAlmostEqual(mwu["u_baseline"], 2.5)
        self.assertAlmostEqual(mwu["u_candidate"], 6.5)
        self.assertAlmostEqual(mwu["z_score"], 2.0 / math.sqrt(4.5), places=9)

        # CLI reports unmatched baseline/candidate group strata cleanly without failing.
        b_csv = self.tmp_dir / "b_strata.csv"
        c_csv = self.tmp_dir / "c_strata.csv"
        b_csv.write_text("suite,val\nshared,10\nshared,12\nonly_base,99\n", encoding="utf-8")
        c_csv.write_text("suite,val\nshared,8\nshared,9\nonly_cand,77\n", encoding="utf-8")
        out_buf = io.StringIO()
        with mock.patch("sys.stdout", out_buf):
            rc = main([
                "--baseline", str(b_csv), "--candidate", str(c_csv),
                "--column", "val", "--group-by", "suite", "--resamples", "100",
            ])
        self.assertEqual(rc, 0)
        doc = json.loads(out_buf.getvalue())
        self.assertEqual(list(doc["groups"].keys()), ["shared"])
        self.assertEqual(doc["unmatched_baseline_groups"], ["only_base"])
        self.assertEqual(doc["unmatched_candidate_groups"], ["only_cand"])

        # Disjoint group sets return exit code 2.
        c_disjoint = self.tmp_dir / "c_disjoint.csv"
        c_disjoint.write_text("suite,val\nother,8\n", encoding="utf-8")
        err_buf = io.StringIO()
        with mock.patch("sys.stderr", err_buf):
            rc_err = main([
                "--baseline", str(b_csv), "--candidate", str(c_disjoint),
                "--column", "val", "--group-by", "suite",
            ])
        self.assertEqual(rc_err, 2)


if __name__ == "__main__":
    unittest.main()
