"""Unit tests for Cartesian completeness and repeat-determinism auditor."""

import csv
import hashlib
import json
import math
import os
import pathlib
import tempfile
import unittest

from tools.completeness_audit import (
    CELL_FAILED,
    CELL_INVALID,
    CELL_MISSING,
    CELL_NOT_RUN,
    ERR_DUPLICATE_CELL,
    ERR_INVALID_ACCURACY_FLAG,
    ERR_INVALID_STATUS,
    ERR_INVALID_TOKEN_COUNT,
    ERR_MALFORMED_JSON,
    ERR_MISSING_CELL,
    ERR_MISSING_TIMESTAMP,
    ERR_NEGATIVE_ENERGY,
    ERR_NEGATIVE_LATENCY,
    ERR_NON_FINITE_ENERGY,
    ERR_NON_FINITE_LATENCY,
    ERR_OUTPUT_HASH_MISMATCH,
    ERR_TIMESTAMP_ORDER,
    ERR_UNSCHEDULED_CELL,
    audit_jsonl_lines,
    build_manifest,
    compute_sample_stats,
    format_markdown_summary,
    main,
)


def _make_valid_record(
    request_id: str,
    config_id: str,
    repeat_id: int,
    latency_s: float = 1.25,
    energy_j: float = 4.50,
    output_tokens: int = 30,
    output_text: str = "deterministic final answer 42",
    passed: bool = True,
    start_ts: float = 1000.0,
) -> dict[str, object]:
    digest = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    return {
        "request_id": request_id, "config_id": config_id, "repeat_id": repeat_id,
        "status": "valid", "start_ts": start_ts, "end_ts": start_ts + latency_s,
        "latency_s": latency_s, "energy_j": energy_j, "output_tokens": output_tokens,
        "output_text": output_text, "output_sha256": digest, "passed": passed,
    }


class TestCompletenessAudit(unittest.TestCase):
    """Test suite covering grid completeness, error codes, CV math, and CLI."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.tmpdir = pathlib.Path(self._tmp.name)
        self.manifest = build_manifest({
            "manifest_id": "grid_3rep",
            "request_ids": ["req_01", "req_02", "req_03"],
            "config_ids": ["cfg_fast", "cfg_balanced"],
            "num_repeats": 3,
        })

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_complete_three_repeat_grid_passes_and_is_deterministic(self) -> None:
        lines: list[str] = []
        for req_idx, req_id in enumerate(self.manifest.request_ids):
            for cfg_idx, cfg_id in enumerate(self.manifest.config_ids):
                for rep_id in self.manifest.repeat_ids:
                    rec = _make_valid_record(
                        request_id=req_id, config_id=cfg_id, repeat_id=rep_id,
                        latency_s=1.0 + 0.2 * cfg_idx + 0.05 * rep_id,
                        energy_j=3.0 + 0.5 * cfg_idx + 0.15 * rep_id,
                        output_tokens=25 + req_idx * 5,
                        output_text=f"canonical_output::{req_id}::{cfg_id}",
                        passed=(req_idx % 2 == 0),
                    )
                    lines.append(json.dumps(rec))

        report = audit_jsonl_lines(self.manifest, lines)
        self.assertEqual(report.expected_cells, 18)
        self.assertEqual(report.observed_records, 18)
        self.assertEqual(report.valid_cells, 18)
        self.assertEqual(report.missing_cells, 0)
        self.assertEqual(report.invalid_cells, 0)
        self.assertEqual(report.failed_cells, 0)
        self.assertEqual(report.not_run_cells, 0)
        self.assertEqual(report.duplicate_count, 0)
        self.assertEqual(report.unscheduled_count, 0)
        self.assertTrue(report.is_complete)
        self.assertTrue(report.all_repeats_deterministic)
        self.assertEqual(len(report.nondeterministic_groups), 0)

        first_group = report.group_audits[0]
        self.assertEqual(first_group.valid_repeats, 3)
        self.assertTrue(first_group.sha256_equivalent)
        self.assertTrue(first_group.exact_text_equivalent)
        self.assertTrue(first_group.accuracy_equivalent)
        self.assertAlmostEqual(first_group.pass_rate or 0.0, 1.0)

        md_text = format_markdown_summary(report)
        self.assertIn("grid_3rep", md_text)
        self.assertIn("cfg_fast", md_text)
        self.assertIn("9/9", md_text)

    def test_missing_duplicate_and_unscheduled_cells_flagged(self) -> None:
        lines: list[str] = []
        for req_id in self.manifest.request_ids:
            for cfg_id in self.manifest.config_ids:
                for rep_id in self.manifest.repeat_ids:
                    if (req_id, cfg_id, rep_id) == ("req_03", "cfg_balanced", 2):
                        continue
                    lines.append(json.dumps(_make_valid_record(req_id, cfg_id, rep_id)))

        lines.append(json.dumps(_make_valid_record("req_01", "cfg_fast", 0, latency_s=1.4)))
        lines.append(json.dumps(_make_valid_record("req_99", "cfg_fast", 0)))
        lines.append(json.dumps(_make_valid_record("req_01", "cfg_fast", 9)))
        lines.append("{malformed json line")

        report = audit_jsonl_lines(self.manifest, lines)
        self.assertFalse(report.is_complete)
        self.assertEqual(report.missing_cells, 1)
        self.assertEqual(report.invalid_cells, 1)
        self.assertEqual(report.valid_cells, 16)
        self.assertEqual(report.duplicate_count, 1)
        self.assertEqual(report.unscheduled_count, 2)
        self.assertEqual(report.malformed_line_count, 1)

        violation_codes = [v.error_code for v in report.violations]
        self.assertIn(ERR_DUPLICATE_CELL, violation_codes)
        self.assertIn(ERR_UNSCHEDULED_CELL, violation_codes)
        self.assertIn(ERR_MALFORMED_JSON, violation_codes)

        by_key = {(c.request_id, c.config_id, c.repeat_id): c for c in report.cell_results}
        missing_cell = by_key[("req_03", "cfg_balanced", 2)]
        self.assertEqual(missing_cell.state, CELL_MISSING)
        self.assertEqual(missing_cell.error_codes, (ERR_MISSING_CELL,))

        dup_cell = by_key[("req_01", "cfg_fast", 0)]
        self.assertEqual(dup_cell.state, CELL_INVALID)
        self.assertIn(ERR_DUPLICATE_CELL, dup_cell.error_codes)
        self.assertEqual(len(dup_cell.line_numbers), 2)

    def test_structured_error_codes_for_invalid_failed_and_not_run_cells(self) -> None:
        small_manifest = build_manifest({
            "request_ids": ["r1", "r2", "r3"], "config_ids": ["c1"], "repeat_ids": [0, 1, 2],
        })
        r_failed = {"request_id": "r1", "config_id": "c1", "repeat_id": 0, "status": "failed"}
        r_not_run = {"request_id": "r1", "config_id": "c1", "repeat_id": 1, "status": "not_run"}
        r_bad_status = {"request_id": "r1", "config_id": "c1", "repeat_id": 2, "status": "mystery"}

        r_neg_lat = _make_valid_record("r2", "c1", 0)
        r_neg_lat["latency_s"] = -0.5
        r_nan_eng = _make_valid_record("r2", "c1", 1)
        r_nan_eng["energy_j"] = float("nan")
        r_neg_eng = _make_valid_record("r2", "c1", 2)
        r_neg_eng["energy_j"] = -1.2
        r_bad_ts = _make_valid_record("r3", "c1", 0)
        r_bad_ts["end_ts"] = 999.0
        r_missing_ts = _make_valid_record("r3", "c1", 1)
        del r_missing_ts["start_ts"]

        r_bad_multi = _make_valid_record("r3", "c1", 2)
        r_bad_multi["output_sha256"] = "0" * 64
        r_bad_multi["output_tokens"] = 0
        r_bad_multi["passed"] = "yes"
        r_bad_multi["latency_s"] = float("inf")

        records = [
            r_failed, r_not_run, r_bad_status, r_neg_lat, r_nan_eng,
            r_neg_eng, r_bad_ts, r_missing_ts, r_bad_multi,
        ]
        report = audit_jsonl_lines(small_manifest, [json.dumps(r, allow_nan=True) for r in records])
        self.assertEqual(report.failed_cells, 1)
        self.assertEqual(report.not_run_cells, 1)
        self.assertEqual(report.invalid_cells, 7)
        self.assertFalse(report.is_complete)

        by_key = {(c.request_id, c.config_id, c.repeat_id): c for c in report.cell_results}
        self.assertEqual(by_key[("r1", "c1", 0)].state, CELL_FAILED)
        self.assertEqual(by_key[("r1", "c1", 1)].state, CELL_NOT_RUN)
        self.assertIn(ERR_INVALID_STATUS, by_key[("r1", "c1", 2)].error_codes)
        self.assertIn(ERR_NEGATIVE_LATENCY, by_key[("r2", "c1", 0)].error_codes)
        self.assertIn(ERR_NON_FINITE_ENERGY, by_key[("r2", "c1", 1)].error_codes)
        self.assertIn(ERR_NEGATIVE_ENERGY, by_key[("r2", "c1", 2)].error_codes)
        self.assertIn(ERR_TIMESTAMP_ORDER, by_key[("r3", "c1", 0)].error_codes)
        self.assertIn(ERR_MISSING_TIMESTAMP, by_key[("r3", "c1", 1)].error_codes)

        multi_err = by_key[("r3", "c1", 2)].error_codes
        self.assertIn(ERR_OUTPUT_HASH_MISMATCH, multi_err)
        self.assertIn(ERR_INVALID_TOKEN_COUNT, multi_err)
        self.assertIn(ERR_INVALID_ACCURACY_FLAG, multi_err)
        self.assertIn(ERR_NON_FINITE_LATENCY, multi_err)

    def test_nondeterminism_detection_across_repeats(self) -> None:
        manifest = build_manifest({
            "request_ids": ["q_det", "q_text_drift", "q_acc_drift"],
            "config_ids": ["cfg_a"], "num_repeats": 3,
        })
        lines: list[str] = []
        for rep in range(3):
            lines.append(json.dumps(_make_valid_record("q_det", "cfg_a", rep, output_text="same", passed=True)))
            drifted_text = "same" if rep < 2 else "divergent token tail"
            lines.append(json.dumps(_make_valid_record("q_text_drift", "cfg_a", rep, output_text=drifted_text, passed=True)))
            lines.append(json.dumps(_make_valid_record("q_acc_drift", "cfg_a", rep, output_text="identical_str", passed=(rep == 0))))

        report = audit_jsonl_lines(manifest, lines)
        self.assertTrue(report.is_complete)
        self.assertFalse(report.all_repeats_deterministic)
        self.assertEqual(report.nondeterministic_groups, (("q_text_drift", "cfg_a"), ("q_acc_drift", "cfg_a")))

        groups = {(g.request_id, g.config_id): g for g in report.group_audits}
        self.assertTrue(groups[("q_det", "cfg_a")].is_deterministic)
        text_drift_grp = groups[("q_text_drift", "cfg_a")]
        self.assertFalse(text_drift_grp.sha256_equivalent)
        self.assertFalse(text_drift_grp.exact_text_equivalent)
        self.assertTrue(text_drift_grp.accuracy_equivalent)
        self.assertEqual(text_drift_grp.distinct_text_count, 2)
        self.assertEqual(len(text_drift_grp.distinct_sha256s), 2)

        acc_drift_grp = groups[("q_acc_drift", "cfg_a")]
        self.assertTrue(acc_drift_grp.sha256_equivalent)
        self.assertTrue(acc_drift_grp.exact_text_equivalent)
        self.assertFalse(acc_drift_grp.accuracy_equivalent)
        self.assertAlmostEqual(acc_drift_grp.pass_rate or 0.0, 1.0 / 3.0)

    def test_unbiased_sample_stddev_and_cv_math(self) -> None:
        stats = compute_sample_stats([2.0, 4.0, 6.0])
        self.assertEqual(stats.count, 3)
        self.assertAlmostEqual(stats.mean or 0.0, 4.0, places=12)
        self.assertAlmostEqual(stats.stddev or 0.0, 2.0, places=12)
        self.assertAlmostEqual(stats.cv or 0.0, 0.5, places=12)
        self.assertAlmostEqual(stats.min_value or 0.0, 2.0, places=12)
        self.assertAlmostEqual(stats.max_value or 0.0, 6.0, places=12)

        single = compute_sample_stats([5.5])
        self.assertEqual(single.count, 1)
        self.assertIsNone(single.stddev)
        self.assertIsNone(single.cv)
        self.assertIsNone(compute_sample_stats([]).mean)

        manifest = build_manifest({"request_ids": ["r1"], "config_ids": ["c1"], "repeat_ids": [0, 1, 2]})
        lines = [
            json.dumps(_make_valid_record("r1", "c1", 0, latency_s=1.0, energy_j=10.0, output_tokens=10)),
            json.dumps(_make_valid_record("r1", "c1", 1, latency_s=2.0, energy_j=20.0, output_tokens=10)),
            json.dumps(_make_valid_record("r1", "c1", 2, latency_s=3.0, energy_j=30.0, output_tokens=10)),
        ]
        report = audit_jsonl_lines(manifest, lines)
        grp = report.group_audits[0]
        self.assertAlmostEqual(grp.energy_per_token_stats.mean or 0.0, 2.0, places=12)
        self.assertAlmostEqual(grp.energy_per_token_stats.stddev or 0.0, 1.0, places=12)
        self.assertAlmostEqual(grp.energy_per_token_stats.cv or 0.0, 0.5, places=12)
        self.assertAlmostEqual(report.config_summaries[0].mean_repeat_ept_cv or 0.0, 0.5, places=12)
        with self.assertRaises(ValueError):
            compute_sample_stats([1.0, math.nan])

    def test_cli_exclusive_output_and_exit_codes(self) -> None:
        manifest_path = self.tmpdir / "manifest.json"
        records_path = self.tmpdir / "records.jsonl"
        out_json = self.tmpdir / "audit_out.json"
        out_csv = self.tmpdir / "audit_groups.csv"
        out_md = self.tmpdir / "audit_summary.md"

        manifest_path.write_text(json.dumps({
            "manifest_id": "cli_grid", "request_ids": ["req_a", "req_b"],
            "config_ids": ["cfg_1"], "repeat_ids": [0, 1, 2],
        }), encoding="utf-8")

        records = [
            _make_valid_record("req_a", "cfg_1", rep, latency_s=1.0 + 0.1 * rep, energy_j=5.0 + rep)
            for rep in range(3)
        ] + [
            _make_valid_record("req_b", "cfg_1", rep, latency_s=2.0 + 0.2 * rep, energy_j=8.0 + rep)
            for rep in range(3)
        ]
        records_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

        rc = main([
            "--manifest", str(manifest_path), "--records", str(records_path),
            "--output", str(out_json), "--csv-output", str(out_csv),
            "--markdown-output", str(out_md), "--strict-determinism",
        ])
        self.assertEqual(rc, 0)
        parsed_out = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertTrue(parsed_out["summary"]["is_complete"])
        self.assertEqual(parsed_out["summary"]["valid_cells"], 6)

        with open(out_csv, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertTrue(out_md.exists())

        rc_collision = main(["--manifest", str(manifest_path), "--records", str(records_path), "--output", str(out_json)])
        self.assertEqual(rc_collision, 1)


if __name__ == "__main__":
    unittest.main()
