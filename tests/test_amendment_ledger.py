"""Unit tests for hash-chained execution scope and rejected-attempt ledger."""

import dataclasses
import json
import os
import pathlib
import tempfile
import unittest

from tools.amendment_ledger import (
    DISPOSITION_ACCEPTED,
    DISPOSITION_PENDING,
    DISPOSITION_REJECTED,
    ERR_AMENDMENT_CAP_EXCEEDED,
    ERR_BROKEN_HASH_LINK,
    ERR_CALIBRATION_SCOPE_MUTATED,
    ERR_CANDIDATE_SET_MUTATED,
    ERR_EVALUATION_SCOPE_MUTATED,
    ERR_INVALID_VERSION_SEQUENCE,
    ERR_INVOCATION_BUDGET_MISMATCH,
    ERR_RECORD_DIGEST_TAMPERED,
    ERR_REJECTED_HAS_ENERGY,
    ERR_REJECTED_MARKED_ELIGIBLE,
    ERR_SCOPE_DIGEST_TAMPERED,
    REASON_BATTERY_DROP,
    REASON_MISSING_END_TS,
    REASON_THERMAL_ABORT,
    ScopeEntry,
    build_invocation_record,
    create_amendment,
    create_genesis_scope,
    main,
    record_hardware_telemetry,
    serialize_ledger_dict,
    verify_ledger,
)


def _build_valid_three_step_ledger() -> tuple[list[ScopeEntry], list]:
    """Construct a valid v0 -> v1 -> v2 chain with rejected attempts and pending slots."""
    v0 = create_genesis_scope(
        ledger_id="mobile_bench_run_01",
        calibration_ids=["cal_01", "cal_02"],
        evaluation_ids=["eval_01", "eval_02", "eval_03"],
        candidates=["cfg_low", "cfg_high"],
        max_invocations=6,
        reason="Initial 6-slot execution budget",
    )

    records = [
        # 4 accepted invocations under v0.
        record_hardware_telemetry(
            "att_01", "v0", "cal_01", "cfg_low", 100.0, 101.2, 3.8, 85.0, 84.0, 36.0
        ),
        record_hardware_telemetry(
            "att_02", "v0", "cal_02", "cfg_high", 105.0, 107.0, 7.2, 84.0, 82.5, 37.5
        ),
        record_hardware_telemetry(
            "att_03", "v0", "eval_01", "cfg_low", 110.0, 111.5, 4.1, 82.5, 81.0, 38.0
        ),
        record_hardware_telemetry(
            "att_04", "v0", "eval_02", "cfg_high", 115.0, 117.1, 7.5, 81.0, 79.0, 39.0
        ),
        # 2 rejected hardware attempts under v0 (battery drop < 20% and missing end_ts).
        record_hardware_telemetry(
            "att_05", "v0", "eval_03", "cfg_low", 120.0, 121.4, 4.3, 21.0, 18.5, 40.0
        ),
        record_hardware_telemetry(
            "att_06", "v0", "eval_03", "cfg_high", 125.0, None, 7.9, 78.0, 77.0, 40.5
        ),
    ]

    # v1 expands max_invocations from 6 -> 8 to replace the 2 rejected hardware attempts.
    v1 = create_amendment(
        v0,
        max_invocations=8,
        reason="Authorize 2 replacement slots after battery drop and missing timestamp",
    )
    records.extend(
        [
            # 1 accepted replacement, 1 thermal abort under v1.
            record_hardware_telemetry(
                "att_07", "v1", "eval_03", "cfg_low", 200.0, 201.3, 4.0, 95.0, 94.0, 35.0
            ),
            record_hardware_telemetry(
                "att_08", "v1", "eval_03", "cfg_high", 205.0, 207.2, 8.1, 94.0, 92.0, 48.5
            ),
        ]
    )

    # v2 expands max_invocations from 8 -> 10 (1 replacement + 1 scheduled pending slot).
    v2 = create_amendment(
        v1,
        max_invocations=10,
        reason="Authorize thermal-abort replacement and 1 pending calibration check",
    )
    records.extend(
        [
            record_hardware_telemetry(
                "att_09", "v2", "eval_03", "cfg_high", 300.0, 302.0, 7.6, 90.0, 88.5, 36.5
            ),
            build_invocation_record(
                attempt_id="att_10",
                amendment_version="v2",
                item_id="cal_01",
                candidate_id="cfg_high",
                disposition=DISPOSITION_PENDING,
            ),
        ]
    )
    return [v0, v1, v2], records


class TestAmendmentLedger(unittest.TestCase):
    """Test multi-step hash-chained amendments, scope immutability, and budget accounting."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.tmpdir = pathlib.Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_valid_three_step_chain_and_rejected_attempt_proofs(self) -> None:
        amendments, records = _build_valid_three_step_ledger()

        # Verify rejected attempts explicitly stripped energy_j=None and eligible_measurement=False.
        rejected = [r for r in records if r.disposition == DISPOSITION_REJECTED]
        self.assertEqual(len(rejected), 3)
        for rec in rejected:
            self.assertFalse(rec.eligible_measurement)
            self.assertIsNone(rec.energy_j)
            self.assertIsNotNone(rec.rejection_reason)

        report = verify_ledger(amendments, records)
        self.assertTrue(report.is_valid)
        self.assertTrue(report.hash_chain_intact)
        self.assertTrue(report.frozen_scope_preserved)
        self.assertTrue(report.rejected_zero_contamination_verified)
        self.assertTrue(report.budget_conserved)
        self.assertEqual(report.chain_length, 3)
        self.assertEqual(report.terminal_version, "v2")
        self.assertEqual(report.accepted_count, 6)
        self.assertEqual(report.rejected_count, 3)
        self.assertEqual(report.pending_count, 1)
        self.assertEqual(report.total_accounted, 10)
        self.assertEqual(report.max_invocations, 10)
        self.assertEqual(
            report.rejection_breakdown,
            {
                REASON_BATTERY_DROP: 1,
                REASON_MISSING_END_TS: 1,
                REASON_THERMAL_ABORT: 1,
            },
        )
        self.assertEqual(len(report.defects), 0)

    def test_scope_mutating_amendments_rejected(self) -> None:
        amendments, records = _build_valid_three_step_ledger()
        v0, v1, v2 = amendments

        # 1. Mutate evaluation_ids on v1 (recomputing hashes so only scope rule triggers).
        v1_bad_eval_draft = dataclasses.replace(
            v1, evaluation_ids=("eval_01", "eval_02", "eval_injected")
        )
        v1_bad_eval = dataclasses.replace(
            v1_bad_eval_draft, entry_sha256=v1_bad_eval_draft.compute_sha256()
        )
        v2_relinked_draft = dataclasses.replace(v2, parent_sha256=v1_bad_eval.entry_sha256)
        v2_relinked = dataclasses.replace(
            v2_relinked_draft, entry_sha256=v2_relinked_draft.compute_sha256()
        )
        rep_eval = verify_ledger([v0, v1_bad_eval, v2_relinked], records)
        self.assertFalse(rep_eval.is_valid)
        self.assertFalse(rep_eval.frozen_scope_preserved)
        self.assertIn(ERR_EVALUATION_SCOPE_MUTATED, [d.error_code for d in rep_eval.defects])

        # 2. Mutate candidates on v1.
        v1_bad_cand_draft = dataclasses.replace(
            v1, candidates=("cfg_low", "cfg_high", "cfg_secret")
        )
        v1_bad_cand = dataclasses.replace(
            v1_bad_cand_draft, entry_sha256=v1_bad_cand_draft.compute_sha256()
        )
        v2_cand_draft = dataclasses.replace(v2, parent_sha256=v1_bad_cand.entry_sha256)
        v2_cand = dataclasses.replace(
            v2_cand_draft, entry_sha256=v2_cand_draft.compute_sha256()
        )
        rep_cand = verify_ledger([v0, v1_bad_cand, v2_cand], records)
        self.assertFalse(rep_cand.is_valid)
        self.assertIn(ERR_CANDIDATE_SET_MUTATED, [d.error_code for d in rep_cand.defects])

        # 3. Mutate calibration_ids on v1.
        v1_bad_cal_draft = dataclasses.replace(v1, calibration_ids=("cal_01",))
        v1_bad_cal = dataclasses.replace(
            v1_bad_cal_draft, entry_sha256=v1_bad_cal_draft.compute_sha256()
        )
        v2_cal_draft = dataclasses.replace(v2, parent_sha256=v1_bad_cal.entry_sha256)
        v2_cal = dataclasses.replace(
            v2_cal_draft, entry_sha256=v2_cal_draft.compute_sha256()
        )
        rep_cal = verify_ledger([v0, v1_bad_cal, v2_cal], records)
        self.assertFalse(rep_cal.is_valid)
        self.assertIn(ERR_CALIBRATION_SCOPE_MUTATED, [d.error_code for d in rep_cal.defects])

        # 4. Overlapping calibration and evaluation IDs rejected at genesis construction.
        with self.assertRaises(ValueError):
            create_genesis_scope(
                ledger_id="bad",
                calibration_ids=["shared_01"],
                evaluation_ids=["shared_01", "eval_02"],
                candidates=["c1"],
                max_invocations=4,
            )

    def test_unlinked_tampered_or_out_of_order_chain_detected(self) -> None:
        amendments, records = _build_valid_three_step_ledger()
        v0, v1, v2 = amendments

        # Unlinked parent_sha256 on v2 (recomputed entry_sha256, wrong parent pointer).
        v2_unlinked_draft = dataclasses.replace(v2, parent_sha256="f" * 64)
        v2_unlinked = dataclasses.replace(
            v2_unlinked_draft, entry_sha256=v2_unlinked_draft.compute_sha256()
        )
        rep_unlinked = verify_ledger([v0, v1, v2_unlinked], records)
        self.assertFalse(rep_unlinked.is_valid)
        self.assertFalse(rep_unlinked.hash_chain_intact)
        self.assertIn(ERR_BROKEN_HASH_LINK, [d.error_code for d in rep_unlinked.defects])

        # Post-hoc tampering with v1 reason without updating entry_sha256.
        v1_tampered = dataclasses.replace(v1, reason="Silently altered reason")
        rep_tampered = verify_ledger([v0, v1_tampered, v2], records)
        self.assertFalse(rep_tampered.is_valid)
        self.assertIn(ERR_SCOPE_DIGEST_TAMPERED, [d.error_code for d in rep_tampered.defects])

        # Out-of-order version sequence (skipping v1).
        rep_skip = verify_ledger([v0, v2], records)
        self.assertFalse(rep_skip.is_valid)
        self.assertIn(ERR_INVALID_VERSION_SEQUENCE, [d.error_code for d in rep_skip.defects])

    def test_rejected_contamination_and_budget_mismatch_detected(self) -> None:
        amendments, records = _build_valid_three_step_ledger()

        # Contaminate a rejected hardware record with eligible_measurement=True and energy_j=4.5.
        contaminated_rec = dataclasses.replace(
            records[4],
            eligible_measurement=True,
            energy_j=4.5,
        )
        contaminated_signed = dataclasses.replace(
            contaminated_rec, record_sha256=contaminated_rec.compute_sha256()
        )
        bad_records = list(records)
        bad_records[4] = contaminated_signed

        rep_contam = verify_ledger(amendments, bad_records)
        self.assertFalse(rep_contam.is_valid)
        self.assertFalse(rep_contam.rejected_zero_contamination_verified)
        codes = [d.error_code for d in rep_contam.defects]
        self.assertIn(ERR_REJECTED_MARKED_ELIGIBLE, codes)
        self.assertIn(ERR_REJECTED_HAS_ENERGY, codes)

        # Tamper with an accepted record's energy_j without updating record_sha256.
        tampered_rec = dataclasses.replace(records[0], energy_j=0.01)
        tampered_list = list(records)
        tampered_list[0] = tampered_rec
        rep_rec_tamper = verify_ledger(amendments, tampered_list)
        self.assertFalse(rep_rec_tamper.is_valid)
        self.assertIn(ERR_RECORD_DIGEST_TAMPERED, [d.error_code for d in rep_rec_tamper.defects])

        # Drop the last pending slot -> accepted + pending + rejected = 9 != max_invocations (10).
        rep_short = verify_ledger(amendments, records[:-1])
        self.assertFalse(rep_short.is_valid)
        self.assertFalse(rep_short.budget_conserved)
        self.assertIn(ERR_INVOCATION_BUDGET_MISMATCH, [d.error_code for d in rep_short.defects])

        # Exceed intermediate v0 cap (v0 max_invocations=6, assign 7 records to v0).
        extra_v0 = dataclasses.replace(records[6], amendment_version="v0")
        extra_v0_signed = dataclasses.replace(extra_v0, record_sha256=extra_v0.compute_sha256())
        over_v0_records = list(records)
        over_v0_records[6] = extra_v0_signed
        rep_cap = verify_ledger(amendments, over_v0_records)
        self.assertFalse(rep_cap.is_valid)
        self.assertIn(ERR_AMENDMENT_CAP_EXCEEDED, [d.error_code for d in rep_cap.defects])

    def test_cli_audit_mode_and_exclusive_output(self) -> None:
        amendments, records = _build_valid_three_step_ledger()
        ledger_path = self.tmpdir / "ledger.json"
        out_json = self.tmpdir / "ledger_audit.json"
        out_md = self.tmpdir / "ledger_audit.md"

        ledger_path.write_text(
            json.dumps(serialize_ledger_dict(amendments, records), indent=2),
            encoding="utf-8",
        )

        rc_ok = main(
            [
                "--ledger",
                str(ledger_path),
                "--output",
                str(out_json),
                "--markdown-output",
                str(out_md),
            ]
        )
        self.assertEqual(rc_ok, 0)
        audit_payload = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertTrue(audit_payload["is_valid"])
        self.assertEqual(audit_payload["total_accounted"], 10)
        self.assertTrue(out_md.exists())

        # Exclusive --output collision must return exit code 1.
        rc_collision = main(["--ledger", str(ledger_path), "--output", str(out_json)])
        self.assertEqual(rc_collision, 1)

        # Invalid ledger (dropped record) must return exit code 2.
        bad_ledger_path = self.tmpdir / "bad_ledger.json"
        bad_out_json = self.tmpdir / "bad_audit.json"
        bad_ledger_path.write_text(
            json.dumps(serialize_ledger_dict(amendments, records[:-1]), indent=2),
            encoding="utf-8",
        )
        rc_invalid = main(["--ledger", str(bad_ledger_path), "--output", str(bad_out_json)])
        self.assertEqual(rc_invalid, 2)


if __name__ == "__main__":
    unittest.main()
