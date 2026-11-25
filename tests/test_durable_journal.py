"""Unit tests for runtime_support.durable_journal."""

import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import unittest

from runtime_support.durable_journal import (
    CellSpec,
    DurableJournal,
    InvalidTransitionError,
    JournalClosedError,
    JournalCorruptionError,
    JournalExistsError,
    JournalLockError,
    JournalValidationError,
    MODE_MEASURE,
    MODE_REUSE,
    MatrixPlan,
    PlanMismatchError,
    ReuseAuditError,
    STATE_ACCEPTED,
    STATE_INVALID,
    STATE_NOT_RUN,
    canonical_json_sha256,
)


class FakeMonotonicNs:
    """Deterministic nanosecond clock for journal tests."""

    def __init__(self, start_ns: int = 1_700_000_000_000_000_000, step_ns: int = 1_000_000) -> None:
        self._now = start_ns
        self._step = step_ns

    def __call__(self) -> int:
        val = self._now
        self._now += self._step
        return val


class DurableJournalTests(unittest.TestCase):
    """Comprehensive tests for crash-safe durable matrix journaling."""

    def setUp(self) -> None:
        tmp_root = os.environ.get("TMPDIR")
        if tmp_root:
            pathlib.Path(tmp_root).mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.TemporaryDirectory(dir=tmp_root)
        self.addCleanup(self._tmpdir.cleanup)
        self.base_dir = pathlib.Path(self._tmpdir.name)
        self.journal_dir = self.base_dir / "matrix_journal"
        self.clock = FakeMonotonicNs()

    def _make_sample_plan(self, *, artifact_file: pathlib.Path | None = None) -> MatrixPlan:
        reuse_metrics = {"latency_ms": 14.25, "tokens_out": 64}
        reuse_bundle: dict[str, object] = {
            "audit": {"passed": True, "auditor": "preflight_gate_v1"},
            "metrics": reuse_metrics,
            "metrics_sha256": canonical_json_sha256(reuse_metrics),
        }
        if artifact_file is not None:
            reuse_bundle["artifact_path"] = str(artifact_file)
            reuse_bundle["artifact_sha256"] = hashlib.sha256(artifact_file.read_bytes()).hexdigest()

        cells = [
            CellSpec("req_alpha", "cfg_b1", mode=MODE_REUSE, parameters={"batch": 1}, reuse_bundle=reuse_bundle),
            CellSpec("req_alpha", "cfg_b4", mode=MODE_MEASURE, parameters={"batch": 4}),
            CellSpec("req_beta", "cfg_b4", mode=MODE_MEASURE, parameters={"batch": 4}),
            CellSpec("req_gamma", "cfg_b8", mode=MODE_MEASURE, parameters={"batch": 8}),
        ]
        return MatrixPlan("matrix_suite_01", cells, metadata={"suite": "core_latency"})

    def test_fresh_creation_and_full_completion_accounting(self) -> None:
        plan = self._make_sample_plan()
        with DurableJournal.open(self.journal_dir, plan, clock_ns=self.clock) as journal:
            self.assertTrue(journal.plan_path.is_file())
            self.assertTrue(journal.state_path.is_file())
            self.assertTrue(journal.events_path.is_file())

            initial = journal.summary()
            self.assertEqual((initial.total_cells, initial.reused_accepted, initial.not_run), (4, 1, 3))
            self.assertFalse(initial.is_complete)

            pending_keys = [spec.key for spec in journal.iter_pending_cells()]
            self.assertEqual(
                pending_keys,
                [("req_alpha", "cfg_b4"), ("req_beta", "cfg_b4"), ("req_gamma", "cfg_b8")],
            )

            journal.start_cell("req_alpha", "cfg_b4", worker_id="worker_0")
            self.assertEqual(journal.summary().inflight, 1)
            journal.complete_cell(
                "req_alpha", "cfg_b4",
                result={"latency_ms": 21.5, "tokens_out": 64}, audit={"passed": True},
            )

            journal.start_cell("req_beta", "cfg_b4", worker_id="worker_0")
            journal.fail_cell("req_beta", "cfg_b4", reason="runner_nonzero_exit", diagnostics={"rc": 2})

            journal.start_cell("req_gamma", "cfg_b8", worker_id="worker_1")
            journal.invalidate_cell("req_gamma", "cfg_b8", reason="thermal_exceeded", diagnostics={"skin_c": 43.2})

            final = journal.summary()
            self.assertEqual((final.not_run, final.inflight, final.accepted), (0, 0, 2))
            self.assertEqual((final.reused_accepted, final.measured_accepted), (1, 1))
            self.assertEqual((final.failed, final.invalid, final.terminal_count), (1, 1, 4))
            self.assertTrue(final.is_complete)
            self.assertFalse(final.all_accepted)
            self.assertEqual(list(journal.iter_pending_cells()), [])
            self.assertEqual(journal.read_events()[0]["event"], "journal_created")

    def test_fresh_open_refuses_to_clobber_existing_journal(self) -> None:
        plan = self._make_sample_plan()
        with DurableJournal.open(self.journal_dir, plan, clock_ns=self.clock):
            pass
        with self.assertRaises(JournalExistsError):
            DurableJournal.open(self.journal_dir, plan, resume=False, clock_ns=self.clock)

    def test_strict_state_transitions_and_validation_guards(self) -> None:
        with self.assertRaises(JournalValidationError):
            CellSpec(request_id="", config_id="cfg_1")
        with self.assertRaises(JournalValidationError):
            CellSpec(request_id=" padded ", config_id="cfg_1")
        with self.assertRaises(JournalValidationError):
            MatrixPlan("dup_plan", [CellSpec("r1", "c1"), CellSpec("r1", "c1")])

        plan = self._make_sample_plan()
        with DurableJournal.open(self.journal_dir, plan, clock_ns=self.clock) as journal:
            with self.assertRaises(InvalidTransitionError):
                journal.start_cell("req_alpha", "cfg_b1")
            with self.assertRaises(InvalidTransitionError):
                journal.complete_cell("req_alpha", "cfg_b4", result={"ok": True})
            with self.assertRaises(InvalidTransitionError):
                journal.fail_cell("req_alpha", "cfg_b4", reason="premature")
            with self.assertRaises(InvalidTransitionError):
                journal.invalidate_cell("req_alpha", "cfg_b4", reason="premature")

            journal.start_cell("req_alpha", "cfg_b4")
            with self.assertRaises(InvalidTransitionError):
                journal.start_cell("req_alpha", "cfg_b4")
            journal.complete_cell("req_alpha", "cfg_b4", result={"latency_ms": 10.0})
            with self.assertRaises(InvalidTransitionError):
                journal.fail_cell("req_alpha", "cfg_b4", reason="late_failure")

        with self.assertRaises(JournalClosedError):
            journal.summary()

    def test_crash_recovery_reconciles_corrupt_tail_and_recovers_inflight(self) -> None:
        plan = self._make_sample_plan()
        with DurableJournal.open(self.journal_dir, plan, clock_ns=self.clock) as journal:
            journal.start_cell("req_alpha", "cfg_b4", worker_id="pid_100")
            journal.complete_cell("req_alpha", "cfg_b4", result={"latency_ms": 19.1})
            journal.start_cell("req_beta", "cfg_b4", worker_id="pid_100")

        torn_fragment = b'{"event":"cell_transition","seq":999,"partial":'
        with open(self.journal_dir / "events.jsonl", "ab") as raw_events:
            raw_events.write(torn_fragment)

        with DurableJournal.open(
            self.journal_dir, plan, resume=True,
            inflight_recovery=STATE_NOT_RUN, clock_ns=self.clock,
        ) as recovered:
            self.assertEqual(recovered.truncated_tail_bytes, len(torn_fragment))
            self.assertEqual(recovered.recovered_cells, (("req_beta", "cfg_b4"),))

            beta_rec = recovered.get_cell("req_beta", "cfg_b4")
            self.assertEqual((beta_rec.state, beta_rec.attempts, beta_rec.interrupted_count), (STATE_NOT_RUN, 1, 1))
            self.assertEqual(recovered.get_cell("req_alpha", "cfg_b4").state, STATE_ACCEPTED)

            for spec in recovered.pending_cells():
                recovered.start_cell(spec.request_id, spec.config_id, worker_id="pid_200")
                recovered.complete_cell(spec.request_id, spec.config_id, result={"latency_ms": 22.0})

            summary = recovered.summary()
            self.assertTrue(summary.is_complete and summary.all_accepted)
            self.assertEqual((summary.accepted, summary.interrupted_recoveries), (4, 1))
            self.assertEqual(recovered.get_cell("req_beta", "cfg_b4").attempts, 2)

            event_names = [e["event"] for e in recovered.read_events()]
            self.assertIn("events_tail_reconciled", event_names)
            self.assertIn("inflight_recovered", event_names)
            self.assertIn("journal_resumed", event_names)

    def test_crash_recovery_can_mark_interrupted_inflight_invalid(self) -> None:
        plan = self._make_sample_plan()
        with DurableJournal.open(self.journal_dir, plan, clock_ns=self.clock) as journal:
            journal.start_cell("req_beta", "cfg_b4", worker_id="crashed_worker")

        with DurableJournal.open(
            self.journal_dir, plan, resume=True,
            inflight_recovery=STATE_INVALID, clock_ns=self.clock,
        ) as recovered:
            rec = recovered.get_cell("req_beta", "cfg_b4")
            self.assertEqual((rec.state, rec.interrupted_count), (STATE_INVALID, 1))
            self.assertIsNotNone(rec.error_reason)

    def test_plan_mismatch_rejected_on_resume(self) -> None:
        plan = self._make_sample_plan()
        with DurableJournal.open(self.journal_dir, plan, clock_ns=self.clock):
            pass

        altered_cells = list(plan.cells)
        altered_cells[1] = CellSpec("req_alpha", "cfg_b4", mode=MODE_MEASURE, parameters={"batch": 16})
        altered_plan = MatrixPlan(plan.plan_id, altered_cells, metadata=plan.metadata)
        with self.assertRaises(PlanMismatchError):
            DurableJournal.open(self.journal_dir, altered_plan, resume=True, clock_ns=self.clock)

    def test_reuse_bundle_audit_and_tamper_rejection(self) -> None:
        for bad_passed in (False, 1, "true", None):
            with self.assertRaises(ReuseAuditError):
                CellSpec(
                    "r1", "c1", mode=MODE_REUSE,
                    reuse_bundle={"audit": {"passed": bad_passed}, "metrics": {"x": 1}},
                )

        with self.assertRaises(ReuseAuditError):
            CellSpec(
                "r1", "c1", mode=MODE_REUSE,
                reuse_bundle={
                    "audit": {"passed": True},
                    "metrics": {"latency_ms": 99.9},
                    "metrics_sha256": canonical_json_sha256({"latency_ms": 10.0}),
                },
            )

        artifact_path = self.base_dir / "reuse_capture.json"
        artifact_path.write_bytes(b'{"verified_energy_mj": 125.4}\n')
        plan = self._make_sample_plan(artifact_file=artifact_path)
        with DurableJournal.open(self.journal_dir, plan, clock_ns=self.clock):
            pass

        artifact_path.write_bytes(b'{"verified_energy_mj": 12.0}\n')
        with self.assertRaises(ReuseAuditError):
            DurableJournal.open(self.journal_dir, plan, resume=True, clock_ns=self.clock)

        artifact_path.write_bytes(b'{"verified_energy_mj": 125.4}\n')
        state_path = self.journal_dir / "state.json"
        state_obj = json.loads(state_path.read_text(encoding="utf-8"))
        state_obj["cells"][0]["audit"]["passed"] = False
        state_path.write_text(json.dumps(state_obj), encoding="utf-8")
        with self.assertRaises(ReuseAuditError):
            DurableJournal.open(self.journal_dir, plan, resume=True, clock_ns=self.clock)

    def test_mid_file_events_corruption_raises(self) -> None:
        plan = self._make_sample_plan()
        with DurableJournal.open(self.journal_dir, plan, clock_ns=self.clock) as journal:
            journal.start_cell("req_alpha", "cfg_b4")
            journal.complete_cell("req_alpha", "cfg_b4", result={"latency_ms": 5.0})

        events_file = self.journal_dir / "events.jsonl"
        lines = events_file.read_text(encoding="utf-8").splitlines(keepends=True)
        lines[0] = "{corrupt_first_line\n"
        events_file.write_text("".join(lines), encoding="utf-8")
        with self.assertRaises(JournalCorruptionError):
            DurableJournal.open(self.journal_dir, plan, resume=True, clock_ns=self.clock)

    def test_concurrent_lock_contention_fails_across_processes(self) -> None:
        plan = self._make_sample_plan()
        pkg_root = str(pathlib.Path(__file__).resolve().parents[1])
        env = dict(os.environ, PYTHONPATH=pkg_root, PYTHONDONTWRITEBYTECODE="1")
        child_code = (
            "import os\n"
            "from runtime_support.durable_journal import CellSpec, DurableJournal, JournalLockError, MatrixPlan\n"
            "plan = MatrixPlan('matrix_suite_01', [CellSpec('req_alpha', 'cfg_b4')])\n"
            "try:\n"
            f"    DurableJournal.open({str(self.journal_dir)!r}, plan, resume=True)\n"
            "except JournalLockError:\n"
            "    os._exit(42)\n"
            "os._exit(0)\n"
        )
        with DurableJournal.open(self.journal_dir, plan, clock_ns=self.clock):
            with self.assertRaises(JournalLockError):
                DurableJournal.open(self.journal_dir, plan, resume=True, clock_ns=self.clock)
            proc = subprocess.run(
                ["python3", "-B", "-c", child_code],
                env=env, capture_output=True, text=True, timeout=10.0, check=False,
            )
            self.assertEqual(proc.returncode, 42, msg=proc.stderr)

    def test_nan_rejection_and_state_authoritativeness_over_uncommitted_event(self) -> None:
        plan = self._make_sample_plan()
        with DurableJournal.open(self.journal_dir, plan, clock_ns=self.clock) as journal:
            journal.start_cell("req_alpha", "cfg_b4")
            with self.assertRaises(JournalValidationError):
                journal.complete_cell("req_alpha", "cfg_b4", result={"latency_ms": float("nan")})
            journal.complete_cell("req_alpha", "cfg_b4", result={"latency_ms": 17.5})

        # Append a syntactically valid trailing event indicating req_gamma completed,
        # while authoritative state.json still records req_gamma as not_run.
        bogus_event = json.dumps({
            "seq": 99, "ts_ns": 1_700_000_000_999_000_000,
            "event": "cell_transition", "plan_id": plan.plan_id,
            "request_id": "req_gamma", "config_id": "cfg_b8",
            "from_state": "inflight", "to_state": "accepted",
        }) + "\n"
        with open(self.journal_dir / "events.jsonl", "a", encoding="utf-8") as handle:
            handle.write(bogus_event)

        with DurableJournal.open(self.journal_dir, plan, resume=True, clock_ns=self.clock) as resumed:
            # Authoritative state.json wins: req_gamma remains not_run and pending.
            self.assertEqual(resumed.get_cell("req_gamma", "cfg_b8").state, STATE_NOT_RUN)
            self.assertIn(("req_gamma", "cfg_b8"), [c.key for c in resumed.pending_cells()])


if __name__ == "__main__":
    unittest.main()
