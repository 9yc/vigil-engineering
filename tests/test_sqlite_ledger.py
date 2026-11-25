"""Unit tests for runtime_support.sqlite_ledger."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from runtime_support.sqlite_ledger import (
    ArtifactVerificationError,
    BackoffPolicy,
    InvalidParameterError,
    JobNotFoundError,
    JobStatus,
    LeaseConflictError,
    RunNotFoundError,
    SqliteExecutionLedger,
    canonicalize_json_mapping,
)


class SyntheticClock:
    """Deterministic controllable clock for lease and backoff testing."""

    def __init__(self, initial_s: float = 1_700_000_000.0) -> None:
        self._lock = threading.Lock()
        self._now = float(initial_s)

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, delta_s: float) -> float:
        with self._lock:
            self._now += float(delta_s)
            return self._now


class SqliteLedgerTest(unittest.TestCase):
    """Comprehensive behavioral tests for SqliteExecutionLedger."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.base_dir = Path(self._tmpdir.name)
        self.db_path = self.base_dir / "ledger.sqlite3"
        self.clock = SyntheticClock()
        self.ledger = SqliteExecutionLedger(
            self.db_path,
            busy_timeout_ms=5000,
            backoff_policy=BackoffPolicy(base_delay_s=5.0, factor=2.0, max_delay_s=60.0),
            clock=self.clock,
        )

    def tearDown(self) -> None:
        self.ledger.close()
        self._tmpdir.cleanup()

    def test_idempotent_run_registration_and_conflict_rejection(self) -> None:
        r1 = self.ledger.register_run(
            "run-alpha", "matrix_sweep",
            config={"threads": 8, "weights": [1, 2, 3], "alpha": 0.25}, metadata={"owner": "ci"},
        )
        self.clock.advance(10.0)
        r2 = self.ledger.register_run(
            "run-alpha", "matrix_sweep",
            config={"alpha": 0.25, "weights": [1, 2, 3], "threads": 8},
        )
        self.assertEqual(r1.run_id, r2.run_id)
        self.assertEqual(r1.config_sha256, r2.config_sha256)
        self.assertEqual(r1.created_at_s, r2.created_at_s)

        with self.assertRaises(InvalidParameterError):
            self.ledger.register_run("run-alpha", "matrix_sweep", config={"threads": 16})
        with self.assertRaises(InvalidParameterError):
            self.ledger.register_run("run-alpha", "different_name", config=r1.config)

    def test_duplicate_job_idempotency_and_canonical_parameters(self) -> None:
        self.ledger.register_run("run-dedup", "sweep")
        job_a, created_a = self.ledger.enqueue_job(
            "run-dedup", "compile_kernel",
            {"tile_m": 64, "tile_n": 128, "flags": {"unroll": True, "vec": 4}, "zero": -0.0},
            priority=10,
        )
        self.assertTrue(created_a)

        job_b, created_b = self.ledger.enqueue_job(
            "run-dedup", "compile_kernel",
            {"zero": 0.0, "flags": {"vec": 4, "unroll": True}, "tile_n": 128, "tile_m": 64},
            priority=99,
        )
        self.assertFalse(created_b)
        self.assertEqual(job_a.job_id, job_b.job_id)
        self.assertEqual(job_a.params_sha256, job_b.params_sha256)
        self.assertEqual(job_b.priority, 10)

        _, created_c = self.ledger.enqueue_job(
            "run-dedup", "compile_kernel", {"tile_m": 32}, dedup_key="custom-key-1"
        )
        self.assertTrue(created_c)
        with self.assertRaises(InvalidParameterError):
            self.ledger.enqueue_job(
                "run-dedup", "compile_kernel", {"tile_m": 16}, dedup_key="custom-key-1"
            )

    def test_corrupt_parameter_rejection(self) -> None:
        self.ledger.register_run("run-bad", "validation")
        invalid_payloads = [
            {"bad_nan": float("nan")}, {"bad_inf": float("inf")}, {"bad_neg_inf": float("-inf")},
            {"nested": [1, 2, {"raw_bytes": b"abc"}]}, {123: "non_string_key"},
            {"surrogate": "bad\ud800char"}, {"unserializable": object()},
        ]
        for bad in invalid_payloads:
            with self.subTest(payload=repr(bad)):
                with self.assertRaises(InvalidParameterError):
                    self.ledger.enqueue_job("run-bad", "eval", bad)  # type: ignore[arg-type]

        with self.assertRaises(InvalidParameterError):
            canonicalize_json_mapping(["not", "a", "dict"])  # type: ignore[arg-type]
        with self.assertRaises(InvalidParameterError):
            self.ledger.enqueue_job("run-bad", "  bad_space  ", {"x": 1})
        with self.assertRaises(RunNotFoundError):
            self.ledger.enqueue_job("missing-run", "eval", {"x": 1})

    def test_priority_ordering_heartbeat_and_lease_expiration_reclamation(self) -> None:
        self.ledger.register_run("run-lease", "lease_test")
        j_low, _ = self.ledger.enqueue_job("run-lease", "step", {"id": "low"}, priority=1, max_attempts=3)
        j_high, _ = self.ledger.enqueue_job("run-lease", "step", {"id": "high"}, priority=20, max_attempts=3)

        claimed = self.ledger.claim_next_job("worker-1", lease_s=30.0)
        assert claimed is not None
        self.assertEqual(claimed.job_id, j_high.job_id)
        self.assertEqual(claimed.status, JobStatus.LEASED)
        self.assertEqual(claimed.attempt, 1)

        self.clock.advance(20.0)
        renewed = self.ledger.renew_lease(
            claimed.job_id, "worker-1", lease_s=30.0, lease_token=claimed.lease_token
        )
        self.assertAlmostEqual(renewed.lease_expires_at_s or 0.0, self.clock() + 30.0)

        with self.assertRaises(LeaseConflictError):
            self.ledger.complete_job(claimed.job_id, "worker-2")

        self.clock.advance(31.0)
        with self.assertRaises(LeaseConflictError):
            self.ledger.complete_job(claimed.job_id, "worker-1", lease_token=claimed.lease_token)

        reclaimed = self.ledger.reclaim_expired_leases()
        self.assertEqual([r.job_id for r in reclaimed], [j_high.job_id])
        self.assertEqual(reclaimed[0].status, JobStatus.PENDING)
        self.assertAlmostEqual(reclaimed[0].available_at_s, self.clock() + 5.0)

        next_claim = self.ledger.claim_next_job("worker-2", lease_s=15.0)
        assert next_claim is not None
        self.assertEqual(next_claim.job_id, j_low.job_id)
        self.ledger.complete_job(
            next_claim.job_id, "worker-2", lease_token=next_claim.lease_token,
            result_summary={"latency_ms": 12.5},
        )

        self.clock.advance(3.0)
        self.assertIsNone(self.ledger.claim_next_job("worker-2", lease_s=15.0))

        self.clock.advance(2.5)
        reclaimed_claim = self.ledger.claim_next_job("worker-2", lease_s=15.0)
        assert reclaimed_claim is not None
        self.assertEqual(reclaimed_claim.job_id, j_high.job_id)
        self.assertEqual(reclaimed_claim.attempt, 2)
        done = self.ledger.complete_job(
            reclaimed_claim.job_id, "worker-2", lease_token=reclaimed_claim.lease_token,
            result_summary={"ok": True},
        )
        self.assertEqual(done.status, JobStatus.SUCCEEDED)

        events = [e["event_type"] for e in self.ledger.get_job_events(j_high.job_id)]
        self.assertEqual(
            events, ["enqueued", "leased", "heartbeat", "lease_expired_retry", "leased", "succeeded"]
        )

    def test_retry_backoff_exhaustion_transitions_to_dead_letter(self) -> None:
        self.ledger.register_run("run-retry", "retry_test")
        job, _ = self.ledger.enqueue_job("run-retry", "fragile_op", {"seed": 7}, max_attempts=3)

        c1 = self.ledger.claim_next_job("w1", lease_s=20.0)
        assert c1 is not None and c1.attempt == 1
        f1 = self.ledger.fail_job(c1.job_id, "w1", "transient io timeout")
        self.assertEqual(f1.status, JobStatus.PENDING)
        self.assertAlmostEqual(f1.available_at_s, self.clock() + 5.0)

        self.clock.advance(5.0)
        c2 = self.ledger.claim_next_job("w1", lease_s=20.0)
        assert c2 is not None and c2.attempt == 2
        f2 = self.ledger.fail_job(c2.job_id, "w1", "transient io timeout 2")
        self.assertEqual(f2.status, JobStatus.PENDING)
        self.assertAlmostEqual(f2.available_at_s, self.clock() + 10.0)

        self.clock.advance(10.0)
        c3 = self.ledger.claim_next_job("w1", lease_s=20.0)
        assert c3 is not None and c3.attempt == 3
        f3 = self.ledger.fail_job(c3.job_id, "w1", "exhausted retries")
        self.assertEqual(f3.status, JobStatus.DEAD_LETTER)
        self.assertIsNotNone(f3.completed_at_s)
        self.assertEqual(f3.last_error, "exhausted retries")
        self.assertEqual(job.job_id, f3.job_id)

        job2, _ = self.ledger.enqueue_job("run-retry", "single_shot", {"seed": 99}, max_attempts=1)
        c_single = self.ledger.claim_next_job("w1", lease_s=10.0)
        assert c_single is not None and c_single.job_id == job2.job_id
        self.clock.advance(12.0)
        reclaimed = self.ledger.reclaim_expired_leases()
        self.assertEqual(len(reclaimed), 1)
        self.assertEqual(reclaimed[0].status, JobStatus.DEAD_LETTER)

    def test_artifact_checksum_verification_audit_and_manifests(self) -> None:
        self.ledger.register_run("run-art", "artifact_pipeline", config={"rev": 3})
        job, _ = self.ledger.enqueue_job("run-art", "render", {"frame": 0})
        claimed = self.ledger.claim_next_job("worker-a", lease_s=60.0)
        assert claimed is not None

        out_file = self.base_dir / "frame_000.bin"
        payload = b"deterministic-binary-artifact-payload\n" * 128
        out_file.write_bytes(payload)
        expected_sha = hashlib.sha256(payload).hexdigest()

        with self.assertRaises(ArtifactVerificationError):
            self.ledger.register_artifact(
                claimed.job_id, "primary_output", out_file, worker_id="worker-a", expected_sha256="0" * 64
            )
        with self.assertRaises(ArtifactVerificationError):
            self.ledger.register_artifact(
                claimed.job_id, "primary_output", out_file, worker_id="worker-a", expected_size=len(payload) + 1
            )
        with self.assertRaises(ArtifactVerificationError):
            self.ledger.register_artifact(
                claimed.job_id, "missing_output", self.base_dir / "does_not_exist.bin"
            )

        art = self.ledger.register_artifact(
            claimed.job_id, "primary_output", out_file, worker_id="worker-a",
            lease_token=claimed.lease_token, expected_sha256=expected_sha.upper(),
            expected_size=len(payload), metadata={"format": "raw"},
        )
        art_dup = self.ledger.register_artifact(claimed.job_id, "primary_output", out_file)
        self.assertEqual(art.artifact_id, art_dup.artifact_id)
        self.ledger.complete_job(claimed.job_id, "worker-a", lease_token=claimed.lease_token)

        progress = self.ledger.get_progress_summary("run-art")
        self.assertEqual(progress.total_jobs, 1)
        self.assertEqual(progress.succeeded_jobs, 1)
        self.assertEqual(progress.total_artifacts, 1)
        self.assertEqual(progress.total_artifact_bytes, len(payload))
        self.assertAlmostEqual(progress.completion_ratio, 1.0)

        audit_ok = self.ledger.verify_registered_artifacts(run_id="run-art")
        self.assertTrue(audit_ok.ok)
        self.assertEqual(audit_ok.valid_ids, (art.artifact_id,))

        json_a = self.base_dir / "manifest_a.json"
        json_b = self.base_dir / "manifest_b.json"
        sha_a = self.ledger.export_manifest_json(json_a, run_id="run-art")
        sha_b = self.ledger.export_manifest_json(json_b, run_id="run-art")
        self.assertEqual(sha_a, sha_b)
        self.assertEqual(json_a.read_bytes(), json_b.read_bytes())
        parsed = json.loads(json_a.read_text(encoding="utf-8"))
        self.assertEqual(parsed["jobs"][0]["artifacts"][0]["sha256"], expected_sha)

        csv_path = self.base_dir / "manifest.csv"
        self.assertEqual(self.ledger.export_manifest_csv(csv_path, run_id="run-art"), 1)
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(rows[0]["artifact_sha256"], expected_sha)

        out_file.write_bytes(payload + b"corrupted")
        audit_corrupt = self.ledger.verify_registered_artifacts(run_id="run-art")
        self.assertFalse(audit_corrupt.ok)
        self.assertEqual(audit_corrupt.mismatched_ids, (art.artifact_id,))

        out_file.unlink()
        audit_missing = self.ledger.verify_registered_artifacts(run_id="run-art")
        self.assertEqual(audit_missing.missing_ids, (art.artifact_id,))

    def test_concurrent_multithread_worker_claims_without_double_leasing(self) -> None:
        self.ledger.register_run("run-mt", "parallel_workers")
        total_jobs = 48
        specs = [{"task_type": "work_unit", "params": {"idx": idx}, "priority": idx % 5} for idx in range(total_jobs)]
        self.assertEqual(len(self.ledger.enqueue_batch("run-mt", specs)), total_jobs)

        claimed_by_worker: dict[str, list[str]] = {}
        errors: list[BaseException] = []
        lock = threading.Lock()
        barrier = threading.Barrier(6)

        def worker_loop(worker_name: str) -> None:
            local_ids: list[str] = []
            try:
                barrier.wait(timeout=5.0)
                while True:
                    job = self.ledger.claim_next_job(worker_name, lease_s=30.0, run_id="run-mt")
                    if job is None:
                        break
                    local_ids.append(job.job_id)
                    self.ledger.complete_job(
                        job.job_id, worker_name, lease_token=job.lease_token, result_summary={"worker": worker_name}
                    )
            except BaseException as exc:  # pragma: no cover
                with lock:
                    errors.append(exc)
            finally:
                with lock:
                    claimed_by_worker[worker_name] = local_ids

        threads = [threading.Thread(target=worker_loop, args=(f"worker-{i}",)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        self.assertEqual(errors, [])
        all_claimed = [jid for ids in claimed_by_worker.values() for jid in ids]
        self.assertEqual(len(all_claimed), total_jobs)
        self.assertEqual(len(set(all_claimed)), total_jobs)

        summary = self.ledger.get_progress_summary("run-mt")
        self.assertEqual(summary.succeeded_jobs, total_jobs)
        self.assertEqual(summary.leased_jobs, 0)
        self.assertEqual(summary.pending_jobs, 0)
        with self.assertRaises(JobNotFoundError):
            self.ledger.complete_job("nonexistent-job", "worker-0")


if __name__ == "__main__":
    unittest.main()
