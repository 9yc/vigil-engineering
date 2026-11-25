import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest


class TestArenaVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifact_dir = Path(os.environ["ARTIFACT_DIR"])
        cls.binary = artifact_dir / "arena_verifier"
        if not cls.binary.is_file():
            raise RuntimeError(f"Missing compiled binary at {cls.binary}")
        cls.tmp_root = Path(os.environ["TMPDIR"])
        cls.tmp_root.mkdir(parents=True, exist_ok=True)

    def _run_verifier(self, events, strategy, capacity=65536, min_block=16, extra_args=None):
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as td:
            trace_file = Path(td) / "workload.jsonl"
            lines = [json.dumps(ev) for ev in events]
            trace_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            cmd = [
                str(self.binary),
                "--trace",
                str(trace_file),
                "--strategy",
                strategy,
                "--capacity",
                str(capacity),
                "--min-block",
                str(min_block),
            ]
            if extra_args:
                cmd.extend(extra_args)
            return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def test_alignment_guarantees_across_strategies(self) -> None:
        events = [
            {"op": "alloc", "id": "u0", "size": 13, "align": 1},
            {"op": "aligned_alloc", "id": "a16", "size": 37, "align": 16},
            {"op": "alloc", "id": "u1", "size": 7, "align": 1},
            {"op": "aligned_alloc", "id": "a64", "size": 100, "align": 64},
            {"op": "alloc", "id": "u2", "size": 19, "align": 1},
            {"op": "aligned_alloc", "id": "a256", "size": 200, "align": 256},
            {"op": "alloc", "id": "u3", "size": 33, "align": 1},
            {"op": "aligned_alloc", "id": "a4096", "size": 512, "align": 4096},
        ]
        expected_align = [1, 16, 1, 64, 1, 256, 1, 4096]

        for strat in ("freelist_first_fit", "freelist_best_fit", "buddy"):
            proc = self._run_verifier(events, strat, capacity=16384, min_block=16)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            report = json.loads(proc.stdout)
            self.assertEqual(report["accepted"], 8)
            self.assertEqual(report["failed"], 0)
            self.assertTrue(report["invariants_verified"])
            self.assertGreater(report["alignment_overhead_bytes"], 0)

            spans = []
            for ev_out, req, align in zip(report["events"], events, expected_align):
                self.assertEqual(ev_out["status"], "accepted")
                off = ev_out["payload_offset"]
                raw = ev_out["raw_offset"]
                res = ev_out["reserved_bytes"]
                self.assertEqual(off % align, 0, f"{strat} misaligned {off} for {align}")
                self.assertGreaterEqual(off, raw)
                self.assertLessEqual(off + req["size"], raw + res)
                spans.append((raw, raw + res))

            spans.sort()
            for idx in range(1, len(spans)):
                self.assertLessEqual(spans[idx - 1][1], spans[idx][0])

    def test_buddy_split_and_merge_cascade(self) -> None:
        events = [
            {"op": "alloc", "id": "b0", "size": 16, "align": 16},
            {"op": "alloc", "id": "b1", "size": 16, "align": 16},
            {"op": "alloc", "id": "b2", "size": 32, "align": 16},
            {"op": "alloc", "id": "b3", "size": 64, "align": 16},
            {"op": "free", "id": "b1"},
            {"op": "free", "id": "b0"},
            {"op": "free", "id": "b2"},
            {"op": "free", "id": "b3"},
            {"op": "alloc", "id": "full", "size": 1024, "align": 1024},
        ]
        proc = self._run_verifier(events, "buddy", capacity=1024, min_block=16)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["accepted"], 5)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["events"][-1]["status"], "accepted")
        self.assertEqual(report["events"][-1]["payload_offset"], 0)
        self.assertEqual(report["peak_reserved_bytes"], 1024)
        self.assertAlmostEqual(report["external_fragmentation"], 0.0, places=6)

    def test_best_fit_beats_first_fit_on_fragmented_holes(self) -> None:
        # Layout in 1024-byte arena:
        # [0, 300) hole A | [300, 364) pinned | [364, 514) hole B (150B) | [514, 1024) pinned
        events = [
            {"op": "alloc", "id": "hole_a", "size": 300},
            {"op": "alloc", "id": "pin_1", "size": 64},
            {"op": "alloc", "id": "hole_b", "size": 150},
            {"op": "alloc", "id": "pin_2", "size": 510},
            {"op": "free", "id": "hole_a"},
            {"op": "free", "id": "hole_b"},
            {"op": "alloc", "id": "fit_140", "size": 140},
            {"op": "alloc", "id": "need_250", "size": 250},
        ]
        ff = json.loads(self._run_verifier(events, "freelist_first_fit", capacity=1024).stdout)
        bf = json.loads(self._run_verifier(events, "freelist_best_fit", capacity=1024).stdout)

        self.assertEqual(ff["accepted"], 5)
        self.assertEqual(ff["failed"], 1)
        self.assertEqual(ff["events"][-1]["status"], "failed")
        self.assertAlmostEqual(ff["external_fragmentation"], 1.0 - (160.0 / 310.0), places=5)

        self.assertEqual(bf["accepted"], 6)
        self.assertEqual(bf["failed"], 0)
        self.assertEqual(bf["events"][-2]["payload_offset"], 364)
        self.assertEqual(bf["events"][-1]["payload_offset"], 0)
        self.assertAlmostEqual(bf["external_fragmentation"], 1.0 - (50.0 / 60.0), places=5)

    def test_double_free_and_invalid_handle_rejection(self) -> None:
        events = [
            {"op": "alloc", "id": "x", "size": 128, "align": 32},
            {"op": "free", "id": "x"},
            {"op": "free", "id": "x"},
            {"op": "free", "id": "never_allocated"},
            {"op": "free", "handle": 999999},
        ]
        for strat in ("freelist_first_fit", "freelist_best_fit", "buddy"):
            proc = self._run_verifier(events, strat, capacity=4096)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            rep = json.loads(proc.stdout)
            self.assertEqual(rep["freed"], 1)
            self.assertEqual(rep["double_frees"], 1)
            self.assertEqual(rep["invalid_frees"], 3)
            self.assertEqual(rep["events"][2]["status"], "double_free")
            self.assertEqual(rep["events"][3]["status"], "invalid_handle")
            self.assertEqual(rep["events"][4]["status"], "invalid_handle")

        strict_proc = self._run_verifier(
            events, "freelist_best_fit", capacity=4096, extra_args=["--strict-free"]
        )
        self.assertNotEqual(strict_proc.returncode, 0)

    def test_randomized_trace_cross_validation_and_exclusive_output(self) -> None:
        rng = random.Random(20261125)
        events = []
        live_ids = []
        alignments = [1, 16, 32, 64, 128, 256]
        for step in range(180):
            if not live_ids or (len(live_ids) < 28 and rng.random() < 0.62):
                aid = f"r_{step}"
                events.append(
                    {"op": "alloc", "id": aid, "size": rng.randint(1, 700), "align": rng.choice(alignments)}
                )
                live_ids.append(aid)
            else:
                victim = live_ids.pop(rng.randrange(len(live_ids)))
                events.append({"op": "free", "id": victim})
        for victim in live_ids:
            events.append({"op": "free", "id": victim})

        with tempfile.TemporaryDirectory(dir=self.tmp_root) as td:
            trace_path = Path(td) / "random.jsonl"
            out_path = Path(td) / "all_report.json"
            trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

            cmd = [
                str(self.binary),
                "--trace",
                str(trace_path),
                "--strategy",
                "all",
                "--capacity",
                "32768",
                "--min-block",
                "16",
                "--output",
                str(out_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            bundle = json.loads(out_path.read_text(encoding="utf-8"))

            for name in ("freelist_first_fit", "freelist_best_fit", "buddy"):
                sub = bundle["strategies"][name]
                self.assertTrue(sub["invariants_verified"])
                self.assertEqual(sub["live_allocations"], 0)
                self.assertEqual(sub["live_reserved_bytes"], 0)
                self.assertEqual(sub["total_free_bytes"], 32768)
                self.assertEqual(sub["largest_free_bytes"], 32768)
                self.assertTrue(math.isclose(sub["external_fragmentation"], 0.0, abs_tol=1e-9))
                self.assertGreaterEqual(sub["peak_reserved_bytes"], sub["peak_payload_bytes"])

            clobber = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertNotEqual(clobber.returncode, 0)

    def test_invalid_size_and_non_power_of_two_alignment_rejected(self) -> None:
        events = [
            {"op": "alloc", "id": "zero_sz", "size": 0, "align": 16},
            {"op": "aligned_alloc", "id": "bad_align", "size": 64, "align": 24},
            {"op": "alloc", "id": "oversize", "size": 8192, "align": 64},
            {"op": "alloc", "id": "valid_ok", "size": 256, "align": 64},
        ]
        for strat in ("freelist_first_fit", "freelist_best_fit", "buddy"):
            proc = self._run_verifier(events, strat, capacity=4096)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            rep = json.loads(proc.stdout)
            self.assertEqual(rep["accepted"], 1)
            self.assertEqual(rep["failed"], 3)
            self.assertEqual([e["status"] for e in rep["events"]], ["failed", "failed", "failed", "accepted"])


if __name__ == "__main__":
    unittest.main()
