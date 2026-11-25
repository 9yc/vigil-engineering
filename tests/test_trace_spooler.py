import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class TestTraceSpooler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifact_dir = Path(os.environ["ARTIFACT_DIR"])
        cls.bin_path = artifact_dir / "trace_spooler"
        if not cls.bin_path.is_file():
            raise RuntimeError(f"Missing binary at {cls.bin_path}")
        cls.tmp_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
        cls.tmp_root.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.work_dir = tempfile.TemporaryDirectory(dir=self.tmp_root)
        self.root = Path(self.work_dir.name)

    def tearDown(self):
        self.work_dir.cleanup()

    def run_cli(self, args, expect_ok=True):
        proc = subprocess.run(
            [str(self.bin_path), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if expect_ok and proc.returncode != 0:
            self.fail(f"CLI failed ({proc.returncode}): {proc.stderr}\n{proc.stdout}")
        if not expect_ok and proc.returncode == 0:
            self.fail(f"Expected CLI failure, but succeeded: {proc.stdout}")
        return proc

    def build_sample_jsonl(self, path, count=150):
        records = []
        with path.open("w", encoding="utf-8") as fh:
            for i in range(count):
                rec = {
                    "timestamp_ns": 1_000_000_000 + i * 2500,
                    "type": (i % 4) + 1,
                    "payload": f"evt-{i:04d}|msg=\"hello_{i}\"|len={'x' * (i % 29)}",
                }
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
        return records

    def test_multisegment_rotation_sparse_query_and_roundtrip_export(self):
        src_jsonl = self.root / "events.jsonl"
        log_dir = self.root / "framed_store"
        export_jsonl = self.root / "exported.jsonl"
        expected = self.build_sample_jsonl(src_jsonl, count=180)

        ing = json.loads(self.run_cli([
            "ingest",
            "--input", str(src_jsonl),
            "--log-dir", str(log_dir),
            "--max-segment-bytes", "2200",
            "--max-segment-records", "30",
            "--index-stride", "4",
        ]).stdout)
        self.assertEqual(ing["records_ingested"], 180)
        self.assertGreaterEqual(ing["segments_created"], 6)
        self.assertEqual(ing["first_seq"], 1)
        self.assertEqual(ing["last_seq"], 180)

        flog_files = sorted(log_dir.glob("segment_*.flog"))
        fidx_files = sorted(log_dir.glob("segment_*.fidx"))
        self.assertEqual(len(flog_files), ing["segments_created"])
        self.assertEqual(len(fidx_files), ing["segments_created"])

        ver = json.loads(self.run_cli(["verify", "--log-dir", str(log_dir)]).stdout)
        self.assertTrue(ver["valid"])
        self.assertEqual(ver["records_verified"], 180)
        self.assertEqual(ver["corrupt_frames"], 0)
        self.assertFalse(ver["truncated_tail_detected"])

        q_proc = self.run_cli([
            "query",
            "--log-dir", str(log_dir),
            "--seq-start", "45",
            "--seq-end", "92",
            "--type", "3",
        ])
        queried = [json.loads(line) for line in q_proc.stdout.strip().splitlines()]
        want_subset = [
            (idx + 1, r)
            for idx, r in enumerate(expected)
            if 45 <= (idx + 1) <= 92 and r["type"] == 3
        ]
        self.assertEqual(len(queried), len(want_subset))
        for got, (seq_no, orig) in zip(queried, want_subset):
            self.assertEqual(got["seq"], seq_no)
            self.assertEqual(got["timestamp_ns"], orig["timestamp_ns"])
            self.assertEqual(got["type"], orig["type"])
            self.assertEqual(got["payload"], orig["payload"])

        self.run_cli([
            "export",
            "--log-dir", str(log_dir),
            "--output", str(export_jsonl),
        ])
        exported = [json.loads(line) for line in export_jsonl.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(exported), 180)
        for i, (got, orig) in enumerate(zip(exported, expected), start=1):
            self.assertEqual(got["seq"], i)
            self.assertEqual(got["timestamp_ns"], orig["timestamp_ns"])
            self.assertEqual(got["type"], orig["type"])
            self.assertEqual(got["payload"], orig["payload"])

        self.run_cli([
            "export",
            "--log-dir", str(log_dir),
            "--output", str(export_jsonl),
        ], expect_ok=False)

    def test_single_bit_payload_and_header_corruption_detected(self):
        src_jsonl = self.root / "src.jsonl"
        self.build_sample_jsonl(src_jsonl, count=60)

        for label, flip_offset in [("payload_bitflip", -7), ("header_bitflip", 30)]:
            with self.subTest(corruption=label):
                log_dir = self.root / f"log_{label}"
                self.run_cli([
                    "ingest",
                    "--input", str(src_jsonl),
                    "--log-dir", str(log_dir),
                    "--max-segment-records", "25",
                ])
                seg1 = sorted(log_dir.glob("segment_*.flog"))[1]
                raw = bytearray(seg1.read_bytes())
                idx = (len(raw) + flip_offset) if flip_offset < 0 else flip_offset
                raw[idx] ^= 0x04
                seg1.write_bytes(bytes(raw))

                ver_proc = self.run_cli(["verify", "--log-dir", str(log_dir)], expect_ok=False)
                rep = json.loads(ver_proc.stdout)
                self.assertFalse(rep["valid"])
                self.assertGreaterEqual(rep["corrupt_frames"], 1)

                self.run_cli([
                    "export",
                    "--log-dir", str(log_dir),
                    "--output", str(self.root / f"bad_out_{label}.jsonl"),
                ], expect_ok=False)

    def test_truncated_tail_detection_and_crash_safe_recovery(self):
        src_jsonl = self.root / "crash_src.jsonl"
        expected = self.build_sample_jsonl(src_jsonl, count=75)
        log_dir = self.root / "crash_log"

        self.run_cli([
            "ingest",
            "--input", str(src_jsonl),
            "--log-dir", str(log_dir),
            "--max-segment-records", "30",
            "--index-stride", "3",
        ])

        last_seg = sorted(log_dir.glob("segment_*.flog"))[-1]
        orig_bytes = last_seg.read_bytes()
        last_seg.write_bytes(orig_bytes[:-11])

        unrecovered = self.run_cli(["verify", "--log-dir", str(log_dir)], expect_ok=False)
        rep_fail = json.loads(unrecovered.stdout)
        self.assertFalse(rep_fail["valid"])
        self.assertTrue(rep_fail["truncated_tail_detected"])
        self.assertFalse(rep_fail["recovered_tail"])

        rec_proc = self.run_cli(["verify", "--log-dir", str(log_dir), "--recover-tail"])
        rep_ok = json.loads(rec_proc.stdout)
        self.assertTrue(rep_ok["valid"])
        self.assertTrue(rep_ok["truncated_tail_detected"])
        self.assertTrue(rep_ok["recovered_tail"])
        self.assertEqual(rep_ok["records_verified"], 74)
        self.assertGreater(rep_ok["truncated_bytes_removed"], 0)

        recheck = json.loads(self.run_cli(["verify", "--log-dir", str(log_dir)]).stdout)
        self.assertTrue(recheck["valid"])
        self.assertEqual(recheck["records_verified"], 74)
        self.assertFalse(recheck["truncated_tail_detected"])

        rec_export = self.root / "recovered.jsonl"
        self.run_cli([
            "export",
            "--log-dir", str(log_dir),
            "--output", str(rec_export),
        ])
        rows = [json.loads(line) for line in rec_export.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 74)
        self.assertEqual(rows[-1]["seq"], 74)
        self.assertEqual(rows[-1]["payload"], expected[73]["payload"])

    def test_custom_initial_seq_corrupt_index_fallback_and_validation(self):
        good_jsonl = self.root / "good.jsonl"
        expected = self.build_sample_jsonl(good_jsonl, count=40)
        d_seq = self.root / "custom_seq"
        ing = json.loads(self.run_cli([
            "ingest",
            "--input", str(good_jsonl),
            "--log-dir", str(d_seq),
            "--initial-seq", "1000",
            "--max-segment-records", "15",
            "--index-stride", "3",
        ]).stdout)
        self.assertEqual(ing["first_seq"], 1000)
        self.assertEqual(ing["last_seq"], 1039)

        idx0 = sorted(d_seq.glob("segment_*.fidx"))[0]
        idx_bytes = bytearray(idx0.read_bytes())
        idx_bytes[0] = ord("Z")
        idx0.write_bytes(bytes(idx_bytes))

        q_out = self.run_cli([
            "query",
            "--log-dir", str(d_seq),
            "--seq-start", "1010",
            "--seq-end", "1014",
        ]).stdout
        rows = [json.loads(line) for line in q_out.strip().splitlines()]
        self.assertEqual([r["seq"] for r in rows], [1010, 1011, 1012, 1013, 1014])
        self.assertEqual(rows[0]["payload"], expected[10]["payload"])

        self.run_cli([
            "query",
            "--log-dir", str(d_seq),
            "--seq-start", "1020",
            "--seq-end", "1010",
        ], expect_ok=False)

        bad_jsonl = self.root / "bad.jsonl"
        bad_jsonl.write_text("{\"timestamp_ns\": -5, \"type\": 1, \"payload\": \"x\"}\n", encoding="utf-8")
        self.run_cli(["ingest", "--input", str(bad_jsonl), "--log-dir", str(self.root / "d1")], expect_ok=False)
        self.run_cli(["ingest", "--input", str(good_jsonl), "--log-dir", str(d_seq)], expect_ok=False)


if __name__ == "__main__":
    unittest.main()
