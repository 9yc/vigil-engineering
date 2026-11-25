import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest


def resolve_binary(name: str) -> Path:
    artifact_dir = os.environ.get("ARTIFACT_DIR")
    if not artifact_dir:
        raise RuntimeError("ARTIFACT_DIR environment variable is required")
    binary = Path(artifact_dir) / name
    if not binary.is_file():
        raise FileNotFoundError(f"Expected compiled binary at {binary}")
    return binary


def resolve_tmp_base() -> Path:
    raw = os.environ.get("TMPDIR")
    if raw:
        base = Path(raw)
        base.mkdir(parents=True, exist_ok=True)
        return base
    return Path(tempfile.gettempdir())


def parse_framed_samples(raw_text: str):
    lines = raw_text.splitlines()
    idx = 0
    frames = []
    while idx < len(lines):
        if lines[idx] != "__SAMPLE_BEGIN__":
            raise AssertionError(f"Unexpected frame start at line {idx}: {lines[idx]!r}")
        idx += 1
        if idx >= len(lines) or not lines[idx].startswith("__TS_BEGIN_NS__="):
            raise AssertionError(f"Missing __TS_BEGIN_NS__ at line {idx}")
        ts_begin = int(lines[idx].split("=", 1)[1])
        idx += 1
        body_lines = []
        while idx < len(lines) and not lines[idx].startswith("__TS_END_NS__="):
            body_lines.append(lines[idx])
            idx += 1
        if idx >= len(lines):
            raise AssertionError("Missing __TS_END_NS__ before EOF")
        ts_end = int(lines[idx].split("=", 1)[1])
        idx += 1
        if idx >= len(lines) or lines[idx] != "__SAMPLE_END__":
            raise AssertionError(f"Missing __SAMPLE_END__ at line {idx}")
        idx += 1
        frames.append({"ts_begin_ns": ts_begin, "ts_end_ns": ts_end, "body": "\n".join(body_lines)})
    return frames


def read_schedule_csv(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["scheduled_ns", "dispatch_ns", "end_ns", "rc", "boottime_ns", "realtime_ns"]
        if reader.fieldnames != expected:
            raise AssertionError(f"Unexpected CSV columns {reader.fieldnames}, wanted {expected}")
        return [{k: int(v) for k, v in row.items()} for row in reader]


class PeriodicCmdSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bin_path = resolve_binary("periodic_cmd_sampler")
        cls.tmp_base = resolve_tmp_base()

    def setUp(self):
        self._tmp_ctx = tempfile.TemporaryDirectory(dir=str(self.tmp_base))
        self.work_dir = Path(self._tmp_ctx.name)

    def tearDown(self):
        self._tmp_ctx.cleanup()

    def test_multi_tick_sampling_frames_csv_and_timerslack(self):
        out_path = self.work_dir / "samples.raw"
        csv_path = self.work_dir / "schedule.csv"
        meta_path = self.work_dir / "meta.json"
        done_path = self.work_dir / "sampler.done"
        counter_file = self.work_dir / "tick_counter.txt"

        probe_script = (
            f"n=$(cat '{counter_file}' 2>/dev/null || echo 0); "
            f"n=$((n + 1)); echo \"$n\" > '{counter_file}'; printf 'tick=%d\\n' \"$n\""
        )
        cmd = [
            str(self.bin_path), "--output", str(out_path), "--csv-output", str(csv_path),
            "--meta-output", str(meta_path), "--done-file", str(done_path),
            "--interval-ms", "30", "--max-samples", "4", "--timerslack-ns", "10000",
            "--shell-cmd", probe_script,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertEqual(done_path.read_text(encoding="utf-8"), "complete\n")

        frames = parse_framed_samples(out_path.read_text(encoding="utf-8"))
        self.assertEqual([f["body"] for f in frames], ["tick=1", "tick=2", "tick=3", "tick=4"])
        rows = read_schedule_csv(csv_path)
        self.assertEqual(len(rows), 4)

        interval_ns = 30 * 1_000_000
        for i, (frame, row) in enumerate(zip(frames, rows)):
            self.assertEqual(frame["ts_begin_ns"], row["dispatch_ns"])
            self.assertEqual(frame["ts_end_ns"], row["end_ns"])
            self.assertLessEqual(row["scheduled_ns"], row["dispatch_ns"] + 2_000_000)
            self.assertLessEqual(row["dispatch_ns"], row["end_ns"])
            self.assertEqual(row["rc"], 0)
            self.assertGreater(row["boottime_ns"], 0)
            self.assertGreater(row["realtime_ns"], 1_500_000_000_000_000_000)
            if i > 0:
                self.assertEqual(row["scheduled_ns"] - rows[i - 1]["scheduled_ns"], interval_ns)
                self.assertGreaterEqual(row["dispatch_ns"], rows[i - 1]["dispatch_ns"])

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertTrue(meta["succeeded"])
        self.assertEqual(meta["stop_reason"], "max_samples")
        self.assertEqual(meta["counts"]["total_samples"], 4)
        self.assertEqual(meta["counts"]["periodic_samples"], 4)
        self.assertFalse(meta["counts"]["final_bracket_captured"])
        self.assertTrue(meta["timerslack"]["modified"])
        self.assertEqual(meta["timerslack"]["effective_ns"], 10000)

    def test_stop_file_graceful_shutdown_captures_final_bracket_and_done_marker(self):
        out_path = self.work_dir / "stop_samples.raw"
        csv_path = self.work_dir / "stop_schedule.csv"
        meta_path = self.work_dir / "stop_meta.json"
        stop_path = self.work_dir / "stop.sentinel"
        done_path = self.work_dir / "stop.done"

        cmd = [
            str(self.bin_path), "--output", str(out_path), "--schedule-csv", str(csv_path),
            "--meta-output", str(meta_path), "--stop-file", str(stop_path),
            "--done-file", str(done_path), "--interval-ms", "40", "--", "/bin/echo", "alive",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(0.11)
            self.assertFalse(done_path.exists())
            stop_path.write_text("stop\n", encoding="utf-8")
            _, stderr_text = proc.communicate(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

        self.assertEqual(proc.returncode, 0, msg=stderr_text)
        self.assertEqual(done_path.read_text(encoding="utf-8"), "complete\n")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["stop_reason"], "stop_file")
        self.assertTrue(meta["counts"]["final_bracket_captured"])
        self.assertGreaterEqual(meta["counts"]["periodic_samples"], 2)
        self.assertEqual(meta["counts"]["total_samples"], meta["counts"]["periodic_samples"] + 1)

        frames = parse_framed_samples(out_path.read_text(encoding="utf-8"))
        rows = read_schedule_csv(csv_path)
        self.assertEqual(len(frames), meta["counts"]["total_samples"])
        self.assertEqual(len(rows), meta["counts"]["total_samples"])
        self.assertTrue(all(f["body"] == "alive" for f in frames))

    def test_sigterm_clean_shutdown_writes_complete_marker(self):
        out_path = self.work_dir / "sigterm.raw"
        csv_path = self.work_dir / "sigterm.csv"
        meta_path = self.work_dir / "sigterm_meta.json"
        done_path = self.work_dir / "sigterm.done"

        cmd = [
            str(self.bin_path), "--output", str(out_path), "--csv-output", str(csv_path),
            "--meta-output", str(meta_path), "--done-file", str(done_path),
            "--interval-ms", "35", "--", "/bin/echo", "sigterm_probe",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(0.10)
            proc.send_signal(signal.SIGTERM)
            _, stderr_text = proc.communicate(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

        self.assertEqual(proc.returncode, 0, msg=stderr_text)
        self.assertEqual(done_path.read_text(encoding="utf-8"), "complete\n")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["stop_reason"], "signal")
        self.assertEqual(meta["terminating_signal"], int(signal.SIGTERM))
        self.assertTrue(meta["counts"]["final_bracket_captured"])
        self.assertGreaterEqual(len(parse_framed_samples(out_path.read_text(encoding="utf-8"))), 2)

    def test_hung_probe_process_group_killed_at_timeout_and_strict_mode_marks_failed(self):
        out_path = self.work_dir / "timeout.raw"
        csv_path = self.work_dir / "timeout.csv"
        meta_path = self.work_dir / "timeout_meta.json"
        done_path = self.work_dir / "timeout.done"
        grandchild_pid_file = self.work_dir / "grandchild.pid"

        hung_cmd = (
            f"sleep 30 & bg=$!; echo \"$bg\" > '{grandchild_pid_file}'; "
            f"echo 'started_hung_probe'; wait \"$bg\""
        )
        cmd = [
            str(self.bin_path), "--output", str(out_path), "--csv-output", str(csv_path),
            "--meta-output", str(meta_path), "--done-file", str(done_path),
            "--interval-ms", "100", "--probe-timeout-ms", "75", "--max-samples", "2",
            "--fail-on-probe-error", "--shell-cmd", hung_cmd,
        ]
        t0 = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        elapsed_s = time.monotonic() - t0

        self.assertEqual(proc.returncode, 1)
        self.assertLess(elapsed_s, 1.5)
        self.assertEqual(done_path.read_text(encoding="utf-8"), "failed\n")

        rows = read_schedule_csv(csv_path)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["rc"], -int(signal.SIGKILL))
            probe_dur_ms = (row["end_ns"] - row["dispatch_ns"]) / 1_000_000.0
            self.assertGreaterEqual(probe_dur_ms, 65.0)
            self.assertLess(probe_dur_ms, 500.0)

        frames = parse_framed_samples(out_path.read_text(encoding="utf-8"))
        self.assertEqual([f["body"] for f in frames], ["started_hung_probe", "started_hung_probe"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["counts"]["timed_out_samples"], 2)
        self.assertFalse(meta["succeeded"])

        if grandchild_pid_file.is_file():
            bg_pid = int(grandchild_pid_file.read_text(encoding="utf-8").strip())
            gone = False
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    os.kill(bg_pid, 0)
                except ProcessLookupError:
                    gone = True
                    break
                stat_file = Path(f"/proc/{bg_pid}/stat")
                if not stat_file.exists() or ") Z " in stat_file.read_text(encoding="utf-8"):
                    gone = True
                    break
                time.sleep(0.02)
            self.assertTrue(gone)

    def test_exclusive_output_guard_protects_existing_files(self):
        sentinel = "DO_NOT_OVERWRITE\n"
        for idx, flag in enumerate(["--output", "--csv-output", "--done-file"]):
            protected = self.work_dir / f"existing_{idx}.dat"
            protected.write_text(sentinel, encoding="utf-8")
            cmd = [str(self.bin_path), "--max-samples", "1"]
            if flag != "--output":
                cmd.extend(["--output", str(self.work_dir / f"fresh_{idx}.raw")])
            cmd.extend([flag, str(protected), "--", "/bin/true"])
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(protected.read_text(encoding="utf-8"), sentinel)


if __name__ == "__main__":
    unittest.main()
