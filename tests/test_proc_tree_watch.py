import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def _make_stat_line(
    pid,
    comm,
    state="R",
    ppid=1,
    pgrp=100,
    minflt=10,
    majflt=1,
    utime=25,
    stime=5,
    threads=2,
    vsize_bytes=4096000,
    rss_pages=500,
):
    # Fields after ") ": index 0..21+
    after = ["0"] * 30
    after[0] = state
    after[1] = str(ppid)
    after[2] = str(pgrp)
    after[7] = str(minflt)
    after[9] = str(majflt)
    after[11] = str(utime)
    after[12] = str(stime)
    after[17] = str(threads)
    after[20] = str(vsize_bytes)
    after[21] = str(rss_pages)
    return f"{pid} ({comm}) " + " ".join(after) + "\n"


class ProcTreeWatchTests(unittest.TestCase):
    def setUp(self):
        self.binary = Path(os.environ["ARTIFACT_DIR"]) / "proc_tree_watch"
        self.assertTrue(self.binary.is_file(), f"missing binary: {self.binary}")
        self._tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def run_cli(self, *args, timeout=15):
        return subprocess.run(
            [str(self.binary), *map(str, args)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_synthetic_proc_tree_traversal_comm_parens_and_zombies(self):
        proc_dir = self.directory / "mock_proc"
        proc_dir.mkdir()
        # Root 100 -> child 200 ("worker (pool) 1") -> grandchild 300; zombie 400 under 100; unrelated 999
        nodes = {
            100: _make_stat_line(100, "leader", state="S", ppid=1, threads=1, rss_pages=100, utime=10, stime=2),
            200: _make_stat_line(200, "worker (pool) 1", state="R", ppid=100, threads=4, rss_pages=250, utime=30, stime=8),
            300: _make_stat_line(300, "leaf", state="R", ppid=200, threads=2, rss_pages=150, utime=5, stime=1),
            400: _make_stat_line(400, "defunct", state="Z", ppid=100, threads=1, rss_pages=999, utime=99, stime=99),
            999: _make_stat_line(999, "other_service", state="S", ppid=1, threads=8, rss_pages=2000, utime=500, stime=50),
        }
        for pid, content in nodes.items():
            pdir = proc_dir / str(pid)
            pdir.mkdir()
            (pdir / "stat").write_text(content)

        out_json = self.directory / "snap.json"
        res = self.run_cli(
            "snapshot",
            "--proc-root",
            proc_dir,
            "--root-pid",
            "100",
            "--page-size",
            "4096",
            "--output",
            out_json,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        snap = json.loads(out_json.read_text())
        self.assertEqual(snap["processes"], 3)
        self.assertEqual(snap["threads"], 7)
        # (100 + 250 + 150) * 4 KB = 2000 KB; zombie 400 and unrelated 999 excluded
        self.assertEqual(snap["rss_kb"], 2000)
        self.assertEqual(snap["utime_ticks"], 45)
        self.assertEqual(snap["stime_ticks"], 11)

        # Existing output file must not be overwritten
        dup = self.run_cli("snapshot", "--proc-root", proc_dir, "--root-pid", "100", "--output", out_json)
        self.assertNotEqual(dup.returncode, 0)

    def test_normal_execution_trace_csv_and_peak_rss(self):
        report_file = self.directory / "report.json"
        trace_file = self.directory / "trace.csv"
        script = (
            "import time\n"
            "buf = bytearray(8 * 1024 * 1024)\n"
            "for i in range(0, len(buf), 4096): buf[i] = 1\n"
            "time.sleep(0.08)\n"
        )
        res = self.run_cli(
            "run",
            "--poll-ms",
            "10",
            "--report",
            report_file,
            "--trace-csv",
            trace_file,
            "--",
            sys.executable,
            "-c",
            script,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        report = json.loads(report_file.read_text())
        self.assertEqual(report["exit_code"], 0)
        self.assertFalse(report["timed_out"])
        self.assertFalse(report["rss_exceeded"])
        self.assertGreaterEqual(report["peak_rss_kb"], 8192)
        self.assertGreaterEqual(report["samples"], 1)

        with trace_file.open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(all(int(r["rss_kb"]) > 0 for r in rows))

    def test_wall_clock_timeout_kills_process_group(self):
        report_file = self.directory / "timeout_report.json"
        # Child spawns a grandchild sleeper; watchdog must reap/kill the entire group cleanly.
        script = (
            "import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "time.sleep(30)\n"
        )
        res = self.run_cli(
            "run",
            "--poll-ms",
            "10",
            "--timeout-ms",
            "80",
            "--grace-ms",
            "30",
            "--report",
            report_file,
            "--",
            sys.executable,
            "-c",
            script,
        )
        self.assertEqual(res.returncode, 124, res.stderr)
        report = json.loads(report_file.read_text())
        self.assertTrue(report["timed_out"])
        self.assertFalse(report["rss_exceeded"])
        self.assertLess(report["wall_time_ms"], 5000.0)

    def test_rss_budget_exceeded_terminates_allocating_child(self):
        report_file = self.directory / "rss_report.json"
        script = (
            "import time\n"
            "chunks = [bytearray(24 * 1024 * 1024)]\n"
            "for i in range(0, len(chunks[0]), 4096): chunks[0][i] = 7\n"
            "time.sleep(10)\n"
        )
        res = self.run_cli(
            "run",
            "--poll-ms",
            "10",
            "--max-rss-kb",
            "4096",
            "--grace-ms",
            "20",
            "--report",
            report_file,
            "--",
            sys.executable,
            "-c",
            script,
        )
        self.assertEqual(res.returncode, 125, res.stderr)
        report = json.loads(report_file.read_text())
        self.assertTrue(report["rss_exceeded"])
        self.assertGreater(report["peak_rss_kb"], 4096)

    def test_invalid_cli_and_missing_executable_rejected(self):
        self.assertEqual(self.run_cli().returncode, 2)
        self.assertEqual(self.run_cli("run", "--poll-ms", "0", "--", "true").returncode, 2)
        self.assertEqual(
            self.run_cli("run", "--", "/nonexistent_binary_xyz_12345").returncode, 2
        )


if __name__ == "__main__":
    unittest.main()
