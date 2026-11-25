import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.schedule_jitter_audit import (
    audit_scheduler_ticks,
    parse_scheduler_log,
    quantile_linear,
    verify_interval_bracketing,
)


ROOT = Path(__file__).resolve().parents[1]


def _build_log_text(rows, header_comment="# original_timer_slack_ns=50000 set_slack_rc=0 errno=0 nice=0"):
    lines = []
    if header_comment:
        lines.append(header_comment)
    lines.append(
        "scheduled_ns,dispatch_ns,command_end_ns,command_rc,dispatch_boottime_ns,dispatch_realtime_ns"
    )
    for r in rows:
        lines.append(",".join(str(x) for x in r))
    return "\n".join(lines) + "\n"


class ScheduleJitterAuditTests(unittest.TestCase):
    def test_clean_periodic_log_metadata_and_jitter_metrics(self):
        # 5 ticks spaced at 100 ms (100_000_000 ns) with small lateness and 4 ms probe duration
        base_real = 1_700_000_000_000_000_000
        rows = []
        for idx in range(5):
            sched = 1_000_000_000 + idx * 100_000_000
            lateness = (idx + 1) * 200_000  # 0.2 .. 1.0 ms
            disp = sched + lateness
            end = disp + 4_000_000
            boot = 5_000_000_000 + (disp - 1_000_000_000)
            real = base_real + (disp - 1_000_000_000)
            rows.append((sched, disp, end, 0, boot, real))

        meta, ticks = parse_scheduler_log(_build_log_text(rows))
        self.assertEqual(meta["original_timer_slack_ns"], 50000)
        self.assertEqual(meta["set_slack_rc"], 0)
        self.assertEqual(len(ticks), 5)

        summary = audit_scheduler_ticks(
            ticks, nominal_period_ms=100.0, max_gap_ms=120.0, metadata=meta
        )
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["nonzero_rc_count"], 0)
        self.assertEqual(summary["slot_overrun_count"], 0)
        self.assertAlmostEqual(summary["lateness"]["min_ms"], 0.2, places=6)
        self.assertAlmostEqual(summary["lateness"]["max_ms"], 1.0, places=6)
        self.assertAlmostEqual(summary["probe_duration"]["mean_ms"], 4.0, places=6)
        self.assertEqual(summary["suspend_episodes"], [])
        self.assertEqual(summary["ntp_jumps"], [])

    def test_suspend_skew_ntp_jump_overrun_and_rc_failures_detected(self):
        base_real = 1_700_000_000_000_000_000
        # Tick 1 -> Tick 2 experiences a 50 ms kernel suspend (boottime advances +150ms while monotonic advances +100ms)
        # Tick 2 overruns 100ms slot (duration 130ms) and Tick 3 has NTP +80ms jump + command_rc=1
        rows = [
            (100_000_000, 100_500_000, 105_000_000, 0, 200_500_000, base_real),
            (200_000_000, 200_500_000, 335_000_000, 0, 350_500_000, base_real + 150_000_000),
            (300_000_000, 336_000_000, 340_000_000, 1, 486_000_000, base_real + 365_500_000),
        ]
        _, ticks = parse_scheduler_log(_build_log_text(rows))
        summary = audit_scheduler_ticks(ticks, nominal_period_ms=100.0, max_gap_ms=120.0)
        self.assertFalse(summary["passed"])
        self.assertTrue(summary["gap_violation"])
        self.assertEqual(summary["nonzero_rc_indices"], [2])
        self.assertEqual(summary["slot_overrun_indices"], [1])
        self.assertEqual(len(summary["suspend_episodes"]), 1)
        self.assertAlmostEqual(summary["suspend_episodes"][0]["suspend_ms"], 50.0, places=5)
        self.assertEqual(len(summary["ntp_jumps"]), 1)
        self.assertAlmostEqual(summary["ntp_jumps"][0]["wall_jump_ms"], 80.0, places=5)

    def test_window_bracketing_and_cli_exclusive_output(self):
        base_real = 1_700_000_000_000_000_000
        rows = [
            (100_000_000 * i, 100_000_000 * i + 1_000_000, 100_000_000 * i + 3_000_000, 0,
             500_000_000 + 100_000_000 * i, base_real + 100_000_000 * i)
            for i in range(1, 7)
        ]
        text = _build_log_text(rows)
        _, ticks = parse_scheduler_log(text)

        # Midpoints are at 102ms, 202ms, 302ms, 402ms, 502ms, 602ms
        good = verify_interval_bracketing(ticks, 220_000_000, 480_000_000, max_gap_ms=120.0)
        self.assertTrue(good["admissible"])
        self.assertEqual(good["left_tick_index"], 1)
        self.assertEqual(good["right_tick_index"], 4)

        unbracketed = verify_interval_bracketing(ticks, 50_000_000, 480_000_000, max_gap_ms=120.0)
        self.assertFalse(unbracketed["admissible"])
        self.assertIn("missing_leading_bracket_tick", unbracketed["issues"])

        tight_gap = verify_interval_bracketing(ticks, 220_000_000, 480_000_000, max_gap_ms=50.0)
        self.assertFalse(tight_gap["admissible"])
        self.assertIn("internal_gap_exceeds_max_gap", tight_gap["issues"])

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            tmp_dir = Path(tmp)
            log_file = tmp_dir / "scheduler.csv"
            out_file = tmp_dir / "jitter.json"
            log_file.write_text(text)

            cmd = [
                sys.executable,
                "-B",
                str(ROOT / "tools/schedule_jitter_audit.py"),
                str(log_file),
                "--nominal-period-ms",
                "100",
                "--max-gap-ms",
                "120",
                "--window-start-ns",
                "220000000",
                "--window-end-ns",
                "480000000",
                "--output",
                str(out_file),
            ]
            first = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            self.assertEqual(first.returncode, 0, first.stderr)
            parsed = json.loads(out_file.read_text())
            self.assertTrue(parsed["passed"])
            self.assertTrue(parsed["window_bracket"]["admissible"])

            # Existing output is preserved
            second = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(second.returncode, 0)

    def test_malformed_logs_and_invalid_quantiles_rejected(self):
        bad_logs = (
            "",
            "# only comment\n",
            "wrong,columns\n1,2\n",
            # dispatch_ns < scheduled_ns
            _build_log_text([(100, 90, 110, 0, 200, 300)]),
            # command_end_ns < dispatch_ns
            _build_log_text([(100, 110, 105, 0, 200, 300)]),
            # non-increasing dispatch_ns
            _build_log_text([(100, 110, 120, 0, 200, 300), (105, 110, 125, 0, 210, 310)]),
        )
        for bad in bad_logs:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_scheduler_log(bad)

        with self.assertRaises(ValueError):
            quantile_linear([], 0.5)
        with self.assertRaises(ValueError):
            quantile_linear([1.0], 1.5)


if __name__ == "__main__":
    unittest.main()
