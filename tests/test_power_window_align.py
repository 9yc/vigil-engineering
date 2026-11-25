import json
import os
from pathlib import Path
import subprocess
import tempfile
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


class PowerWindowAlignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bin_path = resolve_binary("power_window_align")
        cls.tmp_base = resolve_tmp_base()

    def setUp(self):
        self._tmp_ctx = tempfile.TemporaryDirectory(dir=str(self.tmp_base))
        self.work_dir = Path(self._tmp_ctx.name)

    def tearDown(self):
        self._tmp_ctx.cleanup()

    def test_analytical_trapezoid_and_midpoint_on_linear_and_step_ramps(self):
        # Segment 1 [0ms..400ms]: linear power ramp P(t) = 2.0 + 10.0 * t_sec (V=2.0V, I(t)=1.0 + 5.0*t_sec)
        # Window w_lin = [50ms, 350ms] (0.05s to 0.35s, duration=0.30s):
        # Exact linear integral = (P(0.05) + P(0.35)) / 2 * 0.30 = (2.5 + 5.5)/2 * 0.30 = 1.200000000 J.
        # Samples every 100ms at 0, 100, 200, 300, 400ms -> window endpoints cut mid-interval!
        #
        # Segment 2 [500ms..700ms]: step jump across [500ms, 600ms] from 2W (500ms) to 10W (600ms),
        # held at 10W through 700ms. Window w_step = [500ms, 550ms] (first half of [500ms, 600ms]):
        # - Trapezoid on [500ms, 550ms]: P(500ms)=2W, P(550ms)=6W -> mean 4W * 0.05s = 0.20 J
        # - Midpoint-held on [500ms, 550ms]: midpoint is 550ms, so [500ms, 550ms] holds P(500ms)=2W -> 2W * 0.05s = 0.10 J!
        telem_csv = self.work_dir / "ramp.csv"
        telem_rows = [
            "timestamp_ns,voltage_v,current_a,valid",
            "0,2.0,1.0,1",
            "100000000,2.0,1.5,1",
            "200000000,2.0,2.0,1",
            "300000000,2.0,2.5,1",
            "400000000,2.0,3.0,1",
            "500000000,2.0,1.0,1",
            "600000000,2.0,5.0,1",
            "700000000,2.0,5.0,1",
        ]
        telem_csv.write_text("\n".join(telem_rows) + "\n", encoding="utf-8")

        win_csv = self.work_dir / "windows.csv"
        win_csv.write_text(
            "request_id,start_ns,end_ns,output_tokens\n"
            "w_lin,50000000,350000000,12\n"
            "w_step,500000000,550000000,5\n",
            encoding="utf-8",
        )
        out_json = self.work_dir / "aligned.json"

        proc = subprocess.run(
            [
                str(self.bin_path),
                "--telemetry", str(telem_csv),
                "--windows", str(win_csv),
                "--max-gap-ms", "120",
                "--output", str(out_json),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

        rep = json.loads(out_json.read_text(encoding="utf-8"))
        by_id = {w["request_id"]: w for w in rep["windows"]}

        w_lin = by_id["w_lin"]
        self.assertAlmostEqual(w_lin["gross_trapezoid_j"], 1.2, places=8)
        self.assertAlmostEqual(w_lin["gross_midpoint_j"], 1.2, places=8)
        self.assertAlmostEqual(w_lin["gross_trapezoid_j_per_token"], 0.1, places=8)

        w_step = by_id["w_step"]
        self.assertAlmostEqual(w_step["gross_trapezoid_j"], 0.20, places=8)
        self.assertAlmostEqual(w_step["gross_midpoint_j"], 0.10, places=8)

        agg = rep["aggregate"]
        self.assertEqual(agg["accepted_windows"], 2)
        self.assertAlmostEqual(agg["total_gross_trapezoid_j"], 1.40, places=8)
        self.assertAlmostEqual(agg["total_gross_midpoint_j"], 1.30, places=8)

    def test_jsonl_inputs_and_pre_post_idle_baseline_subtraction(self):
        # Pre-idle [0ms..100ms]: 1.5W (V=3V, I=0.5A); 1ms guard;
        # Active [101ms..301ms] (0.2s): 7.5W (V=3V, I=2.5A) -> gross = 1.50 J; 1ms guard;
        # Post-idle [302ms..402ms]: 2.5W (V=5V, I=0.5A).
        # Lead=100ms, Trail=100ms -> time-weighted idle baseline = (1.5 + 2.5)/2 = 2.0W.
        # Idle energy over 0.2s = 0.40 J -> Net dynamic energy = 1.50 - 0.40 = 1.10 J; 22 tokens -> 0.05 J/token.
        telem_jsonl = self.work_dir / "telem.jsonl"
        samples = [
            {"timestamp_ns": 0, "voltage_v": 3.0, "current_a": 0.5, "valid": True},
            {"timestamp_ns": 50_000_000, "voltage_v": 3.0, "current_a": 0.5, "valid": True},
            {"timestamp_ns": 100_000_000, "voltage_v": 3.0, "current_a": 0.5, "valid": True},
            {"timestamp_ns": 101_000_000, "voltage_v": 3.0, "current_a": 2.5, "valid": True},
            {"timestamp_ns": 201_000_000, "voltage_v": 3.0, "current_a": 2.5, "valid": True},
            {"timestamp_ns": 301_000_000, "voltage_v": 3.0, "current_a": 2.5, "valid": True},
            {"timestamp_ns": 302_000_000, "voltage_v": 5.0, "current_a": 0.5, "valid": True},
            {"timestamp_ns": 352_000_000, "voltage_v": 5.0, "current_a": 0.5, "valid": True},
            {"timestamp_ns": 402_000_000, "voltage_v": 5.0, "current_a": 0.5, "valid": True},
        ]
        telem_jsonl.write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")

        win_jsonl = self.work_dir / "win.jsonl"
        win_jsonl.write_text(
            json.dumps({
                "request_id": "req_idle_comp",
                "start_ns": 101_000_000,
                "end_ns": 301_000_000,
                "output_tokens": 22,
            }) + "\n",
            encoding="utf-8",
        )
        out_json = self.work_dir / "idle_comp.json"

        proc = subprocess.run(
            [
                str(self.bin_path),
                "--telemetry", str(telem_jsonl),
                "--windows", str(win_jsonl),
                "--max-gap-ms", "110",
                "--idle-lead-ms", "100",
                "--idle-trail-ms", "100",
                "--idle-guard-ms", "1",
                "--output", str(out_json),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        rep = json.loads(out_json.read_text(encoding="utf-8"))
        w = rep["windows"][0]
        self.assertTrue(w["baseline_applied"])
        self.assertAlmostEqual(w["idle_baseline_trapezoid_w"], 2.0, places=5)
        self.assertAlmostEqual(w["gross_trapezoid_j"], 1.5, places=5)
        self.assertAlmostEqual(w["net_trapezoid_j"], 1.1, places=5)
        self.assertAlmostEqual(w["net_trapezoid_j_per_token"], 0.05, places=6)

    def test_missing_bracket_excessive_gap_and_invalid_sample_rejections(self):
        telem_csv = self.work_dir / "base_telem.csv"
        telem_csv.write_text(
            "timestamp_ns,voltage_v,current_a,valid\n"
            "10000000,4.0,1.0,1\n"
            "20000000,4.0,1.0,1\n"
            "80000000,4.0,1.0,1\n"
            "90000000,4.0,1.0,0\n"
            "100000000,4.0,1.0,1\n",
            encoding="utf-8",
        )

        cases = [
            ("missing_left", "req1,5000000,15000000,4\n", "missing_left_bracket"),
            ("missing_right", "req2,15000000,110000000,4\n", "missing_right_bracket"),
            ("excessive_gap", "req3,15000000,50000000,4\n", "excessive_sample_gap"),
            ("invalid_inside", "req4,82000000,98000000,4\n", "invalid_sample_in_span"),
        ]
        for label, win_row, expected_code in cases:
            win_path = self.work_dir / f"{label}.csv"
            win_path.write_text("request_id,start_ns,end_ns,output_tokens\n" + win_row, encoding="utf-8")
            out_path = self.work_dir / f"{label}_out.json"
            proc = subprocess.run(
                [
                    str(self.bin_path),
                    "--telemetry", str(telem_csv),
                    "--windows", str(win_path),
                    "--max-gap-ms", "25",
                    "--output", str(out_path),
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0, msg=f"Case {label} unexpectedly succeeded")
            self.assertIn(expected_code, proc.stderr)
            self.assertFalse(out_path.exists())

    def test_exclusive_output_guard_prevents_clobber(self):
        telem_csv = self.work_dir / "t.csv"
        telem_csv.write_text("timestamp_ns,voltage_v,current_a,valid\n0,1.0,1.0,1\n10000000,1.0,1.0,1\n", encoding="utf-8")
        win_csv = self.work_dir / "w.csv"
        win_csv.write_text("request_id,start_ns,end_ns,output_tokens\nr,1000000,9000000,1\n", encoding="utf-8")
        protected = self.work_dir / "protected.json"
        protected.write_text("ORIGINAL_PAYLOAD\n", encoding="utf-8")

        proc = subprocess.run(
            [str(self.bin_path), "--telemetry", str(telem_csv), "--windows", str(win_csv), "--output", str(protected)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(protected.read_text(encoding="utf-8"), "ORIGINAL_PAYLOAD\n")


if __name__ == "__main__":
    unittest.main()
