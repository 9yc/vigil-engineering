import json
import math
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


class CooldownDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bin_path = resolve_binary("cooldown_detector")
        cls.tmp_base = resolve_tmp_base()

    def setUp(self):
        self._tmp_ctx = tempfile.TemporaryDirectory(dir=str(self.tmp_base))
        self.work_dir = Path(self._tmp_ctx.name)

    def tearDown(self):
        self._tmp_ctx.cleanup()

    def test_noiseless_and_noisy_exponential_decay_tau_and_target_recovery(self):
        t_amb_true = 28.5
        t0_true = 76.5
        tau_true = 42.0
        target_c = 34.5
        expected_target_s = -tau_true * math.log((target_c - t_amb_true) / (t0_true - t_amb_true))

        # 1) Noiseless trace up to t=60s (so target ~87.34s lies beyond trace end -> tests extrapolation too!)
        clean_csv = self.work_dir / "clean_decay.csv"
        lines = ["time_s,temp_c"]
        for k in range(61):
            t = float(k)
            temp = t_amb_true + (t0_true - t_amb_true) * math.exp(-t / tau_true)
            lines.append(f"{t:.3f},{temp:.8f}")
        clean_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        clean_out = self.work_dir / "clean_out.json"

        proc = subprocess.run(
            [
                str(self.bin_path),
                "--trace", str(clean_csv),
                "--target-c", str(target_c),
                "--output", str(clean_out),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        rep = json.loads(clean_out.read_text(encoding="utf-8"))
        fit = rep["newton_cooling"]
        self.assertTrue(fit["converged"])
        self.assertAlmostEqual(fit["t_amb_c"], t_amb_true, places=4)
        self.assertAlmostEqual(fit["t_initial_c"], t0_true, places=4)
        self.assertAlmostEqual(fit["tau_s"], tau_true, places=4)
        self.assertLess(fit["rmse_c"], 1e-5)
        self.assertGreater(fit["r_squared"], 0.999999)

        tc = rep["target_cooldown"]
        self.assertTrue(tc["reachable"])
        self.assertFalse(tc["empirical_crossed"])
        self.assertAlmostEqual(tc["predicted_elapsed_from_start_s"], expected_target_s, places=3)
        self.assertAlmostEqual(tc["predicted_remaining_from_end_s"], expected_target_s - 60.0, places=3)

        # 2) Deterministic zero-mean small perturbation over 0..180s
        noisy_jsonl = self.work_dir / "noisy_decay.jsonl"
        noisy_rows = []
        for k in range(181):
            t = float(k)
            ripple = 0.08 * math.sin(1.7 * t) - 0.05 * math.cos(2.9 * t)
            temp = t_amb_true + (t0_true - t_amb_true) * math.exp(-t / tau_true) + ripple
            noisy_rows.append(json.dumps({"time_s": t, "temp_c": round(temp, 6)}))
        noisy_jsonl.write_text("\n".join(noisy_rows) + "\n", encoding="utf-8")
        noisy_out = self.work_dir / "noisy_out.json"

        proc_noisy = subprocess.run(
            [
                str(self.bin_path),
                "--trace", str(noisy_jsonl),
                "--target-c", str(target_c),
                "--output", str(noisy_out),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(proc_noisy.returncode, 0, msg=proc_noisy.stderr)
        nrep = json.loads(noisy_out.read_text(encoding="utf-8"))
        self.assertAlmostEqual(nrep["newton_cooling"]["tau_s"], tau_true, delta=0.8)
        self.assertAlmostEqual(nrep["newton_cooling"]["t_amb_c"], t_amb_true, delta=0.25)
        self.assertTrue(nrep["target_cooldown"]["empirical_crossed"])
        self.assertAlmostEqual(
            nrep["target_cooldown"]["predicted_target_time_s"], expected_target_s, delta=1.2
        )

    def test_plateau_detection_distinguishes_oscillating_from_settled_trace(self):
        # Settled trace: exponential decay settling around 31.0C by ~120s over 200s
        settled_csv = self.work_dir / "settled.csv"
        settled_lines = ["time_s,temp_c"]
        for k in range(201):
            t = float(k)
            temp = 31.0 + 35.0 * math.exp(-t / 18.0)
            settled_lines.append(f"{t:.1f},{temp:.6f}")
        settled_csv.write_text("\n".join(settled_lines) + "\n", encoding="utf-8")
        settled_out = self.work_dir / "settled_out.json"

        proc_ok = subprocess.run(
            [
                str(self.bin_path),
                "--trace", str(settled_csv),
                "--hold-s", "25",
                "--max-slope-c-per-s", "0.015",
                "--max-span-c", "0.40",
                "--output", str(settled_out),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(proc_ok.returncode, 0, msg=proc_ok.stderr)
        s_rep = json.loads(settled_out.read_text(encoding="utf-8"))["plateau"]
        self.assertTrue(s_rep["detected"])
        self.assertTrue(s_rep["settled_at_end"])
        self.assertGreater(s_rep["onset_time_s"], 70.0)
        self.assertAlmostEqual(s_rep["trailing_mean_temp_c"], 31.0, delta=0.05)

        # Oscillating trace: +/- 1.2C sustained thermal hunting -> span ~2.4C > 0.40C
        osc_csv = self.work_dir / "oscillating.csv"
        osc_lines = ["time_s,temp_c"]
        for k in range(120):
            t = float(k)
            temp = 40.0 + 1.2 * math.sin(2.0 * math.pi * t / 10.0)
            osc_lines.append(f"{t:.1f},{temp:.6f}")
        osc_csv.write_text("\n".join(osc_lines) + "\n", encoding="utf-8")
        osc_out = self.work_dir / "osc_out.json"

        proc_osc = subprocess.run(
            [
                str(self.bin_path),
                "--trace", str(osc_csv),
                "--hold-s", "25",
                "--max-slope-c-per-s", "0.05",
                "--max-span-c", "0.40",
                "--output", str(osc_out),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(proc_osc.returncode, 0, msg=proc_osc.stderr)
        o_rep = json.loads(osc_out.read_text(encoding="utf-8"))["plateau"]
        self.assertFalse(o_rep["detected"])
        self.assertFalse(o_rep["settled_at_end"])

    def test_multi_batch_thermal_creep_flagging_and_stable_batches(self):
        creep_csv = self.work_dir / "batches_creep.csv"
        creep_csv.write_text(
            "batch_id,start_time_s,end_time_s,start_temp_c,end_temp_c\n"
            "b0,0,30,32.0,48.0\n"
            "b1,45,75,33.1,49.0\n"
            "b2,90,120,34.3,50.2\n"
            "b3,135,165,35.6,51.4\n",
            encoding="utf-8",
        )
        creep_out = self.work_dir / "creep_out.json"

        proc_creep = subprocess.run(
            [
                str(self.bin_path),
                "--batches", str(creep_csv),
                "--max-total-drift-c", "1.5",
                "--max-drift-per-batch-c", "0.4",
                "--max-step-rise-c", "1.0",
                "--fail-on-creep",
                "--output", str(creep_out),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(proc_creep.returncode, 1)
        bd = json.loads(creep_out.read_text(encoding="utf-8"))["batch_drift"]
        self.assertTrue(bd["thermal_creep_detected"])
        self.assertAlmostEqual(bd["total_start_drift_c"], 3.6, places=5)
        self.assertGreater(bd["start_slope_c_per_batch"], 1.1)
        self.assertIn("total_start_drift_exceeded", bd["creep_reasons"])
        self.assertIn("per_batch_start_slope_exceeded", bd["creep_reasons"])
        self.assertIn("step_start_rise_exceeded", bd["creep_reasons"])

        stable_csv = self.work_dir / "batches_stable.csv"
        stable_csv.write_text(
            "batch_id,start_temp_c,end_temp_c\n"
            "b0,32.0,46.0\n"
            "b1,32.2,46.1\n"
            "b2,32.1,45.9\n"
            "b3,32.3,46.2\n",
            encoding="utf-8",
        )
        stable_out = self.work_dir / "stable_out.json"
        proc_stable = subprocess.run(
            [str(self.bin_path), "--batches", str(stable_csv), "--fail-on-creep", "--output", str(stable_out)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(proc_stable.returncode, 0, msg=proc_stable.stderr)
        sbd = json.loads(stable_out.read_text(encoding="utf-8"))["batch_drift"]
        self.assertFalse(sbd["thermal_creep_detected"])

    def test_cli_error_cases_and_exclusive_output_guard(self):
        # Non-monotonic timestamps rejected
        bad_csv = self.work_dir / "bad.csv"
        bad_csv.write_text("time_s,temp_c\n0,50\n5,45\n5,44\n", encoding="utf-8")
        bad_out = self.work_dir / "bad.json"
        proc_bad = subprocess.run(
            [str(self.bin_path), "--trace", str(bad_csv), "--output", str(bad_out)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(proc_bad.returncode, 2)
        self.assertFalse(bad_out.exists())

        # Pre-existing --output rejected without modification
        good_csv = self.work_dir / "good.csv"
        good_csv.write_text("time_s,temp_c\n0,60\n10,45\n20,36\n30,31\n", encoding="utf-8")
        existing = self.work_dir / "existing.json"
        existing.write_text("KEEP_INTACT\n", encoding="utf-8")
        proc_excl = subprocess.run(
            [str(self.bin_path), "--trace", str(good_csv), "--output", str(existing)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(proc_excl.returncode, 2)
        self.assertEqual(existing.read_text(encoding="utf-8"), "KEEP_INTACT\n")


if __name__ == "__main__":
    unittest.main()
