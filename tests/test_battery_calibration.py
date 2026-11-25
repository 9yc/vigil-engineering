import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.battery_calibration import (
    estimate_dc_internal_resistance,
    fit_linear_power_calibration,
    reconcile_vi_and_coulomb_window,
)


ROOT = Path(__file__).resolve().parents[1]


class BatteryCalibrationTests(unittest.TestCase):
    def test_linear_calibration_ols_recovery_and_validation(self):
        # Exact relation: p_ref = 1.04 * p_raw + 0.12
        pairs = [(1.0, 1.16), (2.5, 2.72), (4.0, 4.28), (6.0, 6.36)]
        cal = fit_linear_power_calibration(pairs)
        self.assertAlmostEqual(cal.gain, 1.04, places=10)
        self.assertAlmostEqual(cal.offset_w, 0.12, places=10)
        self.assertAlmostEqual(cal.r_squared, 1.0, places=10)
        self.assertAlmostEqual(cal.rmse_w, 0.0, places=10)
        self.assertAlmostEqual(cal.apply(5.0), 5.32, places=10)

        for bad in (
            [(1.0, 2.0)],
            [(2.0, 1.0), (2.0, 3.0)],
            [(-0.5, 1.0), (1.0, 2.0)],
            [(1.0, float("nan")), (2.0, 3.0)],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                fit_linear_power_calibration(bad)

    def test_vi_and_coulomb_reconciliation_pass_and_drift_rejection(self):
        # Constant 4.0 V, 1.8 A over 10 s => 18 C = 5000 uAh, 72 J
        vi = [(0.0, 4.0, -1.8), (5.0, 4.0, -1.8), (10.0, 4.0, -1.8)]
        ok = reconcile_vi_and_coulomb_window(
            vi, start_charge_uah=4_000_000, end_charge_uah=3_995_000, max_relative_discrepancy=0.02
        )
        self.assertTrue(ok["consistent"])
        self.assertAlmostEqual(ok["vi_charge_c"], 18.0, places=9)
        self.assertAlmostEqual(ok["counter_charge_c"], 18.0, places=9)
        self.assertAlmostEqual(ok["vi_energy_j"], 72.0, places=9)

        drifted = reconcile_vi_and_coulomb_window(
            vi, start_charge_uah=4_000_000, end_charge_uah=3_993_000, max_relative_discrepancy=0.05
        )
        self.assertFalse(drifted["consistent"])
        self.assertGreater(drifted["relative_charge_diff"], 0.25)

        with self.assertRaises(ValueError):
            reconcile_vi_and_coulomb_window(vi, 3_990_000, 3_995_000)

    def test_dc_internal_resistance_estimation_and_cli(self):
        # Cold bin (~22 C): R_int = 80 mOhm; Warm bin (~37 C): R_int = 50 mOhm; V_ocv = 4.20 V
        samples = [
            (0.0, 4.20 - 0.2 * 0.080, 0.2, 22.0),
            (0.5, 4.20 - 2.2 * 0.080, 2.2, 22.2),
            (1.0, 4.20 - 0.4 * 0.080, 0.4, 22.4),
            # Long 20 s gap should be skipped by max_step_dt_s=1.5
            (21.0, 4.10 - 0.3 * 0.050, 0.3, 36.8),
            (21.5, 4.10 - 2.3 * 0.050, 2.3, 37.0),
            (22.0, 4.10 - 0.3 * 0.050, 0.3, 37.2),
        ]
        res = estimate_dc_internal_resistance(
            samples, min_step_current_a=0.5, max_step_dt_s=1.5, temp_bin_width_c=5.0
        )
        self.assertEqual(res["qualifying_steps"], 4)
        self.assertAlmostEqual(res["mean_mohm"], 65.0, places=6)
        self.assertAlmostEqual(res["temperature_bins"]["20.0_25.0C"]["mean_mohm"], 80.0, places=6)
        self.assertAlmostEqual(res["temperature_bins"]["35.0_40.0C"]["mean_mohm"], 50.0, places=6)

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            tmp_dir = Path(tmp)
            csv_in = tmp_dir / "steps.csv"
            out_json = tmp_dir / "impedance.json"
            lines = ["time_s,voltage_v,current_a,temp_c"] + [
                f"{t},{v},{i},{c}" for t, v, i, c in samples
            ]
            csv_in.write_text("\n".join(lines) + "\n")

            cmd = [
                sys.executable,
                "-B",
                str(ROOT / "tools/battery_calibration.py"),
                "impedance",
                str(csv_in),
                "--min-step-a",
                "0.5",
                "--output",
                str(out_json),
            ]
            first = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            self.assertEqual(first.returncode, 0, first.stderr)
            loaded = json.loads(out_json.read_text())
            self.assertEqual(loaded["qualifying_steps"], 4)

            # Non-destructive output check
            second = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(second.returncode, 0)


if __name__ == "__main__":
    unittest.main()
