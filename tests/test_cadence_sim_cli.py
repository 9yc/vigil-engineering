import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class TestCadenceSimCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifact_dir = Path(os.environ["ARTIFACT_DIR"])
        cls.binary = artifact_dir / "cadence_sim_cli"
        if not cls.binary.is_file():
            raise RuntimeError(f"missing binary: {cls.binary}")
        cls.base_tmp = Path(os.environ["TMPDIR"])
        cls.base_tmp.mkdir(parents=True, exist_ok=True)

    def setUp(self) -> None:
        self._tmp_ctx = tempfile.TemporaryDirectory(dir=self.base_tmp)
        self.work_dir = Path(self._tmp_ctx.name)

    def tearDown(self) -> None:
        self._tmp_ctx.cleanup()

    def _write_csv_trace(self, name: str, rows: list[tuple[str, float, float, float]]) -> Path:
        path = self.work_dir / name
        lines = ["# synthetic request trace", "request_id,arrival_s,compute_units,deadline_s"]
        for req_id, arr_s, cu, ddl_s in rows:
            lines.append(f"{req_id},{arr_s:.6f},{cu:.6f},{ddl_s:.6f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _run_cli(self, extra_args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.binary), *extra_args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_rc_thermal_equilibrium_and_analytical_step_decay(self) -> None:
        amb_c = 24.0
        init_hot_c = 72.0
        c_th = 10.0
        r_th = 5.0
        idle_w = 0.50
        nom_w = 8.50
        nom_rate = 100.0
        tau_s = r_th * c_th
        t_idle_inf = amb_c + idle_w * r_th
        t_nom_inf = amb_c + nom_w * r_th

        idle_gap_s = 35.0
        exec_dur_s = 50.0
        trace_path = self._write_csv_trace(
            "rc_step.csv",
            [("req_step", idle_gap_s, nom_rate * exec_dur_s, 200.0)],
        )
        out_path = self.work_dir / "rc_step_out.json"

        proc = self._run_cli([
            "--input", str(trace_path),
            "--output", str(out_path),
            "--mode", "burst",
            "--ambient-temp-c", str(amb_c),
            "--initial-temp-c", str(init_hot_c),
            "--c-th", str(c_th),
            "--r-th", str(r_th),
            "--idle-power-w", str(idle_w),
            "--nominal-power-w", str(nom_w),
            "--nominal-rate", str(nom_rate),
            "--throttle-temp-c", "150.0",
            "--severe-throttle-temp-c", "160.0",
            "--dt-s", "0.005",
            "--include-outcomes",
        ])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertAlmostEqual(payload["thermal_tau_s"], tau_s, places=9)
        self.assertAlmostEqual(payload["idle_equilibrium_temp_c"], t_idle_inf, places=9)
        self.assertAlmostEqual(payload["nominal_equilibrium_temp_c"], t_nom_inf, places=9)

        temp_after_idle = t_idle_inf + (init_hot_c - t_idle_inf) * math.exp(-idle_gap_s / tau_s)
        expected_final = t_nom_inf + (temp_after_idle - t_nom_inf) * math.exp(-exec_dur_s / tau_s)
        burst_res = payload["results"][0]
        self.assertAlmostEqual(burst_res["final_temp_c"], expected_final, places=7)
        self.assertAlmostEqual(burst_res["throttled_duration_s"], 0.0, places=9)

        long_trace = self._write_csv_trace(
            "rc_eq.csv",
            [("req_eq", 0.0, nom_rate * 600.0, 1000.0)],
        )
        eq_out = self.work_dir / "rc_eq_out.json"
        proc_eq = self._run_cli([
            "--input", str(long_trace),
            "--output", str(eq_out),
            "--mode", "burst",
            "--ambient-temp-c", str(amb_c),
            "--initial-temp-c", str(amb_c),
            "--c-th", str(c_th),
            "--r-th", str(r_th),
            "--idle-power-w", str(idle_w),
            "--nominal-power-w", str(nom_w),
            "--nominal-rate", str(nom_rate),
            "--throttle-temp-c", "150.0",
            "--severe-throttle-temp-c", "160.0",
        ])
        self.assertEqual(proc_eq.returncode, 0, msg=proc_eq.stderr)
        eq_data = json.loads(eq_out.read_text(encoding="utf-8"))["by_mode"]["burst"]
        self.assertAlmostEqual(eq_data["final_temp_c"], t_nom_inf, places=3)
        self.assertAlmostEqual(eq_data["peak_temp_c"], t_nom_inf, places=3)

    def test_burst_thermal_throttling_vs_regulated_unthrottled_execution(self) -> None:
        rows = [(f"r_{i:02d}", i * 0.25, 45.0, 35.0) for i in range(28)]
        trace_path = self._write_csv_trace("thermal_compare.csv", rows)
        out_path = self.work_dir / "compare_out.json"

        proc = self._run_cli([
            "--input", str(trace_path),
            "--output", str(out_path),
            "--modes", "burst,fixed_sleep,thermal_feedback",
            "--ambient-temp-c", "25.0",
            "--initial-temp-c", "42.0",
            "--c-th", "4.0",
            "--r-th", "5.0",
            "--idle-power-w", "0.35",
            "--nominal-power-w", "9.50",
            "--nominal-rate", "100.0",
            "--throttle-temp-c", "54.0",
            "--severe-throttle-temp-c", "58.0",
            "--throttle-scale", "0.40",
            "--severe-scale", "0.18",
            "--power-exponent", "2.0",
            "--throttle-leakage-w", "1.20",
            "--fixed-sleep-s", "0.55",
            "--target-temp-c", "47.0",
            "--hysteresis-c", "1.0",
            "--kp", "0.12",
            "--ki", "0.04",
            "--min-scale", "0.35",
        ])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        by_mode = json.loads(out_path.read_text(encoding="utf-8"))["by_mode"]

        burst = by_mode["burst"]
        sleep_m = by_mode["fixed_sleep"]
        feedback = by_mode["thermal_feedback"]

        self.assertGreater(burst["peak_temp_c"], 54.0)
        self.assertGreater(burst["throttled_duration_s"], 1.5)

        self.assertLess(sleep_m["peak_temp_c"], 54.0)
        self.assertEqual(sleep_m["throttled_duration_s"], 0.0)
        self.assertLess(sleep_m["active_duty_factor"], burst["active_duty_factor"])

        self.assertLess(feedback["peak_temp_c"], 54.0)
        self.assertEqual(feedback["throttled_duration_s"], 0.0)
        self.assertLess(feedback["total_energy_j"], burst["total_energy_j"])
        self.assertEqual(feedback["timely_completion_rate"], 1.0)

    def test_token_bucket_burst_allowance_and_steady_refill(self) -> None:
        rows = [(f"b_{i}", 0.0, 50.0, 10.0) for i in range(6)]
        trace_path = self._write_csv_trace("bucket_burst.csv", rows)
        out_path = self.work_dir / "bucket_out.json"

        proc = self._run_cli([
            "--input", str(trace_path),
            "--output", str(out_path),
            "--modes", "burst,token_bucket",
            "--nominal-rate", "100.0",
            "--throttle-temp-c", "120.0",
            "--severe-throttle-temp-c", "130.0",
            "--bucket-capacity", "150.0",
            "--bucket-rate", "25.0",
            "--include-outcomes",
        ])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        results = {r["mode"]: r for r in json.loads(out_path.read_text(encoding="utf-8"))["results"]}

        burst_outcomes = results["burst"]["outcomes"]
        tb_outcomes = results["token_bucket"]["outcomes"]

        for idx in range(3):
            self.assertAlmostEqual(
                tb_outcomes[idx]["finish_s"], burst_outcomes[idx]["finish_s"], places=5
            )

        transition_wait = tb_outcomes[3]["start_s"] - tb_outcomes[2]["finish_s"]
        self.assertAlmostEqual(transition_wait, 0.50, places=4)
        for idx in (4, 5):
            steady_wait = tb_outcomes[idx]["start_s"] - tb_outcomes[idx - 1]["finish_s"]
            steady_period = tb_outcomes[idx]["finish_s"] - tb_outcomes[idx - 1]["finish_s"]
            self.assertAlmostEqual(steady_wait, 1.50, places=4)
            self.assertAlmostEqual(steady_period, 2.00, places=4)

    def test_deadline_miss_accounting_percentiles_and_jsonl_parity(self) -> None:
        rows = [
            ("q3", 1.00, 100.0, 1.50),
            ("q1", 0.00, 100.0, 1.00),
            ("q2", 0.00, 100.0, 1.50),
            ("q4", 1.00, 100.0, 3.00),
        ]
        csv_path = self._write_csv_trace("unsorted.csv", rows)
        jsonl_path = self.work_dir / "unsorted.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for req_id, arr_s, cu, ddl_s in rows:
                fh.write(json.dumps({
                    "request_id": req_id,
                    "arrival_s": arr_s,
                    "compute_units": cu,
                    "deadline_s": ddl_s,
                }) + "\n")

        out_csv = self.work_dir / "out_csv.json"
        out_jsonl = self.work_dir / "out_jsonl.json"
        common = [
            "--mode", "burst",
            "--nominal-rate", "100.0",
            "--throttle-temp-c", "120.0",
            "--severe-throttle-temp-c", "130.0",
            "--include-outcomes",
        ]
        p_csv = self._run_cli(["--input", str(csv_path), "--output", str(out_csv), *common])
        p_js = self._run_cli(["--input", str(jsonl_path), "--output", str(out_jsonl), *common])
        self.assertEqual(p_csv.returncode, 0, msg=p_csv.stderr)
        self.assertEqual(p_js.returncode, 0, msg=p_js.stderr)

        res_csv = json.loads(out_csv.read_text(encoding="utf-8"))["results"][0]
        res_js = json.loads(out_jsonl.read_text(encoding="utf-8"))["results"][0]
        self.assertEqual(res_csv["total_requests"], 4)
        self.assertEqual(res_csv["timely_requests"], 2)
        self.assertAlmostEqual(res_csv["timely_completion_rate"], 0.50, places=9)
        self.assertAlmostEqual(res_csv["p50_latency_s"], res_js["p50_latency_s"], places=9)
        self.assertAlmostEqual(res_csv["p95_latency_s"], res_js["p95_latency_s"], places=9)
        self.assertAlmostEqual(res_csv["p99_latency_s"], res_js["p99_latency_s"], places=9)
        self.assertAlmostEqual(res_csv["total_energy_j"], res_js["total_energy_j"], places=9)
        self.assertAlmostEqual(res_csv["active_duty_factor"], 1.0, places=6)

        by_id = {o["request_id"]: o for o in res_csv["outcomes"]}
        self.assertTrue(by_id["q1"]["met_deadline"])
        self.assertFalse(by_id["q2"]["met_deadline"])
        self.assertFalse(by_id["q3"]["met_deadline"])
        self.assertTrue(by_id["q4"]["met_deadline"])

    def test_exclusive_output_guard_and_invalid_cli_inputs(self) -> None:
        trace_path = self._write_csv_trace("guard.csv", [("r1", 0.0, 25.0, 2.0)])
        existing_out = self.work_dir / "already_exists.json"
        sentinel = "{\"do_not_clobber\": true}\n"
        existing_out.write_text(sentinel, encoding="utf-8")

        proc_dup = self._run_cli([
            "--input", str(trace_path),
            "--output", str(existing_out),
            "--mode", "burst",
        ])
        self.assertNotEqual(proc_dup.returncode, 0)
        self.assertEqual(existing_out.read_text(encoding="utf-8"), sentinel)

        bad_trace = self._write_csv_trace("bad.csv", [("r_bad", 0.0, -5.0, 1.0)])
        proc_bad = self._run_cli([
            "--input", str(bad_trace),
            "--output", str(self.work_dir / "bad_out.json"),
        ])
        self.assertNotEqual(proc_bad.returncode, 0)

        proc_flag = self._run_cli([
            "--input", str(trace_path),
            "--output", str(self.work_dir / "flag_out.json"),
            "--bogus-flag", "123",
        ])
        self.assertNotEqual(proc_flag.returncode, 0)


if __name__ == "__main__":
    unittest.main()
