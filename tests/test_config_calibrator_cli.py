import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def exact_binom_upper_tail(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, j) * (p ** j) * ((1.0 - p) ** (n - j)) for j in range(k, n + 1))


def exact_binom_lower_tail(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, j) * (p ** j) * ((1.0 - p) ** (n - j)) for j in range(0, k + 1))


class TestConfigCalibratorCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifact_dir = Path(os.environ["ARTIFACT_DIR"])
        cls.binary = artifact_dir / "config_calibrator_cli"
        if not cls.binary.is_file():
            raise RuntimeError(f"missing binary: {cls.binary}")
        cls.base_tmp = Path(os.environ["TMPDIR"])
        cls.base_tmp.mkdir(parents=True, exist_ok=True)

    def setUp(self) -> None:
        self._tmp_ctx = tempfile.TemporaryDirectory(dir=self.base_tmp)
        self.work_dir = Path(self._tmp_ctx.name)

    def tearDown(self) -> None:
        self._tmp_ctx.cleanup()

    def _run_cli(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.binary), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def _write_csv(
        self, name: str, rows: list[tuple[str, str, int, float, float, int]]
    ) -> Path:
        path = self.work_dir / name
        lines = ["request_id,config_id,passed,latency_s,energy_j,tokens"]
        for req_id, cfg_id, passed, lat_s, eng_j, toks in rows:
            lines.append(f"{req_id},{cfg_id},{passed},{lat_s:.6f},{eng_j:.6f},{toks}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_betainc_identities_and_clopper_pearson_exact_binomial_sums(self) -> None:
        queries: list[dict[str, object]] = []
        shape_cases = [
            (0.5, 0.5, 0.25),
            (1.0, 1.0, 0.37),
            (3.5, 1.0, 0.62),
            (1.0, 4.2, 0.41),
            (2.5, 7.5, 0.30),
            (18.0, 6.0, 0.72),
            (65.0, 45.0, 0.58),
        ]
        for a, b, x in shape_cases:
            queries.append({"op": "betainc", "a": a, "b": b, "x": 0.0})
            queries.append({"op": "betainc", "a": a, "b": b, "x": 1.0})
            queries.append({"op": "betainc", "a": a, "b": b, "x": x})
            queries.append({"op": "betainc", "a": b, "b": a, "x": 1.0 - x})
            queries.append({"op": "beta_inv", "p": 0.15, "a": a, "b": b})

        cp_cases: list[tuple[int, int, float]] = []
        for n in (5, 12, 20):
            for alpha in (0.01, 0.05, 0.10):
                cp_cases.append((0, n, alpha))
                cp_cases.append((n, n, alpha))
                for k in (1, n // 2, n - 1):
                    cp_cases.append((k, n, alpha))
        for k, n, alpha in cp_cases:
            queries.append({"op": "cp", "k": k, "n": n, "alpha": alpha})

        batch_in = self.work_dir / "math_queries.jsonl"
        batch_in.write_text(
            "\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8"
        )
        batch_out = self.work_dir / "math_replies.jsonl"
        proc = self._run_cli(["--math-batch", str(batch_in), "--output", str(batch_out)])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

        replies = [json.loads(line) for line in batch_out.read_text(encoding="utf-8").splitlines()]
        cursor = 0
        for a, b, x in shape_cases:
            val_zero = replies[cursor]["value"]
            val_one = replies[cursor + 1]["value"]
            val_x = replies[cursor + 2]["value"]
            val_sym = replies[cursor + 3]["value"]
            inv_x = replies[cursor + 4]["value"]
            cursor += 5

            self.assertEqual(val_zero, 0.0)
            self.assertEqual(val_one, 1.0)
            self.assertAlmostEqual(val_x + val_sym, 1.0, places=13)
            if a == 1.0 and b == 1.0:
                self.assertAlmostEqual(val_x, x, places=14)
            elif b == 1.0:
                self.assertAlmostEqual(val_x, x ** a, places=13)
            elif a == 1.0:
                self.assertAlmostEqual(val_x, 1.0 - ((1.0 - x) ** b), places=13)

            roundtrip_in = self.work_dir / f"rt_{cursor}.jsonl"
            roundtrip_out = self.work_dir / f"rt_out_{cursor}.jsonl"
            roundtrip_in.write_text(
                json.dumps({"op": "betainc", "a": a, "b": b, "x": inv_x}) + "\n",
                encoding="utf-8",
            )
            rt_proc = self._run_cli([
                "--math-batch", str(roundtrip_in), "--output", str(roundtrip_out)
            ])
            self.assertEqual(rt_proc.returncode, 0, msg=rt_proc.stderr)
            rt_val = json.loads(roundtrip_out.read_text(encoding="utf-8").strip())["value"]
            self.assertAlmostEqual(rt_val, 0.15, places=12)

        for k, n, alpha in cp_cases:
            cp_res = replies[cursor]
            cursor += 1
            lo = cp_res["lower"]
            hi = cp_res["upper"]
            if k == 0:
                self.assertEqual(lo, 0.0)
                self.assertAlmostEqual(hi, 1.0 - (alpha ** (1.0 / n)), places=13)
            elif k == n:
                self.assertAlmostEqual(lo, alpha ** (1.0 / n), places=13)
                self.assertEqual(hi, 1.0)
            else:
                self.assertLess(lo, k / n)
                self.assertGreater(hi, k / n)
                self.assertAlmostEqual(exact_binom_upper_tail(k, n, lo), alpha, places=12)
                self.assertAlmostEqual(exact_binom_lower_tail(k, n, hi), alpha, places=12)

    def test_bonferroni_conservatism_vs_point_screening_and_oracle_gap(self) -> None:
        cal_rows: list[tuple[str, str, int, float, float, int]] = []
        eval_rows: list[tuple[str, str, int, float, float, int]] = []

        for i in range(25):
            cal_rows.append((f"c_a_{i}", "cfg_low_power_noisy", 1 if i < 23 else 0, 0.40, 1.20, 64))
        for i in range(200):
            eval_rows.append((f"e_a_{i}", "cfg_low_power_noisy", 1 if i < 168 else 0, 0.40, 1.20, 64))

        for i in range(150):
            cal_rows.append((f"c_b_{i}", "cfg_mid_oracle", 1 if i < 139 else 0, 0.55, 2.00, 64))
        for i in range(200):
            eval_rows.append((f"e_b_{i}", "cfg_mid_oracle", 1 if i < 184 else 0, 0.55, 2.00, 64))

        for i in range(200):
            cal_rows.append((f"c_c_{i}", "cfg_safe_certified", 1 if i < 194 else 0, 0.65, 2.75, 64))
            eval_rows.append((f"e_c_{i}", "cfg_safe_certified", 1 if i < 193 else 0, 0.65, 2.75, 64))

        for i in range(100):
            cal_rows.append((f"c_d_{i}", "cfg_heavy", 1 if i < 99 else 0, 0.70, 4.10, 64))
            eval_rows.append((f"e_d_{i}", "cfg_heavy", 1 if i < 99 else 0, 0.70, 4.10, 64))

        cal_csv = self._write_csv("cal.csv", cal_rows)
        eval_csv = self._write_csv("eval.csv", eval_rows)

        point_out = self.work_dir / "point_report.json"
        cp_out = self.work_dir / "cp_report.json"
        common = [
            "--cal", str(cal_csv),
            "--eval", str(eval_csv),
            "--utility-mode", "hard_deadline",
            "--deadline-s", "1.0",
            "--target", "0.90",
            "--delta", "0.04",
        ]

        p_pt = self._run_cli([*common, "--screening", "point", "--output", str(point_out)])
        p_cp = self._run_cli([*common, "--screening", "clopper_pearson", "--output", str(cp_out)])
        self.assertEqual(p_pt.returncode, 0, msg=p_pt.stderr)
        self.assertEqual(p_cp.returncode, 0, msg=p_cp.stderr)

        rep_pt = json.loads(point_out.read_text(encoding="utf-8"))
        rep_cp = json.loads(cp_out.read_text(encoding="utf-8"))

        self.assertEqual(rep_pt["num_candidates"], 4)
        self.assertAlmostEqual(rep_cp["bonferroni_alpha"], 0.01, places=12)

        self.assertEqual(rep_pt["selected_config_id"], "cfg_low_power_noisy")
        self.assertFalse(rep_pt["held_out_target_attained"])
        self.assertAlmostEqual(rep_pt["held_out_utility"], 168 / 200, places=9)

        self.assertEqual(rep_cp["selected_config_id"], "cfg_safe_certified")
        self.assertTrue(rep_cp["held_out_target_attained"])
        self.assertEqual(rep_cp["oracle_config_id"], "cfg_mid_oracle")
        self.assertAlmostEqual(rep_cp["oracle_mean_energy_j"], 2.00, places=9)
        self.assertAlmostEqual(rep_cp["held_out_mean_energy_j"], 2.75, places=9)
        self.assertAlmostEqual(rep_cp["oracle_energy_gap_j"], 0.75, places=9)
        self.assertAlmostEqual(rep_cp["oracle_energy_ratio"], 2.75 / 2.00, places=9)

    def test_soft_exponential_utility_lexicographical_tiebreak_and_infeasible(self) -> None:
        tau_s = 2.0
        lat_s = 0.20
        cal_rows = []
        for cid in ("cfg_zeta", "cfg_alpha"):
            for i in range(50):
                cal_rows.append((f"r_{cid}_{i}", cid, 1, lat_s, 1.85, 50))

        jsonl_cal = self.work_dir / "soft_cal.jsonl"
        with jsonl_cal.open("w", encoding="utf-8") as fh:
            for req_id, cfg_id, passed, l_s, e_j, toks in cal_rows:
                fh.write(json.dumps({
                    "request_id": req_id,
                    "config_id": cfg_id,
                    "passed": bool(passed),
                    "latency_s": l_s,
                    "energy_j": e_j,
                    "tokens": toks,
                }) + "\n")

        soft_out = self.work_dir / "soft_out.json"
        proc = self._run_cli([
            "--cal", str(jsonl_cal),
            "--output", str(soft_out),
            "--utility-mode", "soft_exp",
            "--tau-s", str(tau_s),
            "--screening", "point",
            "--target", "0.85",
        ])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        rep = json.loads(soft_out.read_text(encoding="utf-8"))
        self.assertTrue(rep["feasible"])
        self.assertEqual(rep["selected_config_id"], "cfg_alpha")
        self.assertAlmostEqual(rep["selected_cal_utility"], math.exp(-lat_s / tau_s), places=12)

        infeas_out = self.work_dir / "infeasible_out.json"
        proc_inf = self._run_cli([
            "--cal", str(jsonl_cal),
            "--output", str(infeas_out),
            "--utility-mode", "soft_exp",
            "--tau-s", str(tau_s),
            "--screening", "clopper_pearson",
            "--target", "0.98",
        ])
        self.assertEqual(proc_inf.returncode, 0, msg=proc_inf.stderr)
        rep_inf = json.loads(infeas_out.read_text(encoding="utf-8"))
        self.assertFalse(rep_inf["feasible"])
        self.assertIsNone(rep_inf["selected_config_id"])

    def test_exclusive_output_and_cli_validation(self) -> None:
        cal_csv = self._write_csv("mini.csv", [("r1", "c1", 1, 0.2, 1.0, 10)])
        existing = self.work_dir / "existing.json"
        sentinel = "{\"keep\": 1}\n"
        existing.write_text(sentinel, encoding="utf-8")

        proc_dup = self._run_cli(["--cal", str(cal_csv), "--output", str(existing)])
        self.assertNotEqual(proc_dup.returncode, 0)
        self.assertEqual(existing.read_text(encoding="utf-8"), sentinel)

        for bad_args in (
            ["--cal", str(cal_csv), "--output", str(self.work_dir / "o1.json"), "--delta", "1.0"],
            ["--cal", str(cal_csv), "--output", str(self.work_dir / "o2.json"), "--target", "0.0"],
            ["--cal", str(cal_csv), "--output", str(self.work_dir / "o3.json"), "--deadline-s", "-1"],
            ["--output", str(self.work_dir / "o4.json")],
        ):
            proc_bad = self._run_cli(bad_args)
            self.assertNotEqual(proc_bad.returncode, 0, msg=f"expected failure for {bad_args}")

    def test_sample_size_bound_tightening_and_energy_per_token(self) -> None:
        rows: list[tuple[str, str, int, float, float, int]] = []
        for n, label in ((20, "cfg_n20"), (100, "cfg_n100"), (500, "cfg_n500")):
            k = int(round(0.95 * n))
            for i in range(n):
                rows.append((f"r_{label}_{i}", label, 1 if i < k else 0, 0.35, 2.50, 125))

        cal_csv = self._write_csv("tightening.csv", rows)
        out_json = self.work_dir / "tightening_out.json"
        proc = self._run_cli([
            "--cal", str(cal_csv),
            "--output", str(out_json),
            "--deadline-s", "0.50",
            "--target", "0.90",
            "--delta", "0.03",
        ])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        rep = json.loads(out_json.read_text(encoding="utf-8"))
        by_cfg = {c["config_id"]: c for c in rep["candidates"]}

        self.assertLess(by_cfg["cfg_n20"]["cp_lower_bound"], by_cfg["cfg_n100"]["cp_lower_bound"])
        self.assertLess(by_cfg["cfg_n100"]["cp_lower_bound"], by_cfg["cfg_n500"]["cp_lower_bound"])
        self.assertGreater(by_cfg["cfg_n20"]["cp_upper_bound"], by_cfg["cfg_n100"]["cp_upper_bound"])
        self.assertAlmostEqual(by_cfg["cfg_n100"]["energy_per_token_j"], 2.50 / 125.0, places=12)


if __name__ == "__main__":
    unittest.main()
