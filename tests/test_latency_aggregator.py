import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest


def exact_percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    if pct <= 0.0:
        return sorted_vals[0]
    if pct >= 100.0:
        return sorted_vals[-1]
    rank = (pct / 100.0) * len(sorted_vals)
    lo = max(0, int(math.floor(rank)) - 1)
    hi = min(len(sorted_vals) - 1, int(math.ceil(rank)) - 1)
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - math.floor(rank)
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


class TestLatencyAggregator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifact_dir = Path(os.environ["ARTIFACT_DIR"])
        cls.bin_path = artifact_dir / "latency_aggregator"
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
            self.fail(f"CLI failed ({proc.returncode}): {proc.stderr}")
        if not expect_ok and proc.returncode == 0:
            self.fail(f"Expected CLI failure, but succeeded: {proc.stdout}")
        return proc

    def test_hdr_percentiles_and_cdf_against_sorted_reference(self):
        rng = random.Random(42)
        in_csv = self.root / "lognormal.csv"
        out_json = self.root / "report.json"

        latencies = []
        with in_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["arrival_s", "duration_ms", "group", "status"])
            t = 0.0
            for _ in range(4000):
                t += rng.expovariate(200.0)
                d = round(rng.lognormvariate(2.5, 0.75), 4)
                latencies.append(d)
                writer.writerow([f"{t:.6f}", f"{d:.4f}", "read_api", "ok"])

        self.run_cli([
            "--input", str(in_csv),
            "--output", str(out_json),
            "--sigfigs", "4",
            "--slo-ms", "45.0",
        ])
        report = json.loads(out_json.read_text(encoding="utf-8"))
        overall = report["overall"]
        self.assertEqual(overall["count"], 4000)
        self.assertEqual(overall["error_count"], 0)
        self.assertEqual(overall["overflow_count"], 0)

        sorted_lat = sorted(latencies)
        mean_ref = sum(sorted_lat) / len(sorted_lat)
        var_ref = sum((x - mean_ref) ** 2 for x in sorted_lat) / len(sorted_lat)
        std_ref = math.sqrt(var_ref)

        self.assertAlmostEqual(overall["min_ms"], sorted_lat[0], places=3)
        self.assertAlmostEqual(overall["max_ms"], sorted_lat[-1], places=3)
        self.assertAlmostEqual(overall["mean_ms"], mean_ref, delta=mean_ref * 0.002)
        self.assertAlmostEqual(overall["stddev_ms"], std_ref, delta=std_ref * 0.003)

        for key, pct in [("P50", 50.0), ("P90", 90.0), ("P95", 95.0), ("P99", 99.0), ("P99.9", 99.9)]:
            ref = exact_percentile(sorted_lat, pct)
            got = overall["percentiles_ms"][key]
            self.assertAlmostEqual(got, ref, delta=max(0.02, ref * 0.015), msg=f"Mismatch on {key}")

        expected_slo = sum(1 for x in latencies if x <= 45.0) / len(latencies)
        self.assertAlmostEqual(overall["slo"]["attainment_ratio"], expected_slo, places=6)

        cdf = overall["cdf"]
        self.assertGreater(len(cdf), 0)
        self.assertLessEqual(len(cdf), 8)
        self.assertAlmostEqual(cdf[-1]["cdf"], 1.0, places=6)
        self.assertEqual(sum(b["count"] for b in cdf), 4000)
        for a, b in zip(cdf, cdf[1:]):
            self.assertLessEqual(a["cdf"], b["cdf"])

    def test_multigroup_jsonl_warmup_and_littles_law(self):
        in_jsonl = self.root / "trace.jsonl"
        rows = []
        for i in range(50):
            rows.append({
                "start_s": i * 0.1,
                "latency_ms": 900.0,
                "group": "warmup_burst",
                "status": "ok",
            })
        for i in range(200):
            arr = 5.0 + i * 0.05
            grp = "alpha" if i % 2 == 0 else "beta"
            dur = 40.0 if grp == "alpha" else 120.0
            status = "error" if (grp == "beta" and i % 20 == 1) else "200"
            rows.append({
                "start_s": arr,
                "latency_ms": dur,
                "group": grp,
                "status": status,
            })

        with in_jsonl.open("w", encoding="utf-8") as fh:
            for item in rows:
                fh.write(json.dumps(item) + "\n")

        proc = self.run_cli([
            "--input", str(in_jsonl),
            "--warmup", "5.0",
            "--window-s", "0.5",
            "--slo-ms", "100.0",
            "--percentiles", "50,95,99",
        ])
        report = json.loads(proc.stdout)
        self.assertEqual(report["warmup_skipped"], 50)
        overall = report["overall"]
        self.assertEqual(overall["count"], 200)
        self.assertEqual(overall["error_count"], 10)
        self.assertIn("alpha", report["groups"])
        self.assertIn("beta", report["groups"])
        self.assertNotIn("warmup_burst", report["groups"])

        conc = overall["concurrency"]
        tput = overall["throughput"]
        expected_little = tput["mean_rps"] * (overall["mean_ms"] / 1000.0)
        self.assertAlmostEqual(conc["littles_law_mean"], expected_little, delta=1e-4)
        self.assertAlmostEqual(conc["littles_law_mean"], conc["time_weighted_mean"], delta=1e-3)
        self.assertGreaterEqual(conc["peak_active"], 2)
        self.assertAlmostEqual(tput["peak_arrival_rps"], 22.0, delta=2.5)

        alpha = report["groups"]["alpha"]
        beta = report["groups"]["beta"]
        self.assertEqual(alpha["count"], 100)
        self.assertEqual(beta["count"], 100)
        self.assertAlmostEqual(alpha["slo"]["attainment_ratio"], 1.0, places=6)
        self.assertAlmostEqual(beta["slo"]["attainment_ratio"], 0.0, places=6)

        by_status = json.loads(self.run_cli([
            "--input", str(in_jsonl),
            "--warmup", "5.0",
            "--group-by", "status",
        ]).stdout)
        self.assertEqual(by_status["groups"]["error"]["count"], 10)
        self.assertEqual(by_status["groups"]["200"]["count"], 190)

    def test_overflow_clamping_and_tail_ratios(self):
        in_csv = self.root / "outliers.csv"
        with in_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["arrival_s", "duration_ms", "group"])
            for i in range(98):
                writer.writerow([f"{i * 0.01:.3f}", "10.0", "core"])
            writer.writerow(["1.000", "2500.0", "core"])
            writer.writerow(["1.010", "5000.0", "core"])

        proc = self.run_cli([
            "--input", str(in_csv),
            "--max-trackable-ms", "1000.0",
        ])
        report = json.loads(proc.stdout)
        overall = report["overall"]
        self.assertEqual(overall["count"], 100)
        self.assertEqual(overall["overflow_count"], 2)
        self.assertAlmostEqual(overall["max_ms"], 1000.0, delta=1.0)
        self.assertAlmostEqual(overall["tail_to_median"]["p99_to_p50"], 100.0, delta=1.0)

    def test_non_destructive_output_refuses_clobber(self):
        in_csv = self.root / "tiny.csv"
        in_csv.write_text("arrival_s,duration_ms\n0.1,5.0\n0.2,6.0\n", encoding="utf-8")
        out_path = self.root / "keep_me.json"
        sentinel = "{\"preserved\": true}\n"
        out_path.write_text(sentinel, encoding="utf-8")

        self.run_cli(["--input", str(in_csv), "--output", str(out_path)], expect_ok=False)
        self.assertEqual(out_path.read_text(encoding="utf-8"), sentinel)

    def test_no_group_mode_and_full_warmup_trim_error(self):
        in_csv = self.root / "short.csv"
        in_csv.write_text("start_s,latency_ms,group\n1.0,12.5,g1\n1.5,17.5,g2\n", encoding="utf-8")
        rep = json.loads(self.run_cli(["--input", str(in_csv), "--group-by", "none"]).stdout)
        self.assertEqual(rep["groups"], {})
        self.assertEqual(rep["overall"]["count"], 2)
        self.assertAlmostEqual(rep["overall"]["mean_ms"], 15.0, places=5)
        self.run_cli(["--input", str(in_csv), "--warmup", "10.0"], expect_ok=False)

    def test_malformed_nan_and_negative_inputs_rejected(self):
        bad_cases = [
            ("neg.csv", "arrival_s,duration_ms\n0.1,-2.5\n"),
            ("nan.csv", "arrival_s,duration_ms\n0.1,NaN\n"),
            ("inf.csv", "arrival_s,duration_ms\ninf,4.0\n"),
            ("cols.csv", "arrival_s,duration_ms\n0.1,4.0,extra\n"),
            ("bad.jsonl", "{\"arrival_s\": 0.1, \"duration_ms\": 5.0\n"),
            ("nan.jsonl", "{\"arrival_s\": 0.1, \"duration_ms\": NaN}\n"),
            ("neg.jsonl", "{\"arrival_s\": 0.1, \"duration_ms\": -0.01}\n"),
        ]
        for filename, content in bad_cases:
            with self.subTest(case=filename):
                p = self.root / filename
                p.write_text(content, encoding="utf-8")
                self.run_cli(["--input", str(p)], expect_ok=False)


if __name__ == "__main__":
    unittest.main()
