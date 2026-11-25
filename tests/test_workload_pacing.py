import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.workload_pacing import (
    analyze_arrival_dispersion,
    analyze_concurrency_and_littles_law,
    evaluate_slo_attainment,
    generate_arrival_schedule,
    rescale_arrival_trace,
)


ROOT = Path(__file__).resolve().parents[1]


class WorkloadPacingTests(unittest.TestCase):
    def test_uniform_poisson_mmpp2_and_pareto_dispersion_hierarchy(self):
        uniform = generate_arrival_schedule(400, 10.0, mode="uniform", seed=7)
        poisson = generate_arrival_schedule(2000, 10.0, mode="poisson", seed=7)
        bursty = generate_arrival_schedule(
            2000, 10.0, mode="mmpp2", seed=7, burst_ratio=12.0, high_prob=0.2, switch_prob=0.1
        )
        heavy = generate_arrival_schedule(2000, 10.0, mode="pareto", seed=7, pareto_alpha=2.2)

        d_uni = analyze_arrival_dispersion(uniform)
        d_poi = analyze_arrival_dispersion(poisson)
        d_bur = analyze_arrival_dispersion(bursty)
        d_par = analyze_arrival_dispersion(heavy)

        self.assertAlmostEqual(d_uni["cv"], 0.0, places=12)
        self.assertAlmostEqual(d_uni["empirical_rate_per_s"], 10.0, places=9)
        self.assertTrue(0.92 <= d_poi["cv"] <= 1.08)
        self.assertGreater(d_bur["cv"], 1.35)
        self.assertGreater(d_bur["fano_factor"], d_poi["fano_factor"])
        self.assertGreater(d_par["cv"], 1.1)

    def test_trace_rescaling_rate_and_burst_exponent(self):
        base = generate_arrival_schedule(200, 4.0, mode="poisson", seed=11)
        d_base = analyze_arrival_dispersion(base)

        doubled = rescale_arrival_trace(base, rate_scale=2.0, burst_exponent=1.0)
        d_doubled = analyze_arrival_dispersion(doubled)
        self.assertAlmostEqual(d_doubled["empirical_rate_per_s"], 2.0 * d_base["empirical_rate_per_s"], places=9)
        self.assertAlmostEqual(d_doubled["cv"], d_base["cv"], places=9)

        # burst_exponent = 0 equalizes all positive gaps into uniform pacing while keeping rate!
        flattened = rescale_arrival_trace(base, rate_scale=1.0, burst_exponent=0.0)
        d_flat = analyze_arrival_dispersion(flattened)
        self.assertAlmostEqual(d_flat["empirical_rate_per_s"], d_base["empirical_rate_per_s"], places=9)
        self.assertAlmostEqual(d_flat["cv"], 0.0, places=9)

        amplified = rescale_arrival_trace(base, rate_scale=1.0, burst_exponent=1.8)
        self.assertGreater(analyze_arrival_dispersion(amplified)["cv"], d_base["cv"])

    def test_exact_littles_law_and_concurrency_watermarks(self):
        # 3 requests over horizon [0, 10]:
        # [0, 4] (4s), [2, 6] (4s), [4, 10] (6s) -> notice departure at 4 happens before arrival at 4!
        # Active count: [0,2)->1, [2,4)->2, [4,6)->2, [6,10)->1 => peak=2, total area = 2 + 4 + 4 + 4 = 14s
        intervals = [(0.0, 4.0), (2.0, 6.0), (4.0, 10.0)]
        q = analyze_concurrency_and_littles_law(intervals)
        self.assertEqual(q["peak_concurrency"], 2)
        self.assertAlmostEqual(q["busy_fraction"], 1.0)
        self.assertAlmostEqual(q["time_avg_concurrency"], 1.4)
        self.assertAlmostEqual(q["littles_law_concurrency"], 1.4)
        self.assertLess(q["littles_law_residual"], 1e-12)

    def test_slo_attainment_and_cli(self):
        records = [
            {"status": "ok", "ttft_ms": 80.0, "e2e_ms": 400.0},
            {"status": "ok", "ttft_ms": 150.0, "e2e_ms": 450.0},
            {"status": "ok", "ttft_ms": 90.0, "e2e_ms": 900.0},
            {"status": "http_error", "ttft_ms": 10.0, "e2e_ms": 20.0},
        ]
        slo = evaluate_slo_attainment(records, {"ttft_ms": 100.0, "e2e_ms": 500.0})
        self.assertEqual(slo["joint_goodput_requests"], 1)
        self.assertAlmostEqual(slo["joint_attainment"], 0.25)
        self.assertAlmostEqual(slo["per_metric_attainment"]["ttft_ms"], 0.50)
        self.assertAlmostEqual(slo["per_metric_attainment"]["e2e_ms"], 0.50)

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            tmp_dir = Path(tmp)
            csv_path = tmp_dir / "trace.csv"
            out_path = tmp_dir / "analysis.json"
            csv_path.write_text(
                "arrival_s,end_s,ttft_ms,e2e_ms\n"
                "0.0,1.0,50.0,1000.0\n"
                "0.5,1.5,60.0,1000.0\n"
                "2.0,2.5,40.0,500.0\n"
            )
            res = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "tools/workload_pacing.py"),
                    "analyze",
                    str(csv_path),
                    "--ttft-slo-ms",
                    "55.0",
                    "--e2e-slo-ms",
                    "1000.0",
                    "--output",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            parsed = json.loads(out_path.read_text())
            self.assertEqual(parsed["queueing"]["peak_concurrency"], 2)
            self.assertEqual(parsed["slo"]["joint_goodput_requests"], 2)

            # Existing output is never clobbered
            dup = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "tools/workload_pacing.py"),
                    "generate",
                    "--count",
                    "10",
                    "--rate",
                    "5.0",
                    "--output",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(dup.returncode, 0)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            generate_arrival_schedule(0, 5.0)
        with self.assertRaises(ValueError):
            generate_arrival_schedule(10, -1.0)
        with self.assertRaises(ValueError):
            generate_arrival_schedule(10, 5.0, mode="pareto", pareto_alpha=1.0)
        with self.assertRaises(ValueError):
            rescale_arrival_trace([1.0, 0.5])
        with self.assertRaises(ValueError):
            analyze_concurrency_and_littles_law([(2.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
