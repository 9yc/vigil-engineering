import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest


class TestProcTelemetryDump(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifact_dir = Path(os.environ["ARTIFACT_DIR"])
        cls.binary = artifact_dir / "proc_telemetry_dump"
        if not cls.binary.is_file():
            raise RuntimeError(f"Missing compiled binary at {cls.binary}")
        cls.tmp_root = Path(os.environ["TMPDIR"])
        cls.tmp_root.mkdir(parents=True, exist_ok=True)

    def _populate_sysroot(
        self,
        root: Path,
        cpu0_row: str = "cpu0 100 10 40 800 50 0 0 0",
        cpu1_row: str = "cpu1 200 0 50 700 50 0 0 0",
        total_row: str = "cpu  300 10 90 1500 100 0 0 0",
        vol_cs: int = 120,
        nonvol_cs: int = 15,
        psi_mem_some_total: int = 4500,
    ) -> None:
        (root / "proc" / "pressure").mkdir(parents=True, exist_ok=True)
        (root / "proc" / "4242").mkdir(parents=True, exist_ok=True)
        for tz in ("thermal_zone0", "thermal_zone2"):
            (root / "sys" / "class" / "thermal" / tz).mkdir(parents=True, exist_ok=True)
        for sub in ("cpu0/cpufreq", "cpu1/cpufreq", "cpuidle", "cpufreq"):
            (root / "sys" / "devices" / "system" / "cpu" / sub).mkdir(parents=True, exist_ok=True)

        (root / "proc" / "meminfo").write_text(
            "MemTotal:       16384000 kB\nMemFree:         4096000 kB\n"
            "MemAvailable:    8192000 kB\nHugePages_Total:       0\n",
            encoding="utf-8",
        )
        (root / "proc" / "stat").write_text(
            f"{total_row}\n{cpu1_row}\n{cpu0_row}\nintr 9999\nctxt 55555\n", encoding="utf-8"
        )
        (root / "proc" / "4242" / "status").write_text(
            "Name:\tworker_daemon\nState:\tS (sleeping)\nPid:\t4242\n"
            "VmPeak:\t  262144 kB\nVmRSS:\t  131072 kB\nThreads:\t8\n"
            f"voluntary_ctxt_switches:\t{vol_cs}\nnonvoluntary_ctxt_switches:\t{nonvol_cs}\n",
            encoding="utf-8",
        )
        (root / "proc" / "pressure" / "cpu").write_text(
            "some avg10=12.50 avg60=6.25 avg300=1.50 total=98000\n", encoding="utf-8"
        )
        (root / "proc" / "pressure" / "memory").write_text(
            f"some avg10=2.25 avg60=1.10 avg300=0.50 total={psi_mem_some_total}\n"
            "full avg10=0.75 avg60=0.20 avg300=0.05 total=1200\n",
            encoding="utf-8",
        )
        (root / "proc" / "pressure" / "io").write_text(
            "some avg10=4.00 avg60=2.00 avg300=1.00 total=31000\n"
            "full avg10=1.25 avg60=0.60 avg300=0.10 total=8400\n",
            encoding="utf-8",
        )
        (root / "sys" / "class" / "thermal" / "thermal_zone0" / "type").write_text("soc-pkg\n")
        (root / "sys" / "class" / "thermal" / "thermal_zone0" / "temp").write_text("64250\n")
        (root / "sys" / "class" / "thermal" / "thermal_zone2" / "type").write_text("subzero\n")
        (root / "sys" / "class" / "thermal" / "thermal_zone2" / "temp").write_text("-5500\n")

        cpu_base = root / "sys" / "devices" / "system" / "cpu"
        (cpu_base / "cpu0" / "cpufreq" / "scaling_cur_freq").write_text("1800000\n")
        (cpu_base / "cpu0" / "cpufreq" / "scaling_max_freq").write_text("2400000\n")
        (cpu_base / "cpu1" / "cpufreq" / "scaling_cur_freq").write_text("2700000\n")
        (cpu_base / "cpu1" / "cpufreq" / "scaling_max_freq").write_text("3000000\n")

    def test_single_shot_snapshot_psi_thermal_and_cpufreq(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as td:
            sysroot = Path(td) / "root"
            self._populate_sysroot(sysroot)
            res = subprocess.run(
                [str(self.binary), "--sysroot", str(sysroot), "--pid", "4242"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            payload = json.loads(res.stdout)
            self.assertEqual(payload["sample_count"], 1)
            snap = payload["snapshots"][0]
            self.assertEqual(snap["meminfo_kb"]["MemTotal"], 16384000)
            self.assertEqual([c["cpu_index"] for c in snap["per_cpu"]], [0, 1])

            proc_info = snap["process"]
            self.assertEqual(proc_info["name"], "worker_daemon")
            self.assertEqual(proc_info["vm_rss_kb"], 131072)
            self.assertEqual(proc_info["vm_peak_kb"], 262144)
            self.assertEqual(proc_info["threads"], 8)

            psi = snap["pressure"]
            self.assertTrue(psi["cpu"]["some"]["present"])
            self.assertFalse(psi["cpu"]["full"]["present"])
            self.assertAlmostEqual(psi["cpu"]["some"]["avg10"], 12.50, places=5)
            self.assertEqual(psi["io"]["full"]["total_us"], 8400)

            tz = snap["thermal_zones"]
            self.assertEqual([z["zone_id"] for z in tz], [0, 2])
            self.assertAlmostEqual(tz[0]["temp_celsius"], 64.25, places=5)
            self.assertAlmostEqual(tz[1]["temp_celsius"], -5.5, places=5)

            cf = snap["cpufreq"]
            self.assertEqual([c["cpu_id"] for c in cf], [0, 1])
            self.assertAlmostEqual(cf[0]["utilization_of_max"], 0.75, places=5)
            self.assertAlmostEqual(cf[1]["utilization_of_max"], 0.90, places=5)

    def test_two_snapshot_cpu_busy_delta_and_timed_sampling(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as td:
            r0, r1 = Path(td) / "s0", Path(td) / "s1"
            self._populate_sysroot(r0)
            self._populate_sysroot(
                r1,
                cpu0_row="cpu0 160 10 60 900 70 0 0 0",
                cpu1_row="cpu1 320 0 50 750 50 30 0 0",
                total_row="cpu  480 10 110 1650 120 30 0 0",
                vol_cs=155,
                nonvol_cs=22,
                psi_mem_some_total=6000,
            )
            out_file = Path(td) / "delta_out.json"
            res = subprocess.run(
                [
                    str(self.binary),
                    "--sysroot",
                    str(r0),
                    "--compare-sysroot",
                    str(r1),
                    "--pid",
                    "4242",
                    "--output",
                    str(out_file),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            delta = json.loads(out_file.read_text(encoding="utf-8"))["deltas"][0]
            self.assertAlmostEqual(delta["cpu_total"]["busy_fraction"], 0.575, places=6)
            self.assertAlmostEqual(delta["per_cpu"][0]["busy_fraction"], 0.40, places=6)
            self.assertAlmostEqual(delta["per_cpu"][0]["iowait_fraction"], 0.10, places=6)
            self.assertAlmostEqual(delta["per_cpu"][1]["busy_fraction"], 0.75, places=6)
            self.assertAlmostEqual(delta["per_cpu"][1]["irq_fraction"], 0.15, places=6)
            self.assertEqual(delta["delta_voluntary_ctxt_switches"], 35)
            self.assertEqual(delta["delta_nonvoluntary_ctxt_switches"], 7)
            self.assertEqual(delta["psi_some_delta_us"]["memory"], 1500)

            dup = subprocess.run(
                [str(self.binary), "--sysroot", str(r0), "--output", str(out_file)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(dup.returncode, 0)

            live_root = Path(td) / "live"
            self._populate_sysroot(live_root)

            def updater() -> None:
                time.sleep(0.03)
                tmp_stat = live_root / "proc" / "stat.next"
                tmp_stat.write_text(
                    "cpu  400 10 90 1600 100 0 0 0\n"
                    "cpu0 150 10 40 850 50 0 0 0\n"
                    "cpu1 250 0 50 750 50 0 0 0\n",
                    encoding="utf-8",
                )
                os.replace(tmp_stat, live_root / "proc" / "stat")

            worker = threading.Thread(target=updater)
            worker.start()
            timed = subprocess.run(
                [str(self.binary), "--sysroot", str(live_root), "--samples", "2", "--interval-ms", "80"],
                capture_output=True,
                text=True,
                check=False,
            )
            worker.join()
            self.assertEqual(timed.returncode, 0, msg=timed.stderr)
            self.assertAlmostEqual(
                json.loads(timed.stdout)["deltas"][0]["cpu_total"]["busy_fraction"], 0.50, places=5
            )

    def test_malformed_nodes_fail_strict_validation(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as td:
            bad_mem = Path(td) / "bad_mem"
            self._populate_sysroot(bad_mem)
            (bad_mem / "proc" / "meminfo").write_text("MemTotal: 16384xyz kB\n", encoding="utf-8")
            self.assertNotEqual(
                subprocess.run([str(self.binary), "--sysroot", str(bad_mem)], capture_output=True).returncode,
                0,
            )

            bad_psi = Path(td) / "bad_psi"
            self._populate_sysroot(bad_psi)
            (bad_psi / "proc" / "pressure" / "cpu").write_text(
                "some avg10=145.00 avg60=1.00 avg300=0.50 total=100\n", encoding="utf-8"
            )
            self.assertNotEqual(
                subprocess.run([str(self.binary), "--sysroot", str(bad_psi)], capture_output=True).returncode,
                0,
            )

            bad_tz = Path(td) / "bad_tz"
            self._populate_sysroot(bad_tz)
            (bad_tz / "sys" / "class" / "thermal" / "thermal_zone0" / "temp").write_text("64.25\n")
            self.assertNotEqual(
                subprocess.run([str(self.binary), "--sysroot", str(bad_tz)], capture_output=True).returncode,
                0,
            )


if __name__ == "__main__":
    unittest.main()
