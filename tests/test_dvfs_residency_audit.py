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


def write_mock_policy(
    sysroot: Path,
    policy_id: int,
    related_cpus: str,
    governor: str,
    cpuinfo_min: int,
    cpuinfo_max: int,
    scaling_min: int,
    scaling_max: int,
    scaling_cur: int,
    total_trans: int,
    states,
) -> None:
    pdir = sysroot / "sys" / "devices" / "system" / "cpu" / "cpufreq" / f"policy{policy_id}"
    stats_dir = pdir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    (pdir / "related_cpus").write_text(f"{related_cpus}\n", encoding="utf-8")
    (pdir / "affected_cpus").write_text(f"{related_cpus}\n", encoding="utf-8")
    (pdir / "scaling_governor").write_text(f"{governor}\n", encoding="utf-8")
    (pdir / "cpuinfo_min_freq").write_text(f"{cpuinfo_min}\n", encoding="utf-8")
    (pdir / "cpuinfo_max_freq").write_text(f"{cpuinfo_max}\n", encoding="utf-8")
    (pdir / "scaling_min_freq").write_text(f"{scaling_min}\n", encoding="utf-8")
    (pdir / "scaling_max_freq").write_text(f"{scaling_max}\n", encoding="utf-8")
    (pdir / "scaling_cur_freq").write_text(f"{scaling_cur}\n", encoding="utf-8")
    (stats_dir / "total_trans").write_text(f"{total_trans}\n", encoding="utf-8")

    tis_lines = [f"{freq} {ticks}" for freq, ticks in states]
    (stats_dir / "time_in_state").write_text("\n".join(tis_lines) + "\n", encoding="utf-8")


class DvfsResidencyAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bin_path = resolve_binary("dvfs_residency_audit")
        cls.tmp_base = resolve_tmp_base()

    def setUp(self):
        self._tmp_ctx = tempfile.TemporaryDirectory(dir=str(self.tmp_base))
        self.work_dir = Path(self._tmp_ctx.name)

    def tearDown(self):
        self._tmp_ctx.cleanup()

    def _build_tri_cluster_pair(self):
        before_root = self.work_dir / "sysroot_before"
        after_root = self.work_dir / "sysroot_after"

        # policy0: efficiency cluster (CPUs 0-3), unthrottled
        write_mock_policy(
            before_root,
            policy_id=0,
            related_cpus="0 1 2 3",
            governor="schedutil",
            cpuinfo_min=800000,
            cpuinfo_max=2000000,
            scaling_min=800000,
            scaling_max=2000000,
            scaling_cur=1200000,
            total_trans=100,
            states=[(800000, 500), (1200000, 300), (1600000, 150), (2000000, 50)],
        )
        write_mock_policy(
            after_root,
            policy_id=0,
            related_cpus="0 1 2 3",
            governor="schedutil",
            cpuinfo_min=800000,
            cpuinfo_max=2000000,
            scaling_min=800000,
            scaling_max=2000000,
            scaling_cur=1600000,
            total_trans=140,
            # deltas: 800k:+20, 1200k:+30, 1600k:+30, 2000k:+20 (total=100 ticks = 1.0s)
            states=[(800000, 520), (1200000, 330), (1600000, 180), (2000000, 70)],
        )

        # policy4: mid cluster (CPUs 4-6), thermally capped at 75% in after snapshot
        write_mock_policy(
            before_root,
            policy_id=4,
            related_cpus="4 5 6",
            governor="schedutil",
            cpuinfo_min=1000000,
            cpuinfo_max=2800000,
            scaling_min=1000000,
            scaling_max=2800000,
            scaling_cur=2100000,
            total_trans=250,
            states=[(1000000, 100), (1800000, 200), (2100000, 300), (2800000, 400)],
        )
        write_mock_policy(
            after_root,
            policy_id=4,
            related_cpus="4 5 6",
            governor="schedutil",
            cpuinfo_min=1000000,
            cpuinfo_max=2800000,
            scaling_min=1000000,
            scaling_max=2100000,
            scaling_cur=2100000,
            total_trans=265,
            # deltas: 1000k:+10, 1800k:+40, 2100k:+50, 2800k:+0 (total=100 ticks)
            states=[(1000000, 110), (1800000, 240), (2100000, 350), (2800000, 400)],
        )

        # policy7: prime core (CPU 7), pinned performance governor at top bin
        write_mock_policy(
            before_root,
            policy_id=7,
            related_cpus="7",
            governor="performance",
            cpuinfo_min=1200000,
            cpuinfo_max=3200000,
            scaling_min=1200000,
            scaling_max=3200000,
            scaling_cur=3200000,
            total_trans=12,
            states=[(1200000, 50), (2400000, 50), (2800000, 100), (3200000, 200)],
        )
        write_mock_policy(
            after_root,
            policy_id=7,
            related_cpus="7",
            governor="performance",
            cpuinfo_min=1200000,
            cpuinfo_max=3200000,
            scaling_min=1200000,
            scaling_max=3200000,
            scaling_cur=3200000,
            total_trans=14,
            # deltas: 1200k:+0, 2400k:+0, 2800k:+25, 3200k:+75 (total=100 ticks)
            states=[(1200000, 50), (2400000, 50), (2800000, 125), (3200000, 275)],
        )
        return before_root, after_root

    def test_snapshot_and_delta_weighted_mean_cap_throttling_and_asymmetry(self):
        before_root, after_root = self._build_tri_cluster_pair()
        snap_before_json = self.work_dir / "before_snap.json"
        delta_json = self.work_dir / "delta_report.json"

        res_snap = subprocess.run(
            [str(self.bin_path), "snapshot", "--sysroot", str(before_root), "--output", str(snap_before_json)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(res_snap.returncode, 0, msg=res_snap.stderr)
        snap_data = json.loads(snap_before_json.read_text(encoding="utf-8"))
        self.assertEqual([p["policy_id"] for p in snap_data["policies"]], [0, 4, 7])
        self.assertEqual(snap_data["policies"][0]["related_cpus"], [0, 1, 2, 3])

        # Run delta mixing a saved JSON snapshot (--before) with a live sysroot dir (--after)
        res_delta = subprocess.run(
            [
                str(self.bin_path), "delta",
                "--before", str(snap_before_json),
                "--after", str(after_root),
                "--top-fraction", "0.8",
                "--output", str(delta_json),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(res_delta.returncode, 0, msg=res_delta.stderr)
        rep = json.loads(delta_json.read_text(encoding="utf-8"))

        by_id = {p["policy_id"]: p for p in rep["policies"]}
        self.assertEqual(set(by_id.keys()), {0, 4, 7})

        # policy0 expected weighted mean: (20*800k + 30*1200k + 30*1600k + 20*2000k) / 100 = 1,400,000 kHz
        p0 = by_id[0]
        self.assertAlmostEqual(p0["weighted_avg_freq_khz"], 1400000.0, places=3)
        self.assertAlmostEqual(p0["after_cap_ratio"], 1.0, places=6)
        self.assertFalse(p0["cap_throttled"])
        # top_fraction=0.8 * 2000000 = 1600000 -> 1600k (30) + 2000k (20) = 50/100 = 0.50
        self.assertAlmostEqual(p0["upper_band_share"], 0.50, places=6)
        self.assertAlmostEqual(p0["transitions_per_sec"], 40.0, places=4)

        # policy4 expected weighted mean: (10*1000k + 40*1800k + 50*2100k + 0*2800k) / 100 = 1,870,000 kHz
        p4 = by_id[4]
        self.assertAlmostEqual(p4["weighted_avg_freq_khz"], 1870000.0, places=3)
        self.assertAlmostEqual(p4["after_cap_ratio"], 2100000.0 / 2800000.0, places=6)
        self.assertTrue(p4["cap_throttled"])
        # 0.8 * 2800000 = 2240000 -> only 2800k qualifies, which had 0 delta ticks under cap
        self.assertAlmostEqual(p4["upper_band_share"], 0.0, places=6)
        self.assertAlmostEqual(p4["transitions_per_sec"], 15.0, places=4)

        # policy7 expected weighted mean: (25*2800k + 75*3200k) / 100 = 3,100,000 kHz
        p7 = by_id[7]
        self.assertAlmostEqual(p7["weighted_avg_freq_khz"], 3100000.0, places=3)
        # 0.8 * 3200000 = 2560000 -> both 2800k and 3200k qualify -> share = 1.0
        self.assertAlmostEqual(p7["upper_band_share"], 1.0, places=6)

        asym = rep["cluster_asymmetry"]
        self.assertEqual(asym["capped_policy_ids"], [4])
        self.assertTrue(asym["governor_heterogeneous"])
        expected_max_norm = 3100000.0 / 3200000.0
        expected_min_norm = 1870000.0 / 2800000.0
        self.assertTrue(math.isclose(asym["max_normalized_freq_ratio"], expected_max_norm, rel_tol=1e-5))
        self.assertTrue(math.isclose(asym["min_normalized_freq_ratio"], expected_min_norm, rel_tol=1e-5))
        self.assertAlmostEqual(asym["upper_band_share_spread"], 1.0, places=6)

    def test_counter_regression_in_time_in_state_or_transitions_rejected(self):
        before_root, after_root = self._build_tri_cluster_pair()
        # Inject regression into policy4 time_in_state (after ticks < before ticks)
        tis_file = (
            after_root / "sys" / "devices" / "system" / "cpu" / "cpufreq" / "policy4" / "stats" / "time_in_state"
        )
        tis_file.write_text("1000000 110\n1800000 190\n2100000 350\n2800000 400\n", encoding="utf-8")

        out_tis = self.work_dir / "regressed_tis.json"
        proc_tis = subprocess.run(
            [str(self.bin_path), "delta", "--before", str(before_root), "--after", str(after_root), "--output", str(out_tis)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertNotEqual(proc_tis.returncode, 0)
        self.assertIn("regression", proc_tis.stderr.lower())
        self.assertFalse(out_tis.exists())

        # Restore time_in_state, then inject regression into policy0 total_trans
        tis_file.write_text("1000000 110\n1800000 240\n2100000 350\n2800000 400\n", encoding="utf-8")
        trans_file = (
            after_root / "sys" / "devices" / "system" / "cpu" / "cpufreq" / "policy0" / "stats" / "total_trans"
        )
        trans_file.write_text("95\n", encoding="utf-8")

        out_trans = self.work_dir / "regressed_trans.json"
        proc_trans = subprocess.run(
            [str(self.bin_path), "delta", "--before", str(before_root), "--after", str(after_root), "--output", str(out_trans)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertNotEqual(proc_trans.returncode, 0)
        self.assertIn("regression", proc_trans.stderr.lower())
        self.assertFalse(out_trans.exists())

    def test_exclusive_output_protection_rejects_existing_destination(self):
        before_root, after_root = self._build_tri_cluster_pair()
        protected = self.work_dir / "keep.json"
        protected.write_text("{\"untouched\": true}\n", encoding="utf-8")

        for args in [
            ["snapshot", "--sysroot", str(before_root), "--output", str(protected)],
            ["delta", "--before", str(before_root), "--after", str(after_root), "--output", str(protected)],
        ]:
            proc = subprocess.run(
                [str(self.bin_path), *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(protected.read_text(encoding="utf-8"), "{\"untouched\": true}\n")


if __name__ == "__main__":
    unittest.main()
