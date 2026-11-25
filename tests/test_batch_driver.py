"""Unit tests for runtime_support.batch_driver."""

import json
import os
import pathlib
import tempfile
import unittest

from runtime_support.batch_driver import (
    BatchOutputExistsError,
    BatchRowValidationError,
    BatchSlice,
    BatteryFloorError,
    ChargerPluggedError,
    CooldownTimeoutError,
    CooledBatchDriver,
    EnergyValidityError,
    HardwareTelemetry,
    LockedSchedule,
    SafetyGateConfig,
    ScheduleValidationError,
    StrayRunnerGateError,
    ThermalCeilingError,
)


class FakeDriverClock:
    """Injectable monotonic clock + sleep recorder for batch cooldown tests."""

    def __init__(self, start_s: float = 1000.0) -> None:
        self.now_s = start_s
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now_s

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now_s += seconds


class ScriptedTelemetryProbe:
    """Returns scripted HardwareTelemetry samples sequentially."""

    def __init__(self, samples: list[HardwareTelemetry]) -> None:
        self._samples = list(samples)
        self.calls = 0

    def __call__(self) -> HardwareTelemetry:
        idx = min(self.calls, len(self._samples) - 1)
        self.calls += 1
        return self._samples[idx]


def _healthy_sample(
    *,
    soc: float = 80.0,
    bat_c: float = 32.0,
    skin_c: float = 31.5,
    ac: bool = False,
    usb: bool = False,
    wireless: bool = False,
    stray_pids: tuple[int, ...] = (),
) -> HardwareTelemetry:
    return HardwareTelemetry(
        ac_online=ac,
        usb_online=usb,
        wireless_online=wireless,
        battery_soc_pct=soc,
        battery_c=bat_c,
        skin_c=skin_c,
        stray_pids=stray_pids,
    )


def _valid_worker(batch_slice: BatchSlice, _batch_dir: pathlib.Path) -> list[dict[str, object]]:
    return [
        {
            "repeat_idx": cell.repeat_idx,
            "sample_id": cell.sample_id,
            "config_id": cell.config_id,
            "status": "ok",
            "energy_valid": True,
            "charger_unplugged": True,
            "energy_mj": 42.5 + idx,
            "duration_s": 0.45,
        }
        for idx, cell in enumerate(batch_slice.cells)
    ]


class CooledBatchDriverTests(unittest.TestCase):
    """Tests for schedule partitioning, safety gates, cooldowns, and row checks."""

    def setUp(self) -> None:
        tmp_root = os.environ.get("TMPDIR")
        if tmp_root:
            pathlib.Path(tmp_root).mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.TemporaryDirectory(dir=tmp_root)
        self.addCleanup(self._tmpdir.cleanup)
        self.base_dir = pathlib.Path(self._tmpdir.name)
        self.out_dir = self.base_dir / "driver_out"
        self.cfg = SafetyGateConfig(
            min_battery_soc_pct=30.0,
            max_battery_c=40.0,
            max_skin_c=39.0,
            cool_target_c=34.0,
            max_cool_s=30.0,
            cool_poll_s=5.0,
        )

    def _make_schedule(self) -> LockedSchedule:
        return LockedSchedule.from_matrix(
            "sched_main_01",
            base_pairs=[("s1", "cfg_a"), ("s2", "cfg_a"), ("s1", "cfg_b")],
            repeats=2,
        )

    def test_schedule_partitioning_non_overlapping_and_validation(self) -> None:
        sched = self._make_schedule()
        self.assertEqual(len(sched.cells), 6)
        slices = sched.partition(batch_size=2)
        self.assertEqual(len(slices), 3)
        all_coords = [c.coordinate for sl in slices for c in sl.cells]
        self.assertEqual(len(all_coords), 6)
        self.assertEqual(len(set(all_coords)), 6)

        with self.assertRaises(ScheduleValidationError):
            LockedSchedule.from_matrix("dup", [("s1", "c1"), ("s1", "c1")], repeats=1)

    def test_multi_batch_execution_with_intermediate_cooldown_waits(self) -> None:
        sched = self._make_schedule()
        clock = FakeDriverClock()
        # 3 batches of size 2:
        # Batch 0: pre (31.5C), post (36.8C)
        # Cooldown before Batch 1: 36.5C -> 35.1C -> 33.8C (<= 34.0C after 2 sleeps = 10s!)
        # Batch 1: pre (33.7C), post (33.9C)
        # Cooldown before Batch 2: 33.5C (immediately <= 34.0C, 0 sleeps!)
        # Batch 2: pre (33.4C), post (34.2C)
        probe = ScriptedTelemetryProbe([
            _healthy_sample(skin_c=31.5),
            _healthy_sample(skin_c=36.8),
            _healthy_sample(skin_c=36.5),
            _healthy_sample(skin_c=35.1),
            _healthy_sample(skin_c=33.8),
            _healthy_sample(skin_c=33.7),
            _healthy_sample(skin_c=33.9),
            _healthy_sample(skin_c=33.5),
            _healthy_sample(skin_c=33.4),
            _healthy_sample(skin_c=34.2),
        ])

        driver = CooledBatchDriver(
            output_root=self.out_dir,
            schedule=sched,
            batch_size=2,
            safety_config=self.cfg,
            telemetry_probe=probe,
            batch_worker=_valid_worker,
            clock_fn=clock.clock,
            sleep_fn=clock.sleep,
        )
        summary = driver.run()
        self.assertEqual(summary.total_batches, 3)
        self.assertEqual(summary.total_cells, 6)
        self.assertEqual(summary.cooldown_waits, 1)
        self.assertAlmostEqual(summary.total_cooldown_s, 10.0, places=5)
        self.assertEqual(clock.sleeps, [5.0, 5.0])

        for b_dir in summary.batch_dirs:
            self.assertTrue((b_dir / "manifest.json").is_file())
            rows_lines = (b_dir / "verified_rows.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows_lines), 2)

        events = [
            json.loads(line)["event"]
            for line in (self.out_dir / "driver_events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(events[0], "driver_started")
        self.assertIn("cooldown_wait", events)
        self.assertIn("cooldown_complete", events)
        self.assertEqual(events[-1], "driver_completed")

    def test_cooldown_timeout_aborts_fail_closed(self) -> None:
        sched = self._make_schedule()
        clock = FakeDriverClock()
        # Batch 0 completes, then skin stays at 37.5C (> 34.0C) past max_cool_s=30.0s.
        samples = [_healthy_sample(skin_c=31.0), _healthy_sample(skin_c=38.0)] + [
            _healthy_sample(skin_c=37.5) for _ in range(10)
        ]
        driver = CooledBatchDriver(
            output_root=self.out_dir,
            schedule=sched,
            batch_size=3,
            safety_config=self.cfg,
            telemetry_probe=ScriptedTelemetryProbe(samples),
            batch_worker=_valid_worker,
            clock_fn=clock.clock,
            sleep_fn=clock.sleep,
        )
        with self.assertRaises(CooldownTimeoutError):
            driver.run()
        self.assertAlmostEqual(sum(clock.sleeps), 30.0, places=5)
        # Batch 1 directory must never have been created.
        self.assertFalse((self.out_dir / "batches" / "batch_0001").exists())

    def test_charger_plugged_mid_run_and_safety_gate_failures(self) -> None:
        sched = self._make_schedule()
        # USB plugged in mid-run during Batch 0 post-batch safety gate -> ChargerPluggedError!
        usb_mid_run = ScriptedTelemetryProbe([
            _healthy_sample(skin_c=31.0),
            _healthy_sample(skin_c=32.5, usb=True),
        ])
        driver = CooledBatchDriver(
            output_root=self.out_dir / "run_usb",
            schedule=sched,
            batch_size=3,
            safety_config=self.cfg,
            telemetry_probe=usb_mid_run,
            batch_worker=_valid_worker,
        )
        with self.assertRaises(ChargerPluggedError):
            driver.run()

        # Wireless charger plugged during inter-batch cooldown also aborts immediately!
        wireless_in_cool = ScriptedTelemetryProbe([
            _healthy_sample(skin_c=31.0),
            _healthy_sample(skin_c=36.0),
            _healthy_sample(skin_c=35.5, wireless=True),
        ])
        driver_wl = CooledBatchDriver(
            output_root=self.out_dir / "run_wl",
            schedule=sched,
            batch_size=3,
            safety_config=self.cfg,
            telemetry_probe=wireless_in_cool,
            batch_worker=_valid_worker,
        )
        with self.assertRaises(ChargerPluggedError):
            driver_wl.run()

        # Battery floor, thermal ceiling, and stray PID pre-batch rejections:
        for sub, bad_sample, exc_cls in [
            ("low_soc", _healthy_sample(soc=22.0), BatteryFloorError),
            ("hot_skin", _healthy_sample(skin_c=39.0), ThermalCeilingError),
            ("stray_pid", _healthy_sample(stray_pids=(8192,)), StrayRunnerGateError),
        ]:
            drv = CooledBatchDriver(
                output_root=self.out_dir / sub,
                schedule=sched,
                batch_size=3,
                safety_config=self.cfg,
                telemetry_probe=ScriptedTelemetryProbe([bad_sample]),
                batch_worker=_valid_worker,
            )
            with self.assertRaises(exc_cls):
                drv.run()

    def test_missing_row_and_energy_validity_gate_failures(self) -> None:
        sched = self._make_schedule()

        # 1. Missing last row in batch -> BatchRowValidationError
        def incomplete_worker(bs: BatchSlice, bd: pathlib.Path) -> list[dict[str, object]]:
            return _valid_worker(bs, bd)[:-1]

        drv_missing = CooledBatchDriver(
            output_root=self.out_dir / "missing_row",
            schedule=sched,
            batch_size=3,
            safety_config=self.cfg,
            telemetry_probe=ScriptedTelemetryProbe([_healthy_sample()]),
            batch_worker=incomplete_worker,
        )
        with self.assertRaises(BatchRowValidationError):
            drv_missing.run()

        # 2. Invalid energy (energy_valid=False or negative energy_mj) -> EnergyValidityError
        def bad_energy_worker(bs: BatchSlice, bd: pathlib.Path) -> list[dict[str, object]]:
            rows = _valid_worker(bs, bd)
            rows[0]["energy_mj"] = -5.0
            return rows

        drv_energy = CooledBatchDriver(
            output_root=self.out_dir / "bad_energy",
            schedule=sched,
            batch_size=3,
            safety_config=self.cfg,
            telemetry_probe=ScriptedTelemetryProbe([_healthy_sample()]),
            batch_worker=bad_energy_worker,
        )
        with self.assertRaises(EnergyValidityError):
            drv_energy.run()

    def test_refuses_to_clobber_existing_batch_output_directory(self) -> None:
        sched = self._make_schedule()
        driver_first = CooledBatchDriver(
            output_root=self.out_dir,
            schedule=sched,
            batch_size=6,
            safety_config=self.cfg,
            telemetry_probe=ScriptedTelemetryProbe([_healthy_sample()]),
            batch_worker=_valid_worker,
        )
        driver_first.run()

        # Second run targeting same output_root must fail closed before touching batches!
        worker_invocations = 0

        def counting_worker(bs: BatchSlice, bd: pathlib.Path) -> list[dict[str, object]]:
            nonlocal worker_invocations
            worker_invocations += 1
            return _valid_worker(bs, bd)

        driver_second = CooledBatchDriver(
            output_root=self.out_dir,
            schedule=sched,
            batch_size=6,
            safety_config=self.cfg,
            telemetry_probe=ScriptedTelemetryProbe([_healthy_sample()]),
            batch_worker=counting_worker,
        )
        with self.assertRaises(BatchOutputExistsError):
            driver_second.run()
        self.assertEqual(worker_invocations, 0)

        # Pre-existing batches subdir on a fresh path is also rejected.
        fresh_with_batches = self.base_dir / "preexisting_batches_dir"
        (fresh_with_batches / "batches").mkdir(parents=True)
        driver_third = CooledBatchDriver(
            output_root=fresh_with_batches,
            schedule=sched,
            batch_size=6,
            safety_config=self.cfg,
            telemetry_probe=ScriptedTelemetryProbe([_healthy_sample()]),
            batch_worker=counting_worker,
        )
        with self.assertRaises(BatchOutputExistsError):
            driver_third.run()
        self.assertEqual(worker_invocations, 0)


if __name__ == "__main__":
    unittest.main()
