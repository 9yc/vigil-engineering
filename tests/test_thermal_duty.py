import unittest

from runtime_support.thermal_duty import (
    BatteryLowError,
    DutyCycleController,
    ProbeReading,
    ThermalGateConfig,
    ThermalTimeoutError,
)


class FakeClock:
    def __init__(self, initial=100.0):
        self.now = float(initial)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        if seconds < 0:
            raise AssertionError("negative sleep")
        self.now += float(seconds)


class ThermalDutyTests(unittest.TestCase):
    def test_config_hysteresis_and_bounds_validation(self):
        cfg = ThermalGateConfig(max_entry_temp_c=37.0, pause_temp_c=42.0, resume_temp_c=36.0)
        self.assertEqual(cfg.pause_temp_c, 42.0)
        invalid_configs = (
            {"pause_temp_c": 37.0, "resume_temp_c": 37.0},
            {"pause_temp_c": 37.0, "resume_temp_c": 39.0},
            {"pause_temp_c": 40.0, "max_entry_temp_c": 40.0},
            {"min_soc": -0.1},
            {"min_soc": 1.1},
            {"poll_interval_s": 0.0},
            {"max_cooldown_wait_s": 2.0, "min_cooldown_s": 5.0},
            {"max_active_window_s": 0.0},
            {"pause_temp_c": True},
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ThermalGateConfig(**kwargs)

    def test_entry_gate_wait_and_keepalive_invocations(self):
        clock = FakeClock()
        keepalives = []
        temps = iter([40.5, 39.2, 38.4, 37.5])
        cfg = ThermalGateConfig(
            max_entry_temp_c=38.0,
            pause_temp_c=42.0,
            resume_temp_c=36.0,
            poll_interval_s=5.0,
            keepalive_interval_s=10.0,
        )
        ctrl = DutyCycleController(
            cfg,
            lambda: ProbeReading(next(temps), 0.75),
            clock_fn=clock,
            sleep_fn=clock.advance,
            keepalive_fn=lambda: keepalives.append(clock.now),
        )
        entry = ctrl.await_entry_gate()
        self.assertEqual(entry.temperature_c, 37.5)
        self.assertAlmostEqual(ctrl.summary()["entry_wait_s"], 15.0)
        self.assertEqual(keepalives, [110.0])
        clock.advance(20.0)
        summary = ctrl.finish()
        self.assertAlmostEqual(summary["active_elapsed_s"], 20.0)
        self.assertAlmostEqual(summary["paused_elapsed_s"], 0.0)
        self.assertAlmostEqual(summary["duty_factor"], 1.0)

    def test_mid_run_excursion_hysteresis_and_min_cooldown(self):
        clock = FakeClock(0.0)
        # Entry=35C; step1=42.5C triggers cooldown; during cooldown:
        # t=12s -> 36.0C (below resume 37C, but paused_for=2s < min_cooldown_s=6s!)
        # t=14s -> 38.0C (still above resume)
        # t=16s -> 36.5C (paused_for=6s >= 6s and <= 37C => resumes!)
        readings = iter([
            ProbeReading(35.0, 0.80),
            ProbeReading(42.5, 0.79),
            ProbeReading(36.0, 0.79),
            ProbeReading(38.0, 0.78),
            ProbeReading(36.5, 0.78),
        ])
        cfg = ThermalGateConfig(
            max_entry_temp_c=38.0,
            pause_temp_c=42.0,
            resume_temp_c=37.0,
            min_cooldown_s=6.0,
            poll_interval_s=2.0,
            keepalive_interval_s=4.0,
        )
        keepalive_count = 0

        def on_keepalive():
            nonlocal keepalive_count
            keepalive_count += 1

        ctrl = DutyCycleController(
            cfg,
            lambda: next(readings),
            clock_fn=clock,
            sleep_fn=clock.advance,
            keepalive_fn=on_keepalive,
        )
        ctrl.await_entry_gate()
        clock.advance(10.0)
        outcome = ctrl.checkpoint()
        self.assertTrue(outcome["paused"])
        self.assertAlmostEqual(outcome["pause_duration_s"], 6.0)
        clock.advance(14.0)
        summary = ctrl.finish()
        self.assertEqual(summary["cooldown_episodes"], 1)
        self.assertAlmostEqual(summary["active_elapsed_s"], 24.0)
        self.assertAlmostEqual(summary["paused_elapsed_s"], 6.0)
        self.assertAlmostEqual(summary["run_elapsed_s"], 30.0)
        self.assertAlmostEqual(summary["duty_factor"], 0.8)
        self.assertEqual(summary["peak_temperature_c"], 42.5)
        self.assertGreaterEqual(keepalive_count, 1)

    def test_timed_active_slice_forces_duty_cycle_window(self):
        clock = FakeClock(0.0)
        cfg = ThermalGateConfig(
            max_entry_temp_c=38.0,
            pause_temp_c=45.0,
            resume_temp_c=37.0,
            max_active_window_s=10.0,
            min_cooldown_s=4.0,
            poll_interval_s=2.0,
        )
        ctrl = DutyCycleController(
            cfg,
            lambda: ProbeReading(34.0, 0.90),
            clock_fn=clock,
            sleep_fn=clock.advance,
        )
        ctrl.await_entry_gate()
        clock.advance(5.0)
        self.assertFalse(ctrl.checkpoint()["paused"])
        clock.advance(5.0)
        paused = ctrl.checkpoint()
        self.assertTrue(paused["paused"])
        self.assertAlmostEqual(paused["pause_duration_s"], 4.0)
        summary = ctrl.finish()
        self.assertAlmostEqual(summary["active_elapsed_s"], 10.0)
        self.assertAlmostEqual(summary["paused_elapsed_s"], 4.0)

    def test_battery_floor_and_cooldown_timeout_errors(self):
        clock = FakeClock(0.0)
        cfg = ThermalGateConfig(min_soc=0.25, min_cooldown_s=2.0, max_cooldown_wait_s=6.0, poll_interval_s=2.0)

        low_unplugged = DutyCycleController(
            cfg, lambda: ProbeReading(32.0, 0.20, external_power=False),
            clock_fn=clock, sleep_fn=clock.advance,
        )
        with self.assertRaises(BatteryLowError):
            low_unplugged.await_entry_gate()

        low_plugged = DutyCycleController(
            cfg, lambda: ProbeReading(32.0, 0.20, external_power=True),
            clock_fn=clock, sleep_fn=clock.advance,
        )
        low_plugged.await_entry_gate()
        low_plugged.finish()

        hot_forever = DutyCycleController(
            cfg, lambda: ProbeReading(44.0, 0.80),
            clock_fn=clock, sleep_fn=clock.advance,
        )
        with self.assertRaises(ThermalTimeoutError):
            hot_forever.await_entry_gate()


if __name__ == "__main__":
    unittest.main()
