import unittest

from runtime_support.power_telemetry import (
    integrate_coulomb_energy,
    integrate_vi_trapezoidal,
    parse_dumpsys_battery,
    parse_proc_meminfo,
    parse_proc_uptime,
    subtract_standby_baseline,
)


SAMPLE_BATTERY = """Current Battery Service state:
  AC powered: false
  USB powered: true
  Wireless powered: false
  status: 2
  level: 84
  scale: 100
  voltage: 4150
  temperature: 318
  Charge counter: 3920000
"""

SAMPLE_MEMINFO = """MemTotal:        8192000 kB
MemFree:         1024000 kB
MemAvailable:    3276800 kB
Buffers:          256000 kB
Cached:          1792000 kB
SwapTotal:       4096000 kB
SwapFree:        3584000 kB
AnonPages:       4120000 kB
"""


class TelemetryParserTests(unittest.TestCase):
    def test_dumpsys_battery_snapshot_and_soc(self):
        snap = parse_dumpsys_battery(SAMPLE_BATTERY)
        self.assertEqual(snap.level, 84)
        self.assertEqual(snap.scale, 100)
        self.assertAlmostEqual(snap.state_of_charge, 0.84)
        self.assertAlmostEqual(snap.temperature_c, 31.8)
        self.assertAlmostEqual(snap.voltage_v, 4.15)
        self.assertEqual(snap.charge_uah, 3920000)
        self.assertEqual(snap.status, 2)
        self.assertFalse(snap.ac_powered)
        self.assertTrue(snap.usb_powered)
        self.assertFalse(snap.wireless_powered)

    def test_dumpsys_battery_rejects_missing_or_out_of_range_fields(self):
        invalid_payloads = (
            "",
            "level: 50\nscale: 100\ntemperature: 300\n",
            "level: 110\nscale: 100\ntemperature: 300\nvoltage: 4000\n",
            "level: 50\nscale: 0\ntemperature: 300\nvoltage: 4000\n",
            "level: 50\nscale: 100\ntemperature: 300\nvoltage: 0\n",
            "level: 50\nscale: 100\ntemperature: 300\nvoltage: 4000\nAC powered: maybe\n",
        )
        for text in invalid_payloads:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_dumpsys_battery(text)

    def test_proc_meminfo_available_and_fallback_calculation(self):
        parsed = parse_proc_meminfo(SAMPLE_MEMINFO)
        self.assertEqual(parsed["mem_total_kb"], 8192000)
        self.assertEqual(parsed["mem_available_kb"], 3276800)
        self.assertEqual(parsed["mem_used_kb"], 4915200)
        self.assertAlmostEqual(parsed["available_ratio"], 0.4)
        self.assertEqual(parsed["anon_pages_kb"], 4120000)

        fallback_text = "MemTotal: 1000 kB\nMemFree: 200 kB\nBuffers: 50 kB\nCached: 150 kB\n"
        fallback = parse_proc_meminfo(fallback_text)
        self.assertEqual(fallback["mem_available_kb"], 400)
        self.assertEqual(fallback["mem_used_kb"], 600)

    def test_proc_meminfo_and_uptime_validation(self):
        for bad_mem in (
            "",
            "MemFree: 100 kB\n",
            "MemTotal: 100 MB\nMemAvailable: 50 kB\n",
            "MemTotal: 100 kB\nMemAvailable: 200 kB\n",
            "MemTotal: -10 kB\nMemAvailable: 5 kB\n",
        ):
            with self.subTest(bad_mem=bad_mem), self.assertRaises(ValueError):
                parse_proc_meminfo(bad_mem)

        uptime = parse_proc_uptime("1234.50 4321.75\n")
        self.assertEqual(uptime, {"uptime_s": 1234.5, "idle_s": 4321.75})
        for bad_up in ("", "12.0", "12.0 1.0 2.0", "-1.0 5.0", "nan 1.0"):
            with self.subTest(bad_up=bad_up), self.assertRaises(ValueError):
                parse_proc_uptime(bad_up)


class EnergyIntegrationTests(unittest.TestCase):
    def test_coulomb_counter_integration_and_validation(self):
        # 10,000 uAh = 36 C; V_mean = 4.0 V => 144 J over 36 s => 4.0 W
        res = integrate_coulomb_energy(4_000_000, 3_990_000, 4.1, 3.9, 36.0)
        self.assertAlmostEqual(res["discharged_uah"], 10_000.0)
        self.assertAlmostEqual(res["discharged_c"], 36.0)
        self.assertAlmostEqual(res["mean_voltage_v"], 4.0)
        self.assertAlmostEqual(res["energy_j"], 144.0)
        self.assertAlmostEqual(res["mean_power_w"], 4.0)

        with self.assertRaises(ValueError):
            integrate_coulomb_energy(3_900_000, 3_950_000, 4.0, 4.0, 10.0)
        with self.assertRaises(ValueError):
            integrate_coulomb_energy(4_000_000, 3_990_000, 0.0, 4.0, 10.0)
        with self.assertRaises(ValueError):
            integrate_coulomb_energy(True, 0, 4.0, 4.0, 10.0)

    def test_trapezoidal_vi_units_and_subwindow_clipping(self):
        # t=0: 4.0 V * 0.5 A = 2.0 W; t=2: 4.0 V * 1.5 A = 6.0 W; t=4: 4.0 V * 0.5 A = 2.0 W
        samples_mv_ma = [
            (0.0, 4000, -500),
            (2.0, 4000, -1500),
            (4.0, 4000, -500),
        ]
        full = integrate_vi_trapezoidal(
            samples_mv_ma, voltage_unit="mv", current_unit="ma", max_gap_s=2.0
        )
        self.assertAlmostEqual(full["energy_j"], 16.0)
        self.assertAlmostEqual(full["mean_power_w"], 4.0)

        # Subwindow [1.0, 3.0]: P(1)=4W, P(2)=6W, P(3)=4W => area = 5 + 5 = 10 J
        sub = integrate_vi_trapezoidal(
            samples_mv_ma, start_s=1.0, end_s=3.0, voltage_unit="mv", current_unit="ma"
        )
        self.assertAlmostEqual(sub["energy_j"], 10.0)
        self.assertAlmostEqual(sub["mean_power_w"], 5.0)

    def test_trapezoidal_vi_rejects_bad_inputs(self):
        valid = [(0.0, 4.0, 1.0), (1.0, 4.0, 1.0)]
        with self.assertRaises(ValueError):
            integrate_vi_trapezoidal(valid, voltage_unit="kv")
        with self.assertRaises(ValueError):
            integrate_vi_trapezoidal(valid, start_s=-0.1, end_s=0.5)
        with self.assertRaises(ValueError):
            integrate_vi_trapezoidal(valid, max_gap_s=0.5)
        for bad_samples in (
            [(0.0, 4.0, 1.0)],
            [(0.0, 4.0, 1.0), (0.0, 4.0, 1.0)],
            [(0.0, -4.0, 1.0), (1.0, 4.0, 1.0)],
            [(0.0, 4.0), (1.0, 4.0)],
        ):
            with self.subTest(bad_samples=bad_samples), self.assertRaises(ValueError):
                integrate_vi_trapezoidal(bad_samples)

    def test_standby_baseline_subtraction(self):
        net = subtract_standby_baseline(50.0, duration_s=10.0, baseline_power_w=1.2)
        self.assertAlmostEqual(net["standby_energy_j"], 12.0)
        self.assertAlmostEqual(net["net_energy_j"], 38.0)
        self.assertAlmostEqual(net["net_power_w"], 3.8)

        unclamped = subtract_standby_baseline(5.0, duration_s=10.0, baseline_power_w=0.8)
        self.assertAlmostEqual(unclamped["net_energy_j"], -3.0)
        clamped = subtract_standby_baseline(
            5.0, duration_s=10.0, baseline_power_w=0.8, clamp_zero=True
        )
        self.assertEqual(clamped["net_energy_j"], 0.0)


if __name__ == "__main__":
    unittest.main()
