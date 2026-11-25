import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def make_block(
    *,
    ts_begin: int = 1000,
    ts_end: int = 1020,
    ac: str = "false",
    usb: str = "false",
    wireless: str = "false",
    dock: str = "false",
    status: int = 3,
    voltage_mv: int = 3950,
    level: int = 78,
    bat_temp_tenths: int = 312,
    ibat_val: float = -1450.0,
    ibat_status: int = 0,
    vbat_val: float = 3948.0,
    skin_val: float = 35.6,
    pid: int = 4321,
    rss_kb: int = 184320,
    vmhwm_kb: int = 212992,
    closed: bool = True,
) -> str:
    lines = [
        "__SAMPLE_BEGIN__",
        f"__TS_BEGIN__={ts_begin}",
        "Current Battery Service state:",
        f"  AC powered: {ac}",
        f"  USB powered: {usb}",
        f"  Wireless powered: {wireless}",
        f"  Dock powered: {dock}",
        f"  status: {status}",
        f"  level: {level}",
        f"  voltage: {voltage_mv}",
        f"  temperature: {bat_temp_tenths}",
        "Thermal Service current temperatures:",
        f"  Temperature{{mValue={ibat_val}, mType=8, mName=ibat, mStatus={ibat_status}}}",
        f"  Temperature{{mValue={vbat_val}, mType=7, mName=vbat, mStatus=0}}",
        f"  Temperature{{mValue={skin_val}, mType=3, mName=skin, mStatus=0}}",
        "  Temperature{mValue=41.2, mType=0, mName=cpu_cluster0, mStatus=0}",
        f"pid={pid} rss_kb={rss_kb} vmhwm_kb={vmhwm_kb}",
        f"__TS_END__={ts_end}",
    ]
    if closed:
        lines.append("__SAMPLE_END__")
    return "\n".join(lines) + "\n"


class TestTelemetryBlockDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifact_dir = Path(os.environ["ARTIFACT_DIR"])
        cls.binary = artifact_dir / "telemetry_block_decoder"
        if not cls.binary.is_file():
            raise RuntimeError(f"missing binary: {cls.binary}")
        cls.base_tmp = Path(os.environ["TMPDIR"])
        cls.base_tmp.mkdir(parents=True, exist_ok=True)

    def setUp(self) -> None:
        self._tmp_ctx = tempfile.TemporaryDirectory(dir=self.base_tmp)
        self.work_dir = Path(self._tmp_ctx.name)

    def tearDown(self) -> None:
        self._tmp_ctx.cleanup()

    def _run_decoder(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.binary), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_valid_multiblock_decoding_midpoint_timestamps_and_csv_jsonl(self) -> None:
        raw_stream = (
            "preamble noise ignored outside sample markers\n"
            + make_block(
                ts_begin=10000, ts_end=10015, voltage_mv=4100,
                ibat_val=-1200.0, skin_val=34.2, rss_kb=150000, vmhwm_kb=180000,
            )
            + make_block(
                ts_begin=11000, ts_end=11020, voltage_mv=3900,
                ibat_val=2000.0, skin_val=37.8, rss_kb=192000, vmhwm_kb=220000,
            )
        )
        in_path = self.work_dir / "valid_stream.txt"
        in_path.write_text(raw_stream, encoding="utf-8")

        jsonl_out = self.work_dir / "samples.jsonl"
        summary_out = self.work_dir / "summary.json"
        proc = self._run_decoder([
            "--input", str(in_path),
            "--output", str(jsonl_out),
            "--summary-output", str(summary_out),
        ])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

        rows = [json.loads(line) for line in jsonl_out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)

        self.assertAlmostEqual(rows[0]["ts_mid_ms"], 10007.5, places=6)
        self.assertAlmostEqual(rows[0]["sample_span_ms"], 15.0, places=6)
        self.assertAlmostEqual(rows[0]["voltage_v"], 4.10, places=6)
        self.assertAlmostEqual(rows[0]["current_a"], -1.20, places=6)
        self.assertAlmostEqual(rows[0]["power_w"], 4.10 * 1.20, places=6)
        self.assertAlmostEqual(rows[0]["skin_temp_c"], 34.2, places=6)

        self.assertAlmostEqual(rows[1]["ts_mid_ms"], 11010.0, places=6)
        self.assertAlmostEqual(rows[1]["voltage_v"], 3.90, places=6)
        self.assertAlmostEqual(rows[1]["current_a"], 2.00, places=6)
        self.assertAlmostEqual(rows[1]["power_w"], 3.90 * 2.00, places=6)

        summary = json.loads(summary_out.read_text(encoding="utf-8"))
        self.assertEqual(summary["valid_samples"], 2)
        self.assertEqual(summary["rejected_samples"], 0)
        self.assertAlmostEqual(summary["mean_voltage_v"], 4.00, places=6)
        self.assertAlmostEqual(summary["max_skin_temp_c"], 37.8, places=6)
        self.assertEqual(summary["max_rss_kb"], 192000)
        self.assertEqual(summary["max_vmhwm_kb"], 220000)

        csv_out = self.work_dir / "samples.csv"
        proc_csv = self._run_decoder(["--input", str(in_path), "--output", str(csv_out)])
        self.assertEqual(proc_csv.returncode, 0, msg=proc_csv.stderr)
        with csv_out.open("r", encoding="utf-8", newline="") as fh:
            csv_rows = list(csv.DictReader(fh))
        self.assertEqual(len(csv_rows), 2)
        self.assertAlmostEqual(float(csv_rows[0]["ts_mid_ms"]), 10007.5, places=5)
        self.assertEqual(int(csv_rows[1]["vmhwm_kb"]), 220000)

    def test_charger_rails_and_battery_status_rejections(self) -> None:
        stream = "".join([
            make_block(ts_begin=100, ts_end=110, ac="true"),
            make_block(ts_begin=200, ts_end=210, usb="true"),
            make_block(ts_begin=300, ts_end=310, wireless="true"),
            make_block(ts_begin=400, ts_end=410, dock="true"),
            make_block(ts_begin=500, ts_end=510, status=2),
            make_block(ts_begin=600, ts_end=610),
        ])
        in_path = self.work_dir / "power_qc.txt"
        in_path.write_text(stream, encoding="utf-8")
        out_path = self.work_dir / "power_valid.jsonl"
        sum_path = self.work_dir / "power_sum.json"

        proc = self._run_decoder([
            "--input", str(in_path),
            "--output", str(out_path),
            "--summary-output", str(sum_path),
        ])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        summary = json.loads(sum_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["valid_samples"], 1)
        self.assertEqual(summary["rejected_samples"], 5)
        self.assertEqual(summary["rejected_by_reason"]["external_power_connected"], 4)
        self.assertEqual(summary["rejected_by_reason"]["not_discharging"], 1)

        valid_lines = [json.loads(x) for x in out_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(valid_lines), 1)
        self.assertEqual(valid_lines[0]["block_index"], 5)

    def test_voltage_current_sensor_status_and_timestamp_order_qc(self) -> None:
        stream = "".join([
            make_block(ts_begin=100, ts_end=110, voltage_mv=3200, ibat_val=-15000.0),
            make_block(ts_begin=200, ts_end=210, voltage_mv=4600, ibat_val=15000.0),
            make_block(ts_begin=300, ts_end=310, voltage_mv=3199, ibat_val=-1000.0),
            make_block(ts_begin=400, ts_end=410, voltage_mv=4601, ibat_val=-1000.0),
            make_block(ts_begin=500, ts_end=510, voltage_mv=3900, ibat_val=-15050.0),
            make_block(ts_begin=600, ts_end=610, voltage_mv=3900, ibat_val=15100.0),
            make_block(ts_begin=700, ts_end=710, voltage_mv=3900, ibat_val=-1200.0, ibat_status=2),
            make_block(ts_begin=850, ts_end=840, voltage_mv=3900, ibat_val=-1200.0),
        ])
        in_path = self.work_dir / "bounds_qc.txt"
        in_path.write_text(stream, encoding="utf-8")
        out_path = self.work_dir / "bounds_valid.jsonl"
        sum_path = self.work_dir / "bounds_sum.json"

        proc = self._run_decoder([
            "--input", str(in_path),
            "--output", str(out_path),
            "--summary-output", str(sum_path),
        ])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        summary = json.loads(sum_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["valid_samples"], 2)
        self.assertEqual(summary["rejected_samples"], 6)
        self.assertEqual(summary["rejected_by_reason"]["voltage_out_of_range"], 2)
        self.assertEqual(summary["rejected_by_reason"]["current_out_of_range"], 2)
        self.assertEqual(summary["rejected_by_reason"]["ibat_sensor_fault"], 1)
        self.assertEqual(summary["rejected_by_reason"]["invalid_timestamp_order"], 1)

    def test_truncated_tail_handling_with_and_without_allow_partial_tail(self) -> None:
        stream = (
            make_block(ts_begin=1000, ts_end=1010, voltage_mv=4020)
            + make_block(ts_begin=2000, ts_end=2010, voltage_mv=4010)
            + make_block(ts_begin=3000, ts_end=3010, voltage_mv=4000, closed=False)
        )
        in_path = self.work_dir / "partial_tail.txt"
        in_path.write_text(stream, encoding="utf-8")

        strict_out = self.work_dir / "strict.jsonl"
        strict_sum = self.work_dir / "strict_sum.json"
        proc_strict = self._run_decoder([
            "--input", str(in_path),
            "--output", str(strict_out),
            "--summary-output", str(strict_sum),
        ])
        self.assertNotEqual(proc_strict.returncode, 0)
        self.assertFalse(strict_out.exists())
        self.assertFalse(strict_sum.exists())

        lenient_out = self.work_dir / "lenient.jsonl"
        lenient_sum = self.work_dir / "lenient_sum.json"
        proc_lenient = self._run_decoder([
            "--input", str(in_path),
            "--output", str(lenient_out),
            "--summary-output", str(lenient_sum),
            "--allow-partial-tail",
        ])
        self.assertEqual(proc_lenient.returncode, 0, msg=proc_lenient.stderr)
        decoded = [json.loads(x) for x in lenient_out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(decoded), 2)

        summary = json.loads(lenient_sum.read_text(encoding="utf-8"))
        self.assertTrue(summary["allow_partial_tail"])
        self.assertEqual(summary["complete_blocks"], 2)
        self.assertEqual(summary["partial_tail_blocks"], 1)
        self.assertEqual(summary["valid_samples"], 2)
        self.assertEqual(summary["rejected_samples"], 1)
        self.assertEqual(summary["rejected_by_reason"]["truncated_tail"], 1)

    def test_amps_unit_mode_and_exclusive_output_protection(self) -> None:
        in_path = self.work_dir / "amps_mode.txt"
        in_path.write_text(make_block(ibat_val=-2.25, voltage_mv=4000), encoding="utf-8")
        out_path = self.work_dir / "amps.jsonl"
        proc = self._run_decoder([
            "--input", str(in_path),
            "--output", str(out_path),
            "--ibat-unit", "a",
        ])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        sample = json.loads(out_path.read_text(encoding="utf-8").strip())
        self.assertAlmostEqual(sample["current_a"], -2.25, places=6)
        self.assertAlmostEqual(sample["power_w"], 9.0, places=6)

        sentinel = "do_not_overwrite\n"
        guarded = self.work_dir / "guarded.jsonl"
        guarded.write_text(sentinel, encoding="utf-8")
        proc_excl = self._run_decoder(["--input", str(in_path), "--output", str(guarded)])
        self.assertNotEqual(proc_excl.returncode, 0)
        self.assertEqual(guarded.read_text(encoding="utf-8"), sentinel)

        sum_guard = self.work_dir / "guarded_sum.json"
        sum_guard.write_text(sentinel, encoding="utf-8")
        proc_sum_excl = self._run_decoder([
            "--input", str(in_path),
            "--output", str(self.work_dir / "fresh_samples.jsonl"),
            "--summary-output", str(sum_guard),
        ])
        self.assertNotEqual(proc_sum_excl.returncode, 0)
        self.assertEqual(sum_guard.read_text(encoding="utf-8"), sentinel)

    def test_malformed_interior_blocks_orphan_end_and_missing_fields(self) -> None:
        raw_stream = (
            "__SAMPLE_END__\n"
            + make_block(ts_begin=500, ts_end=510, closed=False)
            + "__SAMPLE_BEGIN__\n"
            + "__TS_BEGIN__=700\n"
            + "__TS_END__=710\n"
            + "  AC powered: false\n"
            + "__SAMPLE_END__\n"
            + make_block(ts_begin=900, ts_end=920, voltage_mv=3850, ibat_val=-900.0)
        )
        in_path = self.work_dir / "malformed_mix.txt"
        in_path.write_text(raw_stream, encoding="utf-8")
        out_path = self.work_dir / "mix_valid.jsonl"
        sum_path = self.work_dir / "mix_sum.json"

        proc = self._run_decoder([
            "--input", str(in_path),
            "--output", str(out_path),
            "--summary-output", str(sum_path),
        ])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        summary = json.loads(sum_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["valid_samples"], 1)
        self.assertEqual(summary["rejected_samples"], 3)
        self.assertEqual(summary["rejected_by_reason"]["orphan_sample_end"], 1)
        self.assertEqual(summary["rejected_by_reason"]["unclosed_interior_block"], 1)
        self.assertEqual(summary["rejected_by_reason"]["missing_required_fields"], 1)


if __name__ == "__main__":
    unittest.main()
