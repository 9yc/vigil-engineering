import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from tools.summarize_csv import summarize
from tools.integrate_power import integrate
from tools.validate_measurements import validate
from runtime_support import FrameReader, FrameWriter, LineProcess, parse_output, validate_calls


ROOT = Path(__file__).resolve().parents[1]


class SummaryTests(unittest.TestCase):
    def test_grouped_fixture(self):
        with (ROOT / "examples/synthetic_measurements.csv").open(newline="") as f:
            result = summarize(csv.DictReader(f), "value", ("case",))
        self.assertEqual(result, [
            {"case": "sample-a", "count": 3, "minimum": 2.0,
             "median": 4.0, "maximum": 6.0},
            {"case": "sample-b", "count": 3, "minimum": 1.0,
             "median": 3.0, "maximum": 5.0},
        ])

    def test_ungrouped_even_median(self):
        result = summarize([{"x": "-2"}, {"x": "4"}], "x")
        self.assertEqual(result[0], {
            "count": 2, "minimum": -2, "median": 1, "maximum": 4,
        })

    def test_invalid_rows_fail(self):
        for rows in ([], [{}], [{"x": ""}], [{"x": None}],
                     [{"x": "nan"}], [{"x": "inf"}], [{"x": "invalid"}],
                     [{"x": "1", None: ["extra"]}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize(rows, "x")

    def test_grouping_errors_fail(self):
        for groups in (("case", "case"), ("count",), ("missing",)):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                summarize([{"x": "1", "case": "a", "count": "a"}], "x", groups)


class MeasurementTests(unittest.TestCase):
    def test_complete_measurements(self):
        rows = [{"iteration": str(i), "duration_ms": str(v)}
                for i, v in enumerate((3, 5, 4))]
        self.assertEqual(validate(rows, 3), {
            "validated_rows": 3, "first_iteration": 0, "last_iteration": 2,
            "minimum": 3, "median": 4, "maximum": 5,
        })

    def test_incomplete_duplicate_and_out_of_order(self):
        for indices in ([], [0], [0, 0, 2], [0, 2, 1], [0, 1, 3], [0, 1, 2, 3]):
            rows = [{"iteration": str(i), "duration_ms": "1"} for i in indices]
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                validate(rows, 3)
        with self.assertRaises(ValueError):
            validate([], 0)

    def test_invalid_durations(self):
        for value in ("nan", "inf", "0", "-1", "text", "", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate([{"iteration": "0", "duration_ms": value}], 1)


class PowerTests(unittest.TestCase):
    def test_constant_and_irregular_windows(self):
        self.assertEqual(integrate([(0, 3), (2, 3)])["energy_j"], 6)
        self.assertEqual(integrate([(0, 2), (2, 4), (5, 2)])["energy_j"], 15)
        self.assertEqual(integrate([(0, 0), (1, 0)])["mean_power_w"], 0)

    def test_clipped_window_and_boundaries(self):
        samples = [(0, 2), (1, 4), (2, 2)]
        full = integrate(samples)
        window = integrate(samples, 0.25, 1.25, max_gap=1)
        self.assertEqual(full["energy_j"], 6)
        self.assertEqual(window["energy_j"], 3.5)
        self.assertEqual(window["mean_power_w"], 3.5)
        self.assertEqual(integrate(samples, 0.5, 1.5)["energy_j"], 4)

    def test_invalid_series(self):
        for samples in ([], [(0, 1)], [(0, 1), (0, 2)], [(1, 1), (0, 2)],
                        [(0, -1), (1, 2)], [(0, 1), (float("inf"), 2)],
                        [(0, 1), (1, float("nan"))],
                        [(-1e308, 1), (1e308, 1)]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                integrate(samples)

    def test_invalid_windows_and_gaps(self):
        samples = [(0, 2), (1, 4), (2, 2)]
        for start, end in ((-1, 1), (0, 3), (1, 1), (2, 1), (0, float("nan"))):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                integrate(samples, start, end)
        for gap in (0, -1, 0.5, float("nan")):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                integrate(samples, max_gap=gap)


class BinaryCodecTests(unittest.TestCase):
    def test_fields_and_utf8_roundtrip(self):
        writer = FrameWriter(b"EXAMPLE")
        writer.put("Ii", 7, -2)
        writer.text("你好")
        writer.text("")
        reader = FrameReader(writer.done(), b"EXAMPLE")
        self.assertEqual(reader.get("Ii"), (7, -2))
        self.assertEqual(reader.text(), "你好")
        self.assertEqual(reader.text(), "")
        reader.finish()

    def test_corruption_and_magic_are_rejected(self):
        writer = FrameWriter(b"EXAMPLE")
        writer.put("I", 7)
        data = writer.done()
        broken = bytearray(data)
        broken[7] ^= 1
        for value, magic in ((data, b"OTHER"), (data[:-1], b"EXAMPLE"),
                             (bytes(broken), b"EXAMPLE"), (b"", b"EXAMPLE")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FrameReader(value, magic)

    def test_truncated_fields_and_strings(self):
        short_field = FrameWriter(b"EXAMPLE")
        short_field.put("H", 1)
        with self.assertRaises(ValueError):
            FrameReader(short_field.done(), b"EXAMPLE").get("I")
        short_string = FrameWriter(b"EXAMPLE")
        short_string.put("H", 8)
        short_string.put("B", 65)
        with self.assertRaises(ValueError):
            FrameReader(short_string.done(), b"EXAMPLE").text()

    def test_invalid_utf8_and_oversized_text(self):
        writer = FrameWriter(b"EXAMPLE")
        writer.put("HB", 1, 255)
        with self.assertRaises(UnicodeDecodeError):
            FrameReader(writer.done(), b"EXAMPLE").text()
        with self.assertRaises(ValueError):
            FrameWriter(b"EXAMPLE").text("字" * 22000)

    def test_unconsumed_fields_are_rejected(self):
        writer = FrameWriter(b"EXAMPLE")
        writer.put("II", 1, 2)
        reader = FrameReader(writer.done(), b"EXAMPLE")
        self.assertEqual(reader.get("I"), 1)
        with self.assertRaises(ValueError):
            reader.finish()


class JsonOutputTests(unittest.TestCase):
    def setUp(self):
        self.specs = {"echo": {
            "text": {"type": "string", "required": True},
            "count": {"type": "int"},
            "style": {"type": "string-enum", "enum": ["brief", "full"]},
        }}

    def test_plain_and_embedded_json(self):
        payload = '[{"name":"echo","parameters":{"text":"hello"}}]'
        value, error, fallback = parse_output(payload)
        self.assertIsNone(error)
        self.assertFalse(fallback)
        self.assertEqual(validate_calls(value, self.specs), [])
        parsed, error, fallback = parse_output("```json\n" + payload + "\n```")
        self.assertEqual(parsed, value)
        self.assertIsNone(error)
        self.assertTrue(fallback)

    def test_malformed_and_nonstandard_json(self):
        for raw in ("not JSON", "[", "[NaN]", "[Infinity]"):
            with self.subTest(raw=raw):
                value, error, fallback = parse_output(raw)
                self.assertIsNone(value)
                self.assertIsNotNone(error)
                self.assertFalse(fallback)

    def test_required_unknown_enum_and_boolean_types(self):
        for params in ({}, {"text": "ok", "extra": 1},
                       {"text": "ok", "count": True},
                       {"text": "ok", "style": "unknown"}):
            with self.subTest(params=params):
                self.assertTrue(validate_calls(
                    [{"name": "echo", "parameters": params}], self.specs))
        for value in ({}, [1], [{"name": "missing", "parameters": {}}],
                      [{"name": "echo", "parameters": []}]):
            with self.subTest(value=value):
                self.assertTrue(validate_calls(value, self.specs))

    def test_additional_supported_types(self):
        specs = {"example": {
            "flag": {"type": "bool"}, "value": {"type": "float"},
            "options": {"type": "dictionary"}, "items": {"type": "list"},
        }}
        params = {"flag": False, "value": 1.5, "options": {}, "items": []}
        self.assertEqual(validate_calls([{"name": "example", "parameters": params}], specs), [])
        for invalid in (True, float("inf"), float("nan")):
            params["value"] = invalid
            with self.subTest(invalid=invalid):
                self.assertTrue(validate_calls(
                    [{"name": "example", "parameters": params}], specs))


class ToolIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(os.environ["TMPDIR"])
        self.assertTrue(temporary_root.is_dir())
        directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.node_a = self.directory / "node_a"
        self.node_b = self.directory / "node_b"
        self.node_a.write_text("-12\n")
        self.node_b.write_text("345\n")
        self.output = self.directory / "samples.csv"
        self.sampler = Path(os.environ["SAMPLER_BINARY"])

    def sample(self, count="2"):
        return subprocess.run(
            [str(self.sampler), str(self.output),
             str(self.node_a), str(self.node_b), count],
            capture_output=True, text=True, timeout=10,
        )

    def summary_cli(self, source, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "tools/summarize_csv.py"),
             str(source), *args],
            capture_output=True, text=True, timeout=10,
        )

    def tool_cli(self, name, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "tools" / name), *map(str, args)],
            capture_output=True, text=True, timeout=10,
        )

    def test_sample_and_summarize(self):
        sampled = self.sample("3")
        self.assertEqual(sampled.returncode, 0, sampled.stderr)
        with self.output.open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        self.assertEqual([int(r["value_a_raw"]) for r in rows], [-12] * 3)
        self.assertEqual([int(r["value_b_raw"]) for r in rows], [345] * 3)
        timestamps = [int(r["monotonic_ns"]) for r in rows]
        self.assertTrue(all(b > a for a, b in zip(timestamps, timestamps[1:])))
        summary = self.summary_cli(self.output, "--column", "value_a_raw")
        self.assertEqual(summary.returncode, 0, summary.stderr)
        self.assertEqual(list(csv.DictReader(io.StringIO(summary.stdout))), [
            {"count": "3", "minimum": "-12.0", "median": "-12.0", "maximum": "-12.0"},
        ])

    def test_existing_output_is_preserved(self):
        self.output.write_text("keep this content\n")
        sampled = self.sample("1")
        self.assertNotEqual(sampled.returncode, 0)
        self.assertEqual(self.output.read_text(), "keep this content\n")

    def test_invalid_node_values_fail_without_output(self):
        for value in ("", "12garbage\n", "2147483648\n", "7\x00hidden", "9" * 64):
            self.node_a.write_text(value)
            with self.subTest(value=value):
                self.assertNotEqual(self.sample("1").returncode, 0)
                self.assertFalse(self.output.exists())

    def test_missing_node_fails(self):
        self.node_a.unlink()
        self.assertNotEqual(self.sample("1").returncode, 0)
        self.assertFalse(self.output.exists())

    def test_invalid_sample_count_fails(self):
        for count in ("-1", "x", "3junk", ""):
            with self.subTest(count=count):
                self.assertNotEqual(self.sample(count).returncode, 0)
                self.assertFalse(self.output.exists())

    def test_continuous_sampling_stops_cleanly(self):
        process = subprocess.Popen(
            [str(self.sampler), str(self.output),
             str(self.node_a), str(self.node_b), "0"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.output.exists() and len(self.output.read_text().splitlines()) >= 2:
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.01)
            self.assertIsNone(process.poll())
            self.assertTrue(self.output.exists())
            self.assertGreaterEqual(len(self.output.read_text().splitlines()), 2)
            process.send_signal(signal.SIGTERM)
            _, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stderr)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()

    def test_summary_cli_fixture(self):
        result = self.summary_cli(ROOT / "examples/synthetic_measurements.csv",
                                  "--column", "value", "--group-by", "case")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        self.assertEqual([r["median"] for r in rows], ["4.0", "3.0"])
        self.assertEqual([r["count"] for r in rows], ["3", "3"])

    def test_summary_cli_rejects_bad_headers_and_rows(self):
        source = self.directory / "input.csv"
        for content in ("", "other\n1\n", "value,value\n1,2\n",
                        "value\n", "value\nNaN\n", "value\n1,2\n"):
            source.write_text(content)
            with self.subTest(content=content):
                result = self.summary_cli(source, "--column", "value")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_summary_output_is_created_once(self):
        source = ROOT / "examples/synthetic_measurements.csv"
        first = self.summary_cli(source, "--column", "value",
                                 "--output", str(self.output))
        self.assertEqual(first.returncode, 0, first.stderr)
        previous = self.output.read_text()
        second = self.summary_cli(source, "--column", "value",
                                  "--output", str(self.output))
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(self.output.read_text(), previous)

    def test_measurement_cli(self):
        source = ROOT / "examples/synthetic_durations.csv"
        result = self.tool_cli("validate_measurements.py", source, "--expected", "3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["validated_rows"], 3)
        partial = self.tool_cli("validate_measurements.py", source, "--expected", "4")
        self.assertNotEqual(partial.returncode, 0)
        self.assertEqual(partial.stdout, "")

    def test_power_cli(self):
        result = self.tool_cli(
            "integrate_power.py", ROOT / "examples/synthetic_power.csv",
            "--start", "0.25", "--end", "1.25", "--max-gap", "1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["energy_j"], 3.5)

    def test_new_tools_reject_malformed_csv(self):
        source = self.directory / "malformed.csv"
        for content in ("", "time_s,power_w\n0,2\n1,\n",
                        "time_s,power_w\n0,2,extra\n1,2\n",
                        "time_s,power_w\n0,nan\n1,2\n"):
            source.write_text(content)
            with self.subTest(content=content):
                self.assertNotEqual(
                    self.tool_cli("integrate_power.py", source).returncode, 0)
                self.assertNotEqual(
                    self.tool_cli("validate_measurements.py", source,
                                  "--expected", "2").returncode, 0)

    def test_process_request_and_diagnostics(self):
        child = (
            "import sys\n"
            "print('example diagnostic', file=sys.stderr, flush=True)\n"
            "print('READY', flush=True)\n"
            "for line in sys.stdin:\n"
            "    print('NOTICE observed input', flush=True)\n"
            "    print('OK ' + line.strip(), flush=True)\n"
        )
        transcript = self.directory / "transcript.log"
        stderr = self.directory / "stderr.log"
        with LineProcess([sys.executable, "-u", "-c", child],
                         transcript=transcript, stderr_path=stderr) as link:
            self.assertEqual(link.wait(lambda value: value == "READY", 5), "READY")
            self.assertEqual(link.request("hello", ("OK",), timeout=5), "OK hello")
            self.assertIn("NOTICE observed input", link.stray)
            with self.assertRaises(ValueError):
                link.send("two\nlines")
        self.assertEqual(link.proc.returncode, 0)
        self.assertIn("> hello", transcript.read_text())
        self.assertIn("< OK hello", transcript.read_text())
        self.assertIn("example diagnostic", stderr.read_text())

    def test_process_timeout_and_forced_close(self):
        child = "import time; print('READY', flush=True); time.sleep(60)"
        link = LineProcess([sys.executable, "-u", "-c", child])
        try:
            self.assertEqual(link.wait(lambda value: value == "READY", 5), "READY")
            self.assertIsNone(link.request("PING", ("PONG",), timeout=0.02))
        finally:
            link.close(timeout=0.02)
        self.assertIsNotNone(link.proc.poll())
        self.assertFalse(link.send("after close"))
        link.close()

    def test_process_eof(self):
        with LineProcess([sys.executable, "-u", "-c", "print('READY')"]) as link:
            self.assertEqual(link.wait(lambda value: value == "READY", 5), "READY")
            self.assertIsNone(link.wait(lambda value: True, 5))
            self.assertFalse(link.send("after EOF"))

    def test_process_deadline_with_stray_output(self):
        child = (
            "import time\n"
            "print('READY', flush=True)\n"
            "for i in range(1000): print('NOTICE', flush=True)\n"
            "time.sleep(60)\n"
        )
        link = LineProcess([sys.executable, "-u", "-c", child])
        try:
            self.assertEqual(link.wait(lambda value: value == "READY", 5), "READY")
            start = time.monotonic()
            self.assertIsNone(link.wait(lambda value: time.sleep(0.002) or False, 0.02))
            self.assertLess(time.monotonic() - start, 1)
            self.assertTrue(link.stray)
        finally:
            link.close(timeout=0.02)


if __name__ == "__main__":
    unittest.main()
