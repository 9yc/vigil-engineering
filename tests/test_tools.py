import csv
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from tools.summarize_csv import summarize


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


if __name__ == "__main__":
    unittest.main()
