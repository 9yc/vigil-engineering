import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.structured_call_eval import (
    evaluate_call_sequence,
    evaluate_dataset,
    extract_tool_calls,
    normalize_call_against_schema,
    values_ast_equal,
)


ROOT = Path(__file__).resolve().parents[1]

SAMPLE_SCHEMAS = {
    "set_alarm": {
        "properties": {
            "hour": {"type": "integer"},
            "minute": {"type": "integer"},
            "label": {"type": "string", "default": "Alarm"},
            "enabled": {"type": "boolean", "default": True},
            "days": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["hour", "minute"],
    },
    "query_weather": {
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
            "latitude": {"type": "number"},
        },
        "required": ["city"],
    },
}


class ExtractionTests(unittest.TestCase):
    def test_direct_fenced_openai_and_inline_extraction(self):
        direct = extract_tool_calls('{"name": "query_weather", "arguments": {"city": "Tokyo"}}')
        self.assertEqual(direct.source_kind, "direct")
        self.assertIsNone(direct.error)
        self.assertEqual(direct.calls, ({"name": "query_weather", "arguments": {"city": "Tokyo"}},))

        fenced = extract_tool_calls(
            "Let me invoke the tool:\n```json\n"
            '[{"name": "set_alarm", "parameters": {"hour": 7, "minute": 30}}]\n'
            "```\nDone."
        )
        self.assertEqual(fenced.source_kind, "fenced")
        self.assertEqual(fenced.calls[0]["arguments"], {"hour": 7, "minute": 30})

        openai_fmt = extract_tool_calls(
            '{"tool_calls": [{"type": "function", "function": '
            '{"name": "query_weather", "arguments": "{\\"city\\": \\"Berlin\\"}"}}]}'
        )
        self.assertEqual(openai_fmt.source_kind, "direct")
        self.assertEqual(openai_fmt.calls[0], {"name": "query_weather", "arguments": {"city": "Berlin"}})

        inline = extract_tool_calls(
            'Thought: check weather first -> {"name": "query_weather", "arguments": {"city": "Oslo"}} trailing text'
        )
        self.assertEqual(inline.source_kind, "inline_scan")
        self.assertEqual(inline.calls[0]["arguments"]["city"], "Oslo")

    def test_malformed_and_nonstandard_json_fail_cleanly(self):
        for bad in ("", "   ", "no braces at all", '{"name": "x", "arguments": [1, 2]}', '{"name": "x", "arguments": {"v": NaN}}'):
            with self.subTest(bad=bad):
                res = extract_tool_calls(bad)
                self.assertEqual(res.calls, ())
                self.assertIsNotNone(res.error)


class SchemaAndAstMatcherTests(unittest.TestCase):
    def test_schema_defaults_coercion_and_strict_booleans(self):
        norm = normalize_call_against_schema(
            {"name": "set_alarm", "arguments": {"hour": 6.0, "minute": "15", "enabled": "false"}},
            SAMPLE_SCHEMAS,
            allow_string_coercion=True,
        )
        self.assertEqual(
            norm,
            {
                "name": "set_alarm",
                "arguments": {"hour": 6, "minute": 15, "label": "Alarm", "enabled": False},
            },
        )
        # Strict mode rejects string-encoded int and bool-as-int
        with self.assertRaises(TypeError):
            normalize_call_against_schema(
                {"name": "set_alarm", "arguments": {"hour": "6", "minute": 15}},
                SAMPLE_SCHEMAS,
                allow_string_coercion=False,
            )
        with self.assertRaises(TypeError):
            normalize_call_against_schema(
                {"name": "set_alarm", "arguments": {"hour": True, "minute": 15}},
                SAMPLE_SCHEMAS,
            )
        with self.assertRaises(ValueError):
            normalize_call_against_schema(
                {"name": "query_weather", "arguments": {"city": "Paris", "unit": "kelvin"}},
                SAMPLE_SCHEMAS,
            )

    def test_order_invariant_ast_and_bipartite_matching(self):
        a = {"filters": [{"k": 1, "v": 0.3333333}, {"k": 2, "v": 2.0}], "title": "  hello   world "}
        b = {"title": "hello world", "filters": [{"v": 2, "k": 2}, {"v": 1.0 / 3.0, "k": 1}]}
        self.assertFalse(values_ast_equal(a, b, unordered_lists=False))
        self.assertTrue(values_ast_equal(a, b, unordered_lists=True, rel_tol=1e-5))
        self.assertFalse(values_ast_equal(True, 1))
        self.assertFalse(values_ast_equal(0, False))

        # Duplicate / overlapping calls require bipartite matching rather than greedy set lookup
        preds = [
            {"name": "set_alarm", "arguments": {"hour": 7, "minute": 0}},
            {"name": "set_alarm", "arguments": {"hour": 8, "minute": 0}},
        ]
        golds = [
            {"name": "set_alarm", "arguments": {"hour": 8, "minute": 0}},
            {"name": "set_alarm", "arguments": {"hour": 7, "minute": 0}},
        ]
        unordered_score = evaluate_call_sequence(preds, golds, order_sensitive=False)
        self.assertTrue(unordered_score["exact_match"])
        self.assertEqual(unordered_score["f1"], 1.0)

        ordered_score = evaluate_call_sequence(preds, golds, order_sensitive=True)
        self.assertFalse(ordered_score["exact_match"])
        self.assertEqual(ordered_score["f1"], 0.0)


class DatasetAndCliTests(unittest.TestCase):
    def test_dataset_aggregation_and_cli_roundtrip(self):
        examples = [
            {
                "raw_output": '```json\n{"name": "query_weather", "arguments": {"city": "Seoul"}}\n```',
                "expected": [{"name": "query_weather", "arguments": {"city": "Seoul", "unit": "celsius"}}],
            },
            {
                "raw_output": '[{"name": "set_alarm", "arguments": {"hour": 7, "minute": 30, "days": ["Mon", "Wed"]}}]',
                "expected": [
                    {"name": "set_alarm", "arguments": {"hour": 7, "minute": 30, "days": ["Wed", "Mon"]}},
                    {"name": "query_weather", "arguments": {"city": "Busan"}},
                ],
            },
            {
                "raw_output": "I do not need any tool for this question.",
                "expected": [],
            },
        ]
        report = evaluate_dataset(
            examples, tool_schemas=SAMPLE_SCHEMAS, unordered_lists=True
        )
        self.assertEqual(report["examples"], 3)
        # Example 0 matches via schema default unit='celsius'; Example 1 gets 1 TP + 1 FN; Example 2 has parse failure on empty gold -> 0 FP, 0 FN => exact match!
        self.assertEqual(report["exact_matches"], 2)
        self.assertEqual(report["true_positives"], 2)
        self.assertEqual(report["false_positives"], 0)
        self.assertEqual(report["false_negatives"], 1)
        self.assertAlmostEqual(report["micro_precision"], 1.0)
        self.assertAlmostEqual(report["micro_recall"], 2.0 / 3.0)

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            tmp_dir = Path(tmp)
            in_jsonl = tmp_dir / "cases.jsonl"
            schema_file = tmp_dir / "schemas.json"
            out_json = tmp_dir / "summary.json"
            in_jsonl.write_text("\n".join(json.dumps(row) for row in examples) + "\n")
            schema_file.write_text(json.dumps(SAMPLE_SCHEMAS))

            cmd = [
                sys.executable,
                "-B",
                str(ROOT / "tools/structured_call_eval.py"),
                str(in_jsonl),
                "--schemas",
                str(schema_file),
                "--unordered-lists",
                "--output",
                str(out_json),
            ]
            first = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            self.assertEqual(first.returncode, 0, first.stderr)
            loaded = json.loads(out_json.read_text())
            self.assertEqual(loaded["true_positives"], 2)

            # Second run with existing --output must fail without clobbering
            second = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(second.returncode, 0)


if __name__ == "__main__":
    unittest.main()
