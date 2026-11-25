#!/usr/bin/env python3
"""Structured JSON function-call extraction, AST argument matching and F1 grading.

Extracts single or batched function calls from raw model text (including fenced
code blocks and surrounding commentary), normalizes typed arguments against
caller-provided schemas, compares predicted calls against reference targets with
order-invariant argument AST matching and numeric tolerances, and aggregates
set-level and call-level precision, recall and F1 metrics.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


_FENCED_BLOCK_RE = re.compile(r"```(?:json|tool_call|javascript|js)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _reject_nonstandard_constant(token):
    raise ValueError(f"nonstandard JSON constant: {token}")


@dataclass(frozen=True)
class ExtractedCalls:
    calls: tuple
    source_kind: str
    error: str | None = None


def _normalize_call_record(item):
    """Canonicalize `{name, arguments|parameters}` or OpenAI-style `{function: {...}}`."""
    if not isinstance(item, dict):
        raise ValueError("call entry must be a JSON object")
    if "function" in item and isinstance(item["function"], dict):
        fn = item["function"]
        raw_name = fn.get("name")
        raw_args = fn.get("arguments", fn.get("parameters", {}))
    else:
        raw_name = item.get("name", item.get("tool", item.get("function")))
        if "arguments" in item and "parameters" in item:
            raise ValueError("call entry cannot contain both arguments and parameters")
        raw_args = item.get("arguments", item.get("parameters", {}))

    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("call entry requires a non-empty string function name")
    if isinstance(raw_args, str):
        raw_args = json.loads(raw_args, parse_constant=_reject_nonstandard_constant)
    if not isinstance(raw_args, dict):
        raise ValueError("call arguments must decode to a JSON object")
    return {"name": raw_name.strip(), "arguments": raw_args}


def _coerce_parsed_payload(payload):
    if isinstance(payload, list):
        return tuple(_normalize_call_record(entry) for entry in payload)
    if isinstance(payload, dict):
        if "tool_calls" in payload and isinstance(payload["tool_calls"], list):
            return tuple(_normalize_call_record(entry) for entry in payload["tool_calls"])
        return (_normalize_call_record(payload),)
    raise ValueError("top-level JSON value must be a call object or array of calls")


def _scan_balanced_json(text):
    decoder = json.JSONDecoder(parse_constant=_reject_nonstandard_constant)
    candidates = []
    idx = 0
    length = len(text)
    while idx < length:
        ch = text[idx]
        if ch in "[{":
            try:
                parsed, end_offset = decoder.raw_decode(text[idx:])
                candidates.append(parsed)
                idx += max(1, end_offset)
                continue
            except (json.JSONDecodeError, ValueError):
                pass
        idx += 1
    return candidates


def extract_tool_calls(raw_text):
    """Extract structured tool calls from direct JSON, fenced blocks or inline prose."""
    if not isinstance(raw_text, str):
        return ExtractedCalls((), "invalid", "input must be a string")
    stripped = raw_text.strip()
    if not stripped:
        return ExtractedCalls((), "empty", "empty response")

    try:
        direct = json.loads(stripped, parse_constant=_reject_nonstandard_constant)
        return ExtractedCalls(_coerce_parsed_payload(direct), "direct", None)
    except (json.JSONDecodeError, ValueError) as direct_err:
        last_error = str(direct_err)

    for match in _FENCED_BLOCK_RE.finditer(raw_text):
        block = match.group(1).strip()
        if not block:
            continue
        try:
            parsed = json.loads(block, parse_constant=_reject_nonstandard_constant)
            return ExtractedCalls(_coerce_parsed_payload(parsed), "fenced", None)
        except (json.JSONDecodeError, ValueError) as fence_err:
            last_error = str(fence_err)

    for candidate in _scan_balanced_json(raw_text):
        try:
            return ExtractedCalls(_coerce_parsed_payload(candidate), "inline_scan", None)
        except ValueError as scan_err:
            last_error = str(scan_err)

    return ExtractedCalls((), "failed", last_error)


def _normalize_scalar(value, expected_type, *, allow_string_coercion=False):
    if expected_type == "string":
        if isinstance(value, str):
            return " ".join(value.strip().split())
        raise TypeError("expected string")
    if expected_type == "boolean":
        if isinstance(value, bool):
            return value
        if allow_string_coercion and isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "false"):
                return low == "true"
        raise TypeError("expected boolean")
    if expected_type == "integer":
        if type(value) is int:
            return value
        if type(value) is float and value.is_integer() and math.isfinite(value):
            return int(value)
        if allow_string_coercion and isinstance(value, str):
            return int(value.strip(), 10)
        raise TypeError("expected integer")
    if expected_type in ("number", "float"):
        if type(value) is int:
            return float(value)
        if type(value) is float and math.isfinite(value):
            return value
        if allow_string_coercion and isinstance(value, str):
            parsed = float(value.strip())
            if not math.isfinite(parsed):
                raise ValueError("non-finite float")
            return parsed
        raise TypeError("expected finite number")
    raise ValueError(f"unsupported scalar schema type: {expected_type!r}")


def normalize_call_against_schema(call, tool_schemas, *, allow_string_coercion=False):
    """Validate and canonicalize a `{name, arguments}` call against `tool_schemas`."""
    if not isinstance(call, dict) or "name" not in call or "arguments" not in call:
        raise ValueError("call must contain 'name' and 'arguments'")
    name = call["name"]
    if name not in tool_schemas:
        raise ValueError(f"unknown tool name: {name!r}")
    spec = tool_schemas[name]
    properties = spec.get("properties", {})
    required = set(spec.get("required", ()))
    allow_additional = spec.get("additionalProperties", False)
    raw_args = call["arguments"]
    if not isinstance(raw_args, dict):
        raise ValueError("call arguments must be a dictionary")

    normalized_args = {}
    for req_key in sorted(required):
        if req_key not in raw_args:
            raise ValueError(f"missing required parameter {req_key!r} for {name!r}")

    for key, prop_spec in properties.items():
        if key not in raw_args and "default" in prop_spec:
            normalized_args[key] = prop_spec["default"]

    for key, val in raw_args.items():
        if key not in properties:
            if not allow_additional:
                raise ValueError(f"unexpected parameter {key!r} for {name!r}")
            normalized_args[key] = val
            continue
        prop = properties[key]
        ptype = prop.get("type", "any")
        if ptype in ("string", "boolean", "integer", "number", "float"):
            coerced = _normalize_scalar(
                val, ptype, allow_string_coercion=allow_string_coercion
            )
        elif ptype == "array":
            if not isinstance(val, list):
                raise TypeError(f"parameter {key!r} expected array")
            item_type = prop.get("items", {}).get("type")
            if item_type in ("string", "boolean", "integer", "number", "float"):
                coerced = [
                    _normalize_scalar(v, item_type, allow_string_coercion=allow_string_coercion)
                    for v in val
                ]
            else:
                coerced = list(val)
        elif ptype == "object":
            if not isinstance(val, dict):
                raise TypeError(f"parameter {key!r} expected object")
            coerced = dict(val)
        elif ptype == "any":
            coerced = val
        else:
            raise ValueError(f"unsupported property type {ptype!r}")

        if "enum" in prop and coerced not in prop["enum"]:
            raise ValueError(f"parameter {key!r}={coerced!r} not in enum {prop['enum']}")
        normalized_args[key] = coerced

    return {"name": name, "arguments": normalized_args}


def values_ast_equal(
    predicted,
    expected,
    *,
    rel_tol=1e-5,
    abs_tol=1e-6,
    unordered_lists=False,
    case_insensitive_strings=False,
):
    """Recursively match two JSON-like AST values with numeric and structural rules."""
    if isinstance(expected, bool) or isinstance(predicted, bool):
        return type(predicted) is bool and type(expected) is bool and predicted == expected

    if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
        if not (math.isfinite(predicted) and math.isfinite(expected)):
            return False
        return math.isclose(float(predicted), float(expected), rel_tol=rel_tol, abs_tol=abs_tol)

    if isinstance(expected, str) and isinstance(predicted, str):
        p_norm = " ".join(predicted.strip().split())
        e_norm = " ".join(expected.strip().split())
        if case_insensitive_strings:
            return p_norm.lower() == e_norm.lower()
        return p_norm == e_norm

    if predicted is None or expected is None:
        return predicted is None and expected is None

    if isinstance(expected, dict) and isinstance(predicted, dict):
        if set(predicted.keys()) != set(expected.keys()):
            return False
        return all(
            values_ast_equal(
                predicted[k],
                expected[k],
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                unordered_lists=unordered_lists,
                case_insensitive_strings=case_insensitive_strings,
            )
            for k in expected
        )

    if isinstance(expected, list) and isinstance(predicted, list):
        if len(predicted) != len(expected):
            return False
        if not unordered_lists:
            return all(
                values_ast_equal(
                    p_item,
                    e_item,
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                    unordered_lists=unordered_lists,
                    case_insensitive_strings=case_insensitive_strings,
                )
                for p_item, e_item in zip(predicted, expected)
            )
        return _bipartite_match_count(
            predicted,
            expected,
            lambda a, b: values_ast_equal(
                a,
                b,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                unordered_lists=True,
                case_insensitive_strings=case_insensitive_strings,
            ),
        ) == len(expected)

    return False


def _bipartite_match_count(left_items, right_items, predicate):
    """Maximum cardinality matching via Kuhn's augmenting-path algorithm."""
    matched_right = [-1] * len(right_items)

    def try_augment(left_idx, visited):
        for right_idx, right_val in enumerate(right_items):
            if visited[right_idx]:
                continue
            if not predicate(left_items[left_idx], right_val):
                continue
            visited[right_idx] = True
            if matched_right[right_idx] == -1 or try_augment(matched_right[right_idx], visited):
                matched_right[right_idx] = left_idx
                return True
        return False

    matches = 0
    for left_idx in range(len(left_items)):
        visited = [False] * len(right_items)
        if try_augment(left_idx, visited):
            matches += 1
    return matches


def calls_match(
    pred_call,
    gold_call,
    *,
    rel_tol=1e-5,
    abs_tol=1e-6,
    unordered_lists=False,
):
    if pred_call.get("name") != gold_call.get("name"):
        return False
    return values_ast_equal(
        pred_call.get("arguments"),
        gold_call.get("arguments"),
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        unordered_lists=unordered_lists,
    )


def evaluate_call_sequence(
    predicted_calls,
    expected_calls,
    *,
    order_sensitive=False,
    rel_tol=1e-5,
    abs_tol=1e-6,
    unordered_lists=False,
):
    """Score one example's predicted call list against expected reference calls."""
    preds = list(predicted_calls)
    golds = list(expected_calls)
    if order_sensitive:
        tp = sum(
            1
            for idx in range(min(len(preds), len(golds)))
            if calls_match(
                preds[idx],
                golds[idx],
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                unordered_lists=unordered_lists,
            )
        )
        exact = len(preds) == len(golds) and tp == len(golds)
    else:
        tp = _bipartite_match_count(
            preds,
            golds,
            lambda p, g: calls_match(
                p, g, rel_tol=rel_tol, abs_tol=abs_tol, unordered_lists=unordered_lists
            ),
        )
        exact = len(preds) == len(golds) and tp == len(golds)

    fp = len(preds) - tp
    fn = len(golds) - tp
    precision = tp / len(preds) if preds else (1.0 if not golds else 0.0)
    recall = tp / len(golds) if golds else (1.0 if not preds else 0.0)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "exact_match": exact,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_dataset(
    examples,
    *,
    tool_schemas=None,
    allow_string_coercion=False,
    order_sensitive=False,
    unordered_lists=False,
):
    """Aggregate extraction validity, schema validity, exact match and micro/macro F1."""
    rows = list(examples)
    if not rows:
        raise ValueError("examples must be non-empty")

    total_tp = total_fp = total_fn = 0
    exact_matches = 0
    parse_failures = 0
    schema_failures = 0
    macro_f1_terms = []
    source_counts = {"direct": 0, "fenced": 0, "inline_scan": 0, "empty": 0, "failed": 0, "invalid": 0}

    for index, entry in enumerate(rows):
        if not isinstance(entry, dict) or "expected" not in entry:
            raise ValueError(f"example {index} must be a dict with 'expected'")
        gold_calls = [
            _normalize_call_record(c) for c in entry["expected"]
        ]
        if tool_schemas is not None:
            gold_calls = [
                normalize_call_against_schema(
                    c, tool_schemas, allow_string_coercion=allow_string_coercion
                )
                for c in gold_calls
            ]

        if "raw_output" in entry:
            extracted = extract_tool_calls(entry["raw_output"])
            source_counts[extracted.source_kind] = source_counts.get(extracted.source_kind, 0) + 1
            if extracted.error is not None:
                parse_failures += 1
                pred_calls = []
            else:
                pred_calls = list(extracted.calls)
        elif "predicted" in entry:
            pred_calls = [_normalize_call_record(c) for c in entry["predicted"]]
            source_counts["direct"] += 1
        else:
            raise ValueError(f"example {index} requires 'raw_output' or 'predicted'")

        if pred_calls and tool_schemas is not None:
            validated = []
            try:
                for call in pred_calls:
                    validated.append(
                        normalize_call_against_schema(
                            call,
                            tool_schemas,
                            allow_string_coercion=allow_string_coercion,
                        )
                    )
                pred_calls = validated
            except (ValueError, TypeError):
                schema_failures += 1
                pred_calls = []

        scored = evaluate_call_sequence(
            pred_calls,
            gold_calls,
            order_sensitive=order_sensitive,
            unordered_lists=unordered_lists,
        )
        if scored["exact_match"]:
            exact_matches += 1
        total_tp += scored["true_positives"]
        total_fp += scored["false_positives"]
        total_fn += scored["false_negatives"]
        macro_f1_terms.append(scored["f1"])

    micro_prec = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else (1.0 if total_fn == 0 else 0.0)
    )
    micro_rec = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 1.0
    )
    micro_f1 = (
        2.0 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if (micro_prec + micro_rec) > 0
        else 0.0
    )
    count = len(rows)
    return {
        "examples": count,
        "exact_match_rate": exact_matches / count,
        "exact_matches": exact_matches,
        "parse_failures": parse_failures,
        "schema_failures": schema_failures,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f1": micro_f1,
        "macro_f1": math.fsum(macro_f1_terms) / count,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "extraction_sources": source_counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL file of {raw_output|predicted, expected}")
    parser.add_argument("--schemas", type=Path, help="Optional JSON tool schema dictionary")
    parser.add_argument("--order-sensitive", action="store_true")
    parser.add_argument("--unordered-lists", action="store_true")
    parser.add_argument("--coerce-strings", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        schemas = None
        if args.schemas is not None:
            schemas = json.loads(
                args.schemas.read_text(encoding="utf-8"),
                parse_constant=_reject_nonstandard_constant,
            )
        examples = []
        with args.input.open(encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, 1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                examples.append(
                    json.loads(stripped, parse_constant=_reject_nonstandard_constant)
                )
        summary = evaluate_dataset(
            examples,
            tool_schemas=schemas,
            allow_string_coercion=args.coerce_strings,
            order_sensitive=args.order_sensitive,
            unordered_lists=args.unordered_lists,
        )
        encoded = json.dumps(summary, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as out_file:
                out_file.write(encoded)
        else:
            print(encoded, end="")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"structured_call_eval error: {exc}\n")


if __name__ == "__main__":
    main()
