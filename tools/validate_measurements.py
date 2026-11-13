#!/usr/bin/env python3
"""Validate an ordered, complete CSV of repeated duration measurements."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def validate(rows, expected, index_column="iteration", value_column="duration_ms"):
    if expected <= 0:
        raise ValueError("expected sample count must be positive")
    rows = list(rows)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, received {len(rows)}")
    values = []
    for position, row in enumerate(rows):
        if None in row:
            raise ValueError(f"row {position + 2}: unexpected extra fields")
        try:
            index = int(row[index_column])
            value = float(row[value_column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"row {position + 2}: invalid required field") from exc
        if index != position:
            raise ValueError(f"row {position + 2}: expected iteration {position}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"row {position + 2}: duration must be finite and positive")
        values.append(value)
    return {
        "validated_rows": len(values),
        "first_iteration": 0,
        "last_iteration": expected - 1,
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--index-column", default="iteration")
    parser.add_argument("--value-column", default="duration_ms")
    args = parser.parse_args()
    try:
        with args.input.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source, strict=True)
            fields = reader.fieldnames or []
            if len(set(fields)) != len(fields):
                raise ValueError("duplicate header columns")
            if not all(name in fields for name in (args.index_column, args.value_column)):
                raise ValueError("required columns are missing")
            result = validate(reader, args.expected, args.index_column, args.value_column)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"validation error: {exc}\n")


if __name__ == "__main__":
    main()
