#!/usr/bin/env python3
"""Summarize a numeric CSV column, optionally grouped by categorical columns."""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


METRICS = ("count", "minimum", "median", "maximum")


def summarize(rows, value_column, group_columns=()):
    """Reject invalid rows rather than silently changing the evaluated sample."""
    if len(set(group_columns)) != len(group_columns):
        raise ValueError("group columns must be unique")
    if any(name in METRICS for name in group_columns):
        raise ValueError("group column names must not collide with output metrics")
    grouped = defaultdict(list)
    for row_number, row in enumerate(rows, 2):
        if None in row:
            raise ValueError(f"row {row_number}: more values than header columns")
        required = (value_column, *group_columns)
        if any(name not in row or row[name] in (None, "") for name in required):
            raise ValueError(f"row {row_number}: missing required value")
        try:
            value = float(row[value_column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {row_number}: expected a numeric value") from exc
        if not math.isfinite(value):
            raise ValueError(f"row {row_number}: non-finite value")
        grouped[tuple(row[name] for name in group_columns)].append(value)
    if not grouped:
        raise ValueError("input contains no measurement rows")
    result = []
    for key, values in sorted(grouped.items()):
        result.append({
            **dict(zip(group_columns, key)),
            "count": len(values),
            "minimum": min(values),
            "median": statistics.median(values),
            "maximum": max(values),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--column", required=True, help="numeric input column")
    parser.add_argument("--group-by", nargs="+", default=[], metavar="COLUMN")
    parser.add_argument("--output", type=Path, help="new output CSV; default: stdout")
    args = parser.parse_args()
    try:
        with args.input.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source, strict=True)
            header = reader.fieldnames or []
            if len(set(header)) != len(header):
                raise ValueError("duplicate column names")
            if not all(name in header for name in (args.column, *args.group_by)):
                raise ValueError("required columns are absent from the header")
            result = summarize(reader, args.column, args.group_by)
        fields = [*args.group_by, *METRICS]
        if args.output:
            with args.output.open("x", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                writer.writerows(result)
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"summary error: {exc}\n")


if __name__ == "__main__":
    main()
