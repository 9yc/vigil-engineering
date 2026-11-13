#!/usr/bin/env python3
"""Estimate energy from time_s/power_w CSV samples using midpoint-held values."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def integrate(samples, start=None, end=None, max_gap=None):
    samples = list(samples)
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    if any(not math.isfinite(t) or not math.isfinite(p) or p < 0
           for t, p in samples):
        raise ValueError("timestamps must be finite and power finite and nonnegative")
    gaps = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
    if any(not math.isfinite(gap) or gap <= 0 for gap in gaps):
        raise ValueError("timestamps must be strictly increasing with finite gaps")
    if max_gap is not None:
        if not math.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("max_gap must be finite and positive")
        if max(gaps) > max_gap:
            raise ValueError("sample gap exceeds max_gap")
    start = samples[0][0] if start is None else start
    end = samples[-1][0] if end is None else end
    if not math.isfinite(start) or not math.isfinite(end) or start >= end:
        raise ValueError("window must have finite, increasing bounds")
    if start < samples[0][0] or end > samples[-1][0]:
        raise ValueError("samples do not cover the complete requested window")
    # Adjacent sample midpoints divide held-value intervals. Do not extrapolate
    # beyond the first or last measured timestamp.
    boundaries = [samples[0][0]]
    boundaries.extend(a[0] + (b[0] - a[0]) / 2
                      for a, b in zip(samples, samples[1:]))
    boundaries.append(samples[-1][0])
    energy = math.fsum(
        max(0.0, min(end, right) - max(start, left)) * power
        for left, right, (_, power) in zip(boundaries, boundaries[1:], samples)
    )
    duration = end - start
    if not math.isfinite(energy) or not math.isfinite(duration):
        raise ValueError("numeric range exceeded")
    return {
        "samples": len(samples),
        "start_s": start,
        "end_s": end,
        "duration_s": duration,
        "energy_j": energy,
        "mean_power_w": energy / duration,
        "median_sample_gap_s": statistics.median(gaps),
        "maximum_sample_gap_s": max(gaps),
        "method": "midpoint_hold_without_endpoint_extrapolation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--max-gap", type=float)
    args = parser.parse_args()
    try:
        with args.input.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source, strict=True)
            if reader.fieldnames != ["time_s", "power_w"]:
                raise ValueError("expected exactly time_s,power_w columns, in that order")
            samples = []
            for row in reader:
                if None in row:
                    raise ValueError("extra fields in input row")
                samples.append((float(row["time_s"]), float(row["power_w"])))
        result = integrate(samples, args.start, args.end, args.max_gap)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, OverflowError, csv.Error) as exc:
        parser.exit(2, f"integration error: {exc}\n")


if __name__ == "__main__":
    main()
