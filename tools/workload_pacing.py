#!/usr/bin/env python3
"""Open-loop arrival schedule generation, trace rescaling and queueing metrics.

Generates deterministic synthetic arrival schedules (Poisson, uniform, Pareto
and two-state Markov-modulated bursty processes), rescales empirical arrival
traces in rate or burstiness, computes dispersion statistics (CV, Fano factor),
evaluates exact sweep-line concurrency watermarks, verifies Little's Law
(`L = lambda * W`), and summarizes multi-metric latency SLO attainment.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import random


def _finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _finite_nonneg(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def generate_arrival_schedule(
    count,
    rate_per_s,
    *,
    mode="poisson",
    seed=0,
    burst_ratio=5.0,
    high_prob=0.2,
    switch_prob=0.25,
    pareto_alpha=2.5,
):
    """Generate `count` monotonic arrival timestamps starting at or after `0.0`."""
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    rate = _finite_positive(rate_per_s, "rate_per_s")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    rng = random.Random(seed)
    mean_gap = 1.0 / rate
    gaps = []

    if mode == "uniform":
        gaps = [mean_gap] * count
    elif mode == "poisson":
        for _ in range(count):
            u = 1.0 - rng.random()
            gaps.append(-math.log(u) * mean_gap)
    elif mode == "mmpp2":
        ratio = _finite_positive(burst_ratio, "burst_ratio")
        if ratio < 1.0:
            raise ValueError("burst_ratio must be >= 1.0")
        p_hi = _finite_positive(high_prob, "high_prob")
        p_sw = _finite_positive(switch_prob, "switch_prob")
        if not (0.0 < p_hi < 1.0) or not (0.0 < p_sw <= 1.0):
            raise ValueError("high_prob must be in (0, 1) and switch_prob in (0, 1]")
        # Interpret high_prob as stationary time fraction in burst state:
        # Mean rate = p_hi * (ratio * lambda_lo) + (1 - p_hi) * lambda_lo == rate.
        # Under Palm sampling at arrival epochs, fraction of arrivals in burst is
        # event_hi = (p_hi * ratio) / (p_hi * ratio + (1 - p_hi)).
        lambda_lo = rate / (p_hi * ratio + (1.0 - p_hi))
        low_gap = 1.0 / lambda_lo
        high_gap = low_gap / ratio
        event_hi = (p_hi * ratio) / (p_hi * ratio + (1.0 - p_hi))
        in_burst = rng.random() < event_hi
        p_low_to_high = p_sw * event_hi
        p_high_to_low = p_sw * (1.0 - event_hi)
        for _ in range(count):
            scale = high_gap if in_burst else low_gap
            u = 1.0 - rng.random()
            gaps.append(-math.log(u) * scale)
            if in_burst:
                if rng.random() < p_high_to_low:
                    in_burst = False
            else:
                if rng.random() < p_low_to_high:
                    in_burst = True
    elif mode == "pareto":
        alpha = _finite_positive(pareto_alpha, "pareto_alpha")
        if alpha <= 1.0:
            raise ValueError("pareto_alpha must be > 1.0 for finite mean")
        # Lomax (Pareto Type II) on [0, inf) with mean == mean_gap: scale = mean_gap * (alpha - 1)
        scale = mean_gap * (alpha - 1.0)
        for _ in range(count):
            u = 1.0 - rng.random()
            gaps.append(scale * ((u ** (-1.0 / alpha)) - 1.0))
    else:
        raise ValueError(f"unsupported arrival mode: {mode!r}")

    arrivals = []
    cursor = 0.0
    for idx, gap in enumerate(gaps):
        if idx == 0 and mode == "uniform":
            cursor = 0.0
        else:
            cursor += gap
        arrivals.append(cursor)
    return arrivals


def rescale_arrival_trace(arrivals, *, rate_scale=1.0, burst_exponent=1.0):
    """Rescale inter-arrival gaps by `gap_i' = c * gap_i ** burst_exponent` matching `rate_scale`."""
    times = [_finite_nonneg(t, "arrival") for t in arrivals]
    if len(times) < 2:
        raise ValueError("at least two arrivals are required to rescale")
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("arrivals must be monotonically nondecreasing")
    r_scale = _finite_positive(rate_scale, "rate_scale")
    b_exp = _finite_nonneg(burst_exponent, "burst_exponent")

    orig_span = times[-1] - times[0]
    if orig_span <= 0.0:
        raise ValueError("arrival trace span must be positive")
    gaps = [b - a for a, b in zip(times, times[1:])]
    transformed = [(g ** b_exp) if g > 0.0 else (1.0 if b_exp == 0.0 else 0.0) for g in gaps]
    raw_sum = math.fsum(transformed)
    if raw_sum <= 0.0:
        raise ValueError("transformed inter-arrival sum is zero")

    target_span = orig_span / r_scale
    norm = target_span / raw_sum
    out = [times[0]]
    acc = times[0]
    for step in transformed:
        acc += step * norm
        out.append(acc)
    return out


def analyze_arrival_dispersion(arrivals, *, window_s=1.0):
    """Compute empirical rate, inter-arrival CV, lag-1 autocorrelation and Fano factor."""
    times = [_finite_nonneg(t, "arrival") for t in arrivals]
    if len(times) < 2:
        raise ValueError("at least two arrivals are required")
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("arrivals must be sorted")
    win = _finite_positive(window_s, "window_s")

    span = times[-1] - times[0]
    if span <= 0.0:
        raise ValueError("arrival trace span must be positive")
    gaps = [b - a for a, b in zip(times, times[1:])]
    mean_gap = math.fsum(gaps) / len(gaps)
    var_gap = math.fsum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    std_gap = math.sqrt(var_gap)
    cv = std_gap / mean_gap if mean_gap > 0.0 else 0.0

    if len(gaps) >= 2 and var_gap > 0.0:
        cov1 = math.fsum(
            (gaps[i] - mean_gap) * (gaps[i + 1] - mean_gap)
            for i in range(len(gaps) - 1)
        ) / (len(gaps) - 1)
        lag1_autocorr = cov1 / var_gap
    else:
        lag1_autocorr = 0.0

    num_bins = max(1, int(math.ceil(span / win)))
    bin_counts = [0] * num_bins
    t0 = times[0]
    for t in times:
        idx = min(num_bins - 1, int((t - t0) / win))
        bin_counts[idx] += 1
    mean_cnt = math.fsum(bin_counts) / num_bins
    var_cnt = math.fsum((c - mean_cnt) ** 2 for c in bin_counts) / num_bins
    fano = var_cnt / mean_cnt if mean_cnt > 0.0 else 0.0

    return {
        "count": len(times),
        "span_s": span,
        "empirical_rate_per_s": (len(times) - 1) / span,
        "mean_gap_s": mean_gap,
        "std_gap_s": std_gap,
        "cv": cv,
        "lag1_autocorr": lag1_autocorr,
        "fano_factor": fano,
        "peak_window_arrivals": max(bin_counts),
    }


def analyze_concurrency_and_littles_law(intervals):
    """Sweep `(start_s, end_s)` intervals for exact peak/time-average concurrency & Little's Law."""
    cleaned = []
    for index, pair in enumerate(intervals):
        if len(pair) != 2:
            raise ValueError(f"interval {index} must be a (start_s, end_s) pair")
        s = _finite_nonneg(pair[0], "start_s")
        e = _finite_nonneg(pair[1], "end_s")
        if e < s:
            raise ValueError(f"interval {index} has end_s < start_s")
        cleaned.append((s, e))
    if not cleaned:
        raise ValueError("intervals must be non-empty")

    # Sort ties so departures (-1) at timestamp t happen before arrivals (+1) at timestamp t
    events = []
    durations = []
    for s, e in cleaned:
        durations.append(e - s)
        events.append((s, 1))
        events.append((e, -1))
    events.sort(key=lambda item: (item[0], item[1]))

    horizon_start = min(s for s, _ in cleaned)
    horizon_end = max(e for _, e in cleaned)
    horizon_s = horizon_end - horizon_start
    if horizon_s <= 0.0:
        raise ValueError("total observation horizon must be positive")

    active = 0
    peak_active = 0
    area_s = 0.0
    busy_time_s = 0.0
    prev_t = events[0][0]
    for t, delta in events:
        dt = t - prev_t
        if dt > 0.0:
            area_s += active * dt
            if active > 0:
                busy_time_s += dt
            prev_t = t
        active += delta
        if active > peak_active:
            peak_active = active

    mean_concurrency = area_s / horizon_s
    throughput_per_s = len(cleaned) / horizon_s
    mean_residence_s = math.fsum(durations) / len(cleaned)
    littles_product = throughput_per_s * mean_residence_s

    return {
        "requests": len(cleaned),
        "horizon_s": horizon_s,
        "peak_concurrency": peak_active,
        "busy_fraction": busy_time_s / horizon_s,
        "time_avg_concurrency": mean_concurrency,
        "throughput_per_s": throughput_per_s,
        "mean_residence_s": mean_residence_s,
        "littles_law_concurrency": littles_product,
        "littles_law_residual": abs(mean_concurrency - littles_product),
    }


def evaluate_slo_attainment(records, slo_targets):
    """Compute per-metric and joint goodput SLO attainment across request dicts."""
    rows = list(records)
    if not rows:
        raise ValueError("records must be non-empty")
    if not isinstance(slo_targets, dict) or not slo_targets:
        raise ValueError("slo_targets must be a non-empty dict of metric -> ceiling")
    ceilings = {
        str(k): _finite_positive(v, f"SLO ceiling {k}") for k, v in slo_targets.items()
    }

    per_metric_pass = {k: 0 for k in ceilings}
    joint_pass = 0
    status_ok = 0

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"record {idx} must be a dict")
        ok = row.get("status", "ok") == "ok"
        if ok:
            status_ok += 1
        all_met = ok
        for metric, limit in ceilings.items():
            if metric not in row or row[metric] is None:
                all_met = False
                continue
            val = _finite_nonneg(row[metric], f"record[{idx}].{metric}")
            if ok and val <= limit:
                per_metric_pass[metric] += 1
            else:
                all_met = False
        if all_met:
            joint_pass += 1

    total = len(rows)
    return {
        "total_requests": total,
        "status_ok_requests": status_ok,
        "joint_goodput_requests": joint_pass,
        "joint_attainment": joint_pass / total,
        "per_metric_attainment": {k: per_metric_pass[k] / total for k in ceilings},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--count", type=int, required=True)
    gen.add_argument("--rate", type=float, required=True)
    gen.add_argument("--mode", choices=("poisson", "uniform", "mmpp2", "pareto"), default="poisson")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--output", type=Path)

    ana = sub.add_parser("analyze")
    ana.add_argument("input", type=Path, help="CSV with arrival_s,end_s[,ttft_ms,e2e_ms]")
    ana.add_argument("--ttft-slo-ms", type=float)
    ana.add_argument("--e2e-slo-ms", type=float)
    ana.add_argument("--output", type=Path)

    args = parser.parse_args()
    try:
        if args.cmd == "generate":
            arrivals = generate_arrival_schedule(
                args.count, args.rate, mode=args.mode, seed=args.seed
            )
            payload = {
                "mode": args.mode,
                "arrivals_s": arrivals,
                "dispersion": analyze_arrival_dispersion(arrivals) if len(arrivals) >= 2 else None,
            }
        else:
            with args.input.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, strict=True))
            if not rows:
                raise ValueError("empty input CSV")
            arrivals = [float(r["arrival_s"]) for r in rows]
            intervals = [(float(r["arrival_s"]), float(r["end_s"])) for r in rows]
            payload = {
                "dispersion": analyze_arrival_dispersion(arrivals) if len(arrivals) >= 2 else None,
                "queueing": analyze_concurrency_and_littles_law(intervals),
            }
            targets = {}
            if args.ttft_slo_ms is not None:
                targets["ttft_ms"] = args.ttft_slo_ms
            if args.e2e_slo_ms is not None:
                targets["e2e_ms"] = args.e2e_slo_ms
            if targets:
                typed_rows = [
                    {k: (float(v) if k.endswith("_ms") and v != "" else v) for k, v in r.items()}
                    for r in rows
                ]
                payload["slo"] = evaluate_slo_attainment(typed_rows, targets)

        encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as f:
                f.write(encoded)
        else:
            print(encoded, end="")
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        parser.exit(2, f"workload_pacing error: {exc}\n")


if __name__ == "__main__":
    main()
