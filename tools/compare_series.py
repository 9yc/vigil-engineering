#!/usr/bin/env python3
"""Pure-Python nonparametric benchmark series comparator library and CLI."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


class SeriesComparisonError(ValueError):
    """Raised when series inputs, CLI options, or statistical constraints are invalid."""


_MAD_NORMAL_FACTOR = 0.6744897501960817


def _validate_numeric_series(values: Sequence[float], label: str) -> List[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise SeriesComparisonError(f"{label} must be a sequence of numbers")
    if not values:
        raise SeriesComparisonError(f"{label} must be non-empty")
    cleaned: List[float] = []
    for idx, raw in enumerate(values):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise SeriesComparisonError(f"{label}[{idx}] must be int or float, got {type(raw).__name__}")
        fval = float(raw)
        if not math.isfinite(fval):
            raise SeriesComparisonError(f"{label}[{idx}] is non-finite: {raw!r}")
        cleaned.append(0.0 if fval == 0.0 else fval)
    return cleaned


def quantile_sorted(sorted_vals: Sequence[float], q: float) -> float:
    """Compute exact Type-7 linear-interpolated quantile on pre-sorted values."""
    if not sorted_vals:
        raise SeriesComparisonError("Cannot compute quantile of empty sequence")
    if not math.isfinite(q) or not (0.0 <= q <= 1.0):
        raise SeriesComparisonError(f"Quantile probability q must be in [0, 1], got {q!r}")
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo]))


def quantile(values: Sequence[float], q: float) -> float:
    """Compute exact Type-7 quantile for unsorted finite numeric `values`."""
    cleaned = _validate_numeric_series(values, "values")
    cleaned.sort()
    return quantile_sorted(cleaned, q)


def detect_outliers(
    values: Sequence[float], *, iqr_multiplier: float = 1.5, mad_z_threshold: float = 3.5
) -> Dict[str, Any]:
    """Flag outliers using both Tukey IQR fences and Iglewicz-Hoaglin MAD Z-scores."""
    cleaned = _validate_numeric_series(values, "values")
    if not math.isfinite(iqr_multiplier) or iqr_multiplier <= 0.0:
        raise SeriesComparisonError("iqr_multiplier must be finite and > 0")
    if not math.isfinite(mad_z_threshold) or mad_z_threshold <= 0.0:
        raise SeriesComparisonError("mad_z_threshold must be finite and > 0")

    s = sorted(cleaned)
    q1, med, q3 = quantile_sorted(s, 0.25), quantile_sorted(s, 0.50), quantile_sorted(s, 0.75)
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
    tukey_indices = [i for i, x in enumerate(cleaned) if x < lower_fence or x > upper_fence]

    abs_devs = sorted(abs(x - med) for x in cleaned)
    mad = quantile_sorted(abs_devs, 0.50)
    mad_indices: List[int] = []
    max_modified_z = 0.0

    if mad > 0.0:
        for idx, x in enumerate(cleaned):
            mz = _MAD_NORMAL_FACTOR * abs(x - med) / mad
            max_modified_z = max(max_modified_z, mz)
            if mz > mad_z_threshold:
                mad_indices.append(idx)
    else:
        denom = 1.2533141373155001 * (sum(abs_devs) / float(len(abs_devs)))
        if denom > 0.0:
            for idx, x in enumerate(cleaned):
                mz = abs(x - med) / denom
                max_modified_z = max(max_modified_z, mz)
                if mz > mad_z_threshold:
                    mad_indices.append(idx)

    return {
        "iqr": iqr,
        "mad": mad,
        "mad_normalized_sigma": mad / _MAD_NORMAL_FACTOR if mad > 0.0 else 0.0,
        "mad_outlier_count": len(mad_indices),
        "mad_outlier_indices": mad_indices,
        "mad_outlier_values": [cleaned[i] for i in mad_indices],
        "mad_z_threshold": mad_z_threshold,
        "max_modified_z": max_modified_z,
        "tukey_lower_fence": lower_fence,
        "tukey_outlier_count": len(tukey_indices),
        "tukey_outlier_indices": tukey_indices,
        "tukey_outlier_values": [cleaned[i] for i in tukey_indices],
        "tukey_upper_fence": upper_fence,
    }


def describe_series(values: Sequence[float], label: str = "series") -> Dict[str, Any]:
    """Compute parametric, quantile, dispersion, and outlier summary statistics."""
    cleaned = _validate_numeric_series(values, label)
    s = sorted(cleaned)
    n = len(s)
    mean_val = math.fsum(s) / float(n)
    stdev_val = statistics.stdev(s) if n >= 2 else 0.0
    geom_mean = math.exp(math.fsum(math.log(x) for x in s) / float(n)) if all(x > 0.0 for x in s) else None
    q25, q50, q75 = quantile_sorted(s, 0.25), quantile_sorted(s, 0.50), quantile_sorted(s, 0.75)
    outliers = detect_outliers(cleaned)
    return {
        "count": n, "geometric_mean": geom_mean, "iqr": q75 - q25, "mad": outliers["mad"],
        "max": s[-1], "mean": mean_val, "median": q50, "min": s[0], "outliers": outliers,
        "p05": quantile_sorted(s, 0.05), "p25": q25, "p50": q50, "p75": q75,
        "p95": quantile_sorted(s, 0.95), "p99": quantile_sorted(s, 0.99), "stdev": stdev_val,
    }


def hodges_lehmann_shift(
    baseline: Sequence[float], candidate: Sequence[float], *, paired: bool = False
) -> float:
    """Compute exact Hodges-Lehmann location shift estimator (`candidate - baseline`)."""
    base = _validate_numeric_series(baseline, "baseline")
    cand = _validate_numeric_series(candidate, "candidate")
    if paired:
        if len(base) != len(cand):
            raise SeriesComparisonError(f"Paired Hodges-Lehmann requires equal lengths ({len(base)} vs {len(cand)})")
        diffs = [c - b for b, c in zip(base, cand)]
        walsh = sorted(0.5 * (diffs[i] + diffs[j]) for i in range(len(diffs)) for j in range(i, len(diffs)))
        return quantile_sorted(walsh, 0.50)
    pairwise = sorted(c - b for b in base for c in cand)
    return quantile_sorted(pairwise, 0.50)


def _midranks_with_ties(values: Sequence[float]) -> Tuple[List[float], List[int]]:
    """Assign 1-based average ranks to `values` and collect tie-group sizes `> 1`."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * n
    tie_sizes: List[int] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        if j - i > 1:
            tie_sizes.append(j - i)
        i = j
    return ranks, tie_sizes


def wilcoxon_signed_rank(baseline: Sequence[float], candidate: Sequence[float]) -> Dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test (`candidate - baseline`) with tie correction."""
    base = _validate_numeric_series(baseline, "baseline")
    cand = _validate_numeric_series(candidate, "candidate")
    if len(base) != len(cand):
        raise SeriesComparisonError(f"Wilcoxon signed-rank requires equal lengths ({len(base)} vs {len(cand)})")

    raw_diffs = [c - b for b, c in zip(base, cand)]
    nonzero = [d for d in raw_diffs if d != 0.0]
    n_zero = len(raw_diffs) - len(nonzero)
    nr = len(nonzero)
    if nr == 0:
        return {
            "n_effective": 0, "n_pairs": len(base), "n_zero_diffs": n_zero,
            "p_value": 1.0, "statistic": 0.0, "w_minus": 0.0, "w_plus": 0.0, "z_score": 0.0,
        }

    ranks, tie_sizes = _midranks_with_ties([abs(d) for d in nonzero])
    w_plus = math.fsum(r for d, r in zip(nonzero, ranks) if d > 0.0)
    w_minus = math.fsum(r for d, r in zip(nonzero, ranks) if d < 0.0)
    stat = min(w_plus, w_minus)

    mean_w = nr * (nr + 1) / 4.0
    tie_corr = sum(t * (t * t - 1) for t in tie_sizes) / 48.0
    variance = (nr * (nr + 1) * (2 * nr + 1)) / 24.0 - tie_corr
    if variance <= 0.0:
        z_signed, p_val = 0.0, 1.0
    else:
        sigma = math.sqrt(variance)
        diff = w_plus - mean_w
        z_signed = diff / sigma
        z_cc = max(0.0, abs(diff) - 0.5) / sigma
        p_val = min(1.0, max(0.0, math.erfc(z_cc / math.sqrt(2.0))))

    return {
        "n_effective": nr, "n_pairs": len(base), "n_zero_diffs": n_zero,
        "p_value": p_val, "statistic": stat, "w_minus": w_minus, "w_plus": w_plus, "z_score": z_signed,
    }


def mann_whitney_u(baseline: Sequence[float], candidate: Sequence[float]) -> Dict[str, Any]:
    """Two-sided Mann-Whitney U test with midrank tie correction and normal approximation."""
    base = _validate_numeric_series(baseline, "baseline")
    cand = _validate_numeric_series(candidate, "candidate")
    n1, n2 = len(base), len(cand)
    total_n = n1 + n2

    ranks, tie_sizes = _midranks_with_ties(list(base) + list(cand))
    u_baseline = math.fsum(ranks[:n1]) - (n1 * (n1 + 1)) / 2.0
    u_candidate = (n1 * n2) - u_baseline
    u_stat = min(u_baseline, u_candidate)

    mean_u = 0.5 * n1 * n2
    tie_sum = sum(t * (t * t - 1) for t in tie_sizes)
    variance = (
        (n1 * n2 / 12.0) * ((total_n + 1.0) - tie_sum / float(total_n * (total_n - 1)))
        if total_n > 1 else 0.0
    )
    if variance <= 0.0:
        z_signed, p_val = 0.0, 1.0
    else:
        sigma = math.sqrt(variance)
        diff = u_candidate - mean_u
        z_signed = diff / sigma
        z_cc = max(0.0, abs(diff) - 0.5) / sigma
        p_val = min(1.0, max(0.0, math.erfc(z_cc / math.sqrt(2.0))))

    return {
        "n_baseline": n1, "n_candidate": n2, "p_value": p_val, "statistic": u_stat,
        "u_baseline": u_baseline, "u_candidate": u_candidate, "z_score": z_signed,
    }


def kolmogorov_smirnov_2sample(baseline: Sequence[float], candidate: Sequence[float]) -> Dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov test (`D` supremum + Stephens asymptotic p-value)."""
    base = sorted(_validate_numeric_series(baseline, "baseline"))
    cand = sorted(_validate_numeric_series(candidate, "candidate"))
    n1, n2 = len(base), len(cand)
    i = j = 0
    d_plus = d_minus = 0.0

    while i < n1 or j < n2:
        if i < n1 and j < n2:
            pivot = base[i] if base[i] <= cand[j] else cand[j]
        else:
            pivot = base[i] if i < n1 else cand[j]
        while i < n1 and base[i] <= pivot:
            i += 1
        while j < n2 and cand[j] <= pivot:
            j += 1
        cdf1, cdf2 = i / float(n1), j / float(n2)
        d_plus = max(d_plus, cdf1 - cdf2)
        d_minus = max(d_minus, cdf2 - cdf1)

    d_stat = max(d_plus, d_minus)
    if d_stat <= 0.0:
        p_val = 1.0
    else:
        root_neff = math.sqrt((n1 * n2) / float(n1 + n2))
        lam = (root_neff + 0.12 + 0.11 / root_neff) * d_stat
        acc = 0.0
        for k in range(1, 101):
            exponent = -2.0 * (k * k) * (lam * lam)
            if exponent < -700.0:
                break
            term = math.exp(exponent)
            acc += term if (k % 2 == 1) else -term
            if term < 1e-15:
                break
        p_val = min(1.0, max(0.0, 2.0 * acc))

    return {
        "d_minus": d_minus, "d_plus": d_plus, "d_statistic": d_stat,
        "n_baseline": n1, "n_candidate": n2, "p_value": p_val,
    }


def cliffs_delta(baseline: Sequence[float], candidate: Sequence[float]) -> Dict[str, Any]:
    """Compute Cliff's delta ordinal effect size (`P(cand > base) - P(cand < base)`)."""
    base = sorted(_validate_numeric_series(baseline, "baseline"))
    cand_sorted = sorted(_validate_numeric_series(candidate, "candidate"))
    n1, n2 = len(base), len(cand_sorted)
    greater = less = tied = lo = hi = 0

    for c_val in cand_sorted:
        while lo < n1 and base[lo] < c_val:
            lo += 1
        while hi < n1 and base[hi] <= c_val:
            hi += 1
        greater += lo
        tied += hi - lo
        less += n1 - hi

    total_pairs = n1 * n2
    delta = (greater - less) / float(total_pairs)
    abs_d = abs(delta)
    if abs_d < 0.147:
        magnitude = "negligible"
    elif abs_d < 0.33:
        magnitude = "small"
    elif abs_d < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "delta": delta, "magnitude": magnitude, "pairs_candidate_greater": greater,
        "pairs_candidate_less": less, "pairs_tied": tied, "total_pairs": total_pairs,
    }


@dataclasses.dataclass(frozen=True)
class BootstrapEstimate:
    """Point estimate and deterministic percentile bootstrap confidence interval."""

    estimate: float
    ci_lower: float
    ci_upper: float
    confidence: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "ci_lower": self.ci_lower, "ci_upper": self.ci_upper,
            "confidence": self.confidence, "estimate": self.estimate,
        }


def _percentile_interval(point: float, replicates: List[float], confidence: float) -> BootstrapEstimate:
    replicates.sort()
    alpha = 1.0 - confidence
    return BootstrapEstimate(
        point, quantile_sorted(replicates, 0.5 * alpha),
        quantile_sorted(replicates, 1.0 - 0.5 * alpha), confidence,
    )


def bootstrap_comparison(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    paired: bool = False,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute seeded paired or unpaired bootstrap CIs for mean/median diffs and speedup."""
    base = _validate_numeric_series(baseline, "baseline")
    cand = _validate_numeric_series(candidate, "candidate")
    if not math.isfinite(confidence) or not (0.0 < confidence < 1.0):
        raise SeriesComparisonError("confidence must be strictly between 0.0 and 1.0")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise SeriesComparisonError("resamples must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SeriesComparisonError("seed must be an integer")
    if paired and len(base) != len(cand):
        raise SeriesComparisonError(f"Paired bootstrap requires equal lengths ({len(base)} vs {len(cand)})")

    n1, n2 = len(base), len(cand)
    all_positive = all(x > 0.0 for x in base) and all(y > 0.0 for y in cand)
    s_base, s_cand = sorted(base), sorted(cand)

    point_mean_diff = (math.fsum(cand) / n2) - (math.fsum(base) / n1)
    point_med_diff = quantile_sorted(s_cand, 0.50) - quantile_sorted(s_base, 0.50)
    point_mean_sp = ((math.fsum(base) / n1) / (math.fsum(cand) / n2)) if all_positive else None
    point_med_sp = (quantile_sorted(s_base, 0.50) / quantile_sorted(s_cand, 0.50)) if all_positive else None

    rng = random.Random(seed)
    rep_mean_diff, rep_med_diff, rep_mean_sp, rep_med_sp = [], [], [], []
    for _ in range(resamples):
        if paired:
            idxs = [rng.randrange(n1) for _ in range(n1)]
            b_samp, c_samp = [base[k] for k in idxs], [cand[k] for k in idxs]
        else:
            b_samp = [base[rng.randrange(n1)] for _ in range(n1)]
            c_samp = [cand[rng.randrange(n2)] for _ in range(n2)]
        b_mean, c_mean = math.fsum(b_samp) / n1, math.fsum(c_samp) / n2
        b_samp.sort()
        c_samp.sort()
        b_med, c_med = quantile_sorted(b_samp, 0.50), quantile_sorted(c_samp, 0.50)
        rep_mean_diff.append(c_mean - b_mean)
        rep_med_diff.append(c_med - b_med)
        if all_positive:
            rep_mean_sp.append(b_mean / c_mean)
            rep_med_sp.append(b_med / c_mean)

    return {
        "confidence": confidence,
        "mean_diff": _percentile_interval(point_mean_diff, rep_mean_diff, confidence).to_dict(),
        "median_diff": _percentile_interval(point_med_diff, rep_med_diff, confidence).to_dict(),
        "mean_speedup_ratio": (
            _percentile_interval(point_mean_sp, rep_mean_sp, confidence).to_dict()
            if point_mean_sp is not None else None
        ),
        "median_speedup_ratio": (
            _percentile_interval(point_med_sp, rep_med_sp, confidence).to_dict()
            if point_med_sp is not None else None
        ),
        "paired": paired,
        "resamples": resamples,
        "seed": seed,
    }


def compare_series(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    paired: bool = False,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run full statistical comparison suite between `baseline` and `candidate`."""
    base = _validate_numeric_series(baseline, "baseline")
    cand = _validate_numeric_series(candidate, "candidate")
    if paired and len(base) != len(cand):
        raise SeriesComparisonError(f"Paired comparison requires equal lengths ({len(base)} vs {len(cand)})")
    return {
        "baseline": describe_series(base, "baseline"),
        "bootstrap": bootstrap_comparison(
            base, cand, paired=paired, confidence=confidence, resamples=resamples, seed=seed
        ),
        "candidate": describe_series(cand, "candidate"),
        "cliffs_delta": cliffs_delta(base, cand),
        "hodges_lehmann_shift": hodges_lehmann_shift(base, cand, paired=paired),
        "kolmogorov_smirnov": kolmogorov_smirnov_2sample(base, cand),
        "mann_whitney_u": mann_whitney_u(base, cand),
        "paired": paired,
        "wilcoxon_signed_rank": wilcoxon_signed_rank(base, cand) if paired else None,
    }


def _parse_finite_float(raw: Any, context: str) -> float:
    if isinstance(raw, bool):
        raise SeriesComparisonError(f"Boolean is not a valid numeric measurement at {context}")
    try:
        val = float(raw)
    except (TypeError, ValueError) as exc:
        raise SeriesComparisonError(f"Non-numeric value {raw!r} at {context}") from exc
    if not math.isfinite(val):
        raise SeriesComparisonError(f"Non-finite value {raw!r} at {context}")
    return 0.0 if val == 0.0 else val


def load_series_file(
    file_path: Union[str, Path], *, column: Optional[str] = None, group_by: Optional[str] = None
) -> Dict[str, List[float]]:
    """Load numeric measurements from CSV or JSON, keyed by group (`__all__` if ungrouped)."""
    path = Path(file_path)
    if not path.is_file():
        raise SeriesComparisonError(f"Input file does not exist: {path}")
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".json":
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SeriesComparisonError(f"Invalid JSON in {path}: {exc}") from exc

        if isinstance(doc, list):
            if not doc:
                raise SeriesComparisonError(f"Empty JSON array in {path}")
            if isinstance(doc[0], Mapping):
                if column is None:
                    raise SeriesComparisonError(f"--column required for JSON object array {path}")
                grouped: Dict[str, List[float]] = {}
                for idx, row in enumerate(doc):
                    if not isinstance(row, Mapping) or column not in row:
                        raise SeriesComparisonError(f"Missing column {column!r} at {path}[{idx}]")
                    if group_by is not None and group_by not in row:
                        raise SeriesComparisonError(f"Missing group_by {group_by!r} at {path}[{idx}]")
                    gkey = str(row[group_by]) if group_by is not None else "__all__"
                    grouped.setdefault(gkey, []).append(
                        _parse_finite_float(row[column], f"{path}[{idx}].{column}")
                    )
                return grouped
            if group_by is not None:
                raise SeriesComparisonError(f"Cannot apply --group-by to flat JSON array {path}")
            return {"__all__": [_parse_finite_float(v, f"{path}[{i}]") for i, v in enumerate(doc)]}

        if isinstance(doc, Mapping):
            if group_by is not None:
                raise SeriesComparisonError(f"--group-by requires tabular records, got mapping in {path}")
            key = column or (next(iter(doc.keys())) if len(doc) == 1 else ("values" if "values" in doc else None))
            if key is None or key not in doc or not isinstance(doc[key], list) or not doc[key]:
                raise SeriesComparisonError(f"Valid non-empty list column required in {path}")
            return {"__all__": [_parse_finite_float(v, f"{path}.{key}[{i}]") for i, v in enumerate(doc[key])]}
        raise SeriesComparisonError(f"Unsupported JSON root structure in {path}")

    lines = text.splitlines()
    if not lines:
        raise SeriesComparisonError(f"Empty CSV file: {path}")
    reader = csv.DictReader(lines)
    if not reader.fieldnames:
        raise SeriesComparisonError(f"CSV file {path} has no header row")

    target_col = column
    if target_col is None:
        non_group = [c for c in reader.fieldnames if c != group_by]
        if len(non_group) != 1:
            raise SeriesComparisonError(f"--column required for multi-column CSV {path}")
        target_col = non_group[0]
    if target_col not in reader.fieldnames:
        raise SeriesComparisonError(f"Column {target_col!r} not found in {path}")
    if group_by is not None and group_by not in reader.fieldnames:
        raise SeriesComparisonError(f"Group column {group_by!r} not found in {path}")

    grouped_csv: Dict[str, List[float]] = {}
    for row_idx, row in enumerate(reader, start=2):
        gkey = str(row[group_by]) if group_by is not None else "__all__"
        val = _parse_finite_float(row.get(target_col, ""), f"{path}:line_{row_idx}.{target_col}")
        grouped_csv.setdefault(gkey, []).append(val)
    if not grouped_csv:
        raise SeriesComparisonError(f"CSV file {path} contains zero data rows")
    return grouped_csv


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for `compare_series.py`."""
    parser = argparse.ArgumentParser(description="Statistical comparator for benchmark series.")
    parser.add_argument("--baseline", required=True, help="Path to baseline CSV or JSON file")
    parser.add_argument("--candidate", required=True, help="Path to candidate CSV or JSON file")
    parser.add_argument("--paired", action="store_true", help="Treat observations as 1-to-1 paired")
    parser.add_argument("--column", default=None, help="Numeric column or JSON key")
    parser.add_argument("--group-by", dest="group_by", default=None, help="Optional group-by column")
    parser.add_argument("--confidence", type=float, default=0.95, help="Bootstrap CI level in (0, 1)")
    parser.add_argument("--resamples", type=int, default=2000, help="Bootstrap resample count")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed")
    parser.add_argument("--output", default=None, help="Non-destructive output JSON path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point returning POSIX exit code (`0` on success, `2` on error)."""
    args = build_arg_parser().parse_args(argv)
    try:
        if args.output is not None and Path(args.output).exists():
            raise SeriesComparisonError(f"Refusing to overwrite existing output file: {args.output}")
        base_groups = load_series_file(args.baseline, column=args.column, group_by=args.group_by)
        cand_groups = load_series_file(args.candidate, column=args.column, group_by=args.group_by)

        if args.group_by is None:
            payload: Dict[str, Any] = compare_series(
                base_groups["__all__"], cand_groups["__all__"], paired=args.paired,
                confidence=args.confidence, resamples=args.resamples, seed=args.seed,
            )
        else:
            common = sorted(set(base_groups.keys()) & set(cand_groups.keys()))
            if not common:
                raise SeriesComparisonError("No overlapping group keys between baseline and candidate")
            payload = {
                "group_by": args.group_by,
                "groups": {
                    gkey: compare_series(
                        base_groups[gkey], cand_groups[gkey], paired=args.paired,
                        confidence=args.confidence, resamples=args.resamples, seed=args.seed + i,
                    )
                    for i, gkey in enumerate(common)
                },
                "unmatched_baseline_groups": sorted(set(base_groups.keys()) - set(cand_groups.keys())),
                "unmatched_candidate_groups": sorted(set(cand_groups.keys()) - set(base_groups.keys())),
            }

        rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output is not None:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "x", encoding="utf-8") as fh:
                fh.write(rendered)
        else:
            sys.stdout.write(rendered)
        return 0
    except (SeriesComparisonError, FileExistsError, OSError) as exc:
        sys.stderr.write(f"compare_series error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
