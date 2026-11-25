#!/usr/bin/env python3
"""Paired cluster bootstrap and calibration-subset sensitivity analyzer.

Resamples paired evaluation items with replacement (seed-deterministic) across
candidate configurations to compute point estimates and linear-interpolated
quantile confidence intervals for mean energy, mean latency, absolute savings,
and ratio-of-means relative reduction (100 * (1 - mean_a / mean_b)). Also draws
without-replacement k-of-N calibration subsets to quantify operating-point
selection agreement, target attainment rate, and calibration time/energy cost.
"""

import argparse
import collections
import csv
import dataclasses
import json
import math
import pathlib
import random
import statistics
import sys


@dataclasses.dataclass(frozen=True)
class ItemMeasurement:
    """Single observation for one (item_id, config_id) pair."""

    item_id: str
    config_id: str
    energy_j: float
    latency_s: float
    quality: float = 1.0
    calibration_time_s: float | None = None
    calibration_energy_j: float | None = None

    def effective_cal_time_s(self) -> float:
        return self.latency_s if self.calibration_time_s is None else self.calibration_time_s

    def effective_cal_energy_j(self) -> float:
        return self.energy_j if self.calibration_energy_j is None else self.calibration_energy_j


@dataclasses.dataclass(frozen=True)
class PairedDataset:
    """Validated complete item-by-configuration matrix."""

    item_ids: tuple[str, ...]
    config_ids: tuple[str, ...]
    matrix: dict[str, dict[str, ItemMeasurement]]

    def num_items(self) -> int:
        return len(self.item_ids)

    def num_configs(self) -> int:
        return len(self.config_ids)

    def column(self, config_id: str) -> list[ItemMeasurement]:
        if config_id not in self.matrix:
            raise KeyError(f"Unknown config_id: {config_id!r}")
        by_item = self.matrix[config_id]
        return [by_item[item_id] for item_id in self.item_ids]


@dataclasses.dataclass(frozen=True)
class ConfidenceInterval:
    """Point estimate and linear-interpolated bootstrap percentile interval."""

    point: float
    lower: float
    upper: float
    confidence_level: float
    num_resamples: int

    def to_dict(self) -> dict[str, object]:
        return {
            "point": self.point, "lower": self.lower, "upper": self.upper,
            "confidence_level": self.confidence_level, "num_resamples": self.num_resamples,
        }


@dataclasses.dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired cluster bootstrap comparison between candidate (a) and baseline (b)."""

    config_a: str
    config_b: str
    num_items: int
    num_resamples: int
    confidence_level: float
    seed: int
    mean_energy_a_j: ConfidenceInterval
    mean_energy_b_j: ConfidenceInterval
    mean_latency_a_s: ConfidenceInterval
    mean_latency_b_s: ConfidenceInterval
    abs_energy_savings_j: ConfidenceInterval
    abs_latency_savings_s: ConfidenceInterval
    rel_energy_reduction_pct: ConfidenceInterval
    rel_latency_reduction_pct: ConfidenceInterval
    quality_diff: ConfidenceInterval

    def to_dict(self) -> dict[str, object]:
        return {
            "config_a": self.config_a, "config_b": self.config_b,
            "num_items": self.num_items, "num_resamples": self.num_resamples,
            "confidence_level": self.confidence_level, "seed": self.seed,
            "mean_energy_a_j": self.mean_energy_a_j.to_dict(),
            "mean_energy_b_j": self.mean_energy_b_j.to_dict(),
            "mean_latency_a_s": self.mean_latency_a_s.to_dict(),
            "mean_latency_b_s": self.mean_latency_b_s.to_dict(),
            "abs_energy_savings_j": self.abs_energy_savings_j.to_dict(),
            "abs_latency_savings_s": self.abs_latency_savings_s.to_dict(),
            "rel_energy_reduction_pct": self.rel_energy_reduction_pct.to_dict(),
            "rel_latency_reduction_pct": self.rel_latency_reduction_pct.to_dict(),
            "quality_diff": self.quality_diff.to_dict(),
        }


@dataclasses.dataclass(frozen=True)
class DistributionSummary:
    """Empirical distribution summary across subset draws."""

    mean: float
    stddev: float
    min_value: float
    q05: float
    q50: float
    q95: float
    max_value: float

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean, "stddev": self.stddev, "min": self.min_value,
            "q05": self.q05, "q50": self.q50, "q95": self.q95, "max": self.max_value,
        }


@dataclasses.dataclass(frozen=True)
class SubsetSensitivityPoint:
    """Calibration sensitivity statistics at without-replacement subset size k."""

    subset_size: int
    total_items: int
    num_draws: int
    target_quality: float
    reference_config_id: str
    selection_agreement_rate: float
    target_attainment_rate: float
    complement_attainment_rate: float | None
    selection_frequencies: dict[str, int]
    calibration_time_s: DistributionSummary
    calibration_energy_j: DistributionSummary
    selected_eval_energy_j: DistributionSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "subset_size": self.subset_size, "total_items": self.total_items,
            "num_draws": self.num_draws, "target_quality": self.target_quality,
            "reference_config_id": self.reference_config_id,
            "selection_agreement_rate": self.selection_agreement_rate,
            "target_attainment_rate": self.target_attainment_rate,
            "complement_attainment_rate": self.complement_attainment_rate,
            "selection_frequencies": dict(self.selection_frequencies),
            "calibration_time_s": self.calibration_time_s.to_dict(),
            "calibration_energy_j": self.calibration_energy_j.to_dict(),
            "selected_eval_energy_j": self.selected_eval_energy_j.to_dict(),
        }


def linear_quantile(values: list[float], prob: float) -> float:
    """Compute Type-7 linear-interpolated quantile on a non-empty float sequence."""
    if not values:
        raise ValueError("Cannot compute quantile of empty sequence.")
    if not math.isfinite(prob) or prob < 0.0 or prob > 1.0:
        raise ValueError(f"Quantile probability must lie in [0, 1], got {prob!r}")
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    position = prob * float(n - 1)
    lo_idx, hi_idx = int(math.floor(position)), int(math.ceil(position))
    if lo_idx == hi_idx:
        return ordered[lo_idx]
    weight = position - float(lo_idx)
    return ordered[lo_idx] * (1.0 - weight) + ordered[hi_idx] * weight


def summarize_distribution(values: list[float]) -> DistributionSummary:
    """Summarize draw values with mean, stddev, min, max, and 5/50/95 percentiles."""
    if not values:
        raise ValueError("Cannot summarize an empty draw list.")
    mean_val = statistics.fmean(values)
    std_val = statistics.stdev(values) if len(values) >= 2 else 0.0
    ordered = sorted(values)
    return DistributionSummary(
        mean_val, std_val, ordered[0],
        linear_quantile(ordered, 0.05), linear_quantile(ordered, 0.50),
        linear_quantile(ordered, 0.95), ordered[-1],
    )


def _validate_finite_non_negative(val: object, field_name: str, strictly_positive: bool = False) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"Field '{field_name}' must be numeric, got {val!r}")
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"Field '{field_name}' must be finite, got {val!r}")
    if strictly_positive and fval <= 0.0:
        raise ValueError(f"Field '{field_name}' must be > 0, got {fval}")
    if not strictly_positive and fval < 0.0:
        raise ValueError(f"Field '{field_name}' must be >= 0, got {fval}")
    return fval


def build_paired_dataset(records: list[ItemMeasurement]) -> PairedDataset:
    """Validate pairedness across all items and candidate configurations."""
    if not records:
        raise ValueError("Cannot build PairedDataset from empty record list.")
    seen_pairs: set[tuple[str, str]] = set()
    item_set: set[str] = set()
    config_set: set[str] = set()
    by_config: dict[str, dict[str, ItemMeasurement]] = collections.defaultdict(dict)

    for rec in records:
        if not isinstance(rec.item_id, str) or not rec.item_id.strip():
            raise ValueError(f"Invalid item_id: {rec.item_id!r}")
        if not isinstance(rec.config_id, str) or not rec.config_id.strip():
            raise ValueError(f"Invalid config_id: {rec.config_id!r}")
        _validate_finite_non_negative(rec.energy_j, "energy_j")
        _validate_finite_non_negative(rec.latency_s, "latency_s", strictly_positive=True)
        _validate_finite_non_negative(rec.quality, "quality")
        if rec.calibration_time_s is not None:
            _validate_finite_non_negative(rec.calibration_time_s, "calibration_time_s")
        if rec.calibration_energy_j is not None:
            _validate_finite_non_negative(rec.calibration_energy_j, "calibration_energy_j")

        pair = (rec.item_id, rec.config_id)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate observation for pair ({rec.item_id!r}, {rec.config_id!r}).")
        seen_pairs.add(pair)
        item_set.add(rec.item_id)
        config_set.add(rec.config_id)
        by_config[rec.config_id][rec.item_id] = rec

    sorted_items = tuple(sorted(item_set))
    sorted_configs = tuple(sorted(config_set))
    missing_pairs = [
        (item_id, cfg)
        for cfg in sorted_configs
        for item_id in sorted_items
        if item_id not in by_config[cfg]
    ]
    if missing_pairs:
        sample_str = ", ".join(f"({i}, {c})" for i, c in missing_pairs[:5])
        raise ValueError(
            f"Unpaired dataset: missing {len(missing_pairs)} (item_id, config_id) cells, e.g. {sample_str}"
        )
    return PairedDataset(sorted_items, sorted_configs, {cfg: dict(by_config[cfg]) for cfg in sorted_configs})


def parse_record_dict(raw: dict[str, object]) -> ItemMeasurement:
    """Parse a raw dictionary into a validated ItemMeasurement."""
    item_raw = raw.get("item_id", raw.get("request_id"))
    cfg_raw = raw.get("config_id")
    if not isinstance(item_raw, str) or not item_raw.strip():
        raise ValueError(f"Record missing valid 'item_id'/'request_id': {raw!r}")
    if not isinstance(cfg_raw, str) or not cfg_raw.strip():
        raise ValueError(f"Record missing valid 'config_id': {raw!r}")

    energy_j = _validate_finite_non_negative(raw.get("energy_j"), "energy_j")
    latency_s = _validate_finite_non_negative(raw.get("latency_s"), "latency_s", strictly_positive=True)
    if "quality" in raw and raw["quality"] is not None:
        quality = _validate_finite_non_negative(raw["quality"], "quality")
    elif "passed" in raw and raw["passed"] is not None:
        quality = (
            (1.0 if raw["passed"] else 0.0)
            if isinstance(raw["passed"], bool)
            else _validate_finite_non_negative(raw["passed"], "passed")
        )
    else:
        quality = 1.0

    cal_time = (
        _validate_finite_non_negative(raw["calibration_time_s"], "calibration_time_s")
        if raw.get("calibration_time_s") not in (None, "")
        else None
    )
    cal_energy = (
        _validate_finite_non_negative(raw["calibration_energy_j"], "calibration_energy_j")
        if raw.get("calibration_energy_j") not in (None, "")
        else None
    )
    return ItemMeasurement(item_raw.strip(), cfg_raw.strip(), energy_j, latency_s, quality, cal_time, cal_energy)


def load_paired_dataset(path: str | pathlib.Path) -> PairedDataset:
    """Load paired observations from JSON, JSONL, or CSV file."""
    file_path = pathlib.Path(path)
    suffix = file_path.suffix.lower()
    records: list[ItemMeasurement] = []

    if suffix == ".csv":
        with open(file_path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                converted: dict[str, object] = {
                    "item_id": row.get("item_id") or row.get("request_id"),
                    "config_id": row.get("config_id"),
                    "energy_j": float(row["energy_j"]) if row.get("energy_j") else None,
                    "latency_s": float(row["latency_s"]) if row.get("latency_s") else None,
                }
                if row.get("quality") not in (None, ""):
                    converted["quality"] = float(row["quality"])
                elif row.get("passed") not in (None, ""):
                    converted["passed"] = row["passed"].strip().lower() in ("true", "1", "yes")
                if row.get("calibration_time_s") not in (None, ""):
                    converted["calibration_time_s"] = float(row["calibration_time_s"])
                if row.get("calibration_energy_j") not in (None, ""):
                    converted["calibration_energy_j"] = float(row["calibration_energy_j"])
                records.append(parse_record_dict(converted))
    elif suffix == ".jsonl":
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError("Each JSONL line must decode to an object.")
                records.append(parse_record_dict(parsed))
    else:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        raw_list = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(raw_list, list):
            raise ValueError("JSON input must be a list of records or {'records': [...]}.")
        for entry in raw_list:
            if not isinstance(entry, dict):
                raise ValueError("Each JSON record must be an object.")
            records.append(parse_record_dict(entry))
    return build_paired_dataset(records)


def _compute_pair_metrics(
    eng_a: list[float], eng_b: list[float], lat_a: list[float], lat_b: list[float],
    qual_a: list[float], qual_b: list[float], indices: list[int],
) -> tuple[float, float, float, float, float, float, float, float, float]:
    n = float(len(indices))
    sum_ea = sum_eb = sum_la = sum_lb = sum_qa = sum_qb = 0.0
    for idx in indices:
        sum_ea += eng_a[idx]
        sum_eb += eng_b[idx]
        sum_la += lat_a[idx]
        sum_lb += lat_b[idx]
        sum_qa += qual_a[idx]
        sum_qb += qual_b[idx]

    mean_ea, mean_eb = sum_ea / n, sum_eb / n
    mean_la, mean_lb = sum_la / n, sum_lb / n
    mean_qa, mean_qb = sum_qa / n, sum_qb / n
    if mean_eb <= 0.0 or mean_lb <= 0.0:
        raise ValueError("Baseline mean energy and latency must be > 0 for ratio-of-means.")
    return (
        mean_ea, mean_eb, mean_la, mean_lb,
        mean_eb - mean_ea, mean_lb - mean_la,
        100.0 * (1.0 - (mean_ea / mean_eb)),
        100.0 * (1.0 - (mean_la / mean_lb)),
        mean_qa - mean_qb,
    )


def run_paired_bootstrap(
    dataset: PairedDataset, config_a: str, config_b: str,
    num_resamples: int = 2000, confidence_level: float = 0.95, seed: int = 42,
) -> PairedBootstrapResult:
    """Compute paired cluster bootstrap CIs comparing config_a vs baseline config_b."""
    if config_a not in dataset.config_ids or config_b not in dataset.config_ids:
        raise ValueError(f"Unknown configuration in ({config_a!r}, {config_b!r}).")
    if isinstance(num_resamples, bool) or not isinstance(num_resamples, int) or num_resamples < 1:
        raise ValueError(f"num_resamples must be a positive integer, got {num_resamples!r}")
    if not math.isfinite(confidence_level) or confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError(f"confidence_level must lie in (0, 1), got {confidence_level!r}")

    col_a, col_b = dataset.column(config_a), dataset.column(config_b)
    n_items = dataset.num_items()
    eng_a, eng_b = [m.energy_j for m in col_a], [m.energy_j for m in col_b]
    lat_a, lat_b = [m.latency_s for m in col_a], [m.latency_s for m in col_b]
    qual_a, qual_b = [m.quality for m in col_a], [m.quality for m in col_b]

    point_vals = _compute_pair_metrics(eng_a, eng_b, lat_a, lat_b, qual_a, qual_b, list(range(n_items)))
    rng = random.Random(seed)
    boot_buckets: list[list[float]] = [[] for _ in range(9)]
    for _ in range(num_resamples):
        draw_indices = [rng.randrange(n_items) for _ in range(n_items)]
        rep_vals = _compute_pair_metrics(eng_a, eng_b, lat_a, lat_b, qual_a, qual_b, draw_indices)
        for bucket_idx, val in enumerate(rep_vals):
            boot_buckets[bucket_idx].append(val)

    alpha = 1.0 - confidence_level
    lo_prob, hi_prob = alpha / 2.0, 1.0 - (alpha / 2.0)
    cis = [
        ConfidenceInterval(
            point_vals[i], linear_quantile(boot_buckets[i], lo_prob),
            linear_quantile(boot_buckets[i], hi_prob), confidence_level, num_resamples,
        )
        for i in range(9)
    ]
    return PairedBootstrapResult(
        config_a, config_b, n_items, num_resamples, confidence_level, seed,
        cis[0], cis[1], cis[2], cis[3], cis[4], cis[5], cis[6], cis[7], cis[8],
    )


def select_operating_point_on_indices(
    dataset: PairedDataset, indices: list[int], target_quality: float
) -> str:
    """Select minimum-energy config meeting target_quality on the given item indices."""
    if not indices:
        raise ValueError("Cannot select an operating point on an empty index subset.")
    n = float(len(indices))
    feasible: list[tuple[float, float, str]] = []
    fallback: list[tuple[float, float, float, str]] = []
    for cfg_id in dataset.config_ids:
        col = dataset.column(cfg_id)
        mean_q = sum(col[i].quality for i in indices) / n
        mean_e = sum(col[i].energy_j for i in indices) / n
        mean_l = sum(col[i].latency_s for i in indices) / n
        if mean_q + 1e-12 >= target_quality:
            feasible.append((mean_e, mean_l, cfg_id))
        fallback.append((-mean_q, mean_e, mean_l, cfg_id))
    if feasible:
        feasible.sort()
        return feasible[0][2]
    fallback.sort()
    return fallback[0][3]


def run_calibration_sensitivity(
    dataset: PairedDataset, subset_sizes: list[int],
    target_quality: float = 0.90, num_draws: int = 500, seed: int = 42,
) -> list[SubsetSensitivityPoint]:
    """Quantify k-of-N without-replacement calibration subset stability and cost."""
    n_items = dataset.num_items()
    if isinstance(num_draws, bool) or not isinstance(num_draws, int) or num_draws < 1:
        raise ValueError(f"num_draws must be a positive integer, got {num_draws!r}")
    if not math.isfinite(target_quality) or target_quality < 0.0 or target_quality > 1.0:
        raise ValueError(f"target_quality must lie in [0, 1], got {target_quality!r}")
    if not subset_sizes:
        raise ValueError("subset_sizes must be non-empty.")
    for k in subset_sizes:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1 or k > n_items:
            raise ValueError(f"Each subset size k must be an integer in [1, {n_items}], got {k!r}")

    all_indices = list(range(n_items))
    reference_cfg = select_operating_point_on_indices(dataset, all_indices, target_quality)
    cols_by_cfg = {cfg: dataset.column(cfg) for cfg in dataset.config_ids}
    full_quality_by_cfg = {cfg: statistics.fmean(m.quality for m in col) for cfg, col in cols_by_cfg.items()}
    full_energy_by_cfg = {cfg: statistics.fmean(m.energy_j for m in col) for cfg, col in cols_by_cfg.items()}

    results: list[SubsetSensitivityPoint] = []
    for k in subset_sizes:
        rng = random.Random(seed + k * 1009)
        agree_count = attain_count = comp_attain_count = 0
        freq: collections.Counter[str] = collections.Counter()
        cal_times: list[float] = []
        cal_energies: list[float] = []
        selected_energies: list[float] = []

        for _ in range(num_draws):
            subset_idx = rng.sample(all_indices, k)
            chosen_cfg = select_operating_point_on_indices(dataset, subset_idx, target_quality)
            freq[chosen_cfg] += 1
            if chosen_cfg == reference_cfg:
                agree_count += 1
            if full_quality_by_cfg[chosen_cfg] + 1e-12 >= target_quality:
                attain_count += 1

            if k < n_items:
                chosen_set = set(subset_idx)
                comp_idx = [i for i in all_indices if i not in chosen_set]
                chosen_col = cols_by_cfg[chosen_cfg]
                if sum(chosen_col[i].quality for i in comp_idx) / float(len(comp_idx)) + 1e-12 >= target_quality:
                    comp_attain_count += 1
                eval_e = sum(chosen_col[i].energy_j for i in comp_idx) / float(len(comp_idx))
            else:
                eval_e = full_energy_by_cfg[chosen_cfg]
            selected_energies.append(eval_e)

            draw_time = draw_energy = 0.0
            for cfg_id in dataset.config_ids:
                col = cols_by_cfg[cfg_id]
                for i in subset_idx:
                    draw_time += col[i].effective_cal_time_s()
                    draw_energy += col[i].effective_cal_energy_j()
            cal_times.append(draw_time)
            cal_energies.append(draw_energy)

        comp_rate = comp_attain_count / float(num_draws) if k < n_items else None
        ordered_freq = {cfg: freq[cfg] for cfg in dataset.config_ids if freq[cfg] > 0}
        results.append(
            SubsetSensitivityPoint(
                k, n_items, num_draws, target_quality, reference_cfg,
                agree_count / float(num_draws), attain_count / float(num_draws), comp_rate,
                ordered_freq, summarize_distribution(cal_times),
                summarize_distribution(cal_energies), summarize_distribution(selected_energies),
            )
        )
    return results


def format_bootstrap_markdown(
    bootstrap: PairedBootstrapResult, sensitivity: list[SubsetSensitivityPoint]
) -> str:
    """Render paired bootstrap CIs and calibration subset curve in Markdown."""
    lines = [
        f"# Paired Cluster Bootstrap: {bootstrap.config_a} vs {bootstrap.config_b}", "",
        f"- Paired items: {bootstrap.num_items}, resamples: {bootstrap.num_resamples}, "
        f"confidence: {bootstrap.confidence_level:.2%}, seed: {bootstrap.seed}", "",
        "| Metric | Point | Lower CI | Upper CI |", "| :--- | ---: | ---: | ---: |",
    ]
    rows = [
        ("Mean Energy A (J)", bootstrap.mean_energy_a_j),
        ("Mean Energy B (J)", bootstrap.mean_energy_b_j),
        ("Abs Energy Savings B-A (J)", bootstrap.abs_energy_savings_j),
        ("Rel Energy Reduction (%)", bootstrap.rel_energy_reduction_pct),
        ("Mean Latency A (s)", bootstrap.mean_latency_a_s),
        ("Mean Latency B (s)", bootstrap.mean_latency_b_s),
        ("Abs Latency Savings B-A (s)", bootstrap.abs_latency_savings_s),
        ("Rel Latency Reduction (%)", bootstrap.rel_latency_reduction_pct),
        ("Quality Diff A-B", bootstrap.quality_diff),
    ]
    for label, ci in rows:
        lines.append(f"| {label} | {ci.point:.4f} | {ci.lower:.4f} | {ci.upper:.4f} |")
    if sensitivity:
        lines.extend([
            "", "## Calibration Subset Sensitivity", "",
            "| k | Agreement | Attainment | Mean Cal Time (s) | Mean Cal Energy (J) |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ])
        for pt in sensitivity:
            lines.append(
                f"| {pt.subset_size} | {pt.selection_agreement_rate:.3f} | "
                f"{pt.target_attainment_rate:.3f} | {pt.calibration_time_s.mean:.2f} | "
                f"{pt.calibration_energy_j.mean:.2f} |"
            )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired cluster bootstrap and k-of-N subset analyzer.")
    parser.add_argument("--input", required=True, help="Path to paired JSON, JSONL, or CSV dataset.")
    parser.add_argument("--candidate", required=True, help="Candidate configuration ID (config_a).")
    parser.add_argument("--baseline", required=True, help="Baseline configuration ID (config_b).")
    parser.add_argument("--resamples", type=int, default=2000, help="Bootstrap replicates.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level in (0, 1).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic PRNG seed.")
    parser.add_argument("--subset-sizes", default=None, help="Optional comma-separated k values.")
    parser.add_argument("--subset-draws", type=int, default=500, help="Subset draws per k.")
    parser.add_argument("--target-quality", type=float, default=0.90, help="Target quality floor.")
    parser.add_argument("--output", required=True, help="Output JSON path (opened exclusively).")
    parser.add_argument("--markdown-output", default=None, help="Optional Markdown path (exclusive).")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint writing exclusive JSON/Markdown outputs."""
    args = _build_arg_parser().parse_args(argv)
    try:
        dataset = load_paired_dataset(args.input)
        boot_result = run_paired_bootstrap(
            dataset, args.candidate, args.baseline, args.resamples, args.confidence, args.seed
        )
        sensitivity_results: list[SubsetSensitivityPoint] = []
        if args.subset_sizes:
            k_values = [int(tok.strip()) for tok in args.subset_sizes.split(",") if tok.strip()]
            sensitivity_results = run_calibration_sensitivity(
                dataset, k_values, args.target_quality, args.subset_draws, args.seed
            )
        payload = {
            "bootstrap": boot_result.to_dict(),
            "calibration_sensitivity": [pt.to_dict() for pt in sensitivity_results],
        }
        with open(args.output, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if args.markdown_output is not None:
            with open(args.markdown_output, "x", encoding="utf-8") as md_handle:
                md_handle.write(format_bootstrap_markdown(boot_result, sensitivity_results))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"paired_bootstrap error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
