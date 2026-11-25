#!/usr/bin/env python3
"""Clopper-Pearson simultaneous operating-point selector and oracle-gap evaluator.

Implements pure-Python regularized incomplete beta (Modified Lentz continued
fraction + bisection quantile solver) for exact one-sided and Bonferroni
simultaneous Clopper-Pearson lower confidence bounds, multi-objective Pareto
frontier construction, hard-deadline vs continuous exponential-decay utility
scoring, baseline comparison, and disjoint split-resampling oracle-gap analysis.
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

MODE_HARD_DEADLINE = "hard_deadline"
MODE_EXP_DECAY = "exp_decay"

SELECTOR_CLOPPER_PEARSON = "clopper_pearson_simultaneous"
SELECTOR_POINT = "point_screening"
SELECTOR_MIN_LATENCY = "min_latency"
SELECTOR_MIN_ENERGY = "min_energy"
SELECTOR_MIN_EPT = "min_energy_per_token"

_LENTZ_TINY = 1.0e-30
_LENTZ_EPS = 1.0e-14
_LENTZ_MAX_ITER = 300


@dataclasses.dataclass(frozen=True)
class TrialObservation:
    """Single item measurement for a candidate configuration."""

    item_id: str
    config_id: str
    passed: bool
    latency_s: float
    energy_j: float
    output_tokens: int = 1

    def energy_per_token_j(self) -> float:
        return self.energy_j / float(self.output_tokens)


@dataclasses.dataclass(frozen=True)
class CandidateStats:
    """Aggregated utility, Clopper-Pearson bound, and cost metrics for one candidate."""

    config_id: str
    num_trials: int
    utility_sum: float
    mean_utility: float
    cp_lower_bound: float
    alpha_per_candidate: float
    mean_energy_j: float
    mean_latency_s: float
    mean_energy_per_token_j: float
    pass_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "config_id": self.config_id, "num_trials": self.num_trials,
            "utility_sum": self.utility_sum, "mean_utility": self.mean_utility,
            "cp_lower_bound": self.cp_lower_bound,
            "alpha_per_candidate": self.alpha_per_candidate,
            "mean_energy_j": self.mean_energy_j, "mean_latency_s": self.mean_latency_s,
            "mean_energy_per_token_j": self.mean_energy_per_token_j, "pass_rate": self.pass_rate,
        }


@dataclasses.dataclass(frozen=True)
class SelectionOutcome:
    """Selected configuration and feasibility status under a selection rule."""

    selector_name: str
    selected_config_id: str
    feasible_on_calibration: bool
    calibration_stats: CandidateStats

    def to_dict(self) -> dict[str, object]:
        return {
            "selector_name": self.selector_name, "selected_config_id": self.selected_config_id,
            "feasible_on_calibration": self.feasible_on_calibration, "calibration_stats": self.calibration_stats.to_dict(),
        }


@dataclasses.dataclass(frozen=True)
class SplitMethodSummary:
    """Held-out evaluation summary for one selector across repeated splits."""

    selector_name: str
    num_splits: int
    calibration_feasible_rate: float
    held_out_attainment_rate: float
    oracle_match_rate: float
    mean_eval_utility: float
    mean_eval_energy_j: float
    mean_eval_latency_s: float
    mean_oracle_gap_pct: float
    median_oracle_gap_pct: float
    p95_oracle_gap_pct: float

    def to_dict(self) -> dict[str, object]:
        return {
            "selector_name": self.selector_name, "num_splits": self.num_splits,
            "calibration_feasible_rate": self.calibration_feasible_rate, "held_out_attainment_rate": self.held_out_attainment_rate,
            "oracle_match_rate": self.oracle_match_rate, "mean_eval_utility": self.mean_eval_utility,
            "mean_eval_energy_j": self.mean_eval_energy_j, "mean_eval_latency_s": self.mean_eval_latency_s,
            "mean_oracle_gap_pct": self.mean_oracle_gap_pct, "median_oracle_gap_pct": self.median_oracle_gap_pct,
            "p95_oracle_gap_pct": self.p95_oracle_gap_pct,
        }


@dataclasses.dataclass(frozen=True)
class SplitEvaluationReport:
    """Full split-resampling comparison report across selectors and baselines."""

    total_items: int
    calibration_size: int
    evaluation_size: int
    num_candidates: int
    num_splits: int
    utility_mode: str
    deadline_s: float | None
    tau_s: float | None
    target_utility: float
    delta: float
    seed: int
    all_splits_disjoint: bool
    method_summaries: tuple[SplitMethodSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_items": self.total_items, "calibration_size": self.calibration_size,
            "evaluation_size": self.evaluation_size, "num_candidates": self.num_candidates,
            "num_splits": self.num_splits, "utility_mode": self.utility_mode,
            "deadline_s": self.deadline_s, "tau_s": self.tau_s, "target_utility": self.target_utility,
            "delta": self.delta, "seed": self.seed, "all_splits_disjoint": self.all_splits_disjoint,
            "method_summaries": [m.to_dict() for m in self.method_summaries],
        }


def _betacf(a: float, b: float, x: float) -> float:
    """Evaluate continued fraction for regularized incomplete beta via Modified Lentz."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 / (max(abs(1.0 - (qab * x / qap)), _LENTZ_TINY) if (1.0 - (qab * x / qap)) >= 0 else -max(abs(1.0 - (qab * x / qap)), _LENTZ_TINY))
    h = d
    for m in range(1, _LENTZ_MAX_ITER + 1):
        mf, m2f = float(m), float(2 * m)
        for aa in (mf * (b - mf) * x / ((qam + m2f) * (a + m2f)), -(a + mf) * (qab + mf) * x / ((a + m2f) * (qap + m2f))):
            d = 1.0 + aa * d
            if abs(d) < _LENTZ_TINY:
                d = _LENTZ_TINY
            c = 1.0 + aa / c
            if abs(c) < _LENTZ_TINY:
                c = _LENTZ_TINY
            d = 1.0 / d
            delta = d * c
            h *= delta
        if abs(delta - 1.0) < _LENTZ_EPS:
            return h
    raise RuntimeError(f"Lentz continued fraction failed to converge for a={a}, b={b}, x={x}.")


def betainc(a: float, b: float, x: float) -> float:
    """Compute regularized incomplete beta function I_x(a, b) in pure Python."""
    if not math.isfinite(a) or not math.isfinite(b) or a <= 0.0 or b <= 0.0:
        raise ValueError(f"Beta shape parameters must be finite and > 0, got a={a}, b={b}")
    if not math.isfinite(x) or x < 0.0 or x > 1.0:
        raise ValueError(f"Evaluation point x must lie in [0, 1], got x={x}")
    if x == 0.0 or x == 1.0:
        return x

    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - (front * _betacf(b, a, 1.0 - x) / b)


def betaincinv(a: float, b: float, prob: float, tol: float = 1.0e-12) -> float:
    """Invert I_x(a, b) = prob on [0, 1] using monotone bisection."""
    if not math.isfinite(a) or not math.isfinite(b) or a <= 0.0 or b <= 0.0:
        raise ValueError(f"Beta shape parameters must be finite and > 0, got a={a}, b={b}")
    if not math.isfinite(prob) or prob < 0.0 or prob > 1.0:
        raise ValueError(f"Target probability must lie in [0, 1], got {prob}")
    if prob == 0.0 or prob == 1.0:
        return prob

    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        val = betainc(a, b, mid)
        if abs(val - prob) <= tol or (hi - lo) <= tol:
            return mid
        if val < prob:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson_lower_bound(success_sum: float, num_trials: int, alpha: float) -> float:
    """Compute one-sided Clopper-Pearson lower bound at failure level alpha."""
    if isinstance(num_trials, bool) or not isinstance(num_trials, int) or num_trials < 1:
        raise ValueError(f"num_trials must be a positive integer, got {num_trials!r}")
    if not math.isfinite(success_sum) or success_sum < 0.0 or success_sum > float(num_trials) + 1e-9:
        raise ValueError(f"success_sum must lie in [0, {num_trials}], got {success_sum!r}")
    if not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")

    clamped_sum = min(max(success_sum, 0.0), float(num_trials))
    if clamped_sum <= 0.0:
        return 0.0
    return betaincinv(clamped_sum, float(num_trials) - clamped_sum + 1.0, alpha)


def compute_item_utility(
    obs: TrialObservation, utility_mode: str,
    deadline_s: float | None = None, tau_s: float | None = None,
) -> float:
    """Score a single observation under hard-deadline or exponential-decay utility."""
    if utility_mode == MODE_HARD_DEADLINE:
        if deadline_s is None or not math.isfinite(deadline_s) or deadline_s <= 0.0:
            raise ValueError(f"deadline_s must be finite and > 0 for hard_deadline, got {deadline_s!r}")
        return 1.0 if (obs.passed and obs.latency_s <= deadline_s) else 0.0
    if utility_mode == MODE_EXP_DECAY:
        if tau_s is None or not math.isfinite(tau_s) or tau_s <= 0.0:
            raise ValueError(f"tau_s must be finite and > 0 for exp_decay, got {tau_s!r}")
        return math.exp(-obs.latency_s / tau_s) if obs.passed else 0.0
    raise ValueError(f"Unsupported utility_mode: {utility_mode!r}")


def summarize_candidates(
    by_config: dict[str, dict[str, TrialObservation]],
    item_ids: list[str] | tuple[str, ...],
    config_ids: list[str] | tuple[str, ...],
    utility_mode: str,
    delta: float = 0.05,
    deadline_s: float | None = None,
    tau_s: float | None = None,
) -> list[CandidateStats]:
    """Compute utility, Bonferroni simultaneous CP bound (delta / M), and costs."""
    if not item_ids or not config_ids:
        raise ValueError("item_ids and config_ids must be non-empty.")
    if not math.isfinite(delta) or delta <= 0.0 or delta >= 1.0:
        raise ValueError(f"delta must lie in (0, 1), got {delta!r}")

    alpha_each = delta / float(len(config_ids))
    n_trials = len(item_ids)
    summaries: list[CandidateStats] = []
    for cfg_id in config_ids:
        cfg_map = by_config[cfg_id]
        util_sum = eng_sum = lat_sum = ept_sum = 0.0
        pass_count = 0
        for item_id in item_ids:
            obs = cfg_map[item_id]
            util_sum += compute_item_utility(obs, utility_mode, deadline_s, tau_s)
            eng_sum += obs.energy_j
            lat_sum += obs.latency_s
            ept_sum += obs.energy_per_token_j()
            if obs.passed:
                pass_count += 1
        cp_lb = clopper_pearson_lower_bound(util_sum, n_trials, alpha_each)
        summaries.append(
            CandidateStats(
                cfg_id, n_trials, util_sum, util_sum / float(n_trials), cp_lb, alpha_each,
                eng_sum / float(n_trials), lat_sum / float(n_trials),
                ept_sum / float(n_trials), pass_count / float(n_trials),
            )
        )
    return summaries


def build_pareto_frontier(
    candidates: list[CandidateStats], use_cp_bound: bool = False, include_latency: bool = True
) -> list[CandidateStats]:
    """Prune dominated candidates over (minimize energy, [minimize latency], maximize utility)."""
    if not candidates:
        return []

    def _util(c: CandidateStats) -> float:
        return c.cp_lower_bound if use_cp_bound else c.mean_utility

    def _dominates(other: CandidateStats, target: CandidateStats) -> bool:
        u_o, u_t = _util(other), _util(target)
        if include_latency:
            no_worse = (
                other.mean_energy_j <= target.mean_energy_j + 1e-12
                and other.mean_latency_s <= target.mean_latency_s + 1e-12
                and u_o >= u_t - 1e-12
            )
            better = (
                other.mean_energy_j < target.mean_energy_j - 1e-12
                or other.mean_latency_s < target.mean_latency_s - 1e-12
                or u_o > u_t + 1e-12
            )
            return no_worse and better
        return (
            other.mean_energy_j <= target.mean_energy_j + 1e-12
            and u_o >= u_t - 1e-12
            and (other.mean_energy_j < target.mean_energy_j - 1e-12 or u_o > u_t + 1e-12)
        )

    frontier = [
        cand for cand in candidates
        if not any(other.config_id != cand.config_id and _dominates(other, cand) for other in candidates)
    ]
    frontier.sort(key=lambda c: (c.mean_energy_j, c.mean_latency_s, c.config_id))
    return frontier


def select_all_methods(
    candidates: list[CandidateStats], target_utility: float
) -> dict[str, SelectionOutcome]:
    """Run point, simultaneous Clopper-Pearson, and 3 cost baselines on calibration stats."""
    if not candidates:
        raise ValueError("Candidate list must be non-empty.")
    if not math.isfinite(target_utility) or target_utility < 0.0 or target_utility > 1.0:
        raise ValueError(f"target_utility must lie in [0, 1], got {target_utility!r}")

    cp_feas = [c for c in candidates if c.cp_lower_bound + 1e-12 >= target_utility]
    if cp_feas:
        cp_pick = min(cp_feas, key=lambda c: (c.mean_energy_j, c.mean_latency_s, c.config_id))
        cp_ok = True
    else:
        cp_pick = min(candidates, key=lambda c: (-c.cp_lower_bound, c.mean_energy_j, c.mean_latency_s, c.config_id))
        cp_ok = False

    pt_feas = [c for c in candidates if c.mean_utility + 1e-12 >= target_utility]
    if pt_feas:
        pt_pick = min(pt_feas, key=lambda c: (c.mean_energy_j, c.mean_latency_s, c.config_id))
        pt_ok = True
    else:
        pt_pick = min(candidates, key=lambda c: (-c.mean_utility, c.mean_energy_j, c.mean_latency_s, c.config_id))
        pt_ok = False

    min_lat = min(candidates, key=lambda c: (c.mean_latency_s, c.mean_energy_j, c.config_id))
    min_eng = min(candidates, key=lambda c: (c.mean_energy_j, c.mean_latency_s, c.config_id))
    min_ept = min(candidates, key=lambda c: (c.mean_energy_per_token_j, c.mean_energy_j, c.config_id))

    return {
        SELECTOR_CLOPPER_PEARSON: SelectionOutcome(SELECTOR_CLOPPER_PEARSON, cp_pick.config_id, cp_ok, cp_pick),
        SELECTOR_POINT: SelectionOutcome(SELECTOR_POINT, pt_pick.config_id, pt_ok, pt_pick),
        SELECTOR_MIN_LATENCY: SelectionOutcome(
            SELECTOR_MIN_LATENCY, min_lat.config_id, min_lat.mean_utility + 1e-12 >= target_utility, min_lat
        ),
        SELECTOR_MIN_ENERGY: SelectionOutcome(
            SELECTOR_MIN_ENERGY, min_eng.config_id, min_eng.mean_utility + 1e-12 >= target_utility, min_eng
        ),
        SELECTOR_MIN_EPT: SelectionOutcome(
            SELECTOR_MIN_EPT, min_ept.config_id, min_ept.mean_utility + 1e-12 >= target_utility, min_ept
        ),
    }


def compute_oracle_energy_gap_pct(selected_energy_j: float, oracle_energy_j: float) -> float:
    """Compute held-out relative energy gap 100 * (E_selected / E_oracle - 1)."""
    if not math.isfinite(selected_energy_j) or selected_energy_j < 0.0:
        raise ValueError(f"selected_energy_j must be finite and >= 0, got {selected_energy_j!r}")
    if not math.isfinite(oracle_energy_j) or oracle_energy_j <= 0.0:
        raise ValueError(f"oracle_energy_j must be finite and > 0, got {oracle_energy_j!r}")
    return 100.0 * ((selected_energy_j / oracle_energy_j) - 1.0)


def _linear_percentile(values: list[float], prob: float) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    pos = prob * float(n - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return ordered[lo] if lo == hi else ordered[lo] * (1.0 - (pos - lo)) + ordered[hi] * (pos - lo)


def validate_trial_table(
    observations: list[TrialObservation],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, dict[str, TrialObservation]]]:
    """Validate paired (item_id, config_id) observations and return indexed table."""
    if not observations:
        raise ValueError("Observation list must be non-empty.")
    seen: set[tuple[str, str]] = set()
    items: set[str] = set()
    configs: set[str] = set()
    by_config: dict[str, dict[str, TrialObservation]] = collections.defaultdict(dict)

    for obs in observations:
        if not isinstance(obs.item_id, str) or not obs.item_id.strip():
            raise ValueError(f"Invalid item_id: {obs.item_id!r}")
        if not isinstance(obs.config_id, str) or not obs.config_id.strip():
            raise ValueError(f"Invalid config_id: {obs.config_id!r}")
        if not isinstance(obs.passed, bool):
            raise ValueError(f"Field 'passed' must be bool, got {obs.passed!r}")
        if isinstance(obs.latency_s, bool) or not math.isfinite(obs.latency_s) or obs.latency_s <= 0.0:
            raise ValueError(f"Field 'latency_s' must be finite and > 0, got {obs.latency_s!r}")
        if isinstance(obs.energy_j, bool) or not math.isfinite(obs.energy_j) or obs.energy_j <= 0.0:
            raise ValueError(f"Field 'energy_j' must be finite and > 0, got {obs.energy_j!r}")
        if isinstance(obs.output_tokens, bool) or not isinstance(obs.output_tokens, int) or obs.output_tokens < 1:
            raise ValueError(f"Field 'output_tokens' must be positive int, got {obs.output_tokens!r}")
        pair = (obs.item_id, obs.config_id)
        if pair in seen:
            raise ValueError(f"Duplicate observation for pair {pair!r}.")
        seen.add(pair)
        items.add(obs.item_id)
        configs.add(obs.config_id)
        by_config[obs.config_id][obs.item_id] = obs

    sorted_items, sorted_configs = tuple(sorted(items)), tuple(sorted(configs))
    for cfg in sorted_configs:
        for item_id in sorted_items:
            if item_id not in by_config[cfg]:
                raise ValueError(f"Unpaired observation table: missing ({item_id!r}, {cfg!r}).")
    return sorted_items, sorted_configs, {c: dict(by_config[c]) for c in sorted_configs}


def evaluate_split_resampling(
    observations: list[TrialObservation], calibration_size: int, utility_mode: str,
    target_utility: float, delta: float = 0.05, deadline_s: float | None = None,
    tau_s: float | None = None, num_splits: int = 200, seed: int = 42,
) -> SplitEvaluationReport:
    """Evaluate disjoint calibration/evaluation splits and held-out oracle energy gaps."""
    item_ids, config_ids, by_config = validate_trial_table(observations)
    n_total = len(item_ids)
    if isinstance(calibration_size, bool) or not isinstance(calibration_size, int) or not (1 <= calibration_size < n_total):
        raise ValueError(f"calibration_size must lie in [1, {n_total - 1}], got {calibration_size!r}")
    if isinstance(num_splits, bool) or not isinstance(num_splits, int) or num_splits < 1:
        raise ValueError(f"num_splits must be >= 1, got {num_splits!r}")

    method_order = (
        SELECTOR_CLOPPER_PEARSON, SELECTOR_POINT,
        SELECTOR_MIN_LATENCY, SELECTOR_MIN_ENERGY, SELECTOR_MIN_EPT,
    )
    cal_feas = {m: 0 for m in method_order}
    attain = {m: 0 for m in method_order}
    oracle_matches = {m: 0 for m in method_order}
    eval_utils: dict[str, list[float]] = {m: [] for m in method_order}
    eval_engs: dict[str, list[float]] = {m: [] for m in method_order}
    eval_lats: dict[str, list[float]] = {m: [] for m in method_order}
    oracle_gaps: dict[str, list[float]] = {m: [] for m in method_order}

    rng = random.Random(seed)
    all_disjoint = True
    item_list = list(item_ids)

    for _ in range(num_splits):
        cal_ids = rng.sample(item_list, calibration_size)
        cal_set = set(cal_ids)
        eval_ids = [item_id for item_id in item_list if item_id not in cal_set]
        if not cal_set.isdisjoint(set(eval_ids)) or len(cal_ids) + len(eval_ids) != n_total:
            all_disjoint = False
            raise AssertionError("Calibration and evaluation sets must partition all items.")

        cal_stats = summarize_candidates(by_config, cal_ids, config_ids, utility_mode, delta, deadline_s, tau_s)
        eval_stats = summarize_candidates(by_config, eval_ids, config_ids, utility_mode, delta, deadline_s, tau_s)
        eval_by_cfg = {s.config_id: s for s in eval_stats}

        oracle_feas = [s for s in eval_stats if s.mean_utility + 1e-12 >= target_utility]
        oracle_cfg = (
            min(oracle_feas, key=lambda s: (s.mean_energy_j, s.mean_latency_s, s.config_id))
            if oracle_feas
            else min(eval_stats, key=lambda s: (-s.mean_utility, s.mean_energy_j, s.mean_latency_s, s.config_id))
        )

        picks = select_all_methods(cal_stats, target_utility=target_utility)
        for m_name in method_order:
            outcome = picks[m_name]
            chosen_eval = eval_by_cfg[outcome.selected_config_id]
            if outcome.feasible_on_calibration:
                cal_feas[m_name] += 1
            if chosen_eval.mean_utility + 1e-12 >= target_utility:
                attain[m_name] += 1
            if chosen_eval.config_id == oracle_cfg.config_id:
                oracle_matches[m_name] += 1
            eval_utils[m_name].append(chosen_eval.mean_utility)
            eval_engs[m_name].append(chosen_eval.mean_energy_j)
            eval_lats[m_name].append(chosen_eval.mean_latency_s)
            oracle_gaps[m_name].append(
                compute_oracle_energy_gap_pct(chosen_eval.mean_energy_j, oracle_cfg.mean_energy_j)
            )

    denom = float(num_splits)
    summaries = [
        SplitMethodSummary(
            m_name, num_splits, cal_feas[m_name] / denom, attain[m_name] / denom,
            oracle_matches[m_name] / denom, statistics.fmean(eval_utils[m_name]),
            statistics.fmean(eval_engs[m_name]), statistics.fmean(eval_lats[m_name]),
            statistics.fmean(oracle_gaps[m_name]), _linear_percentile(oracle_gaps[m_name], 0.50),
            _linear_percentile(oracle_gaps[m_name], 0.95),
        )
        for m_name in method_order
    ]
    return SplitEvaluationReport(
        n_total, calibration_size, n_total - calibration_size, len(config_ids),
        num_splits, utility_mode, deadline_s, tau_s, target_utility, delta, seed,
        all_disjoint, tuple(summaries),
    )


def load_observations(path: str | pathlib.Path) -> list[TrialObservation]:
    """Load trial observations from JSON, JSONL, or CSV."""
    file_path = pathlib.Path(path)
    suffix = file_path.suffix.lower()
    raw_rows: list[dict[str, object]] = []
    if suffix == ".csv":
        with open(file_path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_rows.append({
                    "item_id": row.get("item_id") or row.get("request_id"),
                    "config_id": row.get("config_id"),
                    "passed": (row.get("passed") or "").strip().lower() in ("true", "1", "yes"),
                    "latency_s": float(row["latency_s"]),
                    "energy_j": float(row["energy_j"]),
                    "output_tokens": int(row.get("output_tokens") or 1),
                })
    elif suffix == ".jsonl":
        raw_rows = [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        parsed = json.loads(file_path.read_text(encoding="utf-8"))
        raw_rows = parsed["observations"] if isinstance(parsed, dict) else parsed

    return [
        TrialObservation(
            str(entry.get("item_id", entry.get("request_id"))), str(entry["config_id"]),
            bool(entry["passed"]), float(entry["latency_s"]), float(entry["energy_j"]),
            int(entry.get("output_tokens", 1)),
        )
        for entry in raw_rows
    ]


def format_selector_markdown(
    frontier: list[CandidateStats], split_report: SplitEvaluationReport
) -> str:
    """Format Pareto frontier and split-resampling comparison in Markdown."""
    lines = [
        "# Operating-Point Selection and Oracle-Gap Evaluation", "",
        f"- Mode: `{split_report.utility_mode}`, target utility: {split_report.target_utility:.3f}, "
        f"delta: {split_report.delta:.3f}",
        f"- Split: {split_report.calibration_size} cal / {split_report.evaluation_size} eval "
        f"({split_report.num_splits} splits)", "",
        "## Full-Dataset Pareto Frontier", "",
        "| Config | Mean Utility | CP Lower Bound | Mean Energy (J) | Mean Latency (s) |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for cand in frontier:
        lines.append(
            f"| {cand.config_id} | {cand.mean_utility:.4f} | {cand.cp_lower_bound:.4f} | "
            f"{cand.mean_energy_j:.3f} | {cand.mean_latency_s:.3f} |"
        )
    lines.extend([
        "", "## Split-Resampling Held-Out Performance", "",
        "| Selector | Attainment | Oracle Match | Mean Energy (J) | Mean Oracle Gap (%) |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])
    for m in split_report.method_summaries:
        lines.append(
            f"| {m.selector_name} | {m.held_out_attainment_rate:.3f} | {m.oracle_match_rate:.3f} | "
            f"{m.mean_eval_energy_j:.3f} | {m.mean_oracle_gap_pct:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simultaneous Clopper-Pearson selector and oracle-gap evaluator.")
    parser.add_argument("--input", required=True, help="Path to JSON, JSONL, or CSV observations.")
    parser.add_argument("--mode", choices=[MODE_HARD_DEADLINE, MODE_EXP_DECAY], default=MODE_HARD_DEADLINE)
    parser.add_argument("--deadline-s", type=float, default=2.0, help="Deadline seconds.")
    parser.add_argument("--tau-s", type=float, default=2.0, help="Decay time constant.")
    parser.add_argument("--target-utility", type=float, default=0.85, help="Target utility threshold.")
    parser.add_argument("--delta", type=float, default=0.05, help="Simultaneous error budget delta.")
    parser.add_argument("--calibration-size", type=int, required=True, help="Calibration split size.")
    parser.add_argument("--num-splits", type=int, default=200, help="Number of disjoint random splits.")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    parser.add_argument("--output", required=True, help="Exclusive JSON output path.")
    parser.add_argument("--markdown-output", default=None, help="Optional exclusive Markdown path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint writing exclusive JSON and optional Markdown summaries."""
    args = _build_arg_parser().parse_args(argv)
    try:
        observations = load_observations(args.input)
        item_ids, config_ids, by_config = validate_trial_table(observations)
        full_stats = summarize_candidates(
            by_config, item_ids, config_ids, args.mode, args.delta, args.deadline_s, args.tau_s
        )
        frontier = build_pareto_frontier(full_stats, use_cp_bound=False, include_latency=True)
        cp_frontier = build_pareto_frontier(full_stats, use_cp_bound=True, include_latency=True)
        split_report = evaluate_split_resampling(
            observations, args.calibration_size, args.mode, args.target_utility,
            args.delta, args.deadline_s, args.tau_s, args.num_splits, args.seed,
        )
        payload = {
            "full_dataset_candidates": [c.to_dict() for c in full_stats],
            "pareto_frontier_point": [c.to_dict() for c in frontier],
            "pareto_frontier_cp": [c.to_dict() for c in cp_frontier],
            "split_evaluation": split_report.to_dict(),
        }
        with open(args.output, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if args.markdown_output is not None:
            with open(args.markdown_output, "x", encoding="utf-8") as md_handle:
                md_handle.write(format_selector_markdown(frontier, split_report))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"operating_point_selector error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
