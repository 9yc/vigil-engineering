#!/usr/bin/env python3
"""Cartesian measurement completeness and across-repeat determinism auditor.

Validates observed JSONL measurement logs against an expected Cartesian grid
of (request_id, config_id, repeat_id) cells, detects duplicate or unscheduled
observations, classifies cell execution states with structured error codes,
audits across-repeat output SHA-256, exact string, and boolean accuracy
determinism, and computes unbiased sample statistics (mean, ddof=1 stddev, CV)
for latency, energy, and energy-per-token.
"""

import argparse
import collections
import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import statistics
import sys

CELL_VALID, CELL_FAILED, CELL_NOT_RUN, CELL_INVALID, CELL_MISSING = (
    "valid", "failed", "not_run", "invalid", "missing",
)
ERR_MISSING_CELL, ERR_DUPLICATE_CELL, ERR_UNSCHEDULED_CELL = (
    "MISSING_CELL", "DUPLICATE_CELL", "UNSCHEDULED_CELL",
)
ERR_MALFORMED_JSON, ERR_INVALID_KEY_TYPES, ERR_INVALID_STATUS = (
    "MALFORMED_JSON", "INVALID_KEY_TYPES", "INVALID_STATUS",
)
ERR_MISSING_TIMESTAMP, ERR_NON_FINITE_TIMESTAMP, ERR_TIMESTAMP_ORDER = (
    "MISSING_TIMESTAMP", "NON_FINITE_TIMESTAMP", "TIMESTAMP_ORDER",
)
ERR_MISSING_LATENCY, ERR_NON_FINITE_LATENCY, ERR_NEGATIVE_LATENCY = (
    "MISSING_LATENCY", "NON_FINITE_LATENCY", "NEGATIVE_LATENCY",
)
ERR_MISSING_ENERGY, ERR_NON_FINITE_ENERGY, ERR_NEGATIVE_ENERGY = (
    "MISSING_ENERGY", "NON_FINITE_ENERGY", "NEGATIVE_ENERGY",
)
ERR_INVALID_TOKEN_COUNT, ERR_MISSING_OUTPUT_TEXT = "INVALID_TOKEN_COUNT", "MISSING_OUTPUT_TEXT"
ERR_OUTPUT_HASH_MISMATCH, ERR_INVALID_ACCURACY_FLAG = "OUTPUT_HASH_MISMATCH", "INVALID_ACCURACY_FLAG"

_VALID_STATUS_ALIASES = frozenset({"valid", "ok", "success", "completed"})
_FAILED_STATUS_ALIASES = frozenset({"failed", "error", "aborted", "timeout"})
_NOT_RUN_STATUS_ALIASES = frozenset({"not_run", "skipped", "omitted"})


@dataclasses.dataclass(frozen=True)
class ManifestSpec:
    """Expected Cartesian measurement grid specification."""

    request_ids: tuple[str, ...]
    config_ids: tuple[str, ...]
    repeat_ids: tuple[int, ...]
    manifest_id: str = "default_grid"

    def expected_cells(self) -> list[tuple[str, str, int]]:
        return [(r, c, k) for r in self.request_ids for c in self.config_ids for k in self.repeat_ids]

    def expected_cell_set(self) -> set[tuple[str, str, int]]:
        return set(self.expected_cells())

    def total_cells(self) -> int:
        return len(self.request_ids) * len(self.config_ids) * len(self.repeat_ids)


@dataclasses.dataclass(frozen=True)
class GridViolation:
    """Structural log violation (malformed JSON, duplicate, or out-of-grid tuple)."""

    line_number: int
    error_code: str
    message: str
    cell_key: tuple[str, str, int] | None = None

    def to_dict(self) -> dict[str, object]:
        key_obj = (
            {"request_id": self.cell_key[0], "config_id": self.cell_key[1], "repeat_id": self.cell_key[2]}
            if self.cell_key is not None else None
        )
        return {"line_number": self.line_number, "error_code": self.error_code, "message": self.message, "cell_key": key_obj}


@dataclasses.dataclass(frozen=True)
class CellAuditResult:
    """Classification and validated metrics for one expected Cartesian cell."""

    request_id: str
    config_id: str
    repeat_id: int
    state: str
    error_codes: tuple[str, ...]
    line_numbers: tuple[int, ...]
    latency_s: float | None = None
    energy_j: float | None = None
    output_tokens: int | None = None
    energy_per_token_j: float | None = None
    output_text: str | None = None
    output_sha256: str | None = None
    passed: bool | None = None
    start_ts: float | None = None
    end_ts: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id, "config_id": self.config_id, "repeat_id": self.repeat_id,
            "state": self.state, "error_codes": list(self.error_codes), "line_numbers": list(self.line_numbers),
            "latency_s": self.latency_s, "energy_j": self.energy_j, "output_tokens": self.output_tokens,
            "energy_per_token_j": self.energy_per_token_j, "output_sha256": self.output_sha256,
            "passed": self.passed, "start_ts": self.start_ts, "end_ts": self.end_ts,
        }


@dataclasses.dataclass(frozen=True)
class MetricSampleStats:
    """Unbiased sample mean, ddof=1 standard deviation, and CV."""

    count: int
    mean: float | None
    stddev: float | None
    cv: float | None
    min_value: float | None
    max_value: float | None

    def to_dict(self) -> dict[str, object]:
        return {"count": self.count, "mean": self.mean, "stddev": self.stddev, "cv": self.cv, "min": self.min_value, "max": self.max_value}


@dataclasses.dataclass(frozen=True)
class RepeatGroupAudit:
    """Across-repeat determinism and sample dispersion for (request_id, config_id)."""

    request_id: str
    config_id: str
    expected_repeats: int
    valid_repeats: int
    state_counts: dict[str, int]
    sha256_equivalent: bool
    exact_text_equivalent: bool
    accuracy_equivalent: bool
    is_deterministic: bool
    distinct_sha256s: tuple[str, ...]
    distinct_text_count: int
    distinct_accuracy_values: tuple[bool, ...]
    pass_rate: float | None
    latency_stats: MetricSampleStats
    energy_stats: MetricSampleStats
    energy_per_token_stats: MetricSampleStats

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id, "config_id": self.config_id,
            "expected_repeats": self.expected_repeats, "valid_repeats": self.valid_repeats,
            "state_counts": dict(self.state_counts), "sha256_equivalent": self.sha256_equivalent,
            "exact_text_equivalent": self.exact_text_equivalent, "accuracy_equivalent": self.accuracy_equivalent,
            "is_deterministic": self.is_deterministic, "distinct_sha256s": list(self.distinct_sha256s),
            "distinct_text_count": self.distinct_text_count, "distinct_accuracy_values": list(self.distinct_accuracy_values),
            "pass_rate": self.pass_rate, "latency_s": self.latency_stats.to_dict(),
            "energy_j": self.energy_stats.to_dict(), "energy_per_token_j": self.energy_per_token_stats.to_dict(),
        }


@dataclasses.dataclass(frozen=True)
class ConfigAuditSummary:
    """Aggregated completeness, determinism, and dispersion per config_id."""

    config_id: str
    expected_cells: int
    valid_cells: int
    failed_cells: int
    not_run_cells: int
    invalid_cells: int
    missing_cells: int
    deterministic_groups: int
    total_groups: int
    mean_repeat_latency_cv: float | None
    mean_repeat_energy_cv: float | None
    mean_repeat_ept_cv: float | None
    overall_latency_stats: MetricSampleStats
    overall_energy_stats: MetricSampleStats
    overall_ept_stats: MetricSampleStats

    def to_dict(self) -> dict[str, object]:
        return {
            "config_id": self.config_id, "expected_cells": self.expected_cells, "valid_cells": self.valid_cells,
            "failed_cells": self.failed_cells, "not_run_cells": self.not_run_cells, "invalid_cells": self.invalid_cells,
            "missing_cells": self.missing_cells, "deterministic_groups": self.deterministic_groups, "total_groups": self.total_groups,
            "mean_repeat_latency_cv": self.mean_repeat_latency_cv, "mean_repeat_energy_cv": self.mean_repeat_energy_cv,
            "mean_repeat_ept_cv": self.mean_repeat_ept_cv, "overall_latency_s": self.overall_latency_stats.to_dict(),
            "overall_energy_j": self.overall_energy_stats.to_dict(), "overall_energy_per_token_j": self.overall_ept_stats.to_dict(),
        }


@dataclasses.dataclass(frozen=True)
class CompletenessReport:
    """Full Cartesian audit outcome across manifest cells and observed records."""

    manifest_id: str
    expected_cells: int
    observed_records: int
    valid_cells: int
    failed_cells: int
    not_run_cells: int
    invalid_cells: int
    missing_cells: int
    duplicate_count: int
    unscheduled_count: int
    malformed_line_count: int
    is_complete: bool
    all_repeats_deterministic: bool
    nondeterministic_groups: tuple[tuple[str, str], ...]
    violations: tuple[GridViolation, ...]
    cell_results: tuple[CellAuditResult, ...]
    group_audits: tuple[RepeatGroupAudit, ...]
    config_summaries: tuple[ConfigAuditSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            "summary": {
                "expected_cells": self.expected_cells, "observed_records": self.observed_records,
                "valid_cells": self.valid_cells, "failed_cells": self.failed_cells, "not_run_cells": self.not_run_cells,
                "invalid_cells": self.invalid_cells, "missing_cells": self.missing_cells,
                "duplicate_count": self.duplicate_count, "unscheduled_count": self.unscheduled_count,
                "malformed_line_count": self.malformed_line_count, "is_complete": self.is_complete,
                "all_repeats_deterministic": self.all_repeats_deterministic,
                "nondeterministic_group_count": len(self.nondeterministic_groups),
                "nondeterministic_groups": [{"request_id": r, "config_id": c} for r, c in self.nondeterministic_groups],
            },
            "violations": [v.to_dict() for v in self.violations],
            "config_summaries": [cs.to_dict() for cs in self.config_summaries],
            "group_audits": [ga.to_dict() for ga in self.group_audits],
            "cell_results": [cr.to_dict() for cr in self.cell_results],
        }


def compute_sample_stats(values: list[float]) -> MetricSampleStats:
    """Compute unbiased mean, sample stddev (ddof=1), and coefficient of variation."""
    if not values:
        return MetricSampleStats(0, None, None, None, None, None)
    if any(not math.isfinite(val) for val in values):
        raise ValueError("Cannot compute sample statistics on non-finite value.")
    mean_val, min_val, max_val = statistics.fmean(values), min(values), max(values)
    if len(values) < 2:
        return MetricSampleStats(1, mean_val, None, None, min_val, max_val)
    stddev_val = statistics.stdev(values)
    cv_val = (stddev_val / mean_val) if mean_val > 0.0 else (0.0 if mean_val == 0.0 and stddev_val == 0.0 else None)
    return MetricSampleStats(len(values), mean_val, stddev_val, cv_val, min_val, max_val)


def _validate_unique_strings(items: object, axis_name: str) -> tuple[str, ...]:
    if not isinstance(items, list) or not items:
        raise ValueError(f"Manifest '{axis_name}' must be a non-empty list of strings.")
    seen: set[str] = set()
    for raw in items:
        if not isinstance(raw, str) or not raw.strip() or raw in seen:
            raise ValueError(f"Manifest '{axis_name}' has invalid or duplicate entry: {raw!r}")
        seen.add(raw)
    return tuple(items)


def build_manifest(payload: dict[str, object]) -> ManifestSpec:
    """Validate and construct a ManifestSpec from a parsed dictionary."""
    if not isinstance(payload, dict):
        raise ValueError("Manifest payload must be a JSON object.")
    request_ids = _validate_unique_strings(payload.get("request_ids"), "request_ids")
    config_ids = _validate_unique_strings(payload.get("config_ids"), "config_ids")
    if payload.get("repeat_ids") is not None:
        rep_raw = payload["repeat_ids"]
        if not isinstance(rep_raw, list) or not rep_raw:
            raise ValueError("Manifest 'repeat_ids' must be a non-empty list of integers.")
        seen_rep: set[int] = set()
        for item in rep_raw:
            if isinstance(item, bool) or not isinstance(item, int) or item < 0 or item in seen_rep:
                raise ValueError(f"Invalid or duplicate repeat_id: {item!r}")
            seen_rep.add(item)
        repeat_ids = tuple(rep_raw)
    elif "num_repeats" in payload:
        num_rep = payload["num_repeats"]
        if isinstance(num_rep, bool) or not isinstance(num_rep, int) or num_rep <= 0:
            raise ValueError(f"Manifest 'num_repeats' must be a positive integer: {num_rep!r}")
        repeat_ids = tuple(range(num_rep))
    else:
        raise ValueError("Manifest must specify either 'repeat_ids' or 'num_repeats'.")
    manifest_id = payload.get("manifest_id", "default_grid")
    if not isinstance(manifest_id, str) or not manifest_id.strip():
        raise ValueError("Manifest 'manifest_id' must be a non-empty string.")
    return ManifestSpec(request_ids, config_ids, repeat_ids, manifest_id.strip())


def load_manifest(path: str | pathlib.Path) -> ManifestSpec:
    """Load a Cartesian grid manifest from a JSON file."""
    return build_manifest(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


def _is_real_number(val: object) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _extract_cell_key(record: dict[str, object]) -> tuple[tuple[str, str, int] | None, str | None]:
    req_id, cfg_id, rep_id = record.get("request_id"), record.get("config_id"), record.get("repeat_id")
    if not isinstance(req_id, str) or not req_id.strip() or not isinstance(cfg_id, str) or not cfg_id.strip():
        return None, "Fields 'request_id' and 'config_id' must be non-empty strings."
    if isinstance(rep_id, bool) or not isinstance(rep_id, int):
        return None, "Field 'repeat_id' must be an integer."
    return (req_id, cfg_id, rep_id), None


def _validate_single_record(
    req_id: str, cfg_id: str, rep_id: int, record: dict[str, object], line_number: int
) -> CellAuditResult:
    """Classify a single observed record and collect all structured error codes."""
    errors: list[str] = []
    raw_status = record.get("status", CELL_VALID)
    lowered = raw_status.strip().lower() if isinstance(raw_status, str) else ""
    if lowered in _VALID_STATUS_ALIASES:
        normalized_status = CELL_VALID
    elif lowered in _FAILED_STATUS_ALIASES:
        return CellAuditResult(req_id, cfg_id, rep_id, CELL_FAILED, (), (line_number,))
    elif lowered in _NOT_RUN_STATUS_ALIASES:
        return CellAuditResult(req_id, cfg_id, rep_id, CELL_NOT_RUN, (), (line_number,))
    else:
        errors.append(ERR_INVALID_STATUS)
        normalized_status = CELL_INVALID

    start_raw, end_raw = record.get("start_ts"), record.get("end_ts")
    start_ts = end_ts = None
    if start_raw is None or end_raw is None:
        errors.append(ERR_MISSING_TIMESTAMP)
    elif not _is_real_number(start_raw) or not _is_real_number(end_raw):
        errors.append(ERR_NON_FINITE_TIMESTAMP)
    elif not math.isfinite(float(start_raw)) or not math.isfinite(float(end_raw)):
        errors.append(ERR_NON_FINITE_TIMESTAMP)
    elif float(end_raw) < float(start_raw):
        errors.append(ERR_TIMESTAMP_ORDER)
    else:
        start_ts, end_ts = float(start_raw), float(end_raw)

    lat_raw, eng_raw = record.get("latency_s"), record.get("energy_j")
    latency_s = energy_j = None
    if lat_raw is None:
        errors.append(ERR_MISSING_LATENCY)
    elif not _is_real_number(lat_raw) or not math.isfinite(float(lat_raw)):
        errors.append(ERR_NON_FINITE_LATENCY)
    elif float(lat_raw) <= 0.0:
        errors.append(ERR_NEGATIVE_LATENCY)
    else:
        latency_s = float(lat_raw)

    if eng_raw is None:
        errors.append(ERR_MISSING_ENERGY)
    elif not _is_real_number(eng_raw) or not math.isfinite(float(eng_raw)):
        errors.append(ERR_NON_FINITE_ENERGY)
    elif float(eng_raw) < 0.0:
        errors.append(ERR_NEGATIVE_ENERGY)
    else:
        energy_j = float(eng_raw)

    tok_raw = record.get("output_tokens", 1)
    output_tokens = tok_raw if (not isinstance(tok_raw, bool) and isinstance(tok_raw, int) and tok_raw > 0) else None
    if output_tokens is None:
        errors.append(ERR_INVALID_TOKEN_COUNT)

    text_raw, sha_raw = record.get("output_text"), record.get("output_sha256")
    output_text = output_sha256 = None
    if not isinstance(text_raw, str):
        errors.append(ERR_MISSING_OUTPUT_TEXT)
    else:
        output_text = text_raw
        output_sha256 = hashlib.sha256(text_raw.encode("utf-8")).hexdigest()
        if sha_raw is not None and (not isinstance(sha_raw, str) or sha_raw.strip().lower() != output_sha256):
            errors.append(ERR_OUTPUT_HASH_MISMATCH)

    acc_raw = record.get("passed", record.get("correct"))
    passed = acc_raw if isinstance(acc_raw, bool) else None
    if passed is None:
        errors.append(ERR_INVALID_ACCURACY_FLAG)

    if errors or normalized_status == CELL_INVALID:
        return CellAuditResult(
            req_id, cfg_id, rep_id, CELL_INVALID, tuple(errors), (line_number,),
            latency_s, energy_j, output_tokens, None, output_text, output_sha256, passed, start_ts, end_ts,
        )
    assert energy_j is not None and output_tokens is not None
    return CellAuditResult(
        req_id, cfg_id, rep_id, CELL_VALID, (), (line_number,),
        latency_s, energy_j, output_tokens, energy_j / float(output_tokens),
        output_text, output_sha256, passed, start_ts, end_ts,
    )


def audit_jsonl_lines(
    manifest: ManifestSpec, lines: list[str], allow_not_run: bool = False
) -> CompletenessReport:
    """Audit raw JSONL lines against the expected Cartesian manifest grid."""
    expected_set = manifest.expected_cell_set()
    observed_by_cell: dict[tuple[str, str, int], list[tuple[int, dict[str, object]]]] = (
        collections.defaultdict(list)
    )
    violations: list[GridViolation] = []
    observed_records = duplicate_count = unscheduled_count = malformed_line_count = 0

    for idx, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            malformed_line_count += 1
            violations.append(GridViolation(idx, ERR_MALFORMED_JSON, f"Invalid JSON syntax: {exc.msg}"))
            continue
        if not isinstance(parsed, dict):
            malformed_line_count += 1
            violations.append(GridViolation(idx, ERR_MALFORMED_JSON, "Record must be a JSON object."))
            continue

        observed_records += 1
        cell_key, key_err = _extract_cell_key(parsed)
        if cell_key is None:
            malformed_line_count += 1
            violations.append(GridViolation(idx, ERR_INVALID_KEY_TYPES, key_err or "Invalid key types."))
            continue
        if cell_key not in expected_set:
            unscheduled_count += 1
            violations.append(GridViolation(idx, ERR_UNSCHEDULED_CELL, f"Unscheduled cell {cell_key!r}.", cell_key))
            continue
        if cell_key in observed_by_cell:
            duplicate_count += 1
            first_line = observed_by_cell[cell_key][0][0]
            violations.append(GridViolation(idx, ERR_DUPLICATE_CELL, f"Duplicate {cell_key!r} (seen line {first_line}).", cell_key))
        observed_by_cell[cell_key].append((idx, parsed))

    cell_results: list[CellAuditResult] = []
    cells_by_group: dict[tuple[str, str], list[CellAuditResult]] = collections.defaultdict(list)
    state_totals: collections.Counter[str] = collections.Counter()

    for req_id, cfg_id, rep_id in manifest.expected_cells():
        entries = observed_by_cell.get((req_id, cfg_id, rep_id))
        if not entries:
            res = CellAuditResult(req_id, cfg_id, rep_id, CELL_MISSING, (ERR_MISSING_CELL,), ())
        elif len(entries) > 1:
            line_nums = tuple(line_no for line_no, _ in entries)
            first_audit = _validate_single_record(req_id, cfg_id, rep_id, entries[0][1], entries[0][0])
            merged = (ERR_DUPLICATE_CELL,) + tuple(c for c in first_audit.error_codes if c != ERR_DUPLICATE_CELL)
            res = CellAuditResult(req_id, cfg_id, rep_id, CELL_INVALID, merged, line_nums)
        else:
            res = _validate_single_record(req_id, cfg_id, rep_id, entries[0][1], entries[0][0])
        cell_results.append(res)
        cells_by_group[(req_id, cfg_id)].append(res)
        state_totals[res.state] += 1

    group_audits: list[RepeatGroupAudit] = []
    nondeterministic_groups: list[tuple[str, str]] = []
    groups_by_config: dict[str, list[RepeatGroupAudit]] = collections.defaultdict(list)

    for req_id in manifest.request_ids:
        for cfg_id in manifest.config_ids:
            group_cells = cells_by_group[(req_id, cfg_id)]
            g_counts = {s: 0 for s in (CELL_VALID, CELL_FAILED, CELL_NOT_RUN, CELL_INVALID, CELL_MISSING)}
            valid_in_group: list[CellAuditResult] = []
            for cell in group_cells:
                g_counts[cell.state] += 1
                if cell.state == CELL_VALID:
                    valid_in_group.append(cell)

            sha_seen: list[str] = []
            texts_seen: set[str] = set()
            acc_seen: list[bool] = []
            latencies, energies, epts = [], [], []
            for cell in valid_in_group:
                assert cell.output_sha256 is not None and cell.output_text is not None and cell.passed is not None
                assert cell.latency_s is not None and cell.energy_j is not None and cell.energy_per_token_j is not None
                if cell.output_sha256 not in sha_seen:
                    sha_seen.append(cell.output_sha256)
                texts_seen.add(cell.output_text)
                if cell.passed not in acc_seen:
                    acc_seen.append(cell.passed)
                latencies.append(cell.latency_s)
                energies.append(cell.energy_j)
                epts.append(cell.energy_per_token_j)

            sha_eq, text_eq, acc_eq = len(sha_seen) <= 1, len(texts_seen) <= 1, len(acc_seen) <= 1
            deterministic = sha_eq and text_eq and acc_eq
            if not deterministic:
                nondeterministic_groups.append((req_id, cfg_id))
            pass_rate = sum(1 for c in valid_in_group if c.passed) / float(len(valid_in_group)) if valid_in_group else None
            grp_audit = RepeatGroupAudit(
                req_id, cfg_id, len(manifest.repeat_ids), len(valid_in_group), g_counts,
                sha_eq, text_eq, acc_eq, deterministic, tuple(sha_seen), len(texts_seen),
                tuple(acc_seen), pass_rate, compute_sample_stats(latencies),
                compute_sample_stats(energies), compute_sample_stats(epts),
            )
            group_audits.append(grp_audit)
            groups_by_config[cfg_id].append(grp_audit)

    config_summaries: list[ConfigAuditSummary] = []
    for cfg_id in manifest.config_ids:
        cfg_groups = groups_by_config[cfg_id]
        cfg_cells = [c for c in cell_results if c.config_id == cfg_id]
        c_counts: collections.Counter[str] = collections.Counter(c.state for c in cfg_cells)
        lat_cvs = [g.latency_stats.cv for g in cfg_groups if g.latency_stats.cv is not None]
        eng_cvs = [g.energy_stats.cv for g in cfg_groups if g.energy_stats.cv is not None]
        ept_cvs = [g.energy_per_token_stats.cv for g in cfg_groups if g.energy_per_token_stats.cv is not None]
        all_lat = [c.latency_s for c in cfg_cells if c.state == CELL_VALID and c.latency_s is not None]
        all_eng = [c.energy_j for c in cfg_cells if c.state == CELL_VALID and c.energy_j is not None]
        all_ept = [c.energy_per_token_j for c in cfg_cells if c.state == CELL_VALID and c.energy_per_token_j is not None]
        config_summaries.append(ConfigAuditSummary(
            cfg_id, len(cfg_cells), c_counts[CELL_VALID], c_counts[CELL_FAILED], c_counts[CELL_NOT_RUN],
            c_counts[CELL_INVALID], c_counts[CELL_MISSING], sum(1 for g in cfg_groups if g.is_deterministic),
            len(cfg_groups), statistics.fmean(lat_cvs) if lat_cvs else None,
            statistics.fmean(eng_cvs) if eng_cvs else None, statistics.fmean(ept_cvs) if ept_cvs else None,
            compute_sample_stats(all_lat), compute_sample_stats(all_eng), compute_sample_stats(all_ept),
        ))

    expected_total = manifest.total_cells()
    valid_total, not_run_total = state_totals[CELL_VALID], state_totals[CELL_NOT_RUN]
    is_complete = (
        (valid_total + (not_run_total if allow_not_run else 0)) == expected_total
        and duplicate_count == 0 and unscheduled_count == 0 and malformed_line_count == 0
    )
    return CompletenessReport(
        manifest.manifest_id, expected_total, observed_records, valid_total, state_totals[CELL_FAILED],
        not_run_total, state_totals[CELL_INVALID], state_totals[CELL_MISSING], duplicate_count,
        unscheduled_count, malformed_line_count, is_complete, len(nondeterministic_groups) == 0,
        tuple(nondeterministic_groups), tuple(violations), tuple(cell_results), tuple(group_audits), tuple(config_summaries),
    )


def audit_jsonl_file(
    manifest: ManifestSpec, records_path: str | pathlib.Path, allow_not_run: bool = False
) -> CompletenessReport:
    """Read a JSONL measurement file and run the Cartesian completeness audit."""
    lines = pathlib.Path(records_path).read_text(encoding="utf-8").splitlines()
    return audit_jsonl_lines(manifest, lines, allow_not_run=allow_not_run)


def write_group_csv(report: CompletenessReport, csv_path: str | pathlib.Path) -> None:
    """Export per-(request_id, config_id) repeat statistics exclusively to CSV."""
    fieldnames = [
        "request_id", "config_id", "expected_repeats", "valid_repeats", "is_deterministic",
        "sha256_equivalent", "exact_text_equivalent", "accuracy_equivalent", "pass_rate",
        "latency_mean_s", "latency_stddev_s", "latency_cv", "energy_mean_j", "energy_stddev_j",
        "energy_cv", "ept_mean_j", "ept_stddev_j", "ept_cv",
    ]
    with open(csv_path, "x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for g in report.group_audits:
            writer.writerow({
                "request_id": g.request_id, "config_id": g.config_id,
                "expected_repeats": g.expected_repeats, "valid_repeats": g.valid_repeats,
                "is_deterministic": g.is_deterministic, "sha256_equivalent": g.sha256_equivalent,
                "exact_text_equivalent": g.exact_text_equivalent,
                "accuracy_equivalent": g.accuracy_equivalent, "pass_rate": g.pass_rate,
                "latency_mean_s": g.latency_stats.mean, "latency_stddev_s": g.latency_stats.stddev,
                "latency_cv": g.latency_stats.cv, "energy_mean_j": g.energy_stats.mean,
                "energy_stddev_j": g.energy_stats.stddev, "energy_cv": g.energy_stats.cv,
                "ept_mean_j": g.energy_per_token_stats.mean,
                "ept_stddev_j": g.energy_per_token_stats.stddev,
                "ept_cv": g.energy_per_token_stats.cv,
            })


def format_markdown_summary(report: CompletenessReport) -> str:
    """Format a compact Markdown audit table across configurations."""
    lines = [
        f"# Completeness Audit: {report.manifest_id}", "",
        f"- Expected cells: {report.expected_cells}, Valid cells: {report.valid_cells}",
        f"- Missing / Invalid / Failed / Not-run: {report.missing_cells} / "
        f"{report.invalid_cells} / {report.failed_cells} / {report.not_run_cells}",
        f"- Duplicates / Unscheduled / Malformed: {report.duplicate_count} / "
        f"{report.unscheduled_count} / {report.malformed_line_count}",
        f"- Grid complete: {report.is_complete}, All repeats deterministic: {report.all_repeats_deterministic}",
        "", "| Config | Valid/Expected | Det. Groups | Mean Lat CV | Mean Eng CV | Mean EPT CV |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cfg in report.config_summaries:
        lat_cv = f"{cfg.mean_repeat_latency_cv:.4f}" if cfg.mean_repeat_latency_cv is not None else "n/a"
        eng_cv = f"{cfg.mean_repeat_energy_cv:.4f}" if cfg.mean_repeat_energy_cv is not None else "n/a"
        ept_cv = f"{cfg.mean_repeat_ept_cv:.4f}" if cfg.mean_repeat_ept_cv is not None else "n/a"
        lines.append(
            f"| {cfg.config_id} | {cfg.valid_cells}/{cfg.expected_cells} | "
            f"{cfg.deterministic_groups}/{cfg.total_groups} | {lat_cv} | {eng_cv} | {ept_cv} |"
        )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Cartesian (request_id, config_id, repeat_id) completeness and repeat determinism."
    )
    parser.add_argument("--manifest", required=True, help="Path to expected grid JSON manifest.")
    parser.add_argument("--records", required=True, help="Path to observed JSONL measurement log.")
    parser.add_argument("--output", required=True, help="Output JSON path (opened exclusively).")
    parser.add_argument("--csv-output", default=None, help="Optional CSV group summary path (exclusive).")
    parser.add_argument("--markdown-output", default=None, help="Optional Markdown path (exclusive).")
    parser.add_argument("--strict-determinism", action="store_true", help="Fail on across-repeat drift.")
    parser.add_argument("--allow-not-run", action="store_true", help="Permit explicit not_run cells.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint returning 0 on audit pass, 2 on audit failure, 1 on runtime/IO error."""
    args = _build_arg_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        report = audit_jsonl_file(manifest, args.records, allow_not_run=args.allow_not_run)
        with open(args.output, "x", encoding="utf-8") as out_handle:
            out_handle.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
        if args.csv_output is not None:
            write_group_csv(report, args.csv_output)
        if args.markdown_output is not None:
            with open(args.markdown_output, "x", encoding="utf-8") as md_handle:
                md_handle.write(format_markdown_summary(report))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"completeness_audit error: {exc}\n")
        return 1
    if not report.is_complete or (args.strict_determinism and not report.all_repeats_deterministic):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
