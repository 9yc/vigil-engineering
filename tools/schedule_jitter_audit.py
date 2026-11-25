#!/usr/bin/env python3
"""Audit periodic sampler dispatch logs for timer jitter, overruns and clock skew.

Parses scheduler CSV logs containing monotonic, boottime and realtime nanosecond
stamps alongside optional `# key=value` kernel timer-slack metadata comments,
computes dispatch lateness, probe execution durations, inter-sample gaps, kernel
suspend skew (`CLOCK_BOOTTIME` minus `CLOCK_MONOTONIC` deltas) and NTP wall-clock
jumps, and verifies strict bracketing of target measurement intervals.
"""

import argparse
import csv
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path


REQUIRED_COLUMNS = (
    "scheduled_ns",
    "dispatch_ns",
    "command_end_ns",
    "command_rc",
    "dispatch_boottime_ns",
    "dispatch_realtime_ns",
)


@dataclass(frozen=True)
class DispatchTick:
    index: int
    scheduled_ns: int
    dispatch_ns: int
    command_end_ns: int
    command_rc: int
    boottime_ns: int
    realtime_ns: int

    @property
    def lateness_ns(self) -> int:
        return self.dispatch_ns - self.scheduled_ns

    @property
    def duration_ns(self) -> int:
        return self.command_end_ns - self.dispatch_ns


def _parse_int_strict(raw: str, label: str) -> int:
    if raw is None:
        raise ValueError(f"missing integer field {label}")
    token = raw.strip()
    if not token:
        raise ValueError(f"empty integer field {label}")
    try:
        return int(token, 10)
    except ValueError as exc:
        raise ValueError(f"invalid integer for {label}: {raw!r}") from exc


def quantile_linear(values, q: float) -> float:
    """Compute Type-7 linear-interpolated quantile `q` in `[0.0, 1.0]`."""
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q):
        raise ValueError("quantile q must be a finite number")
    if not (0.0 <= q <= 1.0):
        raise ValueError("quantile q must lie in [0.0, 1.0]")
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("cannot compute quantile of empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


def _summarize_ns_as_ms(values_ns) -> dict:
    items_ms = [v / 1_000_000.0 for v in values_ns]
    if not items_ms:
        raise ValueError("cannot summarize empty series")
    return {
        "count": len(items_ms),
        "min_ms": min(items_ms),
        "p50_ms": quantile_linear(items_ms, 0.50),
        "p90_ms": quantile_linear(items_ms, 0.90),
        "p95_ms": quantile_linear(items_ms, 0.95),
        "p99_ms": quantile_linear(items_ms, 0.99),
        "max_ms": max(items_ms),
        "mean_ms": math.fsum(items_ms) / len(items_ms),
    }


def parse_scheduler_log(text: str) -> tuple[dict, list[DispatchTick]]:
    """Parse optional `# k=v` header comments and strict CSV dispatch rows."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("scheduler log text must be a non-empty string")

    metadata: dict[str, int | str] = {}
    csv_lines: list[str] = []
    seen_csv_header = False

    for line_no, raw_line in enumerate(text.splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if seen_csv_header:
                raise ValueError(f"comment line {line_no} appears after CSV header")
            body = stripped[1:].strip()
            for token in body.split():
                if "=" not in token:
                    raise ValueError(f"malformed metadata token at line {line_no}: {token!r}")
                key, _, val = token.partition("=")
                key = key.strip()
                val = val.strip()
                if not key or not val:
                    raise ValueError(f"empty metadata key/value at line {line_no}")
                try:
                    metadata[key] = int(val, 10)
                except ValueError:
                    metadata[key] = val
            continue
        seen_csv_header = True
        csv_lines.append(raw_line)

    if not csv_lines:
        raise ValueError("scheduler log contains no CSV content")

    reader = csv.DictReader(io.StringIO("\n".join(csv_lines) + "\n"), strict=True)
    if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED_COLUMNS:
        raise ValueError(
            f"expected exact columns {','.join(REQUIRED_COLUMNS)}, "
            f"got {reader.fieldnames!r}"
        )

    ticks: list[DispatchTick] = []
    for idx, row in enumerate(reader):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"malformed field count at row {idx}")
        sched = _parse_int_strict(row["scheduled_ns"], "scheduled_ns")
        disp = _parse_int_strict(row["dispatch_ns"], "dispatch_ns")
        end = _parse_int_strict(row["command_end_ns"], "command_end_ns")
        rc = _parse_int_strict(row["command_rc"], "command_rc")
        boot = _parse_int_strict(row["dispatch_boottime_ns"], "dispatch_boottime_ns")
        real = _parse_int_strict(row["dispatch_realtime_ns"], "dispatch_realtime_ns")

        if sched < 0 or disp < 0 or end < 0 or boot < 0 or real <= 0:
            raise ValueError(f"negative or invalid clock value at row {idx}")
        if disp < sched:
            raise ValueError(f"dispatch_ns precedes scheduled_ns at row {idx}")
        if end < disp:
            raise ValueError(f"command_end_ns precedes dispatch_ns at row {idx}")
        if ticks:
            prev = ticks[-1]
            if sched < prev.scheduled_ns:
                raise ValueError(f"non-monotonic scheduled_ns at row {idx}")
            if disp <= prev.dispatch_ns:
                raise ValueError(f"non-increasing dispatch_ns at row {idx}")
            if boot < prev.boottime_ns:
                raise ValueError(f"non-monotonic dispatch_boottime_ns at row {idx}")

        ticks.append(
            DispatchTick(
                index=idx,
                scheduled_ns=sched,
                dispatch_ns=disp,
                command_end_ns=end,
                command_rc=rc,
                boottime_ns=boot,
                realtime_ns=real,
            )
        )

    if not ticks:
        raise ValueError("scheduler log contains header but zero dispatch rows")
    return metadata, ticks


def audit_scheduler_ticks(
    ticks: list[DispatchTick],
    *,
    nominal_period_ms: float | None = None,
    max_gap_ms: float | None = None,
    suspend_threshold_ms: float = 5.0,
    ntp_jump_threshold_ms: float = 20.0,
    metadata: dict | None = None,
) -> dict:
    """Compute timing jitter, overrun, suspend-skew and health diagnostics."""
    if not ticks:
        raise ValueError("ticks must be non-empty")
    if suspend_threshold_ms <= 0 or not math.isfinite(suspend_threshold_ms):
        raise ValueError("suspend_threshold_ms must be finite and positive")
    if ntp_jump_threshold_ms <= 0 or not math.isfinite(ntp_jump_threshold_ms):
        raise ValueError("ntp_jump_threshold_ms must be finite and positive")

    period_ns: int | None = None
    if nominal_period_ms is not None:
        if isinstance(nominal_period_ms, bool) or not math.isfinite(nominal_period_ms) or nominal_period_ms <= 0:
            raise ValueError("nominal_period_ms must be finite and positive")
        period_ns = int(round(nominal_period_ms * 1_000_000.0))
    elif len(ticks) >= 2:
        sched_steps = [
            b.scheduled_ns - a.scheduled_ns
            for a, b in zip(ticks, ticks[1:])
            if b.scheduled_ns > a.scheduled_ns
        ]
        if sched_steps:
            period_ns = int(round(quantile_linear(sched_steps, 0.50)))

    lateness_ns = [t.lateness_ns for t in ticks]
    durations_ns = [t.duration_ns for t in ticks]
    nonzero_rc_indices = [t.index for t in ticks if t.command_rc != 0]

    overrun_indices: list[int] = []
    if period_ns is not None and period_ns > 0:
        overrun_indices = [
            t.index for t in ticks if (t.command_end_ns - t.scheduled_ns) > period_ns
        ]

    dispatch_gaps_ns: list[int] = []
    suspend_skews_ns: list[int] = []
    suspend_episodes: list[dict] = []
    ntp_jumps: list[dict] = []
    suspend_thresh_ns = int(round(suspend_threshold_ms * 1_000_000.0))
    ntp_thresh_ns = int(round(ntp_jump_threshold_ms * 1_000_000.0))

    for prev, cur in zip(ticks, ticks[1:]):
        mono_dt = cur.dispatch_ns - prev.dispatch_ns
        boot_dt = cur.boottime_ns - prev.boottime_ns
        real_dt = cur.realtime_ns - prev.realtime_ns
        dispatch_gaps_ns.append(mono_dt)

        # CLOCK_BOOTTIME advances during suspend whereas CLOCK_MONOTONIC freezes.
        suspend_delta = max(0, boot_dt - mono_dt)
        suspend_skews_ns.append(suspend_delta)
        if suspend_delta >= suspend_thresh_ns:
            suspend_episodes.append({
                "from_index": prev.index,
                "to_index": cur.index,
                "suspend_ms": suspend_delta / 1_000_000.0,
            })

        wall_drift = real_dt - boot_dt
        if abs(wall_drift) >= ntp_thresh_ns:
            ntp_jumps.append({
                "from_index": prev.index,
                "to_index": cur.index,
                "wall_jump_ms": wall_drift / 1_000_000.0,
            })

    gap_violation = False
    if max_gap_ms is not None:
        if isinstance(max_gap_ms, bool) or not math.isfinite(max_gap_ms) or max_gap_ms <= 0:
            raise ValueError("max_gap_ms must be finite and positive")
        if dispatch_gaps_ns and (max(dispatch_gaps_ns) / 1_000_000.0) > max_gap_ms:
            gap_violation = True

    span_ns = ticks[-1].command_end_ns - ticks[0].dispatch_ns
    busy_ns = sum(durations_ns)
    passed = (
        not nonzero_rc_indices
        and not suspend_episodes
        and not gap_violation
    )

    return {
        "passed": passed,
        "tick_count": len(ticks),
        "span_ms": span_ns / 1_000_000.0,
        "sampler_duty_fraction": (busy_ns / span_ns) if span_ns > 0 else 0.0,
        "inferred_period_ms": (period_ns / 1_000_000.0) if period_ns is not None else None,
        "nonzero_rc_count": len(nonzero_rc_indices),
        "nonzero_rc_indices": nonzero_rc_indices,
        "slot_overrun_count": len(overrun_indices),
        "slot_overrun_indices": overrun_indices,
        "gap_violation": gap_violation,
        "lateness": _summarize_ns_as_ms(lateness_ns),
        "probe_duration": _summarize_ns_as_ms(durations_ns),
        "dispatch_gap": _summarize_ns_as_ms(dispatch_gaps_ns) if dispatch_gaps_ns else None,
        "max_suspend_skew_ms": (max(suspend_skews_ns) / 1_000_000.0) if suspend_skews_ns else 0.0,
        "suspend_episodes": suspend_episodes,
        "ntp_jumps": ntp_jumps,
        "metadata": dict(metadata) if metadata else {},
    }


def verify_interval_bracketing(
    ticks: list[DispatchTick],
    start_ns: int,
    end_ns: int,
    *,
    max_gap_ms: float,
) -> dict:
    """Verify `[start_ns, end_ns]` is bracketed by clean ticks within `max_gap_ms`."""
    if type(start_ns) is not int or type(end_ns) is not int or start_ns < 0 or end_ns <= start_ns:
        raise ValueError("start_ns and end_ns must be nonnegative integers with start_ns < end_ns")
    if isinstance(max_gap_ms, bool) or not math.isfinite(max_gap_ms) or max_gap_ms <= 0:
        raise ValueError("max_gap_ms must be finite and positive")
    if len(ticks) < 2:
        raise ValueError("at least two ticks are required to bracket an interval")

    max_gap_ns = int(round(max_gap_ms * 1_000_000.0))
    midpoints = [(t.dispatch_ns + t.command_end_ns) // 2 for t in ticks]

    left_candidates = [i for i, m in enumerate(midpoints) if m <= start_ns]
    right_candidates = [i for i, m in enumerate(midpoints) if m >= end_ns]
    issues: list[str] = []

    if not left_candidates:
        issues.append("missing_leading_bracket_tick")
        left_idx = 0
    else:
        left_idx = left_candidates[-1]
        if start_ns - midpoints[left_idx] > max_gap_ns:
            issues.append("leading_bracket_exceeds_max_gap")

    if not right_candidates:
        issues.append("missing_trailing_bracket_tick")
        right_idx = len(ticks) - 1
    else:
        right_idx = right_candidates[0]
        if midpoints[right_idx] - end_ns > max_gap_ns:
            issues.append("trailing_bracket_exceeds_max_gap")

    window_slice = ticks[left_idx : right_idx + 1]
    if any(t.command_rc != 0 for t in window_slice):
        issues.append("probe_command_failed_within_bracket")

    slice_mids = midpoints[left_idx : right_idx + 1]
    max_internal_gap_ns = max(
        (b - a for a, b in zip(slice_mids, slice_mids[1:])),
        default=0,
    )
    if max_internal_gap_ns > max_gap_ns:
        issues.append("internal_gap_exceeds_max_gap")

    for prev, cur in zip(window_slice, window_slice[1:]):
        mono_dt = cur.dispatch_ns - prev.dispatch_ns
        boot_dt = cur.boottime_ns - prev.boottime_ns
        if boot_dt - mono_dt >= 5_000_000:
            issues.append("suspend_detected_within_bracket")
            break

    return {
        "admissible": len(issues) == 0,
        "issues": issues,
        "left_tick_index": left_idx,
        "right_tick_index": right_idx,
        "covered_ticks": len(window_slice),
        "max_internal_gap_ms": max_internal_gap_ns / 1_000_000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Scheduler CSV log path")
    parser.add_argument("--nominal-period-ms", type=float)
    parser.add_argument("--max-gap-ms", type=float)
    parser.add_argument("--window-start-ns", type=int)
    parser.add_argument("--window-end-ns", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        raw = args.input.read_text(encoding="utf-8")
        meta, ticks = parse_scheduler_log(raw)
        report = audit_scheduler_ticks(
            ticks,
            nominal_period_ms=args.nominal_period_ms,
            max_gap_ms=args.max_gap_ms,
            metadata=meta,
        )
        if args.window_start_ns is not None or args.window_end_ns is not None:
            if args.window_start_ns is None or args.window_end_ns is None or args.max_gap_ms is None:
                raise ValueError(
                    "--window-start-ns, --window-end-ns and --max-gap-ms must be set together"
                )
            report["window_bracket"] = verify_interval_bracketing(
                ticks,
                args.window_start_ns,
                args.window_end_ns,
                max_gap_ms=args.max_gap_ms,
            )
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        else:
            print(encoded, end="")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"schedule_jitter_audit error: {exc}\n")


if __name__ == "__main__":
    main()
