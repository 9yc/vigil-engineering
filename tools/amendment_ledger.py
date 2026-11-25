#!/usr/bin/env python3
"""Hash-chained execution scope and rejected-attempt accounting ledger.

Locks an initial benchmark scope (calibration_ids, evaluation_ids, candidates,
max_invocations, SHA-256 digest), records ineligible hardware attempts (such as
battery drop below threshold, missing end timestamp, or thermal abort) with
explicit verification that rejected attempts carry eligible_measurement=False
and energy_j=None, and audits hash-linked amendment chains (v0 -> v1 -> v2)
ensuring frozen evaluation IDs and candidate sets never mutate while conserving
total physical invocations (accepted + pending + rejected == max_invocations).
"""

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import pathlib
import sys

DISPOSITION_ACCEPTED = "accepted"
DISPOSITION_REJECTED = "rejected"
DISPOSITION_PENDING = "pending"

REASON_BATTERY_DROP = "BATTERY_DROP_BELOW_THRESHOLD"
REASON_MISSING_END_TS = "MISSING_END_TIMESTAMP"
REASON_THERMAL_ABORT = "THERMAL_ABORT"
REASON_HARNESS_TIMEOUT = "HARNESS_TIMEOUT"
REASON_TRANSPORT_DISCONNECT = "TRANSPORT_DISCONNECT"

ERR_EMPTY_AMENDMENT_CHAIN = "EMPTY_AMENDMENT_CHAIN"
ERR_INVALID_VERSION_SEQUENCE = "INVALID_VERSION_SEQUENCE"
ERR_GENESIS_HAS_PARENT = "GENESIS_HAS_PARENT"
ERR_BROKEN_HASH_LINK = "BROKEN_HASH_LINK"
ERR_SCOPE_DIGEST_TAMPERED = "SCOPE_DIGEST_TAMPERED"
ERR_OVERLAPPING_SPLIT_IDS = "OVERLAPPING_SPLIT_IDS"
ERR_EVALUATION_SCOPE_MUTATED = "EVALUATION_SCOPE_MUTATED"
ERR_CANDIDATE_SET_MUTATED = "CANDIDATE_SET_MUTATED"
ERR_CALIBRATION_SCOPE_MUTATED = "CALIBRATION_SCOPE_MUTATED"
ERR_INVOCATION_CAP_SHRUNK = "INVOCATION_CAP_SHRUNK"
ERR_DUPLICATE_ATTEMPT_ID = "DUPLICATE_ATTEMPT_ID"
ERR_UNKNOWN_AMENDMENT_VERSION = "UNKNOWN_AMENDMENT_VERSION"
ERR_OUT_OF_SCOPE_ITEM = "OUT_OF_SCOPE_ITEM"
ERR_OUT_OF_SCOPE_CANDIDATE = "OUT_OF_SCOPE_CANDIDATE"
ERR_INVALID_DISPOSITION = "INVALID_DISPOSITION"
ERR_REJECTED_MARKED_ELIGIBLE = "REJECTED_MARKED_ELIGIBLE"
ERR_REJECTED_HAS_ENERGY = "REJECTED_HAS_ENERGY"
ERR_MISSING_REJECTION_REASON = "MISSING_REJECTION_REASON"
ERR_PENDING_HAS_MEASUREMENT = "PENDING_HAS_MEASUREMENT"
ERR_INVALID_ACCEPTED_RECORD = "INVALID_ACCEPTED_RECORD"
ERR_RECORD_DIGEST_TAMPERED = "RECORD_DIGEST_TAMPERED"
ERR_AMENDMENT_CAP_EXCEEDED = "AMENDMENT_CAP_EXCEEDED"
ERR_INVOCATION_BUDGET_MISMATCH = "INVOCATION_BUDGET_MISMATCH"


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_id_tuple(values: list[str] | tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"Field '{field_name}' must be a non-empty list/tuple of strings.")
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in values:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"Field '{field_name}' contains invalid identifier: {raw!r}")
        if raw in seen:
            raise ValueError(f"Field '{field_name}' contains duplicate identifier: {raw!r}")
        seen.add(raw)
        cleaned.append(raw)
    return tuple(cleaned)


@dataclasses.dataclass(frozen=True)
class ScopeEntry:
    """Single hash-linked scope declaration or amendment (v0, v1, v2, ...)."""

    ledger_id: str
    version: str
    parent_sha256: str | None
    calibration_ids: tuple[str, ...]
    evaluation_ids: tuple[str, ...]
    candidates: tuple[str, ...]
    max_invocations: int
    reason: str
    entry_sha256: str

    def canonical_body(self) -> dict[str, object]:
        return {
            "calibration_ids": list(self.calibration_ids),
            "candidates": list(self.candidates),
            "evaluation_ids": list(self.evaluation_ids),
            "ledger_id": self.ledger_id,
            "max_invocations": self.max_invocations,
            "parent_sha256": self.parent_sha256,
            "reason": self.reason,
            "version": self.version,
        }

    def compute_sha256(self) -> str:
        return _canonical_sha256(self.canonical_body())

    def to_dict(self) -> dict[str, object]:
        body = self.canonical_body()
        body["entry_sha256"] = self.entry_sha256
        return body


@dataclasses.dataclass(frozen=True)
class InvocationRecord:
    """Physical invocation accounting record for accepted, rejected, or pending slots."""

    attempt_id: str
    amendment_version: str
    item_id: str
    candidate_id: str
    disposition: str
    eligible_measurement: bool
    energy_j: float | None = None
    latency_s: float | None = None
    rejection_reason: str | None = None
    battery_pct_start: float | None = None
    battery_pct_end: float | None = None
    temperature_c: float | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    record_sha256: str = ""

    def canonical_body(self) -> dict[str, object]:
        return {
            "amendment_version": self.amendment_version, "attempt_id": self.attempt_id,
            "battery_pct_end": self.battery_pct_end, "battery_pct_start": self.battery_pct_start,
            "candidate_id": self.candidate_id, "disposition": self.disposition,
            "eligible_measurement": self.eligible_measurement, "end_ts": self.end_ts,
            "energy_j": self.energy_j, "item_id": self.item_id, "latency_s": self.latency_s,
            "rejection_reason": self.rejection_reason, "start_ts": self.start_ts,
            "temperature_c": self.temperature_c,
        }

    def compute_sha256(self) -> str:
        return _canonical_sha256(self.canonical_body())

    def to_dict(self) -> dict[str, object]:
        body = self.canonical_body()
        body["record_sha256"] = self.record_sha256
        return body


@dataclasses.dataclass(frozen=True)
class LedgerDefect:
    """Structured integrity violation raised during ledger verification."""

    error_code: str
    message: str
    subject_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {"error_code": self.error_code, "message": self.message, "subject_id": self.subject_id}


@dataclasses.dataclass(frozen=True)
class LedgerAuditReport:
    """Verification report covering hash chain, frozen scope, and budget conservation."""

    ledger_id: str
    chain_length: int
    terminal_version: str | None
    terminal_sha256: str | None
    max_invocations: int
    accepted_count: int
    pending_count: int
    rejected_count: int
    total_accounted: int
    budget_conserved: bool
    rejected_zero_contamination_verified: bool
    frozen_scope_preserved: bool
    hash_chain_intact: bool
    is_valid: bool
    rejection_breakdown: dict[str, int]
    defects: tuple[LedgerDefect, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ledger_id": self.ledger_id, "chain_length": self.chain_length,
            "terminal_version": self.terminal_version, "terminal_sha256": self.terminal_sha256,
            "max_invocations": self.max_invocations, "accepted_count": self.accepted_count,
            "pending_count": self.pending_count, "rejected_count": self.rejected_count,
            "total_accounted": self.total_accounted, "budget_conserved": self.budget_conserved,
            "rejected_zero_contamination_verified": self.rejected_zero_contamination_verified,
            "frozen_scope_preserved": self.frozen_scope_preserved,
            "hash_chain_intact": self.hash_chain_intact, "is_valid": self.is_valid,
            "rejection_breakdown": dict(self.rejection_breakdown),
            "defects": [d.to_dict() for d in self.defects],
        }


def create_genesis_scope(
    ledger_id: str,
    calibration_ids: list[str] | tuple[str, ...],
    evaluation_ids: list[str] | tuple[str, ...],
    candidates: list[str] | tuple[str, ...],
    max_invocations: int,
    reason: str = "Initial locked execution scope",
) -> ScopeEntry:
    """Lock initial v0 benchmark scope and compute its genesis SHA-256 digest."""
    if not isinstance(ledger_id, str) or not ledger_id.strip():
        raise ValueError("ledger_id must be a non-empty string.")
    cal_tuple = _validate_id_tuple(calibration_ids, "calibration_ids")
    eval_tuple = _validate_id_tuple(evaluation_ids, "evaluation_ids")
    cand_tuple = _validate_id_tuple(candidates, "candidates")
    overlap = set(cal_tuple) & set(eval_tuple)
    if overlap:
        raise ValueError(f"calibration_ids and evaluation_ids must be disjoint; overlap={sorted(overlap)}")
    if isinstance(max_invocations, bool) or not isinstance(max_invocations, int) or max_invocations < 1:
        raise ValueError(f"max_invocations must be positive int, got {max_invocations!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string.")
    draft = ScopeEntry(
        ledger_id.strip(), "v0", None, cal_tuple, eval_tuple, cand_tuple, max_invocations, reason.strip(), ""
    )
    return dataclasses.replace(draft, entry_sha256=draft.compute_sha256())


def create_amendment(
    previous: ScopeEntry, max_invocations: int, reason: str, version: str | None = None
) -> ScopeEntry:
    """Append a hash-linked amendment preserving frozen calibration/evaluation/candidate sets."""
    if not previous.version.startswith("v") or not previous.version[1:].isdigit():
        raise ValueError(f"Previous version has invalid format: {previous.version!r}")
    expected_version = f"v{int(previous.version[1:]) + 1}"
    actual_version = expected_version if version is None else version
    if actual_version != expected_version:
        raise ValueError(f"Expected sequential amendment version {expected_version!r}, got {actual_version!r}")
    if isinstance(max_invocations, bool) or not isinstance(max_invocations, int) or max_invocations < previous.max_invocations:
        raise ValueError(f"Amended max_invocations ({max_invocations}) < previous ({previous.max_invocations}).")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Amendment reason must be non-empty.")
    draft = ScopeEntry(
        previous.ledger_id, actual_version, previous.entry_sha256,
        previous.calibration_ids, previous.evaluation_ids, previous.candidates,
        max_invocations, reason.strip(), "",
    )
    return dataclasses.replace(draft, entry_sha256=draft.compute_sha256())


def build_invocation_record(
    attempt_id: str, amendment_version: str, item_id: str, candidate_id: str,
    disposition: str, energy_j: float | None = None, latency_s: float | None = None,
    rejection_reason: str | None = None, battery_pct_start: float | None = None,
    battery_pct_end: float | None = None, temperature_c: float | None = None,
    start_ts: float | None = None, end_ts: float | None = None,
) -> InvocationRecord:
    """Create a signed InvocationRecord enforcing disposition-specific eligibility rules."""
    if disposition == DISPOSITION_ACCEPTED:
        eligible = True
        if energy_j is None or not math.isfinite(energy_j) or energy_j < 0.0:
            raise ValueError("Accepted record requires finite non-negative energy_j.")
        if latency_s is None or not math.isfinite(latency_s) or latency_s <= 0.0:
            raise ValueError("Accepted record requires finite positive latency_s.")
        if start_ts is None or end_ts is None or end_ts < start_ts:
            raise ValueError("Accepted record requires start_ts <= end_ts.")
        if rejection_reason is not None:
            raise ValueError("Accepted record must have rejection_reason=None.")
    elif disposition == DISPOSITION_REJECTED:
        eligible = False
        if not isinstance(rejection_reason, str) or not rejection_reason.strip():
            raise ValueError("Rejected attempt requires a non-empty rejection_reason.")
        energy_j = None
    elif disposition == DISPOSITION_PENDING:
        eligible = False
        energy_j = None
        latency_s = None
    else:
        raise ValueError(f"Unsupported disposition: {disposition!r}")

    draft = InvocationRecord(
        attempt_id, amendment_version, item_id, candidate_id, disposition, eligible,
        energy_j, latency_s, rejection_reason, battery_pct_start, battery_pct_end,
        temperature_c, start_ts, end_ts, "",
    )
    return dataclasses.replace(draft, record_sha256=draft.compute_sha256())


def record_hardware_telemetry(
    attempt_id: str, amendment_version: str, item_id: str, candidate_id: str,
    start_ts: float | None, end_ts: float | None, raw_energy_j: float | None,
    battery_pct_start: float | None = None, battery_pct_end: float | None = None,
    temperature_c: float | None = None, min_battery_pct: float = 20.0,
    max_temperature_c: float = 45.0,
) -> InvocationRecord:
    """Classify raw hardware telemetry, stripping energy_j on ineligible runs."""
    rejection: str | None = None
    if start_ts is None or end_ts is None or not math.isfinite(start_ts) or not math.isfinite(end_ts) or end_ts <= start_ts:
        rejection = REASON_MISSING_END_TS
    elif battery_pct_end is not None and battery_pct_end < min_battery_pct:
        rejection = REASON_BATTERY_DROP
    elif temperature_c is not None and temperature_c > max_temperature_c:
        rejection = REASON_THERMAL_ABORT
    elif raw_energy_j is None or not math.isfinite(raw_energy_j) or raw_energy_j < 0.0:
        rejection = REASON_TRANSPORT_DISCONNECT

    if rejection is not None:
        lat = (end_ts - start_ts) if (start_ts is not None and end_ts is not None and end_ts >= start_ts) else None
        return build_invocation_record(
            attempt_id, amendment_version, item_id, candidate_id, DISPOSITION_REJECTED,
            None, lat, rejection, battery_pct_start, battery_pct_end, temperature_c, start_ts, end_ts,
        )
    assert start_ts is not None and end_ts is not None and raw_energy_j is not None
    return build_invocation_record(
        attempt_id, amendment_version, item_id, candidate_id, DISPOSITION_ACCEPTED,
        raw_energy_j, end_ts - start_ts, None, battery_pct_start, battery_pct_end,
        temperature_c, start_ts, end_ts,
    )


def verify_ledger(
    amendments: list[ScopeEntry], records: list[InvocationRecord]
) -> LedgerAuditReport:
    """Verify hash-linked amendment chain, scope immutability, and invocation accounting."""
    defects: list[LedgerDefect] = []
    hash_chain_intact = frozen_scope_preserved = rejected_clean = True

    if not amendments:
        defects.append(LedgerDefect(ERR_EMPTY_AMENDMENT_CHAIN, "Amendment chain is empty."))
        return LedgerAuditReport(
            "unknown", 0, None, None, 0, 0, 0, 0, 0, False, False, False, False, False, {}, tuple(defects)
        )

    genesis = amendments[0]
    version_to_idx: dict[str, int] = {}
    for idx, entry in enumerate(amendments):
        expected_ver = f"v{idx}"
        if entry.version != expected_ver:
            hash_chain_intact = False
            defects.append(LedgerDefect(ERR_INVALID_VERSION_SEQUENCE, f"Expected {expected_ver!r}, got {entry.version!r}.", entry.version))
        version_to_idx[entry.version] = idx
        if entry.entry_sha256 != entry.compute_sha256():
            hash_chain_intact = False
            defects.append(LedgerDefect(ERR_SCOPE_DIGEST_TAMPERED, f"Digest mismatch on {entry.version!r}.", entry.version))
        if set(entry.calibration_ids) & set(entry.evaluation_ids):
            frozen_scope_preserved = False
            defects.append(LedgerDefect(ERR_OVERLAPPING_SPLIT_IDS, f"Overlapping split IDs on {entry.version!r}.", entry.version))

        if idx == 0:
            if entry.parent_sha256 is not None:
                hash_chain_intact = False
                defects.append(LedgerDefect(ERR_GENESIS_HAS_PARENT, "Genesis v0 must have parent_sha256=None.", entry.version))
        else:
            prev = amendments[idx - 1]
            if entry.parent_sha256 != prev.entry_sha256:
                hash_chain_intact = False
                defects.append(LedgerDefect(ERR_BROKEN_HASH_LINK, f"{entry.version!r} parent_sha256 mismatch.", entry.version))
            if entry.evaluation_ids != genesis.evaluation_ids:
                frozen_scope_preserved = False
                defects.append(LedgerDefect(ERR_EVALUATION_SCOPE_MUTATED, f"{entry.version!r} mutated evaluation_ids.", entry.version))
            if entry.candidates != genesis.candidates:
                frozen_scope_preserved = False
                defects.append(LedgerDefect(ERR_CANDIDATE_SET_MUTATED, f"{entry.version!r} mutated candidates.", entry.version))
            if entry.calibration_ids != genesis.calibration_ids:
                frozen_scope_preserved = False
                defects.append(LedgerDefect(ERR_CALIBRATION_SCOPE_MUTATED, f"{entry.version!r} mutated calibration_ids.", entry.version))
            if entry.max_invocations < prev.max_invocations:
                frozen_scope_preserved = False
                defects.append(LedgerDefect(ERR_INVOCATION_CAP_SHRUNK, f"{entry.version!r} decreased max_invocations.", entry.version))

    valid_items = set(genesis.calibration_ids) | set(genesis.evaluation_ids)
    valid_candidates = set(genesis.candidates)
    seen_attempts: set[str] = set()
    counts_by_disp: collections.Counter[str] = collections.Counter()
    counts_by_ver_idx: collections.Counter[int] = collections.Counter()
    rejection_breakdown: collections.Counter[str] = collections.Counter()

    for rec in records:
        if rec.attempt_id in seen_attempts:
            defects.append(LedgerDefect(ERR_DUPLICATE_ATTEMPT_ID, f"Duplicate attempt {rec.attempt_id!r}.", rec.attempt_id))
        seen_attempts.add(rec.attempt_id)
        if rec.record_sha256 != rec.compute_sha256():
            defects.append(LedgerDefect(ERR_RECORD_DIGEST_TAMPERED, f"Digest mismatch on {rec.attempt_id!r}.", rec.attempt_id))
        if rec.amendment_version not in version_to_idx:
            defects.append(LedgerDefect(ERR_UNKNOWN_AMENDMENT_VERSION, f"Unknown version {rec.amendment_version!r}.", rec.attempt_id))
        else:
            counts_by_ver_idx[version_to_idx[rec.amendment_version]] += 1
        if rec.item_id not in valid_items:
            defects.append(LedgerDefect(ERR_OUT_OF_SCOPE_ITEM, f"Out-of-scope item {rec.item_id!r}.", rec.attempt_id))
        if rec.candidate_id not in valid_candidates:
            defects.append(LedgerDefect(ERR_OUT_OF_SCOPE_CANDIDATE, f"Unknown candidate {rec.candidate_id!r}.", rec.attempt_id))

        if rec.disposition == DISPOSITION_REJECTED:
            counts_by_disp[DISPOSITION_REJECTED] += 1
            if rec.eligible_measurement is not False:
                rejected_clean = False
                defects.append(LedgerDefect(ERR_REJECTED_MARKED_ELIGIBLE, f"Rejected {rec.attempt_id!r} marked eligible.", rec.attempt_id))
            if rec.energy_j is not None:
                rejected_clean = False
                defects.append(LedgerDefect(ERR_REJECTED_HAS_ENERGY, f"Rejected {rec.attempt_id!r} has non-null energy_j.", rec.attempt_id))
            if not isinstance(rec.rejection_reason, str) or not rec.rejection_reason.strip():
                rejected_clean = False
                defects.append(LedgerDefect(ERR_MISSING_REJECTION_REASON, f"Rejected {rec.attempt_id!r} missing reason.", rec.attempt_id))
            else:
                rejection_breakdown[rec.rejection_reason] += 1
        elif rec.disposition == DISPOSITION_PENDING:
            counts_by_disp[DISPOSITION_PENDING] += 1
            if rec.eligible_measurement is not False or rec.energy_j is not None:
                defects.append(LedgerDefect(ERR_PENDING_HAS_MEASUREMENT, f"Pending {rec.attempt_id!r} carries measurement.", rec.attempt_id))
        elif rec.disposition == DISPOSITION_ACCEPTED:
            counts_by_disp[DISPOSITION_ACCEPTED] += 1
            valid_acc = (
                rec.eligible_measurement is True
                and rec.energy_j is not None and math.isfinite(rec.energy_j) and rec.energy_j >= 0.0
                and rec.latency_s is not None and math.isfinite(rec.latency_s) and rec.latency_s > 0.0
                and rec.start_ts is not None and rec.end_ts is not None and rec.end_ts >= rec.start_ts
                and rec.rejection_reason is None
            )
            if not valid_acc:
                defects.append(LedgerDefect(ERR_INVALID_ACCEPTED_RECORD, f"Invalid accepted telemetry on {rec.attempt_id!r}.", rec.attempt_id))
        else:
            defects.append(LedgerDefect(ERR_INVALID_DISPOSITION, f"Invalid disposition {rec.disposition!r}.", rec.attempt_id))

    cumulative = 0
    for idx, entry in enumerate(amendments):
        cumulative += counts_by_ver_idx[idx]
        if cumulative > entry.max_invocations:
            defects.append(
                LedgerDefect(ERR_AMENDMENT_CAP_EXCEEDED, f"Cumulative {cumulative} > cap {entry.max_invocations}.", entry.version)
            )

    terminal = amendments[-1]
    acc_cnt = counts_by_disp[DISPOSITION_ACCEPTED]
    pend_cnt = counts_by_disp[DISPOSITION_PENDING]
    rej_cnt = counts_by_disp[DISPOSITION_REJECTED]
    total_accounted = acc_cnt + pend_cnt + rej_cnt
    budget_conserved = total_accounted == terminal.max_invocations and total_accounted == len(records)
    if not budget_conserved:
        defects.append(
            LedgerDefect(
                ERR_INVOCATION_BUDGET_MISMATCH,
                f"Accounted ({total_accounted}) != terminal max_invocations ({terminal.max_invocations}).",
                terminal.version,
            )
        )

    return LedgerAuditReport(
        genesis.ledger_id, len(amendments), terminal.version, terminal.entry_sha256,
        terminal.max_invocations, acc_cnt, pend_cnt, rej_cnt, total_accounted,
        budget_conserved, rejected_clean, frozen_scope_preserved, hash_chain_intact,
        len(defects) == 0, dict(sorted(rejection_breakdown.items())), tuple(defects),
    )


def parse_ledger_dict(payload: dict[str, object]) -> tuple[list[ScopeEntry], list[InvocationRecord]]:
    """Deserialize a ledger JSON payload into ScopeEntry and InvocationRecord lists."""
    raw_amendments = payload.get("amendments")
    raw_records = payload.get("invocations", [])
    if not isinstance(raw_amendments, list) or not isinstance(raw_records, list):
        raise ValueError("Ledger JSON must contain list fields 'amendments' and 'invocations'.")

    amendments = [
        ScopeEntry(
            str(item["ledger_id"]), str(item["version"]),
            item.get("parent_sha256"),  # type: ignore[arg-type]
            tuple(item["calibration_ids"]), tuple(item["evaluation_ids"]),  # type: ignore[arg-type]
            tuple(item["candidates"]), int(item["max_invocations"]),  # type: ignore[arg-type]
            str(item["reason"]), str(item["entry_sha256"]),
        )
        for item in raw_amendments if isinstance(item, dict)
    ]
    records = [
        InvocationRecord(
            str(r["attempt_id"]), str(r["amendment_version"]), str(r["item_id"]),
            str(r["candidate_id"]), str(r["disposition"]), bool(r["eligible_measurement"]),
            float(r["energy_j"]) if r.get("energy_j") is not None else None,  # type: ignore[arg-type]
            float(r["latency_s"]) if r.get("latency_s") is not None else None,  # type: ignore[arg-type]
            r.get("rejection_reason"),  # type: ignore[arg-type]
            float(r["battery_pct_start"]) if r.get("battery_pct_start") is not None else None,  # type: ignore[arg-type]
            float(r["battery_pct_end"]) if r.get("battery_pct_end") is not None else None,  # type: ignore[arg-type]
            float(r["temperature_c"]) if r.get("temperature_c") is not None else None,  # type: ignore[arg-type]
            float(r["start_ts"]) if r.get("start_ts") is not None else None,  # type: ignore[arg-type]
            float(r["end_ts"]) if r.get("end_ts") is not None else None,  # type: ignore[arg-type]
            str(r.get("record_sha256", "")),
        )
        for r in raw_records if isinstance(r, dict)
    ]
    return amendments, records


def serialize_ledger_dict(
    amendments: list[ScopeEntry], records: list[InvocationRecord]
) -> dict[str, object]:
    """Serialize amendment chain and invocation records to a JSON-ready dictionary."""
    return {"amendments": [a.to_dict() for a in amendments], "invocations": [r.to_dict() for r in records]}


def format_ledger_markdown(report: LedgerAuditReport) -> str:
    """Format ledger audit summary as Markdown."""
    lines = [
        f"# Execution Scope Ledger Audit: {report.ledger_id}", "",
        f"- Chain length: {report.chain_length} (head: `{report.terminal_version}`)",
        f"- Hash chain intact: {report.hash_chain_intact}, Frozen scope preserved: {report.frozen_scope_preserved}",
        f"- Rejected zero-contamination verified: {report.rejected_zero_contamination_verified}",
        f"- Invocation accounting: {report.accepted_count} accepted + {report.pending_count} pending "
        f"+ {report.rejected_count} rejected = {report.total_accounted} / {report.max_invocations}",
        f"- Overall valid: {report.is_valid}",
    ]
    if report.rejection_breakdown:
        lines.extend(["", "## Rejected Hardware Attempts", ""])
        for reason, cnt in report.rejection_breakdown.items():
            lines.append(f"- `{reason}`: {cnt}")
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify hash-chained execution scope and invocation accounting.")
    parser.add_argument("--ledger", required=True, help="Input ledger JSON path.")
    parser.add_argument("--output", required=True, help="Exclusive audit report JSON path.")
    parser.add_argument("--markdown-output", default=None, help="Optional exclusive Markdown path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: returns 0 if ledger valid, 2 on integrity defect, 1 on IO error."""
    args = _build_arg_parser().parse_args(argv)
    try:
        payload = json.loads(pathlib.Path(args.ledger).read_text(encoding="utf-8"))
        amendments, records = parse_ledger_dict(payload)
        report = verify_ledger(amendments, records)
        with open(args.output, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
        if args.markdown_output is not None:
            with open(args.markdown_output, "x", encoding="utf-8") as md_handle:
                md_handle.write(format_ledger_markdown(report))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"amendment_ledger error: {exc}\n")
        return 1
    return 0 if report.is_valid else 2


if __name__ == "__main__":
    sys.exit(main())
