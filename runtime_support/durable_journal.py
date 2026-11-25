"""Crash-safe durable cell capture journal for planned measurement matrices.

Coordinates exclusive directory locking via ``fcntl.flock``, atomic JSON snapshot
commits (``plan.json`` and ``state.json``) using same-directory temporary files
followed by ``os.fsync``, ``os.replace``, and parent ``O_DIRECTORY`` ``os.fsync``,
plus fsynced append-only transition records in ``events.jsonl``. On recovery,
``state.json`` is authoritative while torn trailing bytes in ``events.jsonl`` are
truncated to the last complete record boundary.
"""

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import pathlib
import threading
import time

STATE_NOT_RUN, STATE_INFLIGHT = "not_run", "inflight"
STATE_ACCEPTED, STATE_FAILED, STATE_INVALID = "accepted", "failed", "invalid"
VALID_STATES = frozenset({STATE_NOT_RUN, STATE_INFLIGHT, STATE_ACCEPTED, STATE_FAILED, STATE_INVALID})
TERMINAL_STATES = frozenset({STATE_ACCEPTED, STATE_FAILED, STATE_INVALID})
MODE_MEASURE, MODE_REUSE = "measure", "reuse"
VALID_MODES = frozenset({MODE_MEASURE, MODE_REUSE})
INFLIGHT_RECOVERY_MODES = frozenset({STATE_NOT_RUN, STATE_FAILED, STATE_INVALID})
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    STATE_NOT_RUN: frozenset({STATE_INFLIGHT}),
    STATE_INFLIGHT: frozenset({STATE_ACCEPTED, STATE_FAILED, STATE_INVALID}),
    STATE_ACCEPTED: frozenset(), STATE_FAILED: frozenset(), STATE_INVALID: frozenset(),
}
JOURNAL_SCHEMA_VERSION = 1
_TMP_SEQ = 0
_TMP_LOCK = threading.Lock()


class JournalError(Exception): """Base error for durable journal operations."""
class JournalLockError(JournalError): """Raised when the journal directory lock is held."""
class JournalExistsError(JournalError): """Raised when opening a fresh journal over existing files."""
class JournalValidationError(JournalError): """Raised when plan definitions or payloads are invalid."""
class PlanMismatchError(JournalValidationError): """Raised when resume plan diverges from plan.json."""
class ReuseAuditError(JournalValidationError): """Raised when a reuse cell fails audit or digest checks."""
class InvalidTransitionError(JournalError): """Raised when an illegal cell state transition is requested."""
class JournalCorruptionError(JournalError): """Raised when persisted journal files are corrupted."""
class JournalClosedError(JournalError): """Raised when using a closed journal handle."""


def _validate_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise JournalValidationError(f"{field_name} must be a non-empty trimmed string")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise JournalValidationError(f"{field_name} contains control characters")
    return value


def _validate_json_tree(value: object, path: str = "root") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise JournalValidationError(f"Non-finite float at {path}")
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _validate_json_tree(item, f"{path}[{idx}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise JournalValidationError(f"Non-string dict key at {path}")
            _validate_json_tree(item, f"{path}.{key}")
        return
    raise JournalValidationError(f"Unsupported JSON type {type(value).__name__} at {path}")


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize a validated JSON structure into deterministic UTF-8 bytes."""
    _validate_json_tree(payload)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: object) -> str:
    """Return lowercase hexadecimal SHA-256 of canonical JSON encoding."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_file_sha256(file_path: pathlib.Path) -> str:
    """Compute SHA-256 digest of an on-disk artifact file."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as handle:
            while chunk := handle.read(65536):
                hasher.update(chunk)
    except OSError as exc:
        raise ReuseAuditError(f"Unable to read reuse artifact {file_path}: {exc}") from exc
    return hasher.hexdigest()


def fsync_directory(directory: pathlib.Path) -> None:
    """Flush directory metadata entries to stable storage via O_DIRECTORY."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    dir_fd = os.open(os.fspath(directory), flags)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(data):
        step = os.write(fd, view[written:])
        if step <= 0:
            raise OSError("Zero-byte write during durable file commit")
        written += step


def atomic_write_json(target_path: pathlib.Path, payload: dict[str, object]) -> None:
    """Atomically write ``payload`` to ``target_path`` with file and parent fsync."""
    global _TMP_SEQ
    if not isinstance(payload, dict):
        raise JournalValidationError("Top-level atomic JSON payload must be a dict")
    encoded = canonical_json_bytes(payload) + b"\n"
    parent = target_path.parent
    with _TMP_LOCK:
        _TMP_SEQ += 1
        seq = _TMP_SEQ
    tmp_path = parent / f".{target_path.name}.{os.getpid()}.{threading.get_ident()}.{seq}.tmp"
    fd = os.open(os.fspath(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    closed = False
    try:
        _write_all(fd, encoded)
        os.fsync(fd)
        os.close(fd)
        closed = True
        os.replace(os.fspath(tmp_path), os.fspath(target_path))
        fsync_directory(parent)
    except BaseException:
        if not closed:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            if tmp_path.exists():
                os.unlink(os.fspath(tmp_path))
        raise


def _validate_reuse_bundle(request_id: str, config_id: str, bundle: dict[str, object]) -> None:
    _validate_json_tree(bundle, "reuse_bundle")
    audit, metrics = bundle.get("audit"), bundle.get("metrics")
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise ReuseAuditError(f"Reuse cell ({request_id}, {config_id}) requires audit.passed is True")
    if metrics is not None and not isinstance(metrics, dict):
        raise ReuseAuditError(f"Reuse cell ({request_id}, {config_id}) metrics must be a dict")
    metrics_sha = bundle.get("metrics_sha256")
    if metrics_sha is not None:
        if not isinstance(metrics_sha, str) or len(metrics_sha) != 64:
            raise ReuseAuditError(f"Malformed metrics_sha256 for ({request_id}, {config_id})")
        if canonical_json_sha256(metrics if metrics is not None else {}) != metrics_sha.lower():
            raise ReuseAuditError(f"Reuse cell ({request_id}, {config_id}) metrics_sha256 mismatch")
    art_path, art_sha = bundle.get("artifact_path"), bundle.get("artifact_sha256")
    if art_path is not None or art_sha is not None:
        if not isinstance(art_path, str) or not art_path.strip() or not isinstance(art_sha, str) or len(art_sha) != 64:
            raise ReuseAuditError(f"Reuse cell ({request_id}, {config_id}) invalid artifact reference")
        if compute_file_sha256(pathlib.Path(art_path)) != art_sha.lower():
            raise ReuseAuditError(f"Reuse cell ({request_id}, {config_id}) artifact digest mismatch")


@dataclasses.dataclass(frozen=True)
class CellSpec:
    """Immutable specification of a single planned matrix cell."""
    request_id: str
    config_id: str
    mode: str = MODE_MEASURE
    parameters: dict[str, object] = dataclasses.field(default_factory=dict)
    reuse_bundle: dict[str, object] | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.request_id, "request_id")
        _validate_identifier(self.config_id, "config_id")
        if self.mode not in VALID_MODES or not isinstance(self.parameters, dict):
            raise JournalValidationError(f"Invalid mode {self.mode!r} or parameters for {self.key!r}")
        _validate_json_tree(self.parameters, "parameters")
        if self.mode == MODE_MEASURE and self.reuse_bundle is not None:
            raise JournalValidationError(f"Cell {self.key!r} mode='measure' forbids reuse_bundle")
        if self.mode == MODE_REUSE:
            if not isinstance(self.reuse_bundle, dict):
                raise ReuseAuditError(f"Cell {self.key!r} mode='reuse' requires reuse_bundle")
            _validate_reuse_bundle(self.request_id, self.config_id, self.reuse_bundle)

    @property
    def key(self) -> tuple[str, str]:
        return (self.request_id, self.config_id)

    def to_plan_dict(self) -> dict[str, object]:
        entry: dict[str, object] = {
            "request_id": self.request_id, "config_id": self.config_id, "mode": self.mode,
            "parameters": json.loads(canonical_json_bytes(self.parameters).decode("utf-8")),
        }
        if self.reuse_bundle is not None:
            norm = json.loads(canonical_json_bytes(self.reuse_bundle).decode("utf-8"))
            norm.setdefault("metrics_sha256", canonical_json_sha256(norm["metrics"])) if "metrics" in norm else None
            entry["reuse_bundle"] = norm
        return entry


@dataclasses.dataclass(frozen=True)
class MatrixPlan:
    """Planned measurement matrix containing unique ``(request_id, config_id)`` cells."""
    plan_id: str
    cells: tuple[CellSpec, ...]
    metadata: dict[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.plan_id, "plan_id")
        if not isinstance(self.cells, (list, tuple)) or not self.cells:
            raise JournalValidationError("MatrixPlan requires a non-empty sequence of CellSpec")
        cell_tuple, seen = tuple(self.cells), set()
        for spec in cell_tuple:
            if not isinstance(spec, CellSpec) or spec.key in seen:
                raise JournalValidationError(f"Invalid or duplicate CellSpec in plan: {spec!r}")
            seen.add(spec.key)
        meta = dict(self.metadata) if self.metadata is not None else {}
        _validate_json_tree(meta, "metadata")
        object.__setattr__(self, "cells", cell_tuple)
        object.__setattr__(self, "metadata", meta)

    def revalidate_reuse_bundles(self) -> None:
        for spec in self.cells:
            if spec.mode == MODE_REUSE and spec.reuse_bundle is not None:
                _validate_reuse_bundle(spec.request_id, spec.config_id, spec.reuse_bundle)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": JOURNAL_SCHEMA_VERSION, "plan_id": self.plan_id,
            "metadata": json.loads(canonical_json_bytes(self.metadata).decode("utf-8")),
            "cells": [spec.to_plan_dict() for spec in self.cells],
        }

    def digest(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclasses.dataclass(frozen=True)
class CellStateRecord:
    """Snapshot of a single matrix cell's durable execution state."""
    request_id: str
    config_id: str
    mode: str
    state: str
    attempts: int
    interrupted_count: int
    updated_at_ns: int
    worker_id: str | None = None
    result: dict[str, object] | None = None
    audit: dict[str, object] | None = None
    error_reason: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.request_id, self.config_id)

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "CellStateRecord":
        if not isinstance(raw, dict):
            raise JournalCorruptionError("Cell state entry must be a JSON object")
        req_id = _validate_identifier(raw.get("request_id"), "state.request_id")
        cfg_id = _validate_identifier(raw.get("config_id"), "state.config_id")
        mode, state = raw.get("mode"), raw.get("state")
        attempts, interrupted, updated_ns = raw.get("attempts"), raw.get("interrupted_count", 0), raw.get("updated_at_ns")
        if mode not in VALID_MODES or state not in VALID_STATES:
            raise JournalCorruptionError(f"Invalid persisted mode/state ({mode!r}, {state!r})")
        for num, non_neg in ((attempts, True), (interrupted, True), (updated_ns, False)):
            if isinstance(num, bool) or not isinstance(num, int) or (non_neg and num < 0):
                raise JournalCorruptionError(f"Invalid integer state field: {num!r}")
        worker_id, err, result, audit = raw.get("worker_id"), raw.get("error_reason"), raw.get("result"), raw.get("audit")
        if (worker_id is not None and not isinstance(worker_id, str)) or (err is not None and not isinstance(err, str)):
            raise JournalCorruptionError("worker_id and error_reason must be str or null")
        if (result is not None and not isinstance(result, dict)) or (audit is not None and not isinstance(audit, dict)):
            raise JournalCorruptionError("result and audit must be object or null")
        return cls(req_id, cfg_id, str(mode), str(state), int(attempts), int(interrupted), int(updated_ns), worker_id, result, audit, err)


@dataclasses.dataclass(frozen=True)
class JournalSummary:
    """Aggregated accounting counters across all planned matrix cells."""
    total_cells: int
    not_run: int
    inflight: int
    accepted: int
    failed: int
    invalid: int
    reused_accepted: int
    measured_accepted: int
    interrupted_recoveries: int
    terminal_count: int = dataclasses.field(init=False)
    is_complete: bool = dataclasses.field(init=False)
    all_accepted: bool = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "terminal_count", self.accepted + self.failed + self.invalid)
        object.__setattr__(self, "is_complete", self.not_run == 0 and self.inflight == 0)
        object.__setattr__(self, "all_accepted", self.accepted == self.total_cells)


class DurableJournal:
    """Crash-safe durable state machine for planned measurement matrices."""

    def __init__(
        self, journal_dir: str | pathlib.Path, plan: MatrixPlan, *,
        resume: bool = False, inflight_recovery: str = STATE_NOT_RUN, clock_ns: object = time.time_ns,
    ) -> None:
        if not isinstance(plan, MatrixPlan) or inflight_recovery not in INFLIGHT_RECOVERY_MODES or not callable(clock_ns):
            raise JournalValidationError("Invalid plan, inflight_recovery, or clock_ns")
        self._dir = pathlib.Path(journal_dir)
        self.plan_path, self.state_path, self.events_path = self._dir / "plan.json", self._dir / "state.json", self._dir / "events.jsonl"
        self._plan, self._clock_ns, self._inflight_recovery = plan, clock_ns, inflight_recovery
        self._lock_fd: int | None = None
        self._events_fd: int | None = None
        self._closed, self._event_seq, self.truncated_tail_bytes = False, 0, 0
        self._recovered: list[tuple[str, str]] = []
        self._spec_by_key = {spec.key: spec for spec in plan.cells}
        self._records: dict[tuple[str, str], CellStateRecord] = {}

        self._dir.mkdir(parents=True, exist_ok=True)
        self._acquire_dir_lock()
        try:
            self._open_existing() if resume else self._initialize_fresh()
        except BaseException:
            self._release_handles()
            self._closed = True
            raise

    @classmethod
    def open(
        cls, journal_dir: str | pathlib.Path, plan: MatrixPlan, *,
        resume: bool = False, inflight_recovery: str = STATE_NOT_RUN, clock_ns: object = time.time_ns,
    ) -> "DurableJournal":
        return cls(journal_dir, plan, resume=resume, inflight_recovery=inflight_recovery, clock_ns=clock_ns)

    @property
    def recovered_cells(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._recovered)

    def __enter__(self) -> "DurableJournal":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._release_handles()

    def _ensure_open(self) -> None:
        if self._closed:
            raise JournalClosedError("DurableJournal handle is closed")

    def _now_ns(self) -> int:
        val = self._clock_ns()
        if isinstance(val, bool) or not isinstance(val, int):
            raise JournalValidationError("clock_ns must return integer nanoseconds")
        return val

    def _acquire_dir_lock(self) -> None:
        fd = os.open(os.fspath(self._dir / ".journal.lock"), os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o600)
        try:
            os.set_inheritable(fd, False)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise JournalLockError(f"Journal directory {self._dir} is locked") from exc
        except BaseException:
            os.close(fd)
            raise
        self._lock_fd = fd

    def _release_handles(self) -> None:
        if self._events_fd is not None:
            for fn in (os.fsync, os.close):
                with contextlib.suppress(OSError):
                    fn(self._events_fd)
            self._events_fd = None
        if self._lock_fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(self._lock_fd)
            self._lock_fd = None

    def _open_events_stream(self) -> None:
        self._events_fd = os.open(os.fspath(self.events_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0), 0o600)
        os.set_inheritable(self._events_fd, False)

    def _append_event(self, event_type: str, payload: dict[str, object]) -> None:
        if self._events_fd is None:
            raise JournalClosedError("Events stream is not open")
        self._event_seq += 1
        _write_all(self._events_fd, canonical_json_bytes({"seq": self._event_seq, "ts_ns": self._now_ns(), "event": event_type, "plan_id": self._plan.plan_id, **payload}) + b"\n")
        os.fsync(self._events_fd)

    def _persist_state(self) -> None:
        atomic_write_json(self.state_path, {"schema_version": JOURNAL_SCHEMA_VERSION, "plan_id": self._plan.plan_id, "plan_digest": self._plan.digest(), "updated_at_ns": self._now_ns(), "cells": [self._records[spec.key].to_dict() for spec in self._plan.cells]})

    def _initialize_fresh(self) -> None:
        if self.plan_path.exists() or self.state_path.exists() or self.events_path.exists():
            raise JournalExistsError(f"Journal files already exist in {self._dir}")
        self._plan.revalidate_reuse_bundles()
        now_ns = self._now_ns()
        for spec in self._plan.cells:
            if spec.mode == MODE_REUSE and spec.reuse_bundle is not None:
                m_obj, a_obj = spec.reuse_bundle.get("metrics"), spec.reuse_bundle.get("audit")
                self._records[spec.key] = CellStateRecord(spec.request_id, spec.config_id, MODE_REUSE, STATE_ACCEPTED, 0, 0, now_ns, "pre_audited_reuse", dict(m_obj) if isinstance(m_obj, dict) else {}, dict(a_obj) if isinstance(a_obj, dict) else None)
            else:
                self._records[spec.key] = CellStateRecord(spec.request_id, spec.config_id, MODE_MEASURE, STATE_NOT_RUN, 0, 0, now_ns)
        atomic_write_json(self.plan_path, self._plan.to_dict())
        self._persist_state()
        self._open_events_stream()
        fsync_directory(self._dir)
        self._append_event("journal_created", {"plan_digest": self._plan.digest(), "total_cells": len(self._plan.cells)})
        for spec in self._plan.cells:
            if spec.mode == MODE_REUSE:
                self._append_event("reuse_registered", {"request_id": spec.request_id, "config_id": spec.config_id, "state": STATE_ACCEPTED})

    @staticmethod
    def _load_json_file(path: pathlib.Path, label: str) -> dict[str, object]:
        if not path.exists():
            raise JournalCorruptionError(f"Missing required {label} at {path}")
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JournalCorruptionError(f"Corrupt {label} at {path}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise JournalCorruptionError(f"{label} root must be a JSON object")
        return parsed

    def _reconcile_events_file(self) -> int:
        if not self.events_path.exists():
            return 0
        raw = self.events_path.read_bytes()
        lines = raw.splitlines(keepends=True)
        valid_offset, max_seq = 0, 0
        for idx, raw_line in enumerate(lines):
            is_last = idx == len(lines) - 1
            if not raw_line.endswith(b"\n"):
                if is_last:
                    break
                raise JournalCorruptionError("Unexpected line boundary in events.jsonl")
            try:
                parsed = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if is_last:
                    break
                raise JournalCorruptionError(f"Mid-file corruption at line {idx + 1}") from exc
            if not isinstance(parsed, dict):
                if is_last:
                    break
                raise JournalCorruptionError(f"Non-object event at line {idx + 1}")
            seq_val = parsed.get("seq")
            max_seq = max(max_seq, seq_val) if isinstance(seq_val, int) and not isinstance(seq_val, bool) else max_seq
            valid_offset += len(raw_line)
        self._event_seq, truncated = max_seq, len(raw) - valid_offset
        if truncated > 0:
            fd = os.open(os.fspath(self.events_path), os.O_WRONLY | getattr(os, "O_CLOEXEC", 0))
            try:
                os.ftruncate(fd, valid_offset)
                os.fsync(fd)
            finally:
                os.close(fd)
            fsync_directory(self._dir)
        return truncated

    def _open_existing(self) -> None:
        self._plan.revalidate_reuse_bundles()
        if canonical_json_bytes(self._load_json_file(self.plan_path, "plan.json")) != canonical_json_bytes(self._plan.to_dict()):
            raise PlanMismatchError("Supplied MatrixPlan diverges from persisted plan.json")
        persisted_state = self._load_json_file(self.state_path, "state.json")
        if persisted_state.get("plan_digest") != self._plan.digest():
            raise PlanMismatchError("Persisted state.json plan_digest mismatch")
        raw_cells = persisted_state.get("cells")
        if not isinstance(raw_cells, list) or len(raw_cells) != len(self._plan.cells):
            raise JournalCorruptionError("state.json cell count mismatch")
        loaded: dict[tuple[str, str], CellStateRecord] = {}
        for item in raw_cells:
            rec = CellStateRecord.from_dict(item)
            spec = self._spec_by_key.get(rec.key)
            if rec.key in loaded or spec is None or rec.mode != spec.mode:
                raise JournalCorruptionError(f"Invalid or duplicate cell {rec.key!r} in state.json")
            if spec.mode == MODE_REUSE and (rec.state != STATE_ACCEPTED or not isinstance(rec.audit, dict) or rec.audit.get("passed") is not True):
                raise ReuseAuditError(f"Persisted reuse cell {rec.key!r} lost passing audit")
            loaded[rec.key] = rec
        self._records = loaded
        self.truncated_tail_bytes = self._reconcile_events_file()
        self._open_events_stream()
        if self.truncated_tail_bytes > 0:
            self._append_event("events_tail_reconciled", {"truncated_bytes": self.truncated_tail_bytes})
        now_ns = self._now_ns()
        for spec in self._plan.cells:
            rec = self._records[spec.key]
            if rec.state == STATE_INFLIGHT:
                self._recovered.append(spec.key)
                reason = "Interrupted inflight recovered" if self._inflight_recovery != STATE_NOT_RUN else None
                self._records[spec.key] = dataclasses.replace(
                    rec, state=self._inflight_recovery, interrupted_count=rec.interrupted_count + 1, updated_at_ns=now_ns, error_reason=reason,
                )
        if self._recovered:
            self._persist_state()
            for req_id, cfg_id in self._recovered:
                rec = self._records[(req_id, cfg_id)]
                self._append_event("inflight_recovered", {"request_id": req_id, "config_id": cfg_id, "recovered_state": rec.state, "interrupted_count": rec.interrupted_count})
        self._append_event("journal_resumed", {"recovered_inflight_count": len(self._recovered), "truncated_tail_bytes": self.truncated_tail_bytes})

    def get_cell(self, request_id: str, config_id: str) -> CellStateRecord:
        self._ensure_open()
        rec = self._records.get((request_id, config_id))
        if rec is None:
            raise JournalValidationError(f"Unknown cell coordinate ({request_id}, {config_id})")
        return rec

    def pending_cells(self) -> list[CellSpec]:
        return list(self.iter_pending_cells())

    def iter_pending_cells(self):
        self._ensure_open()
        for spec in self._plan.cells:
            self._ensure_open()
            if self._records[spec.key].state == STATE_NOT_RUN:
                yield spec

    def start_cell(self, request_id: str, config_id: str, *, worker_id: str = "local") -> CellStateRecord:
        _validate_identifier(worker_id, "worker_id")
        return self._transition(request_id, config_id, STATE_INFLIGHT, worker_id=worker_id)

    def complete_cell(
        self, request_id: str, config_id: str, *, result: dict[str, object], audit: dict[str, object] | None = None,
    ) -> CellStateRecord:
        if not isinstance(result, dict):
            raise JournalValidationError("result must be a dict")
        _validate_json_tree(result, "result")
        if audit is not None:
            if not isinstance(audit, dict) or ("passed" in audit and audit["passed"] is not True):
                raise JournalValidationError("audit must be a dict with passed is True")
            _validate_json_tree(audit, "audit")
        return self._transition(request_id, config_id, STATE_ACCEPTED, result=dict(result), audit=dict(audit) if audit is not None else None)

    def fail_cell(self, request_id: str, config_id: str, *, reason: str, diagnostics: dict[str, object] | None = None) -> CellStateRecord:
        return self._terminate(request_id, config_id, STATE_FAILED, reason, diagnostics)

    def invalidate_cell(self, request_id: str, config_id: str, *, reason: str, diagnostics: dict[str, object] | None = None) -> CellStateRecord:
        return self._terminate(request_id, config_id, STATE_INVALID, reason, diagnostics)

    def _terminate(self, request_id: str, config_id: str, state: str, reason: str, diagnostics: dict[str, object] | None) -> CellStateRecord:
        _validate_identifier(reason, "reason")
        if diagnostics is not None:
            if not isinstance(diagnostics, dict):
                raise JournalValidationError("diagnostics must be a dict")
            _validate_json_tree(diagnostics, "diagnostics")
        return self._transition(request_id, config_id, state, result=dict(diagnostics) if diagnostics is not None else None, error_reason=reason)

    def _transition(
        self, request_id: str, config_id: str, target_state: str, *,
        worker_id: str | None = None, result: dict[str, object] | None = None,
        audit: dict[str, object] | None = None, error_reason: str | None = None,
    ) -> CellStateRecord:
        rec = self.get_cell(request_id, config_id)
        if target_state not in VALID_STATES or target_state not in ALLOWED_TRANSITIONS[rec.state]:
            raise InvalidTransitionError(f"Illegal transition for {rec.key!r}: {rec.state!r} -> {target_state!r}")
        attempts = rec.attempts + 1 if target_state == STATE_INFLIGHT else rec.attempts
        next_worker = worker_id if worker_id is not None else rec.worker_id
        updated = CellStateRecord(rec.request_id, rec.config_id, rec.mode, target_state, attempts, rec.interrupted_count, self._now_ns(), next_worker, result, audit, error_reason)
        self._records[rec.key] = updated
        self._persist_state()
        self._append_event("cell_transition", {"request_id": request_id, "config_id": config_id, "from_state": rec.state, "to_state": target_state, "attempts": attempts, "worker_id": next_worker, "error_reason": error_reason})
        return updated

    def summary(self) -> JournalSummary:
        self._ensure_open()
        counts = {s: 0 for s in VALID_STATES}
        reused_ok, measured_ok, recoveries = 0, 0, 0
        for spec in self._plan.cells:
            rec = self._records[spec.key]
            counts[rec.state] += 1
            recoveries += rec.interrupted_count
            reused_ok += int(rec.state == STATE_ACCEPTED and rec.mode == MODE_REUSE)
            measured_ok += int(rec.state == STATE_ACCEPTED and rec.mode != MODE_REUSE)
        return JournalSummary(len(self._plan.cells), counts[STATE_NOT_RUN], counts[STATE_INFLIGHT], counts[STATE_ACCEPTED], counts[STATE_FAILED], counts[STATE_INVALID], reused_ok, measured_ok, recoveries)

    def read_events(self) -> list[dict[str, object]]:
        self._ensure_open()
        if not self.events_path.exists():
            return []
        out: list[dict[str, object]] = []
        for idx, line in enumerate(self.events_path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise JournalCorruptionError(f"Non-object event at line {idx}")
                out.append(parsed)
        return out
