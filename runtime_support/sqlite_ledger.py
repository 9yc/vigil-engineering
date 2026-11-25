"""Transactional SQLite job execution ledger and SHA-256 artifact registry."""

from __future__ import annotations

import contextlib, csv, dataclasses, hashlib, io, json, math, os, secrets, sqlite3, threading, time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union


class LedgerError(Exception):
    """Base exception for execution ledger operations."""

class InvalidParameterError(LedgerError):
    """Raised when parameters, identifiers, or configurations are malformed."""

class RunNotFoundError(LedgerError):
    """Raised when referencing an unregistered run identifier."""

class JobNotFoundError(LedgerError):
    """Raised when referencing an unknown job identifier."""

class LeaseConflictError(LedgerError):
    """Raised when mutating a job without an active valid worker lease."""

class ArtifactVerificationError(LedgerError):
    """Raised when an artifact fails existence, size, or SHA-256 verification."""


class JobStatus:
    PENDING, LEASED, SUCCEEDED, DEAD_LETTER = "pending", "leased", "succeeded", "dead_letter"
    ALL = frozenset({PENDING, LEASED, SUCCEEDED, DEAD_LETTER})


@dataclasses.dataclass(frozen=True)
class BackoffPolicy:
    base_delay_s: float = 1.0
    factor: float = 2.0
    max_delay_s: float = 300.0
    jitter_ratio: float = 0.0

    def __post_init__(self) -> None:
        checks = (
            (math.isfinite(self.base_delay_s) and self.base_delay_s >= 0.0, "base_delay_s >= 0"),
            (math.isfinite(self.factor) and self.factor >= 1.0, "factor >= 1.0"),
            (math.isfinite(self.max_delay_s) and self.max_delay_s >= 0.0, "max_delay_s >= 0"),
            (math.isfinite(self.jitter_ratio) and 0.0 <= self.jitter_ratio <= 1.0, "jitter_ratio in [0, 1]"),
        )
        for ok, desc in checks:
            if not ok:
                raise InvalidParameterError(f"Invalid BackoffPolicy constraint: {desc}")

    def compute_delay(self, attempt: int, *, job_id: str = "") -> float:
        try:
            bounded = min(self.max_delay_s, self.base_delay_s * (self.factor ** max(0, int(attempt) - 1)))
        except OverflowError:
            bounded = self.max_delay_s
        if self.jitter_ratio > 0.0 and bounded > 0.0:
            unit = int.from_bytes(hashlib.sha256(f"{job_id}:{attempt}".encode()).digest()[:8], "big") / float(1 << 64)
            bounded = max(0.0, min(self.max_delay_s, bounded * (1.0 + self.jitter_ratio * (2.0 * unit - 1.0))))
        return bounded


@dataclasses.dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    config: Dict[str, Any]
    config_json: str
    config_sha256: str
    metadata: Dict[str, Any]
    created_at_s: float
    updated_at_s: float


@dataclasses.dataclass(frozen=True)
class JobRecord:
    job_id: str
    run_id: str
    dedup_key: str
    task_type: str
    params: Dict[str, Any]
    params_json: str
    params_sha256: str
    priority: int
    status: str
    attempt: int
    max_attempts: int
    available_at_s: float
    lease_owner: Optional[str]
    lease_token: Optional[str]
    lease_expires_at_s: Optional[float]
    last_error: Optional[str]
    result_summary: Optional[Dict[str, Any]]
    created_at_s: float
    updated_at_s: float
    completed_at_s: Optional[float]


@dataclasses.dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    job_id: str
    run_id: str
    role: str
    path: str
    size_bytes: int
    sha256: str
    metadata: Dict[str, Any]
    verified_at_s: float


@dataclasses.dataclass(frozen=True)
class RunProgress:
    run_id: str
    total_jobs: int
    pending_ready_jobs: int
    pending_delayed_jobs: int
    leased_jobs: int
    succeeded_jobs: int
    dead_letter_jobs: int
    total_artifacts: int
    total_artifact_bytes: int
    completion_ratio: float

    @property
    def pending_jobs(self) -> int:
        return self.pending_ready_jobs + self.pending_delayed_jobs


@dataclasses.dataclass(frozen=True)
class ArtifactAuditReport:
    checked_count: int
    valid_ids: Tuple[str, ...]
    missing_ids: Tuple[str, ...]
    mismatched_ids: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_ids and not self.mismatched_ids


def _valid_id(value: Any, name: str, *, max_len: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip() != value or len(value) > max_len or "\x00" in value:
        raise InvalidParameterError(f"Invalid identifier for {name}")
    try:
        value.encode("utf-8")
        return value
    except UnicodeEncodeError as exc:
        raise InvalidParameterError(f"{name} contains invalid UTF-8 surrogates") from exc


def _sanitize_json(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidParameterError(f"Non-finite float at {path}: {value!r}")
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
            return value
        except UnicodeEncodeError as exc:
            raise InvalidParameterError(f"Invalid UTF-8 string at {path}") from exc
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise InvalidParameterError(f"Raw binary value not allowed at {path}")
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise InvalidParameterError(f"Object keys must be strings at {path}")
            try:
                k.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise InvalidParameterError(f"Invalid UTF-8 key at {path}") from exc
            out[k] = _sanitize_json(v, f"{path}.{k}")
        return out
    if isinstance(value, Sequence):
        return [_sanitize_json(item, f"{path}[{i}]") for i, item in enumerate(value)]
    raise InvalidParameterError(f"Unsupported type at {path}: {type(value).__name__}")


def canonicalize_json_mapping(data: Optional[Mapping[str, Any]], name: str = "params") -> Tuple[str, str]:
    target: Any = {} if data is None else data
    if not isinstance(target, Mapping):
        raise InvalidParameterError(f"{name} must be a mapping, got {type(target).__name__}")
    canon = json.dumps(_sanitize_json(target, f"$.{name}"), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return canon, hashlib.sha256(canon.encode("utf-8")).hexdigest()


def compute_file_sha256(file_path: Union[str, Path], *, chunk_size: int = 65536) -> Tuple[int, str]:
    resolved = Path(file_path)
    try:
        st = resolved.stat()
    except OSError as exc:
        raise ArtifactVerificationError(f"Artifact file inaccessible: {resolved}") from exc
    if os.path.isdir(resolved) or not os.path.isfile(resolved):
        raise ArtifactVerificationError(f"Artifact path is not a regular file: {resolved}")
    hasher, total = hashlib.sha256(), 0
    try:
        with open(resolved, "rb") as fh:
            while block := fh.read(chunk_size):
                total += len(block)
                hasher.update(block)
    except OSError as exc:
        raise ArtifactVerificationError(f"Failed reading artifact {resolved}: {exc}") from exc
    if total != st.st_size:
        raise ArtifactVerificationError(f"Artifact size changed during read for {resolved}")
    return total, hasher.hexdigest()


def _atomic_write(output_path: Union[str, Path], data: bytes) -> None:
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.{secrets.token_hex(6)}.tmp"
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, name TEXT NOT NULL, config_json TEXT NOT NULL, config_sha256 TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at_s REAL NOT NULL, updated_at_s REAL NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT, dedup_key TEXT NOT NULL, task_type TEXT NOT NULL, params_json TEXT NOT NULL, params_sha256 TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL CHECK (status IN ('pending','leased','succeeded','dead_letter')), attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0), max_attempts INTEGER NOT NULL CHECK (max_attempts >= 1), available_at_s REAL NOT NULL, lease_owner TEXT, lease_token TEXT, lease_expires_at_s REAL, last_error TEXT, result_json TEXT, created_at_s REAL NOT NULL, updated_at_s REAL NOT NULL, completed_at_s REAL, UNIQUE(run_id, dedup_key));
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(status, priority DESC, available_at_s ASC, created_at_s ASC, job_id ASC);
CREATE INDEX IF NOT EXISTS idx_jobs_run_status ON jobs(run_id, status, job_id ASC);
CREATE INDEX IF NOT EXISTS idx_jobs_lease_expiry ON jobs(status, lease_expires_at_s ASC);
CREATE TABLE IF NOT EXISTS job_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE, run_id TEXT NOT NULL, event_type TEXT NOT NULL, attempt INTEGER NOT NULL, worker_id TEXT, detail TEXT, recorded_at_s REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id, event_id ASC);
CREATE TABLE IF NOT EXISTS artifacts (artifact_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE RESTRICT, run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT, role TEXT NOT NULL, path TEXT NOT NULL, size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0), sha256 TEXT NOT NULL, metadata_json TEXT NOT NULL, verified_at_s REAL NOT NULL, UNIQUE(job_id, role, path));
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id, job_id ASC, artifact_id ASC);
"""


class SqliteExecutionLedger:
    """Transactional SQLite ledger for distributed job leasing and artifact tracking."""

    def __init__(self, db_path: Union[str, Path], *, busy_timeout_ms: int = 5000, backoff_policy: Optional[BackoffPolicy] = None, clock: Callable[[], float] = time.time) -> None:
        if busy_timeout_ms < 0:
            raise InvalidParameterError("busy_timeout_ms must be >= 0")
        self._db_path = str(Path(db_path))
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._backoff_policy = backoff_policy or BackoffPolicy()
        self._clock = clock
        self._local = threading.local()
        self._conns_lock = threading.Lock()
        self._open_conns: List[sqlite3.Connection] = []
        self._closed = False
        with self._immediate_tx() as conn:
            for stmt in (s.strip() for s in _SCHEMA_SQL.split(";") if s.strip()):
                conn.execute(stmt)

    @property
    def db_path(self) -> str:
        return self._db_path

    def __enter__(self) -> SqliteExecutionLedger:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._conns_lock:
            self._closed = True
            conns, self._open_conns = list(self._open_conns), []
        for conn in conns:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
        if hasattr(self._local, "conn"):
            delattr(self._local, "conn")

    def _now(self, override_s: Optional[float] = None) -> float:
        val = float(self._clock() if override_s is None else override_s)
        if not math.isfinite(val):
            raise InvalidParameterError("Timestamp must be finite")
        return val

    def _get_conn(self) -> sqlite3.Connection:
        if (conn := getattr(self._local, "conn", None)) is not None:
            return conn
        with self._conns_lock:
            if self._closed:
                raise LedgerError("SqliteExecutionLedger is closed")
            conn = sqlite3.connect(self._db_path, timeout=max(0.1, self._busy_timeout_ms / 1000.0), isolation_level=None, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            for p in (f"busy_timeout = {self._busy_timeout_ms}", "journal_mode = WAL", "synchronous = NORMAL", "foreign_keys = ON"):
                conn.execute(f"PRAGMA {p}")
            self._open_conns.append(conn)
            self._local.conn = conn
            return conn

    @contextlib.contextmanager
    def _immediate_tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._get_conn()
        deadline, sleep_s = time.monotonic() + max(0.05, self._busy_timeout_ms / 1000.0), 0.002
        while True:
            try:
                conn.execute("BEGIN IMMEDIATE")
                break
            except sqlite3.OperationalError as exc:
                if ("locked" in str(exc).lower() or "busy" in str(exc).lower()) and time.monotonic() < deadline:
                    time.sleep(sleep_s)
                    sleep_s = min(0.025, sleep_s * 1.6)
                    continue
                raise
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _row_to_run(r: sqlite3.Row) -> RunRecord:
        return RunRecord(r["run_id"], r["name"], json.loads(r["config_json"]), r["config_json"], r["config_sha256"], json.loads(r["metadata_json"]), float(r["created_at_s"]), float(r["updated_at_s"]))

    @staticmethod
    def _row_to_job(r: sqlite3.Row) -> JobRecord:
        exp, done, rj = r["lease_expires_at_s"], r["completed_at_s"], r["result_json"]
        return JobRecord(
            r["job_id"], r["run_id"], r["dedup_key"], r["task_type"], json.loads(r["params_json"]),
            r["params_json"], r["params_sha256"], int(r["priority"]), r["status"], int(r["attempt"]),
            int(r["max_attempts"]), float(r["available_at_s"]), r["lease_owner"], r["lease_token"],
            float(exp) if exp is not None else None, r["last_error"], json.loads(rj) if rj is not None else None,
            float(r["created_at_s"]), float(r["updated_at_s"]), float(done) if done is not None else None,
        )

    @staticmethod
    def _row_to_artifact(r: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(r["artifact_id"], r["job_id"], r["run_id"], r["role"], r["path"], int(r["size_bytes"]), r["sha256"], json.loads(r["metadata_json"]), float(r["verified_at_s"]))

    @staticmethod
    def _log_event(conn: sqlite3.Connection, jid: str, rid: str, ev: str, att: int, ts: float, wid: Optional[str] = None, detail: Optional[str] = None) -> None:
        conn.execute("INSERT INTO job_events (job_id, run_id, event_type, attempt, worker_id, detail, recorded_at_s) VALUES (?, ?, ?, ?, ?, ?, ?)", (jid, rid, ev, att, wid, detail, ts))

    def register_run(self, run_id: str, name: str, config: Optional[Mapping[str, Any]] = None, *, metadata: Optional[Mapping[str, Any]] = None, now_s: Optional[float] = None) -> RunRecord:
        run_id, name = _valid_id(run_id, "run_id"), _valid_id(name, "name")
        c_json, c_sha = canonicalize_json_mapping(config, "config")
        m_json, _ = canonicalize_json_mapping(metadata, "metadata")
        ts = self._now(now_s)
        with self._immediate_tx() as conn:
            if (ex := conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()) is not None:
                if ex["name"] != name or ex["config_sha256"] != c_sha:
                    raise InvalidParameterError(f"Conflicting re-registration for run {run_id!r}")
                return self._row_to_run(ex)
            conn.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)", (run_id, name, c_json, c_sha, m_json, ts, ts))
            return self._row_to_run(conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone())

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        row = self._get_conn().execute("SELECT * FROM runs WHERE run_id = ?", (_valid_id(run_id, "run_id"),)).fetchone()
        return self._row_to_run(row) if row is not None else None

    def enqueue_job(self, run_id: str, task_type: str, params: Mapping[str, Any], *, priority: int = 0, max_attempts: int = 3, dedup_key: Optional[str] = None, job_id: Optional[str] = None, now_s: Optional[float] = None) -> Tuple[JobRecord, bool]:
        spec = {"task_type": task_type, "params": params, "priority": priority, "max_attempts": max_attempts, "dedup_key": dedup_key, "job_id": job_id}
        return self.enqueue_batch(run_id, [spec], now_s=now_s)[0]

    def enqueue_batch(self, run_id: str, specs: Sequence[Mapping[str, Any]], *, now_s: Optional[float] = None) -> List[Tuple[JobRecord, bool]]:
        run_id, ts = _valid_id(run_id, "run_id"), self._now(now_s)
        if not isinstance(specs, Sequence):
            raise InvalidParameterError("specs must be a sequence of mappings")
        prepared: List[Tuple[str, str, str, str, str, int, int]] = []
        for idx, spec in enumerate(specs):
            if not isinstance(spec, Mapping):
                raise InvalidParameterError(f"specs[{idx}] must be a mapping")
            ttype = _valid_id(spec.get("task_type"), f"specs[{idx}].task_type")
            pjson, psha = canonicalize_json_mapping(spec.get("params"), f"specs[{idx}].params")
            prio, max_att = spec.get("priority", 0), spec.get("max_attempts", 3)
            if isinstance(prio, bool) or not isinstance(prio, int) or isinstance(max_att, bool) or not isinstance(max_att, int) or max_att < 1:
                raise InvalidParameterError(f"Invalid priority or max_attempts in specs[{idx}]")
            dedup = hashlib.sha256(f"{run_id}\x1f{ttype}\x1f{psha}".encode()).hexdigest() if spec.get("dedup_key") is None else _valid_id(spec["dedup_key"], "dedup_key")
            jid = f"job-{dedup[:24]}" if spec.get("job_id") is None else _valid_id(spec["job_id"], "job_id")
            prepared.append((jid, dedup, ttype, pjson, psha, prio, max_att))

        outcomes: List[Tuple[JobRecord, bool]] = []
        ins = "INSERT INTO jobs (job_id, run_id, dedup_key, task_type, params_json, params_sha256, priority, status, attempt, max_attempts, available_at_s, created_at_s, updated_at_s) VALUES (?,?,?,?,?,?,?,?,0,?,?,?,?)"
        with self._immediate_tx() as conn:
            if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                raise RunNotFoundError(f"Unregistered run_id: {run_id!r}")
            for jid, dedup, ttype, pjson, psha, prio, max_att in prepared:
                if (ex := conn.execute("SELECT * FROM jobs WHERE run_id = ? AND dedup_key = ?", (run_id, dedup)).fetchone()) is not None:
                    if ex["task_type"] != ttype or ex["params_sha256"] != psha:
                        raise InvalidParameterError(f"Deduplication key {dedup!r} payload mismatch")
                    outcomes.append((self._row_to_job(ex), False))
                    continue
                if conn.execute("SELECT 1 FROM jobs WHERE job_id = ?", (jid,)).fetchone() is not None:
                    jid = f"{jid}-{secrets.token_hex(4)}"
                conn.execute(ins, (jid, run_id, dedup, ttype, pjson, psha, prio, JobStatus.PENDING, max_att, ts, ts, ts))
                self._log_event(conn, jid, run_id, "enqueued", 0, ts)
                outcomes.append((self._row_to_job(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (jid,)).fetchone()), True))
        return outcomes

    def claim_next_job(self, worker_id: str, lease_s: float, *, run_id: Optional[str] = None, task_types: Optional[Sequence[str]] = None, now_s: Optional[float] = None) -> Optional[JobRecord]:
        worker_id = _valid_id(worker_id, "worker_id")
        if not (math.isfinite(lease_s) and lease_s > 0.0):
            raise InvalidParameterError("lease_s must be finite and > 0")
        ts = self._now(now_s)
        clauses, args = ["status = ?", "available_at_s <= ?"], [JobStatus.PENDING, ts]
        if run_id is not None:
            clauses.append("run_id = ?")
            args.append(_valid_id(run_id, "run_id"))
        if task_types is not None:
            if not (vtypes := [_valid_id(t, "task_type") for t in task_types]):
                return None
            clauses.append(f"task_type IN ({','.join('?' for _ in vtypes)})")
            args.extend(vtypes)
        sql = f"SELECT * FROM jobs WHERE {' AND '.join(clauses)} ORDER BY priority DESC, available_at_s ASC, created_at_s ASC, job_id ASC LIMIT 1"
        with self._immediate_tx() as conn:
            if (cand := conn.execute(sql, args).fetchone()) is None:
                return None
            jid, att, tok, exp = cand["job_id"], int(cand["attempt"]) + 1, secrets.token_hex(12), ts + float(lease_s)
            conn.execute("UPDATE jobs SET status=?, attempt=?, lease_owner=?, lease_token=?, lease_expires_at_s=?, updated_at_s=? WHERE job_id=?", (JobStatus.LEASED, att, worker_id, tok, exp, ts, jid))
            self._log_event(conn, jid, cand["run_id"], "leased", att, ts, worker_id, f"lease_s={lease_s:.3f}")
            return self._row_to_job(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (jid,)).fetchone())

    @staticmethod
    def _verify_lease(conn: sqlite3.Connection, jid: str, wid: str, tok: Optional[str], now_s: float) -> sqlite3.Row:
        if (row := conn.execute("SELECT * FROM jobs WHERE job_id = ?", (jid,)).fetchone()) is None:
            raise JobNotFoundError(f"Unknown job_id: {jid!r}")
        if row["status"] != JobStatus.LEASED or row["lease_owner"] != wid:
            raise LeaseConflictError(f"Job {jid!r} not actively leased by {wid!r}")
        if tok is not None and row["lease_token"] != tok:
            raise LeaseConflictError(f"Stale lease_token for job {jid!r}")
        if row["lease_expires_at_s"] is None or float(row["lease_expires_at_s"]) <= now_s:
            raise LeaseConflictError(f"Lease on job {jid!r} expired")
        return row

    def renew_lease(self, job_id: str, worker_id: str, lease_s: float, *, lease_token: Optional[str] = None, now_s: Optional[float] = None) -> JobRecord:
        jid, wid = _valid_id(job_id, "job_id"), _valid_id(worker_id, "worker_id")
        if not (math.isfinite(lease_s) and lease_s > 0.0):
            raise InvalidParameterError("lease_s must be finite and > 0")
        ts = self._now(now_s)
        with self._immediate_tx() as conn:
            row = self._verify_lease(conn, jid, wid, lease_token, ts)
            exp = ts + float(lease_s)
            conn.execute("UPDATE jobs SET lease_expires_at_s=?, updated_at_s=? WHERE job_id=?", (exp, ts, jid))
            self._log_event(conn, jid, row["run_id"], "heartbeat", int(row["attempt"]), ts, wid, f"exp={exp:.3f}")
            return self._row_to_job(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (jid,)).fetchone())

    def complete_job(self, job_id: str, worker_id: str, *, lease_token: Optional[str] = None, result_summary: Optional[Mapping[str, Any]] = None, now_s: Optional[float] = None) -> JobRecord:
        jid, wid = _valid_id(job_id, "job_id"), _valid_id(worker_id, "worker_id")
        rjson = canonicalize_json_mapping(result_summary, "result_summary")[0] if result_summary is not None else None
        ts = self._now(now_s)
        with self._immediate_tx() as conn:
            row = self._verify_lease(conn, jid, wid, lease_token, ts)
            conn.execute("UPDATE jobs SET status=?, lease_owner=NULL, lease_token=NULL, lease_expires_at_s=NULL, result_json=?, updated_at_s=?, completed_at_s=? WHERE job_id=?", (JobStatus.SUCCEEDED, rjson, ts, ts, jid))
            self._log_event(conn, jid, row["run_id"], "succeeded", int(row["attempt"]), ts, wid)
            return self._row_to_job(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (jid,)).fetchone())

    def _retry_or_dlq(self, conn: sqlite3.Connection, row: sqlite3.Row, reason: str, ts: float, wid: Optional[str], retry_ev: str, dlq_ev: str, *, retryable: bool = True, delay_override: Optional[float] = None) -> JobRecord:
        jid, att, max_att = row["job_id"], int(row["attempt"]), int(row["max_attempts"])
        if retryable and att < max_att:
            delay = float(delay_override) if delay_override is not None else self._backoff_policy.compute_delay(att, job_id=jid)
            conn.execute("UPDATE jobs SET status=?, available_at_s=?, lease_owner=NULL, lease_token=NULL, lease_expires_at_s=NULL, last_error=?, updated_at_s=? WHERE job_id=?", (JobStatus.PENDING, ts + delay, reason, ts, jid))
            self._log_event(conn, jid, row["run_id"], retry_ev, att, ts, wid, f"delay_s={delay:.3f}")
        else:
            conn.execute("UPDATE jobs SET status=?, lease_owner=NULL, lease_token=NULL, lease_expires_at_s=NULL, last_error=?, updated_at_s=?, completed_at_s=? WHERE job_id=?", (JobStatus.DEAD_LETTER, reason, ts, ts, jid))
            self._log_event(conn, jid, row["run_id"], dlq_ev, att, ts, wid, reason)
        return self._row_to_job(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (jid,)).fetchone())

    def fail_job(self, job_id: str, worker_id: str, error_message: str, *, lease_token: Optional[str] = None, retryable: bool = True, retry_delay_s: Optional[float] = None, now_s: Optional[float] = None) -> JobRecord:
        jid, wid = _valid_id(job_id, "job_id"), _valid_id(worker_id, "worker_id")
        if not isinstance(error_message, str) or not error_message.strip():
            raise InvalidParameterError("error_message must be a non-empty string")
        if retry_delay_s is not None and not (math.isfinite(retry_delay_s) and retry_delay_s >= 0.0):
            raise InvalidParameterError("retry_delay_s must be finite and >= 0")
        ts = self._now(now_s)
        with self._immediate_tx() as conn:
            row = self._verify_lease(conn, jid, wid, lease_token, ts)
            return self._retry_or_dlq(conn, row, error_message, ts, wid, "retry_scheduled", "dead_lettered", retryable=retryable, delay_override=retry_delay_s)

    def reclaim_expired_leases(self, now_s: Optional[float] = None, *, run_id: Optional[str] = None) -> List[JobRecord]:
        ts = self._now(now_s)
        clauses, params = ["status = ?", "lease_expires_at_s <= ?"], [JobStatus.LEASED, ts]
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(_valid_id(run_id, "run_id"))
        with self._immediate_tx() as conn:
            rows = conn.execute(f"SELECT * FROM jobs WHERE {' AND '.join(clauses)} ORDER BY job_id ASC", params).fetchall()
            return [self._retry_or_dlq(conn, r, f"Lease expired for {r['lease_owner']!r}", ts, r["lease_owner"], "lease_expired_retry", "lease_expired_dead_letter") for r in rows]

    def register_artifact(self, job_id: str, role: str, file_path: Union[str, Path], *, worker_id: Optional[str] = None, lease_token: Optional[str] = None, expected_sha256: Optional[str] = None, expected_size: Optional[int] = None, metadata: Optional[Mapping[str, Any]] = None, now_s: Optional[float] = None) -> ArtifactRecord:
        jid, role = _valid_id(job_id, "job_id"), _valid_id(role, "role")
        m_json, _ = canonicalize_json_mapping(metadata, "metadata")
        ts, resolved = self._now(now_s), str(Path(file_path).resolve())
        sz, sha = compute_file_sha256(resolved)
        if expected_size is not None:
            if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
                raise InvalidParameterError("expected_size must be a non-negative int")
            if sz != expected_size:
                raise ArtifactVerificationError(f"Size mismatch for {resolved}: {sz} != {expected_size}")
        if expected_sha256 is not None:
            n_exp = _valid_id(expected_sha256, "expected_sha256").lower()
            if len(n_exp) != 64 or any(c not in "0123456789abcdef" for c in n_exp):
                raise InvalidParameterError("expected_sha256 must be a 64-character hex string")
            if sha != n_exp:
                raise ArtifactVerificationError(f"SHA-256 mismatch for {resolved}")

        art_id = "art-" + hashlib.sha256(f"{jid}\x1f{role}\x1f{resolved}".encode()).hexdigest()[:24]
        with self._immediate_tx() as conn:
            if worker_id is not None:
                jrow = self._verify_lease(conn, jid, _valid_id(worker_id, "worker_id"), lease_token, ts)
            elif (jrow := conn.execute("SELECT * FROM jobs WHERE job_id = ?", (jid,)).fetchone()) is None:
                raise JobNotFoundError(f"Unknown job_id: {jid!r}")
            if (ex := conn.execute("SELECT * FROM artifacts WHERE job_id=? AND role=? AND path=?", (jid, role, resolved)).fetchone()) is not None:
                if int(ex["size_bytes"]) != sz or ex["sha256"] != sha:
                    raise ArtifactVerificationError(f"Artifact mutated after registration: {resolved}")
                return self._row_to_artifact(ex)
            conn.execute("INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (art_id, jid, jrow["run_id"], role, resolved, sz, sha, m_json, ts))
            return self._row_to_artifact(conn.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (art_id,)).fetchone())

    def verify_registered_artifacts(self, *, run_id: Optional[str] = None, job_id: Optional[str] = None) -> ArtifactAuditReport:
        records = self.list_artifacts(run_id=run_id, job_id=job_id)
        valid_ids, missing_ids, mismatched_ids = [], [], []
        for rec in records:
            if not os.path.exists(rec.path):
                missing_ids.append(rec.artifact_id)
                continue
            try:
                sz, digest = compute_file_sha256(rec.path)
            except ArtifactVerificationError:
                mismatched_ids.append(rec.artifact_id)
                continue
            (valid_ids if (sz == rec.size_bytes and digest == rec.sha256) else mismatched_ids).append(rec.artifact_id)
        return ArtifactAuditReport(len(records), tuple(valid_ids), tuple(missing_ids), tuple(mismatched_ids))

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        row = self._get_conn().execute("SELECT * FROM jobs WHERE job_id = ?", (_valid_id(job_id, "job_id"),)).fetchone()
        return self._row_to_job(row) if row is not None else None

    @staticmethod
    def _build_where(**filters: Optional[str]) -> Tuple[str, List[str]]:
        clauses, params = [], []
        for col, val in filters.items():
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(val if col == "status" else _valid_id(val, col))
        return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), params

    def list_jobs(self, *, run_id: Optional[str] = None, status: Optional[str] = None) -> List[JobRecord]:
        if status is not None and status not in JobStatus.ALL:
            raise InvalidParameterError(f"Unknown status filter: {status!r}")
        where, params = self._build_where(run_id=run_id, status=status)
        return [self._row_to_job(r) for r in self._get_conn().execute(f"SELECT * FROM jobs {where} ORDER BY run_id ASC, job_id ASC", params).fetchall()]

    def get_job_events(self, job_id: str) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._get_conn().execute("SELECT * FROM job_events WHERE job_id = ? ORDER BY event_id ASC", (_valid_id(job_id, "job_id"),)).fetchall()]

    def list_artifacts(self, *, run_id: Optional[str] = None, job_id: Optional[str] = None) -> List[ArtifactRecord]:
        where, params = self._build_where(run_id=run_id, job_id=job_id)
        return [self._row_to_artifact(r) for r in self._get_conn().execute(f"SELECT * FROM artifacts {where} ORDER BY run_id ASC, job_id ASC, artifact_id ASC", params).fetchall()]

    def get_progress_summary(self, run_id: str, *, now_s: Optional[float] = None) -> RunProgress:
        run_id, ts, conn = _valid_id(run_id, "run_id"), self._now(now_s), self._get_conn()
        if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
            raise RunNotFoundError(f"Unregistered run_id: {run_id!r}")
        sql = "SELECT COUNT(*) AS total, COALESCE(SUM(CASE WHEN status='pending' AND available_at_s<=? THEN 1 ELSE 0 END), 0) AS ready, COALESCE(SUM(CASE WHEN status='pending' AND available_at_s>? THEN 1 ELSE 0 END), 0) AS delayed, COALESCE(SUM(CASE WHEN status='leased' THEN 1 ELSE 0 END), 0) AS leased, COALESCE(SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END), 0) AS succ, COALESCE(SUM(CASE WHEN status='dead_letter' THEN 1 ELSE 0 END), 0) AS dlq FROM jobs WHERE run_id=?"
        ja = conn.execute(sql, (ts, ts, run_id)).fetchone()
        aa = conn.execute("SELECT COUNT(*) AS cnt, COALESCE(SUM(size_bytes), 0) AS bsum FROM artifacts WHERE run_id=?", (run_id,)).fetchone()
        tot, succ = int(ja["total"]), int(ja["succ"])
        return RunProgress(run_id, tot, int(ja["ready"]), int(ja["delayed"]), int(ja["leased"]), succ, int(ja["dlq"]), int(aa["cnt"]), int(aa["bsum"]), (succ / float(tot)) if tot > 0 else 1.0)

    def build_manifest(self, *, run_id: Optional[str] = None) -> Dict[str, Any]:
        conn = self._get_conn()
        if run_id is not None:
            vrun = _valid_id(run_id, "run_id")
            if not (r_rows := conn.execute("SELECT * FROM runs WHERE run_id = ? ORDER BY run_id ASC", (vrun,)).fetchall()):
                raise RunNotFoundError(f"Unregistered run_id: {vrun!r}")
        else:
            r_rows = conn.execute("SELECT * FROM runs ORDER BY run_id ASC").fetchall()
        arts_by_job: Dict[str, List[Dict[str, Any]]] = {}
        for a in self.list_artifacts(run_id=run_id):
            arts_by_job.setdefault(a.job_id, []).append({k: v for k, v in dataclasses.asdict(a).items() if k not in ("job_id", "run_id")})
        runs = [{k: v for k, v in dataclasses.asdict(self._row_to_run(r)).items() if k not in ("config_json", "updated_at_s")} for r in r_rows]
        drop_j = {"params_json", "available_at_s", "lease_owner", "lease_token", "lease_expires_at_s", "updated_at_s"}
        jobs = [{**{k: v for k, v in dataclasses.asdict(j).items() if k not in drop_j}, "artifacts": arts_by_job.get(j.job_id, [])} for j in self.list_jobs(run_id=run_id)]
        return {"schema_version": 1, "runs": runs, "jobs": jobs}

    def export_manifest_json(self, output_path: Union[str, Path], *, run_id: Optional[str] = None) -> str:
        encoded = (json.dumps(self.build_manifest(run_id=run_id), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        _atomic_write(output_path, encoded)
        return hashlib.sha256(encoded).hexdigest()

    def export_manifest_csv(self, output_path: Union[str, Path], *, run_id: Optional[str] = None) -> int:
        if run_id is not None and self.get_run(run_id) is None:
            raise RunNotFoundError(f"Unregistered run_id: {run_id!r}")
        arts_by_job: Dict[str, List[ArtifactRecord]] = {}
        for art in self.list_artifacts(run_id=run_id):
            arts_by_job.setdefault(art.job_id, []).append(art)
        fields = ["run_id", "job_id", "task_type", "status", "priority", "attempt", "max_attempts", "params_sha256", "params_json", "artifact_id", "artifact_role", "artifact_path", "artifact_size_bytes", "artifact_sha256"]
        buf, count = io.StringIO(), 0
        writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for j in self.list_jobs(run_id=run_id):
            base = {k: getattr(j, k) for k in fields[:9]}
            for art in arts_by_job.get(j.job_id, [None]):  # type: ignore[list-item]
                extra = {k: "" for k in fields[9:]} if art is None else {"artifact_id": art.artifact_id, "artifact_role": art.role, "artifact_path": art.path, "artifact_size_bytes": art.size_bytes, "artifact_sha256": art.sha256}
                writer.writerow({**base, **extra})
                count += 1
        _atomic_write(output_path, buf.getvalue().encode("utf-8"))
        return count
