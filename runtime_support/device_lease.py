"""Two-tier fcntl hardware measurement lock coordinator and wakelock guard.

Implements process-safe hardware lease coordination using a shared (``LOCK_SH``)
global gate lock paired with an exclusive (``LOCK_EX``) per-serial hardware lock
over non-inheritable file descriptors, monotonic timeout polling, ``contextvars``
reentrancy rejection, transport-to-hardware ``ro.serialno`` verification, JSONL
lifecycle event logging, and a ``ScopedWakelockGuard`` that refuses pre-existing
shell wakelocks while guaranteeing verified release of owned locks.
"""

import contextvars
import dataclasses
import fcntl
import json
import math
import os
import pathlib
import re
import shlex
import subprocess
import time

_SERIAL_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_TOKEN_SAFE_RE = re.compile(r"^[A-Za-z0-9._:-]+$")

DEFAULT_SHELL_WAKELOCK_PREFIXES: tuple[str, ...] = (
    "shell_", "measure_", "runner_", "power_hold", "TemporaryLock",
)

_ACTIVE_LEASES: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "active_hardware_leases", default=frozenset(),
)


class DeviceLeaseError(Exception):
    """Base error for hardware lease coordination and wakelock guards."""


class LeaseValidationError(DeviceLeaseError):
    """Raised when lease parameters, serial tokens, or timeouts are invalid."""


class LeaseTimeoutError(DeviceLeaseError):
    """Raised when lock acquisition exceeds the monotonic timeout budget."""


class LeaseReentrancyError(DeviceLeaseError):
    """Raised when attempting a nested device lease within the same context."""


class IdentityProbeError(DeviceLeaseError):
    """Raised when querying remote hardware identity fails or returns empty data."""


class SerialMismatchError(IdentityProbeError):
    """Raised when remote ro.serialno does not match expected hardware serial."""


class WakelockError(DeviceLeaseError):
    """Base error for remote wakelock ownership guard failures."""


class ForeignWakelockError(WakelockError):
    """Raised when a pre-existing shell wakelock is already active on hardware."""


class WakelockAcquireError(WakelockError):
    """Raised when acquiring or confirming a bounded remote wakelock fails."""


class WakelockReleaseError(WakelockError):
    """Raised when releasing or confirming removal of an owned wakelock fails."""


def _validate_serial(serial: object, field_name: str = "serial") -> str:
    if not isinstance(serial, str):
        raise LeaseValidationError(f"{field_name} must be str, got {type(serial).__name__}")
    if not serial or serial.strip() != serial or not _SERIAL_SAFE_RE.match(serial):
        raise LeaseValidationError(f"Invalid {field_name} {serial!r}; expected [A-Za-z0-9._-]+")
    return serial


def _validate_non_negative_timeout(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LeaseValidationError(f"{field_name} must be numeric")
    as_float = float(value)
    if math.isnan(as_float) or math.isinf(as_float) or as_float < 0.0:
        raise LeaseValidationError(f"{field_name} must be finite and >= 0.0")
    return as_float


def _validate_positive_interval(value: object, field_name: str) -> float:
    as_float = _validate_non_negative_timeout(value, field_name)
    if as_float <= 0.0:
        raise LeaseValidationError(f"{field_name} must be > 0.0")
    return as_float


def _open_non_inheritable_lock_fd(path: pathlib.Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(os.fspath(path), flags, 0o600)
    try:
        os.set_inheritable(fd, False)
        if os.get_inheritable(fd):
            raise DeviceLeaseError(f"Failed to mark lock fd non-inheritable for {path}")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _poll_flock(
    fd: int, lock_mode: int, *, timeout_s: float, poll_interval_s: float,
    clock_fn: object, sleep_fn: object, resource_label: str,
) -> float:
    deadline_s = float(clock_fn()) + timeout_s
    while True:
        try:
            fcntl.flock(fd, lock_mode | fcntl.LOCK_NB)
            return float(clock_fn())
        except BlockingIOError as exc:
            remaining_s = deadline_s - float(clock_fn())
            if remaining_s <= 0.0:
                raise LeaseTimeoutError(
                    f"Timed out after {timeout_s:.3f}s waiting for {resource_label}"
                ) from exc
            sleep_fn(min(poll_interval_s, remaining_s))


class LeaseEventLogger:
    """Process-safe append-only JSONL lifecycle logger protected by flock."""

    def __init__(self, lock_dir: pathlib.Path, *, clock_ns: object = time.time_ns) -> None:
        self._lock_dir = lock_dir
        self._events_path = lock_dir / "lease_events.jsonl"
        self._guard_path = lock_dir / ".lease_events.lock"
        self._clock_ns = clock_ns

    @property
    def events_path(self) -> pathlib.Path:
        return self._events_path

    def log(self, event_name: str, details: dict[str, object]) -> None:
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        guard_fd = _open_non_inheritable_lock_fd(self._guard_path)
        try:
            fcntl.flock(guard_fd, fcntl.LOCK_EX)
            record = {
                "ts_ns": int(self._clock_ns()), "pid": os.getpid(),
                "event": event_name, **details,
            }
            encoded = json.dumps(
                record, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8") + b"\n"
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
            out_fd = os.open(os.fspath(self._events_path), flags, 0o600)
            try:
                os.set_inheritable(out_fd, False)
                view = memoryview(encoded)
                written = 0
                while written < len(encoded):
                    step = os.write(out_fd, view[written:])
                    if step <= 0:
                        raise OSError("Zero-byte write while appending lease event")
                    written += step
                os.fsync(out_fd)
            finally:
                os.close(out_fd)
        finally:
            try:
                fcntl.flock(guard_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(guard_fd)

    def read_events(self) -> list[dict[str, object]]:
        if not self._events_path.exists():
            return []
        rows: list[dict[str, object]] = []
        for raw_line in self._events_path.read_text(encoding="utf-8").splitlines():
            if raw_line.strip():
                parsed = json.loads(raw_line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
        return rows


def adb_ro_serialno_probe(
    transport_target: str, *, adb_binary: str = "adb", timeout_s: float = 5.0,
) -> str:
    """Default subprocess probe querying ``getprop ro.serialno`` over ADB."""
    _validate_serial(transport_target, "transport_target")
    cmd = [adb_binary, "-s", transport_target, "shell", "getprop", "ro.serialno"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise IdentityProbeError(f"Transport probe failed for {transport_target}: {exc}") from exc
    if proc.returncode != 0:
        raise IdentityProbeError(f"ro.serialno exited {proc.returncode}: {proc.stderr.strip()}")
    actual = proc.stdout.strip()
    if not actual:
        raise IdentityProbeError(f"ro.serialno returned empty serial for {transport_target}")
    return actual


class LeaseHandle:
    """Active two-tier hardware lease holding shared-gate and exclusive-serial locks."""

    def __init__(
        self, *, expected_serial: str, verified_serial: str, transport_target: str,
        holder_id: str, acquired_at_s: float, gate_fd: int, serial_fd: int,
        clock_fn: object, logger: LeaseEventLogger,
    ) -> None:
        self.expected_serial = expected_serial
        self.verified_serial = verified_serial
        self.transport_target = transport_target
        self.holder_id = holder_id
        self.acquired_at_s = acquired_at_s
        self._gate_fd: int | None = gate_fd
        self._serial_fd: int | None = serial_fd
        self._clock_fn = clock_fn
        self._logger = logger
        self._released = False

    @property
    def is_active(self) -> bool:
        return not self._released

    @property
    def fds_non_inheritable(self) -> bool:
        if self._released or self._gate_fd is None or self._serial_fd is None:
            return True
        return (not os.get_inheritable(self._gate_fd)) and (not os.get_inheritable(self._serial_fd))

    def __enter__(self) -> "LeaseHandle":
        if self._released:
            raise DeviceLeaseError("Cannot re-enter a released LeaseHandle")
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self.release()
        return False

    def release(self) -> None:
        """Idempotently release per-serial lock, global gate lock, and context state."""
        if self._released:
            return
        self._released = True
        held_duration_s = max(0.0, float(self._clock_fn()) - self.acquired_at_s)
        current = _ACTIVE_LEASES.get()
        if self.expected_serial in current:
            _ACTIVE_LEASES.set(current - frozenset({self.expected_serial}))

        try:
            if self._serial_fd is not None:
                try:
                    fcntl.flock(self._serial_fd, fcntl.LOCK_UN)
                finally:
                    os.close(self._serial_fd)
                    self._serial_fd = None
        finally:
            if self._gate_fd is not None:
                try:
                    fcntl.flock(self._gate_fd, fcntl.LOCK_UN)
                finally:
                    os.close(self._gate_fd)
                    self._gate_fd = None

        self._logger.log("lease_released", {
            "serial": self.expected_serial, "transport_target": self.transport_target,
            "holder_id": self.holder_id, "held_duration_s": round(held_duration_s, 6),
        })


class GlobalGateBarrier:
    """Exclusive global gate holder (``LOCK_EX``) that drains all device leases."""

    def __init__(
        self, gate_path: pathlib.Path, *, holder_id: str = "barrier",
        timeout_s: float = 10.0, poll_interval_s: float = 0.05,
        clock_fn: object = time.monotonic, sleep_fn: object = time.sleep,
        logger: LeaseEventLogger | None = None,
    ) -> None:
        self._gate_path = gate_path
        self._holder_id = holder_id
        self._timeout_s = _validate_non_negative_timeout(timeout_s, "timeout_s")
        self._poll_interval_s = _validate_positive_interval(poll_interval_s, "poll_interval_s")
        self._clock_fn = clock_fn
        self._sleep_fn = sleep_fn
        self._logger = logger
        self._fd: int | None = None

    def acquire(self) -> "GlobalGateBarrier":
        if self._fd is not None:
            raise LeaseReentrancyError("GlobalGateBarrier is already held")
        fd = _open_non_inheritable_lock_fd(self._gate_path)
        try:
            _poll_flock(
                fd, fcntl.LOCK_EX, timeout_s=self._timeout_s,
                poll_interval_s=self._poll_interval_s, clock_fn=self._clock_fn,
                sleep_fn=self._sleep_fn, resource_label="global_gate_exclusive",
            )
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        if self._logger is not None:
            self._logger.log("global_barrier_acquired", {"holder_id": self._holder_id})
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        if self._logger is not None:
            self._logger.log("global_barrier_released", {"holder_id": self._holder_id})

    def __enter__(self) -> "GlobalGateBarrier":
        return self.acquire()

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self.release()
        return False


class DeviceLeaseCoordinator:
    """Two-tier hardware lease coordinator enforcing serial identity and context isolation."""

    GLOBAL_GATE_FILENAME = "global_gate.lock"

    def __init__(
        self, lock_dir: str | pathlib.Path, *, serial_probe: object = adb_ro_serialno_probe,
        clock_fn: object = time.monotonic, sleep_fn: object = time.sleep,
        wall_clock_ns: object = time.time_ns,
    ) -> None:
        if not callable(serial_probe) or not callable(clock_fn) or not callable(sleep_fn):
            raise LeaseValidationError("serial_probe, clock_fn, and sleep_fn must be callable")
        self._lock_dir = pathlib.Path(lock_dir)
        self._devices_dir = self._lock_dir / "devices"
        self._serial_probe = serial_probe
        self._clock_fn = clock_fn
        self._sleep_fn = sleep_fn
        self._logger = LeaseEventLogger(self._lock_dir, clock_ns=wall_clock_ns)

    @property
    def lock_dir(self) -> pathlib.Path:
        return self._lock_dir

    @property
    def events_path(self) -> pathlib.Path:
        return self._logger.events_path

    def read_events(self) -> list[dict[str, object]]:
        return self._logger.read_events()

    def global_barrier(
        self, *, holder_id: str = "maintenance", timeout_s: float = 10.0, poll_interval_s: float = 0.05,
    ) -> GlobalGateBarrier:
        return GlobalGateBarrier(
            self._lock_dir / self.GLOBAL_GATE_FILENAME, holder_id=holder_id,
            timeout_s=timeout_s, poll_interval_s=poll_interval_s,
            clock_fn=self._clock_fn, sleep_fn=self._sleep_fn, logger=self._logger,
        )

    def acquire(
        self, expected_serial: str, *, transport_target: str | None = None,
        holder_id: str = "local", timeout_s: float = 10.0, poll_interval_s: float = 0.05,
    ) -> LeaseHandle:
        """Acquire shared global gate + exclusive serial lock and verify ro.serialno."""
        _validate_serial(expected_serial, "expected_serial")
        target = expected_serial if transport_target is None else transport_target
        _validate_serial(target, "transport_target")
        _validate_serial(holder_id, "holder_id")
        budget_s = _validate_non_negative_timeout(timeout_s, "timeout_s")
        interval_s = _validate_positive_interval(poll_interval_s, "poll_interval_s")

        active_set = _ACTIVE_LEASES.get()
        if active_set:
            self._logger.log("lease_reentrancy_rejected", {
                "requested_serial": expected_serial,
                "active_serials": sorted(active_set), "holder_id": holder_id,
            })
            raise LeaseReentrancyError(
                f"Context already holds lease(s) {sorted(active_set)}; nested acquire forbidden"
            )

        self._devices_dir.mkdir(parents=True, exist_ok=True)
        start_s = float(self._clock_fn())
        self._logger.log("lease_acquire_attempt", {
            "serial": expected_serial, "transport_target": target,
            "holder_id": holder_id, "timeout_s": budget_s,
        })

        gate_fd = _open_non_inheritable_lock_fd(self._lock_dir / self.GLOBAL_GATE_FILENAME)
        serial_fd: int | None = None
        try:
            after_gate_s = _poll_flock(
                gate_fd, fcntl.LOCK_SH, timeout_s=budget_s, poll_interval_s=interval_s,
                clock_fn=self._clock_fn, sleep_fn=self._sleep_fn,
                resource_label=f"global_gate_shared({expected_serial})",
            )
            remaining_s = max(0.0, budget_s - max(0.0, after_gate_s - start_s))
            serial_fd = _open_non_inheritable_lock_fd(self._devices_dir / f"{expected_serial}.lock")
            acquired_s = _poll_flock(
                serial_fd, fcntl.LOCK_EX, timeout_s=remaining_s, poll_interval_s=interval_s,
                clock_fn=self._clock_fn, sleep_fn=self._sleep_fn,
                resource_label=f"device_serial_exclusive({expected_serial})",
            )

            try:
                probed_raw = self._serial_probe(target)
            except IdentityProbeError:
                raise
            except Exception as exc:
                raise IdentityProbeError(f"Probe failed for {target!r}: {exc}") from exc

            if not isinstance(probed_raw, str) or not probed_raw.strip():
                raise IdentityProbeError(f"Probe returned blank serial for {target!r}")
            probed_serial = probed_raw.strip()
            if probed_serial != expected_serial:
                self._logger.log("serial_mismatch", {
                    "expected_serial": expected_serial, "actual_serial": probed_serial,
                    "transport_target": target, "holder_id": holder_id,
                })
                raise SerialMismatchError(
                    f"Transport {target!r} reported ro.serialno={probed_serial!r}, "
                    f"expected {expected_serial!r}"
                )
        except LeaseTimeoutError:
            self._logger.log("lease_acquire_timeout", {
                "serial": expected_serial, "transport_target": target,
                "holder_id": holder_id, "timeout_s": budget_s,
            })
            self._rollback_fds(serial_fd, gate_fd)
            raise
        except IdentityProbeError as exc:
            if not isinstance(exc, SerialMismatchError):
                self._logger.log("identity_probe_failed", {
                    "serial": expected_serial, "transport_target": target,
                    "holder_id": holder_id, "reason": str(exc),
                })
            self._rollback_fds(serial_fd, gate_fd)
            raise
        except BaseException:
            self._rollback_fds(serial_fd, gate_fd)
            raise

        _ACTIVE_LEASES.set(active_set | frozenset({expected_serial}))
        self._logger.log("lease_acquired", {
            "serial": expected_serial, "verified_serial": probed_serial,
            "transport_target": target, "holder_id": holder_id,
        })
        return LeaseHandle(
            expected_serial=expected_serial, verified_serial=probed_serial,
            transport_target=target, holder_id=holder_id, acquired_at_s=acquired_s,
            gate_fd=gate_fd, serial_fd=serial_fd, clock_fn=self._clock_fn, logger=self._logger,
        )

    @staticmethod
    def _rollback_fds(serial_fd: int | None, gate_fd: int | None) -> None:
        for fd in (serial_fd, gate_fd):
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    os.close(fd)
                except OSError:
                    pass


@dataclasses.dataclass(frozen=True)
class ShellWakelockAdapter:
    """Adapter managing ``/sys/power/wake_lock`` via an injectable shell runner."""

    run_shell_fn: object
    wake_lock_node: str = "/sys/power/wake_lock"
    wake_unlock_node: str = "/sys/power/wake_unlock"

    def list_active_tokens(self) -> tuple[str, ...]:
        rc, stdout, stderr = self.run_shell_fn(f"cat {shlex.quote(self.wake_lock_node)}")
        if rc != 0:
            raise WakelockError(f"Query {self.wake_lock_node} failed (rc={rc}): {stderr.strip()}")
        return tuple(tok for tok in stdout.split() if tok)

    def acquire_bounded(self, lock_token: str, timeout_ns: int) -> None:
        payload = shlex.quote(f"{lock_token} {timeout_ns}")
        rc, _, stderr = self.run_shell_fn(f"printf '%s' {payload} > {shlex.quote(self.wake_lock_node)}")
        if rc != 0:
            raise WakelockAcquireError(f"Write {lock_token!r} failed (rc={rc}): {stderr.strip()}")

    def release_token(self, lock_token: str) -> None:
        payload = shlex.quote(lock_token)
        rc, _, stderr = self.run_shell_fn(f"printf '%s' {payload} > {shlex.quote(self.wake_unlock_node)}")
        if rc != 0:
            raise WakelockReleaseError(f"Unlock {lock_token!r} failed (rc={rc}): {stderr.strip()}")


class ScopedWakelockGuard:
    """Ownership-tracked wakelock guard refusing pre-existing shell wakelocks."""

    def __init__(
        self, backend: ShellWakelockAdapter, *, lock_name: str = "measure_runner_hold",
        timeout_s: float = 300.0, shell_prefixes: tuple[str, ...] = DEFAULT_SHELL_WAKELOCK_PREFIXES,
        logger: LeaseEventLogger | None = None,
    ) -> None:
        if not isinstance(lock_name, str) or not _TOKEN_SAFE_RE.match(lock_name):
            raise LeaseValidationError(f"Invalid wakelock token {lock_name!r}")
        valid_timeout = _validate_positive_interval(timeout_s, "timeout_s")
        self._backend = backend
        self._lock_name = lock_name
        self._timeout_ns = int(valid_timeout * 1_000_000_000)
        self._shell_prefixes = tuple(shell_prefixes)
        self._logger = logger
        self._owned = False
        self._entered = False

    @property
    def lock_name(self) -> str:
        return self._lock_name

    @property
    def is_owned(self) -> bool:
        return self._owned

    def _detect_conflicts(self, active_tokens: tuple[str, ...]) -> list[str]:
        return [
            tok for tok in active_tokens
            if tok == self._lock_name or any(tok.startswith(p) for p in self._shell_prefixes)
        ]

    def acquire(self) -> "ScopedWakelockGuard":
        if self._entered or self._owned:
            raise LeaseReentrancyError(f"ScopedWakelockGuard {self._lock_name!r} entered twice")
        conflicts = self._detect_conflicts(self._backend.list_active_tokens())
        if conflicts:
            if self._logger is not None:
                self._logger.log("wakelock_refused_foreign", {
                    "lock_name": self._lock_name, "conflicts": conflicts,
                })
            raise ForeignWakelockError(
                f"Refusing wakelock {self._lock_name!r}; foreign shell wakelocks held: {conflicts}"
            )
        self._backend.acquire_bounded(self._lock_name, self._timeout_ns)
        active_after = self._backend.list_active_tokens()
        if self._lock_name not in active_after:
            if self._logger is not None:
                self._logger.log("wakelock_acquire_unverified", {
                    "lock_name": self._lock_name, "active_tokens": list(active_after),
                })
            raise WakelockAcquireError(f"Wakelock {self._lock_name!r} inactive after bounded write")
        self._owned = True
        self._entered = True
        if self._logger is not None:
            self._logger.log("wakelock_acquired", {
                "lock_name": self._lock_name, "timeout_ns": self._timeout_ns,
            })
        return self

    def release(self) -> None:
        """Release and verify removal strictly when owned by this guard instance."""
        if not self._owned:
            return
        self._owned = False
        self._backend.release_token(self._lock_name)
        remaining = self._backend.list_active_tokens()
        if self._lock_name in remaining:
            if self._logger is not None:
                self._logger.log("wakelock_release_unverified", {
                    "lock_name": self._lock_name, "remaining": list(remaining),
                })
            raise WakelockReleaseError(f"Wakelock {self._lock_name!r} remained active after unlock")
        if self._logger is not None:
            self._logger.log("wakelock_released", {"lock_name": self._lock_name})

    def __enter__(self) -> "ScopedWakelockGuard":
        return self.acquire()

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        if not self._owned:
            return False
        try:
            self.release()
        except BaseException as release_exc:
            if exc_val is not None and isinstance(exc_val, BaseException):
                raise release_exc from exc_val
            raise
        return False
