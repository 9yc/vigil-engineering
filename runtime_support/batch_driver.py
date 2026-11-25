"""Cooled fixed-size batch execution driver with fail-closed safety gates.

Partitions a locked schedule of ``(sample_id, config_id)`` cells across repeats
into non-overlapping fixed-size batches, enforces pre-batch and post-batch
hardware safety gates (AC/USB/Wireless unplugged, battery SoC floor, battery and
skin temperature ceilings, and zero stray runner PIDs), runs a bounded inter-batch
cooldown loop until ``skin_c <= cool_target_c``, validates batch output rows
against the manifest schedule and energy-validity gate, refuses to clobber
existing batch output directories, and writes an exclusive fsynced JSONL log.
"""

import dataclasses
import fcntl
import hashlib
import json
import math
import os
import pathlib
import time


class BatchDriverError(Exception):
    """Base error for cooled batch execution driver failures."""


class ScheduleValidationError(BatchDriverError):
    """Raised when schedule cells, repeats, or batch sizes are malformed."""


class DriverLockError(BatchDriverError):
    """Raised when another driver instance holds the output directory lock."""


class BatchOutputExistsError(BatchDriverError):
    """Raised when batch output directories or logs already exist on disk."""


class SafetyGateError(BatchDriverError):
    """Base error for fail-closed pre-batch or post-batch hardware safety gates."""


class ChargerPluggedError(SafetyGateError):
    """Raised when AC, USB, or Wireless power input is online."""


class BatteryFloorError(SafetyGateError):
    """Raised when battery state-of-charge falls below the configured floor."""


class ThermalCeilingError(SafetyGateError):
    """Raised when battery or skin temperature reaches or exceeds ceiling."""


class StrayRunnerGateError(SafetyGateError):
    """Raised when stray runner PIDs are detected during a safety gate check."""


class CooldownTimeoutError(SafetyGateError):
    """Raised when skin temperature fails to reach cool_target_c within max_cool_s."""


class BatchRowValidationError(BatchDriverError):
    """Raised when batch output rows violate manifest completeness or schema."""


class EnergyValidityError(BatchRowValidationError):
    """Raised when a batch output row fails the hardware energy-validity gate."""


def _check_id(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ScheduleValidationError(f"{name} must be str, got {type(value).__name__}")
    if not value or value.strip() != value:
        raise ScheduleValidationError(f"{name} must be non-empty without outer whitespace")
    return value


def _check_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BatchDriverError(f"{name} must be numeric, got {type(value).__name__}")
    as_float = float(value)
    if math.isnan(as_float) or math.isinf(as_float):
        raise BatchDriverError(f"{name} must be finite, got {value!r}")
    return as_float


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


@dataclasses.dataclass(frozen=True)
class ScheduleCell:
    """Single scheduled measurement cell identified by repeat, sample, and config."""

    sample_id: str
    config_id: str
    repeat_idx: int = 0
    parameters: dict[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_id(self.sample_id, "sample_id")
        _check_id(self.config_id, "config_id")
        if isinstance(self.repeat_idx, bool) or not isinstance(self.repeat_idx, int) or self.repeat_idx < 0:
            raise ScheduleValidationError(f"Invalid repeat_idx {self.repeat_idx!r}")
        if not isinstance(self.parameters, dict):
            raise ScheduleValidationError("parameters must be a dict")

    @property
    def coordinate(self) -> tuple[int, str, str]:
        return (self.repeat_idx, self.sample_id, self.config_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "repeat_idx": self.repeat_idx, "sample_id": self.sample_id,
            "config_id": self.config_id, "parameters": dict(self.parameters),
        }


@dataclasses.dataclass(frozen=True)
class BatchSlice:
    """Non-overlapping fixed-size partition of a locked measurement schedule."""

    batch_index: int
    batch_id: str
    cells: tuple[ScheduleCell, ...]

    def manifest_dict(self, schedule_digest: str) -> dict[str, object]:
        return {
            "batch_index": self.batch_index, "batch_id": self.batch_id,
            "schedule_digest": schedule_digest, "cell_count": len(self.cells),
            "cells": [c.to_dict() for c in self.cells],
        }


class LockedSchedule:
    """Immutable schedule of unique ``(repeat_idx, sample_id, config_id)`` cells."""

    def __init__(
        self, schedule_id: str, cells: list[ScheduleCell] | tuple[ScheduleCell, ...],
    ) -> None:
        _check_id(schedule_id, "schedule_id")
        if not isinstance(cells, (list, tuple)) or not cells:
            raise ScheduleValidationError("LockedSchedule requires at least one ScheduleCell")
        normalized = tuple(cells)
        seen: set[tuple[int, str, str]] = set()
        for cell in normalized:
            if not isinstance(cell, ScheduleCell):
                raise ScheduleValidationError("All schedule entries must be ScheduleCell")
            if cell.coordinate in seen:
                raise ScheduleValidationError(f"Duplicate cell coordinate: {cell.coordinate!r}")
            seen.add(cell.coordinate)
        self.schedule_id = schedule_id
        self.cells = normalized
        self.digest = hashlib.sha256(_canonical_json_bytes({
            "schedule_id": self.schedule_id, "cells": [c.to_dict() for c in self.cells],
        })).hexdigest()

    @classmethod
    def from_matrix(
        cls, schedule_id: str,
        base_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...],
        *, repeats: int = 1,
    ) -> "LockedSchedule":
        """Expand ``(sample_id, config_id)`` pairs across ``repeats`` in order."""
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ScheduleValidationError(f"repeats must be int >= 1, got {repeats!r}")
        if not base_pairs:
            raise ScheduleValidationError("base_pairs must be non-empty")
        seen_pairs: set[tuple[str, str]] = set()
        for pair in base_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ScheduleValidationError("Each base pair must be (sample_id, config_id)")
            if pair in seen_pairs:
                raise ScheduleValidationError(f"Duplicate base pair {pair!r}")
            seen_pairs.add(pair)
        expanded = [
            ScheduleCell(sample_id=s_id, config_id=c_id, repeat_idx=rep)
            for rep in range(repeats)
            for s_id, c_id in base_pairs
        ]
        return cls(schedule_id=schedule_id, cells=expanded)

    def partition(self, batch_size: int) -> tuple[BatchSlice, ...]:
        """Partition schedule into non-overlapping fixed-size ``BatchSlice`` items."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ScheduleValidationError(f"batch_size must be int >= 1, got {batch_size!r}")
        return tuple(
            BatchSlice(batch_idx, f"batch_{batch_idx:04d}", self.cells[off : off + batch_size])
            for batch_idx, off in enumerate(range(0, len(self.cells), batch_size))
        )


@dataclasses.dataclass(frozen=True)
class HardwareTelemetry:
    """Point-in-time power, battery, thermal, and stray-PID hardware telemetry."""

    ac_online: bool
    usb_online: bool
    wireless_online: bool
    battery_soc_pct: float
    battery_c: float
    skin_c: float
    stray_pids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "ac_online": self.ac_online, "usb_online": self.usb_online,
            "wireless_online": self.wireless_online, "battery_soc_pct": self.battery_soc_pct,
            "battery_c": self.battery_c, "skin_c": self.skin_c,
            "stray_pids": list(self.stray_pids),
        }


@dataclasses.dataclass(frozen=True)
class SafetyGateConfig:
    """Fail-closed hardware safety thresholds and inter-batch cooldown parameters."""

    min_battery_soc_pct: float = 30.0
    max_battery_c: float = 40.0
    max_skin_c: float = 39.0
    cool_target_c: float = 34.0
    max_cool_s: float = 300.0
    cool_poll_s: float = 5.0
    min_row_energy_mj: float = 0.001
    max_row_energy_mj: float = 500_000.0

    def __post_init__(self) -> None:
        soc = _check_finite(self.min_battery_soc_pct, "min_battery_soc_pct")
        bat_c = _check_finite(self.max_battery_c, "max_battery_c")
        skin_c = _check_finite(self.max_skin_c, "max_skin_c")
        cool_c = _check_finite(self.cool_target_c, "cool_target_c")
        max_cool = _check_finite(self.max_cool_s, "max_cool_s")
        poll_s = _check_finite(self.cool_poll_s, "cool_poll_s")
        min_e = _check_finite(self.min_row_energy_mj, "min_row_energy_mj")
        max_e = _check_finite(self.max_row_energy_mj, "max_row_energy_mj")
        if not (0.0 <= soc <= 100.0):
            raise ScheduleValidationError("min_battery_soc_pct must be within [0, 100]")
        if bat_c <= 0.0 or skin_c <= 0.0 or cool_c <= 0.0 or cool_c >= skin_c:
            raise ScheduleValidationError("cool_target_c must be > 0 and < max_skin_c")
        if max_cool < 0.0 or poll_s <= 0.0 or min_e <= 0.0 or max_e < min_e:
            raise ScheduleValidationError("Invalid cooldown or row energy validity bounds")


def evaluate_safety_gate(
    telemetry: HardwareTelemetry, cfg: SafetyGateConfig, *,
    stage: str, check_thermal_ceilings: bool = True,
) -> None:
    """Enforce fail-closed power, SoC, stray-PID, and thermal ceiling checks."""
    if not isinstance(telemetry, HardwareTelemetry):
        raise SafetyGateError(f"[{stage}] Telemetry sample is not HardwareTelemetry")
    for field_name in ("ac_online", "usb_online", "wireless_online"):
        if not isinstance(getattr(telemetry, field_name), bool):
            raise ChargerPluggedError(f"[{stage}] {field_name} must be strict bool")
    if telemetry.ac_online or telemetry.usb_online or telemetry.wireless_online:
        raise ChargerPluggedError(
            f"[{stage}] External power connected: ac={telemetry.ac_online}, "
            f"usb={telemetry.usb_online}, wireless={telemetry.wireless_online}"
        )

    soc = _check_finite(telemetry.battery_soc_pct, "battery_soc_pct")
    if soc < cfg.min_battery_soc_pct or soc > 100.0:
        raise BatteryFloorError(f"[{stage}] Battery SoC {soc:.1f}% < {cfg.min_battery_soc_pct:.1f}%")
    if telemetry.stray_pids:
        raise StrayRunnerGateError(f"[{stage}] Stray runner PIDs present: {list(telemetry.stray_pids)}")

    bat_c = _check_finite(telemetry.battery_c, "battery_c")
    skin_c = _check_finite(telemetry.skin_c, "skin_c")
    if check_thermal_ceilings:
        if bat_c >= cfg.max_battery_c:
            raise ThermalCeilingError(f"[{stage}] Battery temp {bat_c:.2f}C >= {cfg.max_battery_c:.2f}C")
        if skin_c >= cfg.max_skin_c:
            raise ThermalCeilingError(f"[{stage}] Skin temp {skin_c:.2f}C >= {cfg.max_skin_c:.2f}C")


def validate_batch_rows(
    batch_slice: BatchSlice, rows: list[dict[str, object]], cfg: SafetyGateConfig,
) -> list[dict[str, object]]:
    """Validate worker output rows against manifest cells and energy-validity gate."""
    if not isinstance(rows, list):
        raise BatchRowValidationError(f"Batch {batch_slice.batch_id} rows must be a list")
    expected_coords = [cell.coordinate for cell in batch_slice.cells]
    expected_set = set(expected_coords)

    by_coord: dict[tuple[int, str, str], dict[str, object]] = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BatchRowValidationError(f"Batch {batch_slice.batch_id} row[{idx}] not a dict")
        rep, s_id, c_id = row.get("repeat_idx"), row.get("sample_id"), row.get("config_id")
        if isinstance(rep, bool) or not isinstance(rep, int) or not isinstance(s_id, str) or not isinstance(c_id, str):
            raise BatchRowValidationError(f"Batch {batch_slice.batch_id} row[{idx}] bad coordinates")
        coord = (rep, s_id, c_id)
        if coord not in expected_set:
            raise BatchRowValidationError(f"Batch {batch_slice.batch_id} unscheduled row {coord!r}")
        if coord in by_coord:
            raise BatchRowValidationError(f"Batch {batch_slice.batch_id} duplicate row {coord!r}")
        if row.get("status") != "ok":
            raise BatchRowValidationError(f"Batch {batch_slice.batch_id} row {coord!r} status != 'ok'")
        if row.get("energy_valid") is not True:
            raise EnergyValidityError(f"Batch {batch_slice.batch_id} row {coord!r} energy_valid != True")
        if row.get("charger_unplugged") is False:
            raise EnergyValidityError(f"Batch {batch_slice.batch_id} row {coord!r} charger connected")
        try:
            energy_mj = _check_finite(row.get("energy_mj"), "energy_mj")
            duration_s = _check_finite(row.get("duration_s"), "duration_s")
        except BatchDriverError as exc:
            raise EnergyValidityError(f"Batch {batch_slice.batch_id} row {coord!r}: {exc}") from exc
        if not (cfg.min_row_energy_mj <= energy_mj <= cfg.max_row_energy_mj):
            raise EnergyValidityError(f"Batch {batch_slice.batch_id} row {coord!r} energy_mj={energy_mj}")
        if duration_s <= 0.0:
            raise EnergyValidityError(f"Batch {batch_slice.batch_id} row {coord!r} duration_s <= 0")
        by_coord[coord] = dict(row)

    missing = [coord for coord in expected_coords if coord not in by_coord]
    if missing:
        raise BatchRowValidationError(f"Batch {batch_slice.batch_id} missing scheduled rows: {missing}")
    return [by_coord[coord] for coord in expected_coords]


@dataclasses.dataclass(frozen=True)
class DriverRunSummary:
    """Summary returned upon successful completion of all scheduled batches."""

    schedule_id: str
    schedule_digest: str
    total_batches: int
    total_cells: int
    cooldown_waits: int
    total_cooldown_s: float
    batch_dirs: tuple[pathlib.Path, ...]


class ExclusiveDriverLog:
    """Exclusive flock-guarded append-only fsynced JSONL log for the batch driver."""

    def __init__(self, output_root: pathlib.Path, *, clock_ns: object = time.time_ns) -> None:
        self._root = output_root
        self._lock_path = output_root / "driver.lock"
        self._log_path = output_root / "driver_events.jsonl"
        self._clock_ns = clock_ns
        self._lock_fd: int | None = None
        self._log_fd: int | None = None
        self._seq = 0

    def open_exclusive(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        lfd = os.open(
            os.fspath(self._lock_path),
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o600,
        )
        try:
            os.set_inheritable(lfd, False)
            fcntl.flock(lfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(lfd)
            raise DriverLockError(f"Driver lock {self._lock_path} is already held") from exc
        except BaseException:
            os.close(lfd)
            raise

        if self._log_path.exists():
            fcntl.flock(lfd, fcntl.LOCK_UN)
            os.close(lfd)
            raise BatchOutputExistsError(f"Driver log already exists at {self._log_path}")

        log_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        try:
            gfd = os.open(os.fspath(self._log_path), log_flags, 0o600)
            os.set_inheritable(gfd, False)
        except BaseException:
            fcntl.flock(lfd, fcntl.LOCK_UN)
            os.close(lfd)
            raise
        self._lock_fd, self._log_fd = lfd, gfd

    def append(self, event: str, payload: dict[str, object]) -> None:
        if self._log_fd is None:
            raise BatchDriverError("Driver log is not open")
        self._seq += 1
        rec = {"seq": self._seq, "ts_ns": int(self._clock_ns()), "event": event, **payload}
        encoded = _canonical_json_bytes(rec) + b"\n"
        view = memoryview(encoded)
        written = 0
        while written < len(encoded):
            n = os.write(self._log_fd, view[written:])
            if n <= 0:
                raise OSError("Zero-byte write to driver_events.jsonl")
            written += n
        os.fsync(self._log_fd)

    def close(self) -> None:
        if self._log_fd is not None:
            for op in (lambda: os.fsync(self._log_fd), lambda: os.close(self._log_fd)):
                try:
                    op()
                except OSError:
                    pass
            self._log_fd = None
        if self._lock_fd is not None:
            for op in (
                lambda: fcntl.flock(self._lock_fd, fcntl.LOCK_UN),
                lambda: os.close(self._lock_fd),
            ):
                try:
                    op()
                except OSError:
                    pass
            self._lock_fd = None


class CooledBatchDriver:
    """Executes fixed-size batches with pre/post safety gates and inter-batch cooling."""

    def __init__(
        self, *, output_root: str | pathlib.Path, schedule: LockedSchedule, batch_size: int,
        safety_config: SafetyGateConfig, telemetry_probe: object, batch_worker: object,
        clock_fn: object = time.monotonic, sleep_fn: object = time.sleep,
        wall_clock_ns: object = time.time_ns,
    ) -> None:
        if not isinstance(schedule, LockedSchedule) or not isinstance(safety_config, SafetyGateConfig):
            raise ScheduleValidationError("Invalid schedule or safety_config type")
        if not all(callable(fn) for fn in (telemetry_probe, batch_worker, clock_fn, sleep_fn)):
            raise ScheduleValidationError("Probe, worker, clock, and sleep must be callable")
        self._output_root = pathlib.Path(output_root)
        self._batches_root = self._output_root / "batches"
        self._schedule = schedule
        self._slices = schedule.partition(batch_size)
        self._cfg = safety_config
        self._probe = telemetry_probe
        self._worker = batch_worker
        self._clock_fn = clock_fn
        self._sleep_fn = sleep_fn
        self._wall_clock_ns = wall_clock_ns

    @property
    def slices(self) -> tuple[BatchSlice, ...]:
        return self._slices

    def _verify_clean_output_tree(self) -> None:
        for path in (self._batches_root, self._output_root / "run_summary.json"):
            if path.exists():
                raise BatchOutputExistsError(f"Output artifact already exists: {path}")

    def _run_inter_batch_cooldown(self, batch_id: str, log: ExclusiveDriverLog) -> tuple[int, float]:
        """Poll telemetry until skin_c <= cool_target_c within max_cool_s."""
        start_s = float(self._clock_fn())
        deadline_s = start_s + self._cfg.max_cool_s
        polls = 0
        while True:
            polls += 1
            sample = self._probe()
            evaluate_safety_gate(
                sample, self._cfg, stage=f"cooldown:{batch_id}", check_thermal_ceilings=False,
            )
            if sample.battery_c >= self._cfg.max_battery_c:
                raise ThermalCeilingError(
                    f"[cooldown:{batch_id}] Battery temp {sample.battery_c:.2f}C >= {self._cfg.max_battery_c:.2f}C"
                )
            if sample.skin_c <= self._cfg.cool_target_c:
                elapsed_s = max(0.0, float(self._clock_fn()) - start_s)
                log.append("cooldown_complete", {
                    "batch_id": batch_id, "polls": polls,
                    "elapsed_s": round(elapsed_s, 6), "skin_c": sample.skin_c,
                })
                return (1 if polls > 1 else 0, elapsed_s)

            remaining_s = deadline_s - float(self._clock_fn())
            if remaining_s <= 0.0:
                log.append("cooldown_timeout", {
                    "batch_id": batch_id, "polls": polls, "skin_c": sample.skin_c,
                    "cool_target_c": self._cfg.cool_target_c, "max_cool_s": self._cfg.max_cool_s,
                })
                raise CooldownTimeoutError(
                    f"Cooldown before {batch_id} timed out after {self._cfg.max_cool_s:.1f}s: "
                    f"skin_c={sample.skin_c:.2f}C > {self._cfg.cool_target_c:.2f}C"
                )
            step_s = min(self._cfg.cool_poll_s, remaining_s)
            log.append("cooldown_wait", {
                "batch_id": batch_id, "poll": polls,
                "skin_c": sample.skin_c, "sleep_s": round(step_s, 6),
            })
            self._sleep_fn(step_s)

    @staticmethod
    def _write_file_atomic(target: pathlib.Path, content: bytes) -> None:
        tmp = target.parent / f".{target.name}.{os.getpid()}.tmp"
        fd = os.open(
            os.fspath(tmp),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600,
        )
        try:
            os.write(fd, content)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(os.fspath(tmp), os.fspath(target))

    def run(self) -> DriverRunSummary:
        """Execute all scheduled batches with fail-closed gates and cooldown loops."""
        self._verify_clean_output_tree()
        driver_log = ExclusiveDriverLog(self._output_root, clock_ns=self._wall_clock_ns)
        driver_log.open_exclusive()
        try:
            os.mkdir(os.fspath(self._batches_root), 0o700)
            driver_log.append("driver_started", {
                "schedule_id": self._schedule.schedule_id,
                "schedule_digest": self._schedule.digest,
                "total_batches": len(self._slices), "total_cells": len(self._schedule.cells),
            })
            cooldown_waits, total_cool_s = 0, 0.0
            completed_dirs: list[pathlib.Path] = []

            for idx, batch_slice in enumerate(self._slices):
                if idx > 0:
                    waited, cool_s = self._run_inter_batch_cooldown(batch_slice.batch_id, driver_log)
                    cooldown_waits += waited
                    total_cool_s += cool_s

                pre_sample = self._probe()
                evaluate_safety_gate(pre_sample, self._cfg, stage=f"pre_batch:{batch_slice.batch_id}")
                driver_log.append("pre_batch_gate_passed", {
                    "batch_id": batch_slice.batch_id, "telemetry": pre_sample.to_dict(),
                })

                batch_dir = self._batches_root / batch_slice.batch_id
                try:
                    os.mkdir(os.fspath(batch_dir), 0o700)
                except FileExistsError as exc:
                    raise BatchOutputExistsError(f"Batch directory exists: {batch_dir}") from exc

                self._write_file_atomic(
                    batch_dir / "manifest.json",
                    _canonical_json_bytes(batch_slice.manifest_dict(self._schedule.digest)) + b"\n",
                )
                driver_log.append("batch_dispatched", {
                    "batch_id": batch_slice.batch_id, "cell_count": len(batch_slice.cells),
                })
                raw_rows = self._worker(batch_slice, batch_dir)

                post_sample = self._probe()
                evaluate_safety_gate(post_sample, self._cfg, stage=f"post_batch:{batch_slice.batch_id}")
                driver_log.append("post_batch_gate_passed", {
                    "batch_id": batch_slice.batch_id, "telemetry": post_sample.to_dict(),
                })

                verified_rows = validate_batch_rows(batch_slice, raw_rows, self._cfg)
                self._write_file_atomic(
                    batch_dir / "verified_rows.jsonl",
                    b"".join(_canonical_json_bytes(row) + b"\n" for row in verified_rows),
                )
                completed_dirs.append(batch_dir)
                driver_log.append("batch_verified", {
                    "batch_id": batch_slice.batch_id, "verified_rows": len(verified_rows),
                })

            summary = DriverRunSummary(
                schedule_id=self._schedule.schedule_id, schedule_digest=self._schedule.digest,
                total_batches=len(self._slices), total_cells=len(self._schedule.cells),
                cooldown_waits=cooldown_waits, total_cooldown_s=round(total_cool_s, 6),
                batch_dirs=tuple(completed_dirs),
            )
            self._write_file_atomic(
                self._output_root / "run_summary.json",
                _canonical_json_bytes({
                    "schedule_id": summary.schedule_id, "schedule_digest": summary.schedule_digest,
                    "total_batches": summary.total_batches, "total_cells": summary.total_cells,
                    "cooldown_waits": summary.cooldown_waits, "total_cooldown_s": summary.total_cooldown_s,
                }) + b"\n",
            )
            driver_log.append("driver_completed", {
                "total_batches": summary.total_batches, "total_cells": summary.total_cells,
                "cooldown_waits": summary.cooldown_waits,
            })
            return summary
        except BaseException as exc:
            driver_log.append("driver_aborted", {"error_type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            driver_log.close()
