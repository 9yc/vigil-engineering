"""Multi-backend on-device CLI runner preflight verifier and execution adapter.

Performs read-only hardware and artifact preflight gates (device serial identity,
battery SoC and thermal readiness, local-vs-remote byte-size equality, shared
library bundle presence, CLI ``--help`` flag capability probing across zero and
non-zero usage exit codes, and ``pidof`` stray-runner collision checks), stages
text or tokenized inputs with ``shlex.quote`` safety, executes bounded runs with
monotonic start/end timestamps, and parses structured timing/token metrics.
"""

import dataclasses
import json
import math
import os
import pathlib
import re
import shlex
import subprocess
import tempfile
import time

_SAFE_SERIAL_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_FLAG_RE = re.compile(r"^--?[A-Za-z0-9][A-Za-z0-9_-]*$")
_KV_METRIC_RE = re.compile(
    r"\b(prompt_tokens|output_tokens|generated_tokens|"
    r"prefill_ms|decode_ms|ttft_ms|total_ms|tokens_per_s|peak_rss_mb)"
    r"\s*[=:]\s*([-+]?\d+(?:\.\d+)?)\b"
)


class PreflightError(Exception):
    """Base error for CLI runner preflight and execution adapter failures."""


class PreflightConfigError(PreflightError):
    """Raised when runner specs, paths, flags, or input payloads are malformed."""


class TransportCommandError(PreflightError):
    """Raised when an underlying device shell or file-push command fails."""


class DeviceReadinessError(PreflightError):
    """Raised when device serial, battery SoC, or thermal readiness fails."""


class ArtifactByteMismatchError(PreflightError):
    """Raised when a remote artifact is missing or differs in byte size."""


class MissingSharedLibraryError(PreflightError):
    """Raised when one or more required shared libraries are absent on device."""


class MissingCliCapabilityError(PreflightError):
    """Raised when CLI --help output lacks required flags or binary cannot execute."""


class StrayRunnerPidError(PreflightError):
    """Raised when pidof detects pre-existing stray runner processes on device."""


class InputStagingError(PreflightError):
    """Raised when staging or verifying text/token inputs on device fails."""


class RunnerExecutionError(PreflightError):
    """Raised when bounded CLI runner execution fails or metrics are invalid."""


@dataclasses.dataclass(frozen=True)
class ShellOutcome:
    """Immutable outcome of a remote shell command execution."""

    returncode: int
    stdout: str
    stderr: str
    duration_s: float = 0.0


class SubprocessDeviceTransport:
    """Default ADB subprocess transport implementing shell and file push."""

    def __init__(self, serial: str, *, adb_binary: str = "adb") -> None:
        if not isinstance(serial, str) or not _SAFE_SERIAL_RE.match(serial):
            raise PreflightConfigError(f"Invalid transport serial {serial!r}")
        self._serial = serial
        self._adb_binary = adb_binary

    def run_shell(self, command: str, *, timeout_s: float = 15.0) -> ShellOutcome:
        if not isinstance(command, str) or not command.strip():
            raise PreflightConfigError("Shell command must be a non-empty string")
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [self._adb_binary, "-s", self._serial, "shell", command],
                capture_output=True, text=True, timeout=timeout_s, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise TransportCommandError(f"ADB shell failed: {exc}") from exc
        return ShellOutcome(proc.returncode, proc.stdout, proc.stderr, max(0.0, time.monotonic() - t0))

    def push_file(self, local_path: pathlib.Path, remote_path: str, *, timeout_s: float = 30.0) -> None:
        try:
            proc = subprocess.run(
                [self._adb_binary, "-s", self._serial, "push", os.fspath(local_path), remote_path],
                capture_output=True, text=True, timeout=timeout_s, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise TransportCommandError(f"ADB push failed: {exc}") from exc
        if proc.returncode != 0:
            raise TransportCommandError(f"ADB push exited {proc.returncode}: {proc.stderr.strip()}")


@dataclasses.dataclass(frozen=True)
class ArtifactPair:
    """Local-to-remote artifact mapping verified by exact byte length."""

    label: str
    local_path: pathlib.Path
    remote_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise PreflightConfigError("ArtifactPair.label must be non-empty")
        if not isinstance(self.local_path, pathlib.Path):
            object.__setattr__(self, "local_path", pathlib.Path(self.local_path))
        if not isinstance(self.remote_path, str) or not self.remote_path.startswith("/"):
            raise PreflightConfigError(f"remote_path must be absolute POSIX path: {self.remote_path!r}")


@dataclasses.dataclass(frozen=True)
class RunnerBackendSpec:
    """Specification for an on-device CLI backend binary and preflight contract."""

    backend_name: str
    expected_serial: str
    remote_binary_path: str
    remote_staging_dir: str
    artifact_pairs: tuple[ArtifactPair, ...] = ()
    remote_lib_dirs: tuple[str, ...] = ()
    required_shared_libs: tuple[str, ...] = ()
    required_cli_flags: tuple[str, ...] = ()
    help_args: tuple[str, ...] = ("--help",)
    input_flag: str = "--input-file"
    extra_process_names: tuple[str, ...] = ()
    min_battery_pct: float = 25.0
    max_thermal_c: float = 42.0
    battery_capacity_node: str = "/sys/class/power_supply/battery/capacity"
    thermal_temp_node: str = "/sys/class/thermal/thermal_zone0/temp"
    thermal_millidegree_scale: float = 1000.0

    def __post_init__(self) -> None:
        for name, val in (("backend_name", self.backend_name), ("expected_serial", self.expected_serial)):
            if not isinstance(val, str) or not _SAFE_SERIAL_RE.match(val):
                raise PreflightConfigError(f"Invalid {name} {val!r}")
        for name, val in (
            ("remote_binary_path", self.remote_binary_path),
            ("remote_staging_dir", self.remote_staging_dir),
        ):
            if not isinstance(val, str) or not val.startswith("/"):
                raise PreflightConfigError(f"{name} must be an absolute path")
        if not _SAFE_FLAG_RE.match(self.input_flag):
            raise PreflightConfigError(f"Invalid input_flag {self.input_flag!r}")
        for flag in self.required_cli_flags:
            if not isinstance(flag, str) or not _SAFE_FLAG_RE.match(flag):
                raise PreflightConfigError(f"Invalid required CLI flag {flag!r}")
        for lib_dir in self.remote_lib_dirs:
            if not isinstance(lib_dir, str) or not lib_dir.startswith("/"):
                raise PreflightConfigError(f"Invalid remote_lib_dir {lib_dir!r}")
        for lib_name in self.required_shared_libs:
            if not isinstance(lib_name, str) or "/" in lib_name or (
                not lib_name.endswith(".so") and ".so." not in lib_name
            ):
                raise PreflightConfigError(f"Invalid shared library basename {lib_name!r}")
        if not (0.0 <= float(self.min_battery_pct) <= 100.0):
            raise PreflightConfigError("min_battery_pct must be within [0, 100]")
        if float(self.max_thermal_c) <= 0.0 or float(self.thermal_millidegree_scale) <= 0.0:
            raise PreflightConfigError("Thermal ceiling and scale must be > 0")


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    """Verified read-only preflight telemetry and artifact audit summary."""

    backend_name: str
    verified_serial: str
    battery_pct: float
    thermal_c: float
    verified_artifacts: dict[str, int]
    verified_shared_libs: tuple[str, ...]
    help_returncode: int
    verified_flags: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class InputPayload:
    """Text or pre-tokenized input payload to stage safely on device."""

    request_id: str
    kind: str
    text: str | None = None
    token_ids: tuple[int, ...] | None = None
    token_format: str = "space"

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not _SAFE_SERIAL_RE.match(self.request_id):
            raise PreflightConfigError(f"Invalid request_id {self.request_id!r}")
        if self.kind == "text":
            if not isinstance(self.text, str) or not self.text or self.token_ids is not None:
                raise PreflightConfigError("kind='text' requires non-empty text and token_ids=None")
        elif self.kind == "tokens":
            if self.text is not None or not isinstance(self.token_ids, (tuple, list)):
                raise PreflightConfigError("kind='tokens' requires token_ids and text=None")
            normalized = tuple(self.token_ids)
            if not normalized:
                raise PreflightConfigError("token_ids must be non-empty")
            for tok in normalized:
                if isinstance(tok, bool) or not isinstance(tok, int) or tok < 0:
                    raise PreflightConfigError(f"Invalid token id {tok!r}; expected int >= 0")
            object.__setattr__(self, "token_ids", normalized)
            if self.token_format not in ("space", "csv", "json"):
                raise PreflightConfigError(f"Unsupported token_format {self.token_format!r}")
        else:
            raise PreflightConfigError(f"Unsupported InputPayload.kind {self.kind!r}")

    def serialize_bytes(self) -> bytes:
        if self.kind == "text":
            assert self.text is not None
            return self.text.encode("utf-8")
        assert self.token_ids is not None
        if self.token_format == "space":
            return (" ".join(str(t) for t in self.token_ids) + "\n").encode("utf-8")
        if self.token_format == "csv":
            return (",".join(str(t) for t in self.token_ids) + "\n").encode("utf-8")
        return (json.dumps(list(self.token_ids), separators=(",", ":")) + "\n").encode("utf-8")


@dataclasses.dataclass(frozen=True)
class ParsedRunMetrics:
    """Structured timing and token generation metrics parsed from CLI output."""

    prompt_tokens: int
    output_tokens: int
    prefill_ms: float
    decode_ms: float
    total_ms: float
    tokens_per_s: float
    ttft_ms: float | None = None
    peak_rss_mb: float | None = None


@dataclasses.dataclass(frozen=True)
class RunnerExecutionOutcome:
    """Complete execution record with monotonic timestamps and parsed metrics."""

    request_id: str
    backend_name: str
    command: str
    staged_remote_path: str
    returncode: int
    start_monotonic_ns: int
    end_monotonic_ns: int
    host_elapsed_ms: float
    stdout: str
    stderr: str
    metrics: ParsedRunMetrics


def _flag_present_in_help(help_text: str, flag: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_-]){re.escape(flag)}(?![A-Za-z0-9_-])", help_text) is not None


def parse_runner_metrics(stdout: str, stderr: str) -> ParsedRunMetrics:
    """Extract and validate structured timing/token metrics from stdout/stderr."""
    combined = f"{stdout}\n{stderr}"
    extracted: dict[str, object] = {}

    for line in combined.splitlines():
        stripped = line.strip()
        if stripped.startswith("METRICS_JSON:"):
            raw_json = stripped.split("METRICS_JSON:", 1)[1].strip()
            try:
                parsed = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                raise RunnerExecutionError(f"Malformed METRICS_JSON line: {exc}") from exc
            if not isinstance(parsed, dict):
                raise RunnerExecutionError("METRICS_JSON payload must be a JSON object")
            extracted.update(parsed)

    for match in _KV_METRIC_RE.finditer(combined):
        key, raw_val = match.group(1), match.group(2)
        if key not in extracted:
            extracted[key] = float(raw_val) if "." in raw_val else int(raw_val)

    if "output_tokens" not in extracted and "generated_tokens" in extracted:
        extracted["output_tokens"] = extracted["generated_tokens"]

    def _req_int(name: str) -> int:
        val = extracted.get(name)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise RunnerExecutionError(f"Missing or non-numeric integer metric {name!r}")
        as_int = int(val)
        if as_int < 0 or float(as_int) != float(val):
            raise RunnerExecutionError(f"Metric {name!r} must be a non-negative integer")
        return as_int

    def _req_float(name: str, *, allow_zero: bool = True) -> float:
        val = extracted.get(name)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise RunnerExecutionError(f"Missing or non-numeric float metric {name!r}")
        as_float = float(val)
        if math.isnan(as_float) or math.isinf(as_float):
            raise RunnerExecutionError(f"Non-finite metric {name!r}")
        if (allow_zero and as_float < 0.0) or (not allow_zero and as_float <= 0.0):
            raise RunnerExecutionError(f"Out-of-range metric {name!r}: {as_float}")
        return as_float

    def _opt_float(name: str) -> float | None:
        if name not in extracted or extracted[name] is None:
            return None
        return _req_float(name, allow_zero=True)

    prompt_tokens = _req_int("prompt_tokens")
    output_tokens = _req_int("output_tokens")
    prefill_ms = _req_float("prefill_ms")
    decode_ms = _req_float("decode_ms")
    total_ms = _req_float("total_ms", allow_zero=False) if "total_ms" in extracted else prefill_ms + decode_ms
    if total_ms <= 0.0:
        raise RunnerExecutionError("total_ms must be strictly positive")
    tps = (
        _req_float("tokens_per_s")
        if "tokens_per_s" in extracted
        else ((output_tokens * 1000.0 / decode_ms) if decode_ms > 0.0 else 0.0)
    )
    return ParsedRunMetrics(
        prompt_tokens=prompt_tokens, output_tokens=output_tokens, prefill_ms=prefill_ms,
        decode_ms=decode_ms, total_ms=total_ms, tokens_per_s=tps,
        ttft_ms=_opt_float("ttft_ms"), peak_rss_mb=_opt_float("peak_rss_mb"),
    )


class RunnerPreflightAdapter:
    """Read-only preflight verifier and bounded CLI execution adapter."""

    def __init__(
        self, spec: RunnerBackendSpec, transport: object, *, clock_ns: object = time.monotonic_ns,
    ) -> None:
        if not isinstance(spec, RunnerBackendSpec):
            raise PreflightConfigError("spec must be a RunnerBackendSpec")
        if not hasattr(transport, "run_shell") or not hasattr(transport, "push_file"):
            raise PreflightConfigError("transport must provide run_shell(...) and push_file(...)")
        if not callable(clock_ns):
            raise PreflightConfigError("clock_ns must be callable")
        self._spec = spec
        self._transport = transport
        self._clock_ns = clock_ns

    @property
    def spec(self) -> RunnerBackendSpec:
        return self._spec

    def _build_env_prefix(self) -> str:
        if not self._spec.remote_lib_dirs:
            return ""
        return f"LD_LIBRARY_PATH={shlex.quote(':'.join(self._spec.remote_lib_dirs))} "

    def _query_remote_byte_size(self, remote_path: str, *, timeout_s: float) -> int | None:
        qpath = shlex.quote(remote_path)
        cmd = f"if [ -f {qpath} ]; then wc -c < {qpath}; else printf 'MISSING\\n'; fi"
        out = self._transport.run_shell(cmd, timeout_s=timeout_s)
        if out.returncode != 0:
            raise TransportCommandError(f"Size probe failed for {remote_path} (rc={out.returncode})")
        token = out.stdout.strip()
        if token == "MISSING" or not token:
            return None
        try:
            size = int(token.split()[0])
        except ValueError as exc:
            raise TransportCommandError(f"Unparseable byte count {token!r} for {remote_path}") from exc
        if size < 0:
            raise TransportCommandError(f"Negative byte count {size} for {remote_path}")
        return size

    def verify_device_readiness(self, *, timeout_s: float = 10.0) -> tuple[str, float, float]:
        """Read-only check of device serial, battery SoC, and thermal ceiling."""
        serial_out = self._transport.run_shell("getprop ro.serialno", timeout_s=timeout_s)
        if serial_out.returncode != 0:
            raise DeviceReadinessError(f"ro.serialno query failed (rc={serial_out.returncode})")
        actual_serial = serial_out.stdout.strip()
        if actual_serial != self._spec.expected_serial:
            raise DeviceReadinessError(
                f"Serial mismatch: expected {self._spec.expected_serial!r}, got {actual_serial!r}"
            )

        bat_out = self._transport.run_shell(
            f"cat {shlex.quote(self._spec.battery_capacity_node)}", timeout_s=timeout_s,
        )
        if bat_out.returncode != 0:
            raise DeviceReadinessError(f"Battery capacity read failed (rc={bat_out.returncode})")
        try:
            battery_pct = float(bat_out.stdout.strip())
        except ValueError as exc:
            raise DeviceReadinessError(f"Invalid battery percentage {bat_out.stdout!r}") from exc
        if math.isnan(battery_pct) or battery_pct < self._spec.min_battery_pct:
            raise DeviceReadinessError(
                f"Battery SoC {battery_pct:.1f}% below floor {self._spec.min_battery_pct:.1f}%"
            )

        therm_out = self._transport.run_shell(
            f"cat {shlex.quote(self._spec.thermal_temp_node)}", timeout_s=timeout_s,
        )
        if therm_out.returncode != 0:
            raise DeviceReadinessError(f"Thermal read failed (rc={therm_out.returncode})")
        try:
            raw_therm = float(therm_out.stdout.strip())
        except ValueError as exc:
            raise DeviceReadinessError(f"Invalid thermal value {therm_out.stdout!r}") from exc
        thermal_c = raw_therm / self._spec.thermal_millidegree_scale if raw_therm > 200.0 else raw_therm
        if math.isnan(thermal_c) or thermal_c > self._spec.max_thermal_c:
            raise DeviceReadinessError(
                f"Device temperature {thermal_c:.2f}C exceeds {self._spec.max_thermal_c:.2f}C"
            )
        return actual_serial, battery_pct, thermal_c

    def verify_artifacts(self, *, timeout_s: float = 10.0) -> dict[str, int]:
        """Verify local-vs-remote exact byte length for every configured artifact."""
        verified_sizes: dict[str, int] = {}
        for pair in self._spec.artifact_pairs:
            if not pair.local_path.is_file():
                raise ArtifactByteMismatchError(f"Local artifact {pair.label!r} missing: {pair.local_path}")
            local_size = pair.local_path.stat().st_size
            if local_size <= 0:
                raise ArtifactByteMismatchError(f"Local artifact {pair.label!r} is empty (0 bytes)")
            remote_size = self._query_remote_byte_size(pair.remote_path, timeout_s=timeout_s)
            if remote_size is None:
                raise ArtifactByteMismatchError(f"Remote artifact {pair.label!r} missing at {pair.remote_path}")
            if remote_size != local_size:
                raise ArtifactByteMismatchError(
                    f"Byte-size mismatch for {pair.label!r}: local={local_size}, remote={remote_size}"
                )
            verified_sizes[pair.label] = local_size
        return verified_sizes

    def verify_shared_libraries(self, *, timeout_s: float = 10.0) -> tuple[str, ...]:
        """Verify remote binary executable bit and shared-library bundle presence."""
        q_bin = shlex.quote(self._spec.remote_binary_path)
        if self._transport.run_shell(f"test -f {q_bin} && test -x {q_bin}", timeout_s=timeout_s).returncode != 0:
            raise MissingCliCapabilityError(f"Runner binary missing or not executable: {self._spec.remote_binary_path}")
        if not self._spec.required_shared_libs:
            return ()
        if not self._spec.remote_lib_dirs:
            raise MissingSharedLibraryError("required_shared_libs specified without remote_lib_dirs")

        resolved, missing = [], []
        for lib_name in self._spec.required_shared_libs:
            checks = [
                f"(test -r {shlex.quote(lib_dir.rstrip('/') + '/' + lib_name)} "
                f"&& test -s {shlex.quote(lib_dir.rstrip('/') + '/' + lib_name)})"
                for lib_dir in self._spec.remote_lib_dirs
            ]
            if self._transport.run_shell(" || ".join(checks), timeout_s=timeout_s).returncode == 0:
                resolved.append(lib_name)
            else:
                missing.append(lib_name)
        if missing:
            raise MissingSharedLibraryError(f"Missing required shared libraries: {missing}")
        return tuple(resolved)

    def verify_cli_capabilities(self, *, timeout_s: float = 10.0) -> tuple[int, tuple[str, ...]]:
        """Probe CLI --help and verify required flags even on non-zero usage exit."""
        quoted_help = " ".join(shlex.quote(arg) for arg in self._spec.help_args)
        cmd = f"{self._build_env_prefix()}{shlex.quote(self._spec.remote_binary_path)} {quoted_help}"
        out = self._transport.run_shell(cmd, timeout_s=timeout_s)
        if out.returncode < 0 or out.returncode in (126, 127):
            raise MissingCliCapabilityError(f"CLI --help failed to launch (rc={out.returncode})")
        help_text = f"{out.stdout}\n{out.stderr}"
        if not help_text.strip():
            raise MissingCliCapabilityError(f"CLI --help produced empty output (rc={out.returncode})")
        missing_flags = [f for f in self._spec.required_cli_flags if not _flag_present_in_help(help_text, f)]
        if missing_flags:
            raise MissingCliCapabilityError(
                f"Runner {self._spec.backend_name!r} --help (rc={out.returncode}) missing flags: {missing_flags}"
            )
        return out.returncode, self._spec.required_cli_flags

    def verify_no_stray_runners(self, *, timeout_s: float = 10.0) -> None:
        """Ensure zero stray runner PIDs exist via read-only pidof probes."""
        candidates = (pathlib.PurePosixPath(self._spec.remote_binary_path).name,) + self._spec.extra_process_names
        detected: dict[str, list[int]] = {}
        for proc_name in candidates:
            out = self._transport.run_shell(f"pidof {shlex.quote(proc_name)}", timeout_s=timeout_s)
            pids = [int(t) for t in out.stdout.strip().split() if t.isdigit()]
            if out.returncode == 0 and pids:
                detected[proc_name] = pids
        if detected:
            raise StrayRunnerPidError(f"Detected stray runner PIDs before execution: {detected}")

    def run_preflight(self, *, timeout_s: float = 10.0) -> PreflightReport:
        """Execute all read-only preflight checks and return a verified report."""
        serial, bat_pct, therm_c = self.verify_device_readiness(timeout_s=timeout_s)
        self.verify_no_stray_runners(timeout_s=timeout_s)
        artifacts = self.verify_artifacts(timeout_s=timeout_s)
        libs = self.verify_shared_libraries(timeout_s=timeout_s)
        help_rc, flags = self.verify_cli_capabilities(timeout_s=timeout_s)
        return PreflightReport(
            backend_name=self._spec.backend_name, verified_serial=serial,
            battery_pct=bat_pct, thermal_c=therm_c, verified_artifacts=artifacts,
            verified_shared_libs=libs, help_returncode=help_rc, verified_flags=flags,
        )

    def stage_input(self, payload: InputPayload, *, timeout_s: float = 15.0) -> str:
        """Stage text or tokenized input to device and verify exact remote byte length."""
        if not isinstance(payload, InputPayload):
            raise PreflightConfigError("payload must be an InputPayload")
        raw_bytes = payload.serialize_bytes()
        ext = "tok" if payload.kind == "tokens" else "txt"
        remote_dir = self._spec.remote_staging_dir.rstrip("/")
        remote_path = f"{remote_dir}/input_{self._spec.backend_name}_{payload.request_id}.{ext}"

        mkdir_out = self._transport.run_shell(f"mkdir -p -- {shlex.quote(remote_dir)}", timeout_s=timeout_s)
        if mkdir_out.returncode != 0:
            raise InputStagingError(f"Failed creating remote staging dir {remote_dir}")

        tmp_root = os.environ.get("TMPDIR")
        if tmp_root:
            pathlib.Path(tmp_root).mkdir(parents=True, exist_ok=True)
        fd, local_tmp_str = tempfile.mkstemp(prefix="stage_in_", suffix=f".{ext}", dir=tmp_root)
        local_tmp = pathlib.Path(local_tmp_str)
        try:
            try:
                os.write(fd, raw_bytes)
            finally:
                os.close(fd)
            self._transport.push_file(local_tmp, remote_path, timeout_s=timeout_s)
        finally:
            try:
                if local_tmp.exists():
                    local_tmp.unlink()
            except OSError:
                pass

        remote_size = self._query_remote_byte_size(remote_path, timeout_s=timeout_s)
        if remote_size != len(raw_bytes):
            raise InputStagingError(
                f"Staged input byte mismatch at {remote_path}: expected={len(raw_bytes)}, remote={remote_size}"
            )
        return remote_path

    def execute_run(
        self, payload: InputPayload, *, cli_args: tuple[str, ...] = (),
        timeout_s: float = 60.0, cleanup_staged_input: bool = True,
    ) -> RunnerExecutionOutcome:
        """Stage input, execute bounded CLI run with shlex quoting, and parse metrics."""
        remote_input_path = self.stage_input(payload, timeout_s=min(timeout_s, 15.0))
        try:
            arg_parts = [
                shlex.quote(self._spec.remote_binary_path),
                shlex.quote(self._spec.input_flag),
                shlex.quote(remote_input_path),
            ]
            for extra in cli_args:
                if not isinstance(extra, str) or not extra:
                    raise PreflightConfigError("All cli_args entries must be non-empty strings")
                arg_parts.append(shlex.quote(extra))
            full_cmd = f"{self._build_env_prefix()}{' '.join(arg_parts)}"

            start_ns = int(self._clock_ns())
            outcome = self._transport.run_shell(full_cmd, timeout_s=timeout_s)
            end_ns = int(self._clock_ns())
            if end_ns < start_ns:
                raise RunnerExecutionError("Monotonic clock moved backwards during run")
            if outcome.returncode != 0:
                raise RunnerExecutionError(
                    f"Runner {self._spec.backend_name!r} failed (rc={outcome.returncode}): {outcome.stderr.strip()}"
                )
            metrics = parse_runner_metrics(outcome.stdout, outcome.stderr)
            return RunnerExecutionOutcome(
                request_id=payload.request_id, backend_name=self._spec.backend_name,
                command=full_cmd, staged_remote_path=remote_input_path,
                returncode=outcome.returncode, start_monotonic_ns=start_ns,
                end_monotonic_ns=end_ns, host_elapsed_ms=(end_ns - start_ns) / 1_000_000.0,
                stdout=outcome.stdout, stderr=outcome.stderr, metrics=metrics,
            )
        finally:
            if cleanup_staged_input:
                self._transport.run_shell(f"rm -f -- {shlex.quote(remote_input_path)}", timeout_s=10.0)
