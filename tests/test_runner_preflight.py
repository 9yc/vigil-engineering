"""Unit tests for runtime_support.runner_preflight."""

import json
import os
import pathlib
import shlex
import tempfile
import unittest

from runtime_support.runner_preflight import (
    ArtifactByteMismatchError,
    ArtifactPair,
    DeviceReadinessError,
    InputPayload,
    MissingCliCapabilityError,
    MissingSharedLibraryError,
    RunnerBackendSpec,
    RunnerExecutionError,
    RunnerPreflightAdapter,
    ShellOutcome,
    StrayRunnerPidError,
)


class MockDeviceTransport:
    """Injectable deterministic device transport simulating shell and POSIX VFS."""

    def __init__(
        self,
        *,
        serial: str = "DEV-RUNNER-7001",
        battery_pct: int = 85,
        thermal_millic: int = 34500,
        remote_files: dict[str, bytes] | None = None,
        executables: set[str] | None = None,
        help_rc: int = 0,
        help_stdout: str = "",
        help_stderr: str = "",
        stray_pids: dict[str, list[int]] | None = None,
        run_rc: int = 0,
        run_stdout: str = "",
        run_stderr: str = "",
    ) -> None:
        self.serial = serial
        self.battery_pct = battery_pct
        self.thermal_millic = thermal_millic
        self.remote_files: dict[str, bytes] = dict(remote_files or {})
        self.executables: set[str] = set(executables or set())
        self.help_rc = help_rc
        self.help_stdout = help_stdout
        self.help_stderr = help_stderr
        self.stray_pids: dict[str, list[int]] = dict(stray_pids or {})
        self.run_rc = run_rc
        self.run_stdout = run_stdout
        self.run_stderr = run_stderr
        self.shell_history: list[str] = []
        self.pushed_snapshots: dict[str, bytes] = {}

    def push_file(self, local_path: pathlib.Path, remote_path: str, *, timeout_s: float = 30.0) -> None:
        _ = timeout_s
        data = local_path.read_bytes()
        self.remote_files[remote_path] = data
        self.pushed_snapshots[remote_path] = data

    def run_shell(self, command: str, *, timeout_s: float = 15.0) -> ShellOutcome:
        _ = timeout_s
        self.shell_history.append(command)
        if command == "getprop ro.serialno":
            return ShellOutcome(0, f"{self.serial}\n", "")
        if command.startswith("cat "):
            target = shlex.split(command)[1]
            if target.endswith("/capacity"):
                return ShellOutcome(0, f"{self.battery_pct}\n", "")
            if target.endswith("/temp"):
                return ShellOutcome(0, f"{self.thermal_millic}\n", "")
        if command.startswith("pidof "):
            pids = self.stray_pids.get(shlex.split(command)[1], [])
            return ShellOutcome(0, " ".join(str(p) for p in pids) + "\n", "") if pids else ShellOutcome(1, "", "")
        if command.startswith("if [ -f ") and "wc -c <" in command:
            path = shlex.split(command)[3]
            if path in self.remote_files:
                return ShellOutcome(0, f"{len(self.remote_files[path])}\n", "")
            return ShellOutcome(0, "MISSING\n", "")
        if command.startswith("test -f ") and "&& test -x " in command:
            bin_path = shlex.split(command)[2]
            return ShellOutcome(0 if (bin_path in self.remote_files and bin_path in self.executables) else 1, "", "")
        if "(test -r " in command and "&& test -s " in command:
            for clause in command.split(" || "):
                candidate = shlex.split(clause.strip().lstrip("(").rstrip(")"))[2]
                payload = self.remote_files.get(candidate)
                if payload is not None and len(payload) > 0:
                    return ShellOutcome(0, "", "")
            return ShellOutcome(1, "", "")
        if command.startswith("mkdir -p -- "):
            return ShellOutcome(0, "", "")
        if command.startswith("rm -f -- "):
            self.remote_files.pop(shlex.split(command)[3], None)
            return ShellOutcome(0, "", "")
        if "--help" in command:
            return ShellOutcome(self.help_rc, self.help_stdout, self.help_stderr)
        return ShellOutcome(self.run_rc, self.run_stdout, self.run_stderr)


class RunnerPreflightTests(unittest.TestCase):
    """Tests for multi-backend CLI preflight verification and execution."""

    def setUp(self) -> None:
        tmp_root = os.environ.get("TMPDIR")
        if tmp_root:
            pathlib.Path(tmp_root).mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.TemporaryDirectory(dir=tmp_root)
        self.addCleanup(self._tmpdir.cleanup)
        self.base_dir = pathlib.Path(self._tmpdir.name)

        self.local_weights = self.base_dir / "weights_int8.bin"
        self.local_weights.write_bytes(b"W" * 4096)
        self.local_vocab = self.base_dir / "tokenizer_lut.bin"
        self.local_vocab.write_bytes(b"V" * 512)
        self._clock_ticks = [10_000_000_000, 10_045_500_000]
        self._tick_idx = 0

    def _next_clock_ns(self) -> int:
        val = self._clock_ticks[min(self._tick_idx, len(self._clock_ticks) - 1)]
        self._tick_idx += 1
        return val

    def _make_spec(self) -> RunnerBackendSpec:
        return RunnerBackendSpec(
            backend_name="native_cpu_v2",
            expected_serial="DEV-RUNNER-7001",
            remote_binary_path="/data/local/tmp/runner/cli_engine",
            remote_staging_dir="/data/local/tmp/runner/staging",
            artifact_pairs=(
                ArtifactPair("weights", self.local_weights, "/data/local/tmp/runner/weights_int8.bin"),
                ArtifactPair("vocab", self.local_vocab, "/data/local/tmp/runner/tokenizer_lut.bin"),
            ),
            remote_lib_dirs=("/data/local/tmp/runner/lib64", "/data/local/tmp/runner/plugins"),
            required_shared_libs=("libruntime_core.so", "libkernel_dispatch.so"),
            required_cli_flags=("--input-file", "--threads", "--max-new-tokens", "--seed"),
            input_flag="--input-file",
            extra_process_names=("cli_engine_daemon",),
        )

    def _make_healthy_transport(self, **overrides: object) -> MockDeviceTransport:
        default_files = {
            "/data/local/tmp/runner/cli_engine": b"\x7fELF_BINARY",
            "/data/local/tmp/runner/weights_int8.bin": b"W" * 4096,
            "/data/local/tmp/runner/tokenizer_lut.bin": b"V" * 512,
            "/data/local/tmp/runner/lib64/libruntime_core.so": b"SO_CORE",
            "/data/local/tmp/runner/plugins/libkernel_dispatch.so": b"SO_DISPATCH",
        }
        default_help = (
            "Usage: cli_engine [options]\n"
            "  --input-file PATH       Input prompt or token file\n"
            "  --threads N             Worker thread count\n"
            "  --max-new-tokens N      Decode budget\n"
            "  --seed INT              RNG seed\n"
        )
        metrics_line = "METRICS_JSON: " + json.dumps({
            "prompt_tokens": 32, "output_tokens": 16, "prefill_ms": 12.4,
            "decode_ms": 30.6, "total_ms": 43.0, "tokens_per_s": 522.87,
            "ttft_ms": 13.1, "peak_rss_mb": 218.5,
        })
        kwargs: dict[str, object] = {
            "remote_files": default_files,
            "executables": {"/data/local/tmp/runner/cli_engine"},
            "help_rc": 0,
            "help_stdout": default_help,
            "run_rc": 0,
            "run_stdout": f"Output text ok\n{metrics_line}\n",
        }
        kwargs.update(overrides)
        return MockDeviceTransport(**kwargs)

    def test_full_preflight_pass_and_text_execution(self) -> None:
        spec = self._make_spec()
        transport = self._make_healthy_transport()
        adapter = RunnerPreflightAdapter(spec, transport, clock_ns=self._next_clock_ns)

        report = adapter.run_preflight()
        self.assertEqual(report.verified_serial, "DEV-RUNNER-7001")
        self.assertAlmostEqual(report.battery_pct, 85.0)
        self.assertAlmostEqual(report.thermal_c, 34.5)
        self.assertEqual(report.verified_artifacts, {"weights": 4096, "vocab": 512})
        self.assertEqual(report.verified_shared_libs, ("libruntime_core.so", "libkernel_dispatch.so"))
        self.assertEqual(report.help_returncode, 0)

        tricky_prompt = "Compute 2+2; $(touch /tmp/pwn) `id` 'single' \"double\""
        payload = InputPayload(request_id="req_001", kind="text", text=tricky_prompt)
        outcome = adapter.execute_run(
            payload, cli_args=("--threads", "4", "--max-new-tokens", "16", "--seed", "7"),
        )
        self.assertEqual(outcome.returncode, 0)
        self.assertAlmostEqual(outcome.host_elapsed_ms, 45.5, places=3)
        self.assertEqual((outcome.metrics.prompt_tokens, outcome.metrics.output_tokens), (32, 16))
        self.assertAlmostEqual(outcome.metrics.total_ms, 43.0)
        self.assertAlmostEqual(outcome.metrics.ttft_ms or 0.0, 13.1)
        self.assertEqual(transport.pushed_snapshots[outcome.staged_remote_path].decode("utf-8"), tricky_prompt)
        self.assertNotIn(outcome.staged_remote_path, transport.remote_files)

    def test_artifact_byte_length_mismatch_failure(self) -> None:
        spec = self._make_spec()
        transport = self._make_healthy_transport()
        transport.remote_files["/data/local/tmp/runner/weights_int8.bin"] = b"W" * 4095
        adapter = RunnerPreflightAdapter(spec, transport)
        with self.assertRaisesRegex(ArtifactByteMismatchError, "local=4096, remote=4095"):
            adapter.run_preflight()

        del transport.remote_files["/data/local/tmp/runner/weights_int8.bin"]
        with self.assertRaisesRegex(ArtifactByteMismatchError, "missing"):
            adapter.run_preflight()

    def test_missing_shared_library_failure(self) -> None:
        spec = self._make_spec()
        transport = self._make_healthy_transport()
        del transport.remote_files["/data/local/tmp/runner/plugins/libkernel_dispatch.so"]
        adapter = RunnerPreflightAdapter(spec, transport)
        with self.assertRaisesRegex(MissingSharedLibraryError, "libkernel_dispatch.so"):
            adapter.run_preflight()

    def test_missing_cli_flag_failure_and_prefix_boundary_check(self) -> None:
        spec = self._make_spec()
        incomplete_help = (
            "Usage: cli_engine\n  --input-file PATH\n  --threads-affinity MASK\n  --max-new-tokens N\n"
        )
        transport = self._make_healthy_transport(help_stdout=incomplete_help)
        adapter = RunnerPreflightAdapter(spec, transport)
        with self.assertRaisesRegex(MissingCliCapabilityError, "--threads"):
            adapter.run_preflight()

    def test_nonzero_exit_help_accepted_when_flags_match_but_127_rejected(self) -> None:
        spec = self._make_spec()
        help_on_stderr = (
            "Error: usage requested (exit 1)\n"
            "Flags: --input-file --threads --max-new-tokens --seed\n"
        )
        transport = self._make_healthy_transport(help_rc=1, help_stdout="", help_stderr=help_on_stderr)
        report = RunnerPreflightAdapter(spec, transport).run_preflight()
        self.assertEqual((report.help_returncode, report.verified_flags), (1, spec.required_cli_flags))

        crash_transport = self._make_healthy_transport(help_rc=127, help_stderr=help_on_stderr)
        with self.assertRaises(MissingCliCapabilityError):
            RunnerPreflightAdapter(spec, crash_transport).run_preflight()

    def test_stray_pid_and_device_readiness_rejection(self) -> None:
        spec = self._make_spec()
        stray_transport = self._make_healthy_transport(stray_pids={"cli_engine": [4120, 4129]})
        with self.assertRaisesRegex(StrayRunnerPidError, "4120"):
            RunnerPreflightAdapter(spec, stray_transport).run_preflight()

        hot_transport = self._make_healthy_transport(thermal_millic=45200)
        with self.assertRaisesRegex(DeviceReadinessError, "exceeds"):
            RunnerPreflightAdapter(spec, hot_transport).run_preflight()

    def test_tokenized_input_staging_and_kv_metric_parsing(self) -> None:
        spec = self._make_spec()
        kv_stderr = (
            "[INFO] prompt_tokens=12 output_tokens=8 prefill_ms=6.5 "
            "decode_ms=16.0 ttft_ms=7.0 peak_rss_mb=140.25\n"
        )
        transport = self._make_healthy_transport(run_stdout="done\n", run_stderr=kv_stderr)
        adapter = RunnerPreflightAdapter(spec, transport, clock_ns=self._next_clock_ns)

        tok_payload = InputPayload(
            request_id="req_tok_02", kind="tokens",
            token_ids=(101, 2054, 2003, 102), token_format="space",
        )
        outcome = adapter.execute_run(
            tok_payload, cli_args=("--threads", "2; echo not_executed"),
            cleanup_staged_input=False,
        )
        self.assertTrue(outcome.staged_remote_path.endswith(".tok"))
        self.assertEqual(transport.remote_files[outcome.staged_remote_path], b"101 2054 2003 102\n")
        self.assertIn("'2; echo not_executed'", outcome.command)
        self.assertEqual((outcome.metrics.prompt_tokens, outcome.metrics.output_tokens), (12, 8))
        self.assertAlmostEqual(outcome.metrics.total_ms, 22.5)
        self.assertAlmostEqual(outcome.metrics.tokens_per_s, 500.0)

        fail_transport = self._make_healthy_transport(run_rc=3, run_stderr="oom_abort")
        fail_adapter = RunnerPreflightAdapter(spec, fail_transport, clock_ns=self._next_clock_ns)
        with self.assertRaises(RunnerExecutionError):
            fail_adapter.execute_run(tok_payload, cleanup_staged_input=True)

    def test_csv_and_json_token_formats_and_malformed_metrics_rejection(self) -> None:
        csv_payload = InputPayload(
            request_id="req_csv", kind="tokens", token_ids=(7, 8, 9), token_format="csv",
        )
        json_payload = InputPayload(
            request_id="req_json", kind="tokens", token_ids=(7, 8, 9), token_format="json",
        )
        self.assertEqual(csv_payload.serialize_bytes(), b"7,8,9\n")
        self.assertEqual(json_payload.serialize_bytes(), b"[7,8,9]\n")

        spec = self._make_spec()
        bad_json_transport = self._make_healthy_transport(
            run_stdout="METRICS_JSON: {not_valid_json\n",
        )
        adapter = RunnerPreflightAdapter(spec, bad_json_transport, clock_ns=self._next_clock_ns)
        with self.assertRaises(RunnerExecutionError):
            adapter.execute_run(csv_payload)

        missing_metric_transport = self._make_healthy_transport(
            run_stdout="prompt_tokens=10 prefill_ms=4.0\n",
        )
        adapter_missing = RunnerPreflightAdapter(
            spec, missing_metric_transport, clock_ns=self._next_clock_ns,
        )
        with self.assertRaises(RunnerExecutionError):
            adapter_missing.execute_run(json_payload)


if __name__ == "__main__":
    unittest.main()
