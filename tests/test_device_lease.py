"""Unit tests for runtime_support.device_lease."""

import os
import pathlib
import subprocess
import tempfile
import threading
import time
import unittest

from runtime_support.device_lease import (
    DeviceLeaseCoordinator,
    ForeignWakelockError,
    IdentityProbeError,
    LeaseReentrancyError,
    LeaseTimeoutError,
    ScopedWakelockGuard,
    SerialMismatchError,
    ShellWakelockAdapter,
    WakelockAcquireError,
    WakelockReleaseError,
)


class FakeSteppingClock:
    """Deterministic monotonic clock driven by simulated sleep calls."""

    def __init__(self, initial_s: float = 100.0) -> None:
        self.now_s = initial_s
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now_s

    def sleep(self, duration_s: float) -> None:
        self.sleeps.append(duration_s)
        self.now_s += duration_s


class FakeWakelockRig:
    """In-memory fake kernel ``/sys/power/wake_lock`` shell simulator."""

    def __init__(
        self,
        initial_tokens: list[str] | None = None,
        *,
        ignore_acquire: bool = False,
        ignore_release: bool = False,
    ) -> None:
        self.active_tokens: list[str] = list(initial_tokens or [])
        self.commands: list[str] = []
        self.acquire_calls = 0
        self.release_calls = 0
        self.ignore_acquire = ignore_acquire
        self.ignore_release = ignore_release

    def run_shell(self, cmd: str) -> tuple[int, str, str]:
        self.commands.append(cmd)
        if cmd.startswith("cat "):
            return (0, " ".join(self.active_tokens) + "\n", "")
        if "> /sys/power/wake_lock" in cmd or "> '/sys/power/wake_lock'" in cmd:
            self.acquire_calls += 1
            inner = cmd.split("printf '%s' ", 1)[1].split(" >", 1)[0].strip("'")
            token = inner.split()[0]
            if not self.ignore_acquire and token not in self.active_tokens:
                self.active_tokens.append(token)
            return (0, "", "")
        if "> /sys/power/wake_unlock" in cmd or "> '/sys/power/wake_unlock'" in cmd:
            self.release_calls += 1
            inner = cmd.split("printf '%s' ", 1)[1].split(" >", 1)[0].strip("'")
            token = inner.strip()
            if not self.ignore_release and token in self.active_tokens:
                self.active_tokens.remove(token)
            return (0, "", "")
        return (1, "", f"unexpected command: {cmd}")


class DeviceLeaseTests(unittest.TestCase):
    """Tests for two-tier hardware leases and scoped wakelock ownership guards."""

    def setUp(self) -> None:
        tmp_root = os.environ.get("TMPDIR")
        if tmp_root:
            pathlib.Path(tmp_root).mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.TemporaryDirectory(dir=tmp_root)
        self.addCleanup(self._tmpdir.cleanup)
        self.base_dir = pathlib.Path(self._tmpdir.name)
        self.lock_dir = self.base_dir / "hw_locks"

    def test_shared_and_exclusive_locks_across_real_child_processes(self) -> None:
        pkg_root = str(pathlib.Path(__file__).resolve().parents[1])
        env = dict(os.environ, PYTHONPATH=pkg_root, PYTHONDONTWRITEBYTECODE="1")
        coordinator = DeviceLeaseCoordinator(self.lock_dir, serial_probe=lambda target: target)

        with coordinator.acquire("DEV-SERIAL-1001", holder_id="parent_worker", timeout_s=1.0) as lease:
            self.assertTrue(lease.is_active and lease.fds_non_inheritable)

            # 1. Child acquiring a DIFFERENT serial succeeds concurrently under LOCK_SH.
            diff_script = (
                "from runtime_support.device_lease import DeviceLeaseCoordinator\n"
                f"coord = DeviceLeaseCoordinator({str(self.lock_dir)!r}, serial_probe=lambda t: t)\n"
                "with coord.acquire('DEV-SERIAL-2002', holder_id='child_ok', timeout_s=1.0):\n"
                "    pass\n"
            )
            child_diff = subprocess.run(
                ["python3", "-B", "-c", diff_script],
                env=env, capture_output=True, text=True, timeout=10.0, check=False,
            )
            self.assertEqual(child_diff.returncode, 0, msg=child_diff.stderr)

            # 2. Child acquiring the SAME serial times out under LOCK_EX.
            same_script = (
                "import os\n"
                "from runtime_support.device_lease import DeviceLeaseCoordinator, LeaseTimeoutError\n"
                f"coord = DeviceLeaseCoordinator({str(self.lock_dir)!r}, serial_probe=lambda t: t)\n"
                "try:\n"
                "    coord.acquire('DEV-SERIAL-1001', holder_id='child_blocked', timeout_s=0.08, poll_interval_s=0.02)\n"
                "except LeaseTimeoutError:\n"
                "    os._exit(37)\n"
                "os._exit(0)\n"
            )
            child_same = subprocess.run(
                ["python3", "-B", "-c", same_script],
                env=env, capture_output=True, text=True, timeout=10.0, check=False,
            )
            self.assertEqual(child_same.returncode, 37, msg=child_same.stderr)

            # 3. Exclusive global gate barrier times out while parent holds LOCK_SH.
            barrier_script = (
                "import os\n"
                "from runtime_support.device_lease import DeviceLeaseCoordinator, LeaseTimeoutError\n"
                f"coord = DeviceLeaseCoordinator({str(self.lock_dir)!r}, serial_probe=lambda t: t)\n"
                "try:\n"
                "    with coord.global_barrier(holder_id='drain', timeout_s=0.06, poll_interval_s=0.02):\n"
                "        pass\n"
                "except LeaseTimeoutError:\n"
                "    os._exit(38)\n"
                "os._exit(0)\n"
            )
            child_barrier = subprocess.run(
                ["python3", "-B", "-c", barrier_script],
                env=env, capture_output=True, text=True, timeout=10.0, check=False,
            )
            self.assertEqual(child_barrier.returncode, 38, msg=child_barrier.stderr)

        with coordinator.global_barrier(holder_id="drain_after", timeout_s=0.5):
            blocked = subprocess.run(
                [
                    "python3", "-B", "-c",
                    (
                        "import os\n"
                        "from runtime_support.device_lease import DeviceLeaseCoordinator, LeaseTimeoutError\n"
                        f"coord = DeviceLeaseCoordinator({str(self.lock_dir)!r}, serial_probe=lambda t: t)\n"
                        "try:\n"
                        "    coord.acquire('DEV-SERIAL-2002', timeout_s=0.06, poll_interval_s=0.02)\n"
                        "except LeaseTimeoutError:\n"
                        "    os._exit(39)\n"
                        "os._exit(0)\n"
                    ),
                ],
                env=env, capture_output=True, text=True, timeout=10.0, check=False,
            )
            self.assertEqual(blocked.returncode, 39, msg=blocked.stderr)

    def test_monotonic_timeout_polling_with_injectable_clock(self) -> None:
        pkg_root = str(pathlib.Path(__file__).resolve().parents[1])
        env = dict(os.environ, PYTHONPATH=pkg_root, PYTHONDONTWRITEBYTECODE="1")
        ready_flag = self.base_dir / "holder_ready"
        holder_script = (
            "import pathlib, time\n"
            "from runtime_support.device_lease import DeviceLeaseCoordinator\n"
            f"coord = DeviceLeaseCoordinator({str(self.lock_dir)!r}, serial_probe=lambda t: t)\n"
            "with coord.acquire('DEV-SERIAL-3003', holder_id='bg_holder', timeout_s=2.0):\n"
            f"    pathlib.Path({str(ready_flag)!r}).write_text('ready', encoding='utf-8')\n"
            "    time.sleep(30.0)\n"
        )
        holder_proc = subprocess.Popen(
            ["python3", "-B", "-c", holder_script],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 5.0
            while not ready_flag.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(ready_flag.exists())

            fake_clock = FakeSteppingClock(initial_s=500.0)
            waiter = DeviceLeaseCoordinator(
                self.lock_dir, serial_probe=lambda t: t,
                clock_fn=fake_clock.clock, sleep_fn=fake_clock.sleep,
            )
            with self.assertRaises(LeaseTimeoutError):
                waiter.acquire("DEV-SERIAL-3003", holder_id="waiter", timeout_s=0.25, poll_interval_s=0.10)
            self.assertEqual([round(x, 6) for x in fake_clock.sleeps], [0.10, 0.10, 0.05])
            self.assertAlmostEqual(fake_clock.now_s, 500.25, places=6)
        finally:
            holder_proc.terminate()
            holder_proc.communicate(timeout=5.0)

    def test_serial_mismatch_and_probe_failure_abort_and_release_locks(self) -> None:
        mismatch_coord = DeviceLeaseCoordinator(self.lock_dir, serial_probe=lambda _t: "DEV-WRONG-9999")
        with self.assertRaises(SerialMismatchError):
            mismatch_coord.acquire("DEV-EXPECTED-1111", transport_target="usb_port_01", timeout_s=0.5)

        blank_coord = DeviceLeaseCoordinator(self.lock_dir, serial_probe=lambda _t: "   ")
        with self.assertRaises(IdentityProbeError):
            blank_coord.acquire("DEV-EXPECTED-1111", timeout_s=0.5)

        valid_coord = DeviceLeaseCoordinator(self.lock_dir, serial_probe=lambda _t: "DEV-EXPECTED-1111")
        with valid_coord.acquire("DEV-EXPECTED-1111", holder_id="valid_run", timeout_s=0.2) as lease:
            self.assertEqual(lease.verified_serial, "DEV-EXPECTED-1111")

        names = [e["event"] for e in valid_coord.read_events()]
        for expected_event in ("serial_mismatch", "identity_probe_failed", "lease_acquired", "lease_released"):
            self.assertIn(expected_event, names)

    def test_contextvars_nested_reentrancy_rejected(self) -> None:
        coordinator = DeviceLeaseCoordinator(self.lock_dir, serial_probe=lambda t: t)
        with coordinator.acquire("DEV-SERIAL-1001", holder_id="outer"):
            with self.assertRaises(LeaseReentrancyError):
                coordinator.acquire("DEV-SERIAL-1001", holder_id="inner_same")
            with self.assertRaises(LeaseReentrancyError):
                coordinator.acquire("DEV-SERIAL-2002", holder_id="inner_other")

            thread_errors: list[BaseException] = []

            def worker() -> None:
                try:
                    with coordinator.acquire("DEV-SERIAL-2002", holder_id="other_thread"):
                        pass
                except BaseException as exc:
                    thread_errors.append(exc)

            th = threading.Thread(target=worker)
            th.start()
            th.join(timeout=5.0)
            self.assertEqual(thread_errors, [])

    def test_pre_existing_foreign_wakelock_refused_without_releasing(self) -> None:
        rig = FakeWakelockRig(initial_tokens=["kernel_suspend_blocker", "shell_stale_session"])
        guard = ScopedWakelockGuard(
            ShellWakelockAdapter(run_shell_fn=rig.run_shell),
            lock_name="measure_runner_hold", timeout_s=60.0,
        )
        with self.assertRaises(ForeignWakelockError):
            with guard:
                self.fail("Guard should not enter when foreign shell wakelock exists")

        self.assertFalse(guard.is_owned)
        self.assertEqual((rig.acquire_calls, rig.release_calls), (0, 0))
        self.assertIn("shell_stale_session", rig.active_tokens)

    def test_wakelock_guaranteed_verified_cleanup_on_exception_and_error_modes(self) -> None:
        rig = FakeWakelockRig(initial_tokens=["power_supply_kern"])
        guard = ScopedWakelockGuard(
            ShellWakelockAdapter(run_shell_fn=rig.run_shell),
            lock_name="measure_runner_hold", timeout_s=120.0,
        )
        with self.assertRaisesRegex(RuntimeError, "simulated_workload_crash"):
            with guard as active_guard:
                self.assertTrue(active_guard.is_owned)
                self.assertIn("measure_runner_hold", rig.active_tokens)
                raise RuntimeError("simulated_workload_crash")

        self.assertFalse(guard.is_owned)
        self.assertEqual((rig.acquire_calls, rig.release_calls), (1, 1))
        self.assertEqual(rig.active_tokens, ["power_supply_kern"])

        broken_acq_rig = FakeWakelockRig(ignore_acquire=True)
        with self.assertRaises(WakelockAcquireError):
            with ScopedWakelockGuard(ShellWakelockAdapter(run_shell_fn=broken_acq_rig.run_shell)):
                pass
        self.assertEqual(broken_acq_rig.release_calls, 0)

        broken_rel_rig = FakeWakelockRig(ignore_release=True)
        with self.assertRaises(WakelockReleaseError):
            with ScopedWakelockGuard(ShellWakelockAdapter(run_shell_fn=broken_rel_rig.run_shell)):
                pass

    def test_wakelock_guard_double_entry_and_exception_chaining_on_release_failure(self) -> None:
        rig = FakeWakelockRig()
        guard = ScopedWakelockGuard(
            ShellWakelockAdapter(run_shell_fn=rig.run_shell),
            lock_name="measure_once_lock",
            timeout_s=30.0,
        )
        with guard:
            with self.assertRaises(LeaseReentrancyError):
                guard.acquire()
        with self.assertRaises(LeaseReentrancyError):
            with guard:
                pass

        # When both body and release fail, WakelockReleaseError chains the body error as __cause__.
        sticky_rig = FakeWakelockRig(ignore_release=True)
        sticky_guard = ScopedWakelockGuard(
            ShellWakelockAdapter(run_shell_fn=sticky_rig.run_shell),
            lock_name="measure_sticky_lock",
            timeout_s=30.0,
        )
        with self.assertRaises(WakelockReleaseError) as ctx:
            with sticky_guard:
                raise ValueError("primary_body_failure")
        self.assertIsInstance(ctx.exception.__cause__, ValueError)
        self.assertIn("primary_body_failure", str(ctx.exception.__cause__))


if __name__ == "__main__":
    unittest.main()
