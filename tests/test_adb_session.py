import hashlib
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import unittest

from runtime_support.adb_session import (
    AdbError,
    AdbSession,
    RetryPolicy,
    TcpStreamForwarder,
    validate_serial,
)


class SerialValidationTests(unittest.TestCase):
    def test_valid_usb_emulator_tcp_and_mdns_serials(self):
        valid = (
            "emulator-5554",
            "USB987654321",
            "192.0.2.15:5555",
            "device-host.local:5555",
            "adb-node-01._adb-tls-connect._tcp",
            "adb-node-01._adb._tcp",
        )
        for serial in valid:
            with self.subTest(serial=serial):
                self.assertEqual(validate_serial(serial), serial)

    def test_invalid_serials_are_rejected(self):
        invalid = (
            "",
            "  emulator-5554",
            "-H",
            "--help",
            "serial with space",
            "serial\nline",
            "192.0.2.999:5555",
            "192.0.2.01:5555",
            "192.0.2.10:0",
            "192.0.2.10:70000",
            "192.0.2.10:notaport",
            ":5555",
            "emulator-0",
            "emulator-abc",
            "._adb-tls-connect._tcp",
            None,
        )
        for serial in invalid:
            with self.subTest(serial=serial), self.assertRaises(ValueError):
                validate_serial(serial)


class AdbSessionTests(unittest.TestCase):
    def setUp(self):
        temp_root = os.environ.get("TMPDIR")
        self._tmp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def test_retry_policy_bounds_and_schedule(self):
        policy = RetryPolicy(max_attempts=4, initial_delay_s=0.2, multiplier=2.5, max_delay_s=1.0)
        self.assertAlmostEqual(policy.delay_for_attempt(1), 0.2)
        self.assertAlmostEqual(policy.delay_for_attempt(2), 0.5)
        self.assertAlmostEqual(policy.delay_for_attempt(3), 1.0)
        for bad in (
            {"max_attempts": 0},
            {"max_attempts": True},
            {"initial_delay_s": -0.1},
            {"multiplier": 0.5},
            {"max_delay_s": 0.05, "initial_delay_s": 0.2},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RetryPolicy(**bad)

    def test_transient_failures_retry_and_recover(self):
        sleeps = []
        clock_values = iter([1.0, 1.05, 2.0, 2.04, 3.0, 3.02])
        responses = [
            (1, "", "error: device offline\n"),
            (1, "", "error: protocol fault (couldn't read status)\n"),
            (0, "ready\n", ""),
        ]

        def fake_runner(argv, timeout_s):
            self.assertEqual(argv, ("adb", "-s", "emulator-5554", "shell", "echo ready"))
            self.assertEqual(timeout_s, 5.0)
            return responses.pop(0)

        session = AdbSession(
            "emulator-5554",
            runner=fake_runner,
            sleep_fn=sleeps.append,
            clock_fn=lambda: next(clock_values),
            retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.25, multiplier=2.0, max_delay_s=1.0),
        )
        result = session.shell("echo ready", timeout_s=5.0, retry=True)
        self.assertEqual(result.stdout, "ready\n")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(sleeps, [0.25, 0.5])

    def test_non_transient_failure_stops_immediately(self):
        calls = 0

        def fake_runner(argv, timeout_s):
            nonlocal calls
            calls += 1
            return 127, "", "/system/bin/sh: missing_bin: inaccessible or not found\n"

        session = AdbSession(runner=fake_runner, sleep_fn=lambda _: self.fail("unexpected sleep"))
        with self.assertRaises(AdbError) as ctx:
            session.shell("missing_bin", retry=True)
        self.assertEqual(calls, 1)
        self.assertEqual(ctx.exception.result.returncode, 127)

        unchecked = session.shell("missing_bin", check=False, retry=True)
        self.assertEqual(unchecked.returncode, 127)

    def test_timeout_retries_then_raises_adb_error(self):
        sleeps = []

        def timeout_runner(argv, timeout_s):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout_s)

        session = AdbSession(
            runner=timeout_runner,
            sleep_fn=sleeps.append,
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.1),
        )
        with self.assertRaises(AdbError):
            session.run(["get-state"], retry=True)
        self.assertEqual(sleeps, [0.1])

    def test_stage_file_pushes_when_missing_and_skips_when_matching(self):
        payload = b"binary-payload-\x00\x01\x02\xff" * 40
        digest = hashlib.sha256(payload).hexdigest()
        local_file = self.directory / "artifact.bin"
        local_file.write_bytes(payload)
        remote_state = {"present": False, "corrupt_after_push": False}
        pushed_commands = []

        def fake_runner(argv, timeout_s):
            if argv[1] == "shell":
                if not remote_state["present"]:
                    return 0, "", ""
                reported_digest = "0" * 64 if remote_state["corrupt_after_push"] else digest
                return 0, f"{len(payload)}\n{reported_digest}  /data/local/tmp/artifact.bin\n", ""
            if argv[1] == "push":
                pushed_commands.append(argv)
                remote_state["present"] = True
                return 0, "1 file pushed.\n", ""
            self.fail(f"unexpected argv: {argv}")

        session = AdbSession(runner=fake_runner, sleep_fn=lambda _: None)
        first = session.stage_file(local_file, "/data/local/tmp/artifact.bin")
        self.assertTrue(first["pushed"])
        self.assertEqual(first["size_bytes"], len(payload))
        self.assertEqual(first["sha256"], digest)
        self.assertEqual(len(pushed_commands), 1)

        second = session.stage_file(local_file, "/data/local/tmp/artifact.bin")
        self.assertFalse(second["pushed"])
        self.assertEqual(len(pushed_commands), 1)

        remote_state["present"] = False
        remote_state["corrupt_after_push"] = True
        with self.assertRaises(AdbError):
            session.stage_file(local_file, "/data/local/tmp/artifact.bin")


class TcpStreamForwarderTests(unittest.TestCase):
    def test_bidirectional_forwarding_and_byte_accounting(self):
        server_threads = []

        def make_remote_pair():
            left, right = socket.socketpair()

            def echo_uppercase():
                try:
                    while True:
                        chunk = right.recv(4096)
                        if not chunk:
                            break
                        right.sendall(chunk.upper())
                finally:
                    right.close()

            worker = threading.Thread(target=echo_uppercase, daemon=True)
            worker.start()
            server_threads.append(worker)
            return left

        with TcpStreamForwarder(make_remote_pair) as forwarder:
            host, port = forwarder.address
            with socket.create_connection((host, port), timeout=5) as client:
                client.sendall(b"ping-stream")
                client.shutdown(socket.SHUT_WR)
                received = b""
                while True:
                    block = client.recv(1024)
                    if not block:
                        break
                    received += block
            self.assertEqual(received, b"PING-STREAM")
            for worker in server_threads:
                worker.join(timeout=2)
            stats = forwarder.stats()
            self.assertEqual(stats["total_connections"], 1)
            self.assertEqual(stats["failed_connections"], 0)
            self.assertEqual(stats["bytes_tx"], 11)
            self.assertEqual(stats["bytes_rx"], 11)

    def test_connector_failure_is_counted_cleanly(self):
        def failing_connector():
            raise OSError("remote forward unavailable")

        with TcpStreamForwarder(failing_connector) as forwarder:
            host, port = forwarder.address
            with socket.create_connection((host, port), timeout=5) as client:
                self.assertEqual(client.recv(64), b"")
            self.assertEqual(forwarder.stats()["failed_connections"], 1)


if __name__ == "__main__":
    unittest.main()
