"""Host-side Android Debug Bridge runner, file staging and stream forwarding.

Subprocess execution and sleep hooks are injectable so callers and tests can
exercise retry policies, remote checksum staging and local TCP forwarding
without hardware attached.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import shlex
import socket
import subprocess
import threading
import time


_SERIAL_TOKEN_RE = re.compile(r"^[A-Za-z0-9._:@-]+$")
_MDNS_SUFFIXES = ("._adb._tcp", "._adb-tls-connect._tcp", "._adb-tls-pairing._tcp")
_TRANSIENT_MARKERS = (
    "device offline",
    "device still authorizing",
    "protocol fault",
    "connection reset",
    "cannot connect to daemon",
    "closed",
)


def validate_serial(serial):
    """Validate an ADB device serial, host:port endpoint or mDNS service name."""
    if not isinstance(serial, str) or not serial or serial != serial.strip():
        raise ValueError("serial must be a non-empty trimmed string")
    if serial.startswith("-"):
        raise ValueError("serial must not start with '-'")
    if not _SERIAL_TOKEN_RE.match(serial):
        raise ValueError("serial contains unsupported characters")
    if any(serial.endswith(suffix) for suffix in _MDNS_SUFFIXES):
        prefix = serial.split(".", 1)[0]
        if not prefix:
            raise ValueError("mDNS serial must have a non-empty instance name")
        return serial
    if ":" in serial:
        host, sep, port_str = serial.rpartition(":")
        if not sep or not host or not port_str.isdigit():
            raise ValueError("endpoint serial must have host:port format")
        port = int(port_str)
        if not (1 <= port <= 65535):
            raise ValueError("endpoint port must be in 1..65535")
        parts = host.split(".")
        if len(parts) == 4 and all(part.isdigit() for part in parts):
            if any(int(part) > 255 or (len(part) > 1 and part.startswith("0"))
                   for part in parts):
                raise ValueError("invalid IPv4 address in endpoint serial")
        return serial
    if serial.startswith("emulator-"):
        port_str = serial.split("-", 1)[1]
        if not port_str.isdigit() or not (1 <= int(port_str) <= 65535):
            raise ValueError("emulator serial must specify a valid port")
    return serial


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_delay_s: float = 0.1
    multiplier: float = 2.0
    max_delay_s: float = 2.0

    def __post_init__(self):
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if not math.isfinite(self.initial_delay_s) or self.initial_delay_s < 0:
            raise ValueError("initial_delay_s must be finite and nonnegative")
        if not math.isfinite(self.multiplier) or self.multiplier < 1.0:
            raise ValueError("multiplier must be finite and at least 1.0")
        if not math.isfinite(self.max_delay_s) or self.max_delay_s < self.initial_delay_s:
            raise ValueError("max_delay_s must be finite and >= initial_delay_s")

    def delay_for_attempt(self, attempt):
        """Return backoff delay in seconds after 1-indexed failed attempt."""
        if type(attempt) is not int or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        raw = self.initial_delay_s * (self.multiplier ** (attempt - 1))
        return min(self.max_delay_s, raw)


@dataclass(frozen=True)
class CommandResult:
    argv: tuple
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    attempts: int


class AdbError(RuntimeError):
    def __init__(self, message, *, result=None):
        super().__init__(message)
        self.result = result


def _default_runner(argv, timeout_s):
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _is_transient_failure(returncode, stderr):
    if returncode == 0:
        return False
    lowered = (stderr or "").lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


class AdbSession:
    """Execute ADB commands, stage verified files and run shell probes."""

    def __init__(
        self,
        serial=None,
        *,
        adb_path="adb",
        runner=None,
        sleep_fn=time.sleep,
        clock_fn=time.monotonic,
        retry_policy=None,
        default_timeout_s=30.0,
    ):
        if not isinstance(adb_path, str) or not adb_path.strip():
            raise ValueError("adb_path must be a non-empty string")
        if not math.isfinite(default_timeout_s) or default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be finite and positive")
        self.serial = validate_serial(serial) if serial is not None else None
        self.adb_path = adb_path
        self._runner = _default_runner if runner is None else runner
        self._sleep = sleep_fn
        self._clock = clock_fn
        self.retry_policy = RetryPolicy() if retry_policy is None else retry_policy
        self.default_timeout_s = float(default_timeout_s)

    def build_argv(self, args):
        items = list(args)
        if not items or any(not isinstance(item, str) or not item for item in items):
            raise ValueError("args must be a non-empty sequence of non-empty strings")
        prefix = [self.adb_path]
        if self.serial is not None:
            prefix.extend(["-s", self.serial])
        return tuple(prefix + items)

    def run(self, args, *, timeout_s=None, check=True, retry=False):
        effective_timeout = self.default_timeout_s if timeout_s is None else timeout_s
        if not math.isfinite(effective_timeout) or effective_timeout <= 0:
            raise ValueError("timeout_s must be finite and positive")
        argv = self.build_argv(args)
        limit = self.retry_policy.max_attempts if retry else 1
        last_timeout = None
        for attempt in range(1, limit + 1):
            started = self._clock()
            try:
                returncode, stdout, stderr = self._runner(argv, effective_timeout)
            except subprocess.TimeoutExpired as exc:
                last_timeout = exc
                if attempt < limit:
                    self._sleep(self.retry_policy.delay_for_attempt(attempt))
                    continue
                raise AdbError(f"command timed out after {attempt} attempt(s): {argv}") from exc
            elapsed = max(0.0, self._clock() - started)
            result = CommandResult(
                argv=argv,
                returncode=int(returncode),
                stdout=stdout,
                stderr=stderr,
                duration_s=elapsed,
                attempts=attempt,
            )
            if result.returncode == 0:
                return result
            if attempt < limit and _is_transient_failure(result.returncode, result.stderr):
                self._sleep(self.retry_policy.delay_for_attempt(attempt))
                continue
            if check:
                detail = result.stderr.strip() or f"exit status {result.returncode}"
                raise AdbError(f"adb command failed ({detail})", result=result)
            return result
        raise AdbError(f"exhausted retries: {last_timeout}")

    def shell(self, command, *, timeout_s=None, check=True, retry=False):
        if not isinstance(command, str) or not command.strip():
            raise ValueError("shell command must be a non-empty string")
        return self.run(["shell", command], timeout_s=timeout_s, check=check, retry=retry)

    def _remote_file_probe(self, remote_path, *, timeout_s=None):
        quoted = shlex.quote(remote_path)
        probe_cmd = f"if [ -f {quoted} ]; then wc -c < {quoted}; sha256sum {quoted}; fi"
        res = self.shell(probe_cmd, timeout_s=timeout_s, check=True, retry=True)
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        if len(lines) < 2:
            return None
        try:
            remote_size = int(lines[0])
        except ValueError:
            return None
        digest_token = lines[1].split()[0].lower()
        if len(digest_token) != 64 or any(c not in "0123456789abcdef" for c in digest_token):
            return None
        return remote_size, digest_token

    def stage_file(self, local_path, remote_path, *, verify_checksum=True, timeout_s=None):
        """Stage local_path at remote_path, skipping transfer if size and SHA-256 match."""
        source = Path(local_path)
        if not source.is_file():
            raise ValueError(f"local_path is not a regular file: {source}")
        if not isinstance(remote_path, str) or not remote_path.startswith("/"):
            raise ValueError("remote_path must be an absolute POSIX path")
        hasher = hashlib.sha256()
        size_bytes = 0
        with source.open("rb") as handle:
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    break
                size_bytes += len(chunk)
                hasher.update(chunk)
        local_sha256 = hasher.hexdigest()

        if verify_checksum:
            existing = self._remote_file_probe(remote_path, timeout_s=timeout_s)
            if existing == (size_bytes, local_sha256):
                return {
                    "pushed": False,
                    "size_bytes": size_bytes,
                    "sha256": local_sha256,
                    "remote_path": remote_path,
                }

        self.run(["push", str(source), remote_path], timeout_s=timeout_s, check=True, retry=True)

        if verify_checksum:
            verified = self._remote_file_probe(remote_path, timeout_s=timeout_s)
            if verified != (size_bytes, local_sha256):
                raise AdbError(
                    f"staged file verification mismatch for {remote_path}: "
                    f"expected {(size_bytes, local_sha256)}, got {verified}"
                )
        return {
            "pushed": True,
            "size_bytes": size_bytes,
            "sha256": local_sha256,
            "remote_path": remote_path,
        }


class TcpStreamForwarder:
    """Accept local TCP clients and bridge bytes to connector-provided sockets."""

    def __init__(self, connector, *, host="127.0.0.1", port=0, chunk_size=16384):
        if not callable(connector):
            raise ValueError("connector must be callable")
        if type(port) is not int or not (0 <= port <= 65535):
            raise ValueError("port must be an integer in 0..65535")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self._connector = connector
        self._chunk_size = chunk_size
        self._lock = threading.Lock()
        self._closed = False
        self._active_sockets = set()
        self._total_connections = 0
        self._failed_connections = 0
        self._bytes_tx = 0
        self._bytes_rx = 0

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind((host, port))
            self._server.listen(16)
            self.address = self._server.getsockname()
        except BaseException:
            self._server.close()
            raise
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _pump(self, src, dst, direction):
        try:
            while True:
                data = src.recv(self._chunk_size)
                if not data:
                    break
                dst.sendall(data)
                with self._lock:
                    if direction == "tx":
                        self._bytes_tx += len(data)
                    else:
                        self._bytes_rx += len(data)
        except OSError:
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def _handle_client(self, client_sock):
        try:
            remote_sock = self._connector()
        except Exception:
            with self._lock:
                self._failed_connections += 1
                self._active_sockets.discard(client_sock)
            client_sock.close()
            return
        with self._lock:
            if self._closed:
                self._active_sockets.discard(client_sock)
                client_sock.close()
                remote_sock.close()
                return
            self._active_sockets.add(remote_sock)
        up = threading.Thread(target=self._pump, args=(client_sock, remote_sock, "tx"), daemon=True)
        down = threading.Thread(target=self._pump, args=(remote_sock, client_sock, "rx"), daemon=True)
        up.start()
        down.start()
        up.join()
        down.join()
        with self._lock:
            self._active_sockets.discard(client_sock)
            self._active_sockets.discard(remote_sock)
        client_sock.close()
        remote_sock.close()

    def _accept_loop(self):
        while True:
            try:
                client_sock, _ = self._server.accept()
            except OSError:
                break
            with self._lock:
                if self._closed:
                    client_sock.close()
                    break
                self._total_connections += 1
                self._active_sockets.add(client_sock)
            worker = threading.Thread(target=self._handle_client, args=(client_sock,), daemon=True)
            worker.start()

    def stats(self):
        with self._lock:
            return {
                "listen_host": self.address[0],
                "listen_port": self.address[1],
                "total_connections": self._total_connections,
                "failed_connections": self._failed_connections,
                "bytes_tx": self._bytes_tx,
                "bytes_rx": self._bytes_rx,
            }

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sockets = list(self._active_sockets)
            self._active_sockets.clear()
        try:
            self._server.close()
        except OSError:
            pass
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        self._accept_thread.join(timeout=2.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
