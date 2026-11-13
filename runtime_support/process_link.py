"""Line-based subprocess transport with a background reader and bounded waits.

One caller owns send/wait/request/close. Child commands and reply prefixes are
provided by the caller. The transport manages the direct child process only.
"""

import math
import queue
import subprocess
import threading
import time


class LineProcess:
    def __init__(self, argv, *, transcript=None, stderr_path=None, env=None):
        self.argv = list(argv)
        self._err = None
        self.tx = None
        self._closed = False
        self.eof = False
        self.stray = []
        self.q = queue.Queue()
        try:
            self._err = open(stderr_path, "a", encoding="utf-8") if stderr_path else None
            self.tx = open(transcript, "a", encoding="utf-8") if transcript else None
            self.proc = subprocess.Popen(
                self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._err if self._err else subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
            )
        except BaseException:
            if self._err:
                self._err.close()
            if self.tx:
                self.tx.close()
            raise
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

    def _reader(self):
        try:
            for line in self.proc.stdout:
                self.q.put(line.rstrip("\n"))
        finally:
            self.q.put(None)

    def _log(self, direction, line):
        if self.tx:
            self.tx.write(f"{time.monotonic_ns()} {direction} {line[:300]}\n")
            self.tx.flush()

    def send(self, line):
        if "\n" in line or "\r" in line:
            raise ValueError("send expects exactly one line without a line terminator")
        if self._closed or self.eof or self.proc.poll() is not None:
            self.eof = True
            return False
        self._log(">", line)
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self.eof = True
            return False
        return True

    def wait(self, predicate, timeout):
        """Return a matching line, or None on timeout/EOF; keep up to 128 stray lines."""
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        while not self.eof and not self._closed:
            remaining = deadline - time.monotonic()
            if remaining < 0:
                return None
            try:
                line = self.q.get(timeout=remaining)
            except queue.Empty:
                return None
            if line is None:
                self.eof = True
                return None
            self._log("<", line)
            if predicate(line):
                return line
            self.stray.append(line)
            del self.stray[:-128]
        return None

    def request(self, line, prefixes, timeout=60.0):
        if not self.send(line):
            return None
        return self.wait(lambda value: value.split(" ", 1)[0] in prefixes, timeout)

    def close(self, timeout=2.0):
        """Send EOF, then wait or kill and reap the direct child; safe to call twice."""
        if self._closed:
            return
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        self._closed = True
        try:
            try:
                self.proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        finally:
            self._reader_thread.join(timeout=2.0)
            if not self._reader_thread.is_alive():
                self.proc.stdout.close()
            if self.tx:
                self.tx.close()
            if self._err:
                self._err.close()
            self.eof = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
