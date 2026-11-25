"""Unit tests for runtime_support.rpc_mux."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

from runtime_support.rpc_mux import (
    FLAG_JSON,
    FrameCRCError,
    FrameProtocolError,
    MsgType,
    RetryPolicy,
    RpcApplicationError,
    RpcCancelledError,
    RpcClient,
    RpcConnectionError,
    RpcContext,
    RpcRemoteError,
    RpcServer,
    RpcTimeoutError,
    decode_frame,
    encode_frame,
    recv_frame,
)


class RpcMuxTest(unittest.TestCase):
    """Tests for binary frame codec, out-of-order multiplexing, and cancellation."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_frame_codec_roundtrip_streaming_and_corrupt_crc_rejection(self) -> None:
        payload = b'{"hello":"world","nums":[1,2,3]}'
        wire = encode_frame(
            MsgType.REQUEST,
            request_id=0x0102030405060708,
            payload=payload,
            deadline_ms=2500,
            flags=FLAG_JSON,
        )
        self.assertIsNone(decode_frame(wire[:10]))
        self.assertIsNone(decode_frame(wire[:-1]))

        decoded = decode_frame(wire + b"extra")
        self.assertIsNotNone(decoded)
        assert decoded is not None
        frame, consumed = decoded
        self.assertEqual(consumed, len(wire))
        self.assertEqual(frame.msg_type, MsgType.REQUEST)
        self.assertTrue(frame.is_json)
        self.assertEqual(frame.request_id, 0x0102030405060708)
        self.assertEqual(frame.deadline_ms, 2500)
        self.assertEqual(frame.payload, payload)

        corrupted = bytearray(wire)
        corrupted[22] ^= 0x01
        with self.assertRaises(FrameCRCError):
            decode_frame(corrupted)

        bad_magic = b"XX" + wire[2:]
        with self.assertRaises(FrameProtocolError):
            decode_frame(bad_magic)

        bad_version = wire[:2] + b"\x99" + wire[3:]
        with self.assertRaises(FrameProtocolError):
            decode_frame(bad_version)

        s1, s2 = socket.socketpair()
        try:
            s1.sendall(corrupted)
            with self.assertRaises(FrameCRCError):
                recv_frame(s2)
        finally:
            s1.close()
            s2.close()

    def test_out_of_order_concurrent_multiplexing_no_head_of_line_blocking(self) -> None:
        slow_entered = threading.Event()
        release_slow = threading.Event()

        server = RpcServer(host="127.0.0.1", port=0, max_workers=6)

        def slow_op(params: dict, ctx: RpcContext) -> dict:
            slow_entered.set()
            release_slow.wait(timeout=5.0)
            ctx.raise_if_cancelled()
            return {"tag": params["tag"], "kind": "slow"}

        def fast_op(params: dict) -> dict:
            return {"value": params["x"] * params["y"], "kind": "fast"}

        def echo_raw(payload: bytes) -> bytes:
            return payload[::-1]

        server.register_handler("slow_op", slow_op)
        server.register_handler("fast_op", fast_op)
        server.register_handler("echo_raw", echo_raw, raw_bytes=True)
        host, port = server.start()  # type: ignore[misc]

        try:
            with RpcClient.connect_tcp(host, port) as client:
                slow_fut = client.call_async("slow_op", {"tag": "first"}, timeout=5.0)
                self.assertTrue(slow_entered.wait(timeout=2.0))
                self.assertFalse(slow_fut.done())

                fast_results: list[int] = [0] * 20
                errors: list[BaseException] = []

                def fire_fast(idx: int) -> None:
                    try:
                        res = client.call("fast_op", {"x": idx, "y": 3}, timeout=2.0)
                        fast_results[idx] = res["value"]
                    except BaseException as exc:  # pragma: no cover
                        errors.append(exc)

                threads = [threading.Thread(target=fire_fast, args=(i,)) for i in range(20)]
                t0 = time.monotonic()
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=2.0)
                fast_elapsed = time.monotonic() - t0

                self.assertEqual(errors, [])
                self.assertEqual(fast_results, [i * 3 for i in range(20)])
                self.assertLess(fast_elapsed, 1.0)
                self.assertFalse(slow_fut.done())

                raw_out = client.call("echo_raw", b"\x00\x01\x02\xff", raw_bytes=True)
                self.assertEqual(raw_out, b"\xff\x02\x01\x00")

                release_slow.set()
                slow_res = slow_fut.result(timeout=2.0)
                self.assertEqual(slow_res, {"tag": "first", "kind": "slow"})
        finally:
            server.shutdown(wait=True)

    def test_per_request_timeout_and_explicit_server_side_cancellation(self) -> None:
        timeout_started = threading.Event()
        timeout_cancelled = threading.Event()
        explicit_started = threading.Event()
        explicit_cancelled = threading.Event()

        server = RpcServer(max_workers=4)

        def watch_cancel(params: dict, ctx: RpcContext) -> dict:
            mode = params["mode"]
            if mode == "timeout":
                timeout_started.set()
                if ctx.cancel_event.wait(timeout=3.0):
                    timeout_cancelled.set()
                    ctx.raise_if_cancelled()
            else:
                explicit_started.set()
                if ctx.cancel_event.wait(timeout=3.0):
                    explicit_cancelled.set()
                    ctx.raise_if_cancelled()
            return {"unexpected": True}

        server.register_handler("watch_cancel", watch_cancel)
        s_srv, s_cli = socket.socketpair()
        server.attach_socket(s_srv)

        try:
            with RpcClient(sock=s_cli) as client:
                with self.assertRaises(RpcTimeoutError):
                    client.call("watch_cancel", {"mode": "timeout"}, timeout=0.08)
                self.assertTrue(timeout_started.is_set())
                self.assertTrue(timeout_cancelled.wait(timeout=2.0))

                fut = client.call_async("watch_cancel", {"mode": "explicit"}, timeout=5.0)
                self.assertTrue(explicit_started.wait(timeout=2.0))
                self.assertTrue(fut.cancel())
                self.assertTrue(explicit_cancelled.wait(timeout=2.0))
                with self.assertRaises(RpcCancelledError):
                    fut.result(timeout=1.0)
        finally:
            server.shutdown(wait=True)

    def test_handler_exceptions_map_to_structured_remote_error_and_retry_policy(self) -> None:
        unix_sock = self.tmp_path / "rpc_mux.sock"
        server = RpcServer(unix_path=unix_sock, max_workers=4)
        flaky_attempts = {"idemp": 0, "non_idemp": 0}
        lock = threading.Lock()

        def boom_unhandled(_params: dict) -> dict:
            raise ValueError("singular matrix encountered")

        def boom_structured(_params: dict) -> dict:
            raise RpcApplicationError(
                "INVALID_ARGUMENT",
                "batch_size must be positive",
                retriable=False,
                details={"field": "batch_size"},
            )

        def flaky_service(params: dict) -> dict:
            key = params["key"]
            with lock:
                flaky_attempts[key] += 1
                n = flaky_attempts[key]
            if n < 3:
                raise RpcApplicationError(
                    "UNAVAILABLE", f"transient overload #{n}", retriable=True
                )
            return {"attempts": n, "status": "recovered"}

        server.register_handler("boom_unhandled", boom_unhandled)
        server.register_handler("boom_structured", boom_structured)
        server.register_handler("flaky_service", flaky_service)
        server.start()

        try:
            policy = RetryPolicy(max_attempts=4, initial_backoff_s=0.01, max_backoff_s=0.03)
            with RpcClient.connect_unix(unix_sock, retry_policy=policy) as client:
                with self.assertRaises(RpcRemoteError) as ctx_unhandled:
                    client.call("boom_unhandled", {})
                self.assertEqual(ctx_unhandled.exception.code, "INTERNAL_ERROR")
                self.assertIn("ValueError", ctx_unhandled.exception.message)
                self.assertFalse(ctx_unhandled.exception.retriable)

                with self.assertRaises(RpcRemoteError) as ctx_app:
                    client.call("boom_structured", {})
                self.assertEqual(ctx_app.exception.code, "INVALID_ARGUMENT")
                self.assertEqual(ctx_app.exception.details, {"field": "batch_size"})

                with self.assertRaises(RpcRemoteError) as ctx_missing:
                    client.call("no_such_method", {})
                self.assertEqual(ctx_missing.exception.code, "METHOD_NOT_FOUND")

                with self.assertRaises(RpcRemoteError):
                    client.call("flaky_service", {"key": "non_idemp"}, idempotent=False)
                self.assertEqual(flaky_attempts["non_idemp"], 1)

                client.mark_idempotent("flaky_service")
                recovered = client.call("flaky_service", {"key": "idemp"}, timeout=2.0)
                self.assertEqual(recovered, {"attempts": 3, "status": "recovered"})
                self.assertEqual(flaky_attempts["idemp"], 3)
        finally:
            server.shutdown(wait=True)

    def test_ping_pong_keepalive_and_clean_shutdown_during_inflight_calls(self) -> None:
        handler_started = threading.Event()
        allow_finish = threading.Event()

        server = RpcServer(host="127.0.0.1", port=0, max_workers=2)

        def inflight_work(params: dict) -> dict:
            handler_started.set()
            allow_finish.wait(timeout=2.0)
            return {"seq": params["seq"], "completed": True}

        server.register_handler("inflight_work", inflight_work)
        host, port = server.start()  # type: ignore[misc]

        client = RpcClient.connect_tcp(host, port, keepalive_interval_s=0.05)
        try:
            fut = client.call_async("inflight_work", {"seq": 42}, timeout=3.0)
            self.assertTrue(handler_started.wait(timeout=2.0))

            rtt_s = client.ping(b"probe-payload", timeout=1.0)
            self.assertGreaterEqual(rtt_s, 0.0)
            self.assertLess(rtt_s, 0.5)

            shutdown_done = threading.Event()

            def trigger_shutdown() -> None:
                server.shutdown(wait=True, timeout=3.0)
                shutdown_done.set()

            st = threading.Thread(target=trigger_shutdown)
            st.start()
            time.sleep(0.05)
            allow_finish.set()

            res = fut.result(timeout=2.0)
            self.assertEqual(res, {"seq": 42, "completed": True})
            st.join(timeout=2.0)
            self.assertTrue(shutdown_done.is_set())

            with self.assertRaises(RpcConnectionError):
                client.call("inflight_work", {"seq": 43}, timeout=0.5)
        finally:
            client.close()
            server.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
