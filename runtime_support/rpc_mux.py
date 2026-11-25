"""Length-prefixed binary RPC frame codec and multiplexed socket transport."""

from __future__ import annotations

import concurrent.futures, contextlib, dataclasses, enum, inspect, json, math, os, selectors, socket, struct, threading, time, zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

FRAME_MAGIC, FRAME_VERSION = b"RM", 1
_HEADER_STRUCT, _TRAILER_STRUCT, _U16_STRUCT = struct.Struct("!2sBBQII"), struct.Struct("!I"), struct.Struct("!H")
HEADER_SIZE, TRAILER_SIZE = _HEADER_STRUCT.size, _TRAILER_STRUCT.size
DEFAULT_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
FLAG_JSON, FLAG_IDEMPOTENT = 0x10, 0x20
_VALID_FLAGS_MASK = FLAG_JSON | FLAG_IDEMPOTENT


class MsgType(enum.IntEnum):
    REQUEST, RESPONSE, ERROR, CANCEL, PING, PONG = 1, 2, 3, 4, 5, 6


class RpcError(Exception):
    """Base exception for RPC codec and transport errors."""

class FrameProtocolError(RpcError):
    """Raised when a binary frame violates header, version, or size rules."""

class FrameCRCError(FrameProtocolError):
    """Raised when a binary frame fails CRC32 trailer validation."""

class RpcConnectionError(RpcError):
    """Raised when the underlying socket closes or fails mid-call."""

class RpcTimeoutError(RpcError):
    """Raised when an RPC call exceeds its per-request deadline."""

class RpcCancelledError(RpcError):
    """Raised when an RPC call is explicitly cancelled."""

class RpcRemoteError(RpcError):
    def __init__(self, code: str, message: str, *, retriable: bool = False, details: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code, self.message, self.retriable, self.details = code, message, bool(retriable), details

class RpcApplicationError(Exception):
    def __init__(self, code: str, message: str, *, retriable: bool = False, details: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.retriable, self.details = code, message, bool(retriable), details

@dataclasses.dataclass(frozen=True)
class Frame:
    msg_type: MsgType
    flags: int
    request_id: int
    deadline_ms: int
    payload: bytes

    @property
    def is_json(self) -> bool: return bool(self.flags & FLAG_JSON)
    @property
    def is_idempotent(self) -> bool: return bool(self.flags & FLAG_IDEMPOTENT)


def _parse_header(view: memoryview, max_bytes: int) -> Tuple[MsgType, int, int, int, int]:
    magic, ver, tf, req_id, deadline_ms, plen = _HEADER_STRUCT.unpack_from(view, 0)
    if magic != FRAME_MAGIC or ver != FRAME_VERSION or plen > max_bytes or ((tf & 0xF0) & ~_VALID_FLAGS_MASK):
        raise FrameProtocolError("Invalid frame header magic, version, flags, or payload length")
    try:
        return MsgType(tf & 0x0F), tf & 0xF0, req_id, deadline_ms, plen
    except ValueError as exc:
        raise FrameProtocolError(f"Unknown message type code: {tf & 0x0F}") from exc

def encode_frame(msg_type: MsgType, request_id: int, payload: bytes = b"", *, deadline_ms: int = 0, flags: int = 0, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    body = bytes(payload)
    if len(body) > max_payload_bytes or not (0 <= request_id < (1 << 64)) or not (0 <= deadline_ms < (1 << 32)) or (flags & ~_VALID_FLAGS_MASK):
        raise FrameProtocolError("Frame parameters out of bounds")
    hdr = _HEADER_STRUCT.pack(FRAME_MAGIC, FRAME_VERSION, (flags & 0xF0) | (int(msg_type) & 0x0F), request_id, deadline_ms, len(body))
    return hdr + body + _TRAILER_STRUCT.pack(zlib.crc32(body, zlib.crc32(hdr)) & 0xFFFFFFFF)

def decode_frame(buffer: Union[bytes, bytearray, memoryview], *, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> Optional[Tuple[Frame, int]]:
    view = memoryview(buffer)
    if len(view) < HEADER_SIZE:
        return None
    msg_type, flags, req_id, deadline_ms, plen = _parse_header(view, max_payload_bytes)
    total = HEADER_SIZE + plen + TRAILER_SIZE
    if len(view) < total:
        return None
    body_slice = view[HEADER_SIZE : HEADER_SIZE + plen]
    if (zlib.crc32(body_slice, zlib.crc32(view[:HEADER_SIZE])) & 0xFFFFFFFF) != _TRAILER_STRUCT.unpack_from(view, HEADER_SIZE + plen)[0]:
        raise FrameCRCError(f"CRC32 mismatch on request_id={req_id}")
    return Frame(msg_type, flags, req_id, deadline_ms, bytes(body_slice)), total

def _pack_request(method: str, body: Any, *, raw_bytes: bool) -> Tuple[bytes, int]:
    if not isinstance(method, str) or not method:
        raise ValueError("method must be a non-empty string")
    if raw_bytes:
        raw = b"" if body is None else body
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise TypeError("raw_bytes request requires bytes-like payload")
        m_utf8 = method.encode("utf-8")
        if len(m_utf8) > 0xFFFF:
            raise ValueError("method name exceeds u16 length")
        return _U16_STRUCT.pack(len(m_utf8)) + m_utf8 + bytes(raw), 0
    return json.dumps({"method": method, "params": body}, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8"), FLAG_JSON

def _unpack_request(frame: Frame) -> Tuple[str, Any, bool]:
    if frame.is_json:
        try:
            doc = json.loads(frame.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FrameProtocolError(f"Malformed JSON request: {exc}") from exc
        if not isinstance(doc, Mapping) or not isinstance(doc.get("method"), str):
            raise FrameProtocolError("JSON request missing string 'method'")
        return doc["method"], doc.get("params"), False
    if len(frame.payload) < _U16_STRUCT.size:
        raise FrameProtocolError("Binary request shorter than method header")
    end = _U16_STRUCT.size + _U16_STRUCT.unpack_from(frame.payload, 0)[0]
    if len(frame.payload) < end:
        raise FrameProtocolError("Binary request truncated inside method name")
    try:
        return frame.payload[_U16_STRUCT.size : end].decode("utf-8"), frame.payload[end:], True
    except UnicodeDecodeError as exc:
        raise FrameProtocolError("Binary request method is not valid UTF-8") from exc

def _unpack_err(frame: Frame) -> RpcRemoteError:
    with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError):
        data = json.loads(frame.payload.decode("utf-8"))
        if isinstance(data, Mapping):
            return RpcRemoteError(str(data.get("code", "REMOTE_ERROR")), str(data.get("message", "Remote error")), retriable=bool(data.get("retriable", False)), details=data.get("details"))
    return RpcRemoteError("PROTOCOL_ERROR", "Unparseable error frame payload")

def _recv_exact(sock: socket.socket, n: int, stop: Optional[threading.Event] = None) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        if stop is not None and stop.is_set() and not buf:
            return None
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            continue
        except OSError as exc:
            if not buf: return None
            raise RpcConnectionError(f"Socket read error: {exc}") from exc
        if not chunk:
            if not buf: return None
            raise RpcConnectionError(f"Unexpected EOF mid-frame ({len(buf)}/{n})")
        buf.extend(chunk)
    return bytes(buf)

def recv_frame(sock: socket.socket, *, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES, stop_event: Optional[threading.Event] = None) -> Optional[Frame]:
    if (hdr := _recv_exact(sock, HEADER_SIZE, stop_event)) is None:
        return None
    plen = _parse_header(memoryview(hdr), max_payload_bytes)[4]
    if (rest := _recv_exact(sock, plen + TRAILER_SIZE)) is None:
        raise RpcConnectionError("Unexpected EOF reading frame body and CRC32 trailer")
    decoded = decode_frame(hdr + rest, max_payload_bytes=max_payload_bytes)
    assert decoded is not None
    return decoded[0]

def _tune_socket(sock: socket.socket) -> None:
    sock.settimeout(0.05)
    if sock.family == socket.AF_INET:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

@dataclasses.dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff_s: float = 0.02
    backoff_multiplier: float = 2.0
    max_backoff_s: float = 0.25
    retryable_codes: Tuple[str, ...] = ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "TRY_AGAIN")

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.initial_backoff_s < 0.0 or self.max_backoff_s < 0.0 or self.backoff_multiplier < 1.0:
            raise ValueError("Invalid RetryPolicy configuration")

    def delay_for_attempt(self, attempt: int) -> float:
        return min(self.max_backoff_s, self.initial_backoff_s * (self.backoff_multiplier ** max(0, attempt - 1)))

class RpcContext:
    def __init__(self, request_id: int, method: str, deadline_monotonic: Optional[float], idempotent: bool) -> None:
        self.request_id, self.method = request_id, method
        self.deadline_monotonic, self.idempotent, self.cancel_event = deadline_monotonic, idempotent, threading.Event()

    def is_cancelled(self) -> bool:
        if not self.cancel_event.is_set() and self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
            self.cancel_event.set()
        return self.cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise RpcCancelledError(f"Request {self.request_id} ({self.method}) cancelled")

class _PeerConn:
    def __init__(self, sock: socket.socket) -> None:
        self.sock, self.closed = sock, False
        self.wlock, self.alock = threading.Lock(), threading.Lock()
        self.active: Dict[int, RpcContext] = {}

    def send(self, raw_frame: bytes) -> None:
        with self.wlock:
            if not self.closed:
                try:
                    self.sock.sendall(raw_frame)
                except OSError:
                    self.closed = True

    def send_err(self, req_id: int, code: str, msg: str, *, retriable: bool = False, details: Any = None) -> None:
        body = json.dumps({"code": code, "details": details, "message": msg, "retriable": bool(retriable)}, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send(encode_frame(MsgType.ERROR, req_id, body, flags=FLAG_JSON))

    def close(self) -> None:
        with self.alock:
            self.closed = True
            for ctx in self.active.values():
                ctx.cancel_event.set()
        with self.wlock:
            with contextlib.suppress(OSError):
                self.sock.shutdown(socket.SHUT_WR)
            with contextlib.suppress(OSError):
                self.sock.close()


class RpcServer:
    """Multi-worker multiplexed RPC server supporting TCP loopback, Unix, and socketpair."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, *, unix_path: Optional[Union[str, Path]] = None, max_workers: int = 8, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._host, self._port, self._max_bytes = host, port, max_payload_bytes
        self._unix_path = str(unix_path) if unix_path is not None else None
        self._handlers: Dict[str, Tuple[Callable[..., Any], bool, bool]] = {}
        self._hlock, self._conns_lock = threading.RLock(), threading.Lock()
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rpc-srv")
        self._listener: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._conns: Set[_PeerConn] = set()
        self._readers: List[threading.Thread] = []
        self._inflight_cv, self._inflight, self._closing = threading.Condition(), 0, False

    @property
    def address(self) -> Union[Tuple[str, int], str]:
        if self._listener is None:
            raise RpcError("Server listener has not been started")
        return self._listener.getsockname()

    def register_handler(self, method: str, fn: Callable[..., Any], *, raw_bytes: bool = False) -> None:
        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be a non-empty string")
        params = list(inspect.signature(fn).parameters.values())
        pos = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        with self._hlock:
            self._handlers[method] = (fn, varargs or len(pos) >= 2, raw_bytes)

    def start(self) -> Union[Tuple[str, int], str]:
        if self._listener is not None:
            return self.address
        if self._unix_path is not None:
            if os.path.exists(self._unix_path):
                os.unlink(self._unix_path)
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(self._unix_path)
        else:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self._host, self._port))
        srv.listen(64)
        srv.setblocking(False)
        self._listener = srv
        self._accept_thread = threading.Thread(target=self._accept_loop, name="rpc-accept", daemon=True)
        self._accept_thread.start()
        return srv.getsockname()

    def attach_socket(self, conn_sock: socket.socket) -> None:
        _tune_socket(conn_sock)
        peer = _PeerConn(conn_sock)
        with self._conns_lock:
            if self._closing:
                return peer.close()
            self._conns.add(peer)
            t = threading.Thread(target=self._conn_loop, args=(peer,), name="rpc-conn", daemon=True)
            self._readers.append(t)
            t.start()

    def _accept_loop(self) -> None:
        assert self._listener is not None
        sel = selectors.DefaultSelector()
        sel.register(self._listener, selectors.EVENT_READ)
        try:
            while not self._stop.is_set():
                for _ in sel.select(timeout=0.05):
                    if not self._stop.is_set():
                        with contextlib.suppress(OSError):
                            self.attach_socket(self._listener.accept()[0])
        finally:
            sel.close()

    def _conn_loop(self, peer: _PeerConn) -> None:
        try:
            while not self._stop.is_set() and not peer.closed:
                try:
                    frame = recv_frame(peer.sock, max_payload_bytes=self._max_bytes, stop_event=self._stop)
                except FrameProtocolError as exc:
                    peer.send_err(0, "PROTOCOL_ERROR", str(exc))
                    break
                except RpcConnectionError:
                    break
                if frame is None:
                    break
                if frame.msg_type == MsgType.PING:
                    peer.send(encode_frame(MsgType.PONG, frame.request_id, frame.payload, max_payload_bytes=self._max_bytes))
                elif frame.msg_type == MsgType.CANCEL:
                    with peer.alock:
                        if (ctx := peer.active.get(frame.request_id)) is not None:
                            ctx.cancel_event.set()
                elif frame.msg_type == MsgType.REQUEST:
                    self._dispatch(peer, frame)
                else:
                    peer.send_err(frame.request_id, "UNEXPECTED_FRAME", f"Unsupported {frame.msg_type.name}")
        finally:
            peer.close()
            with self._conns_lock:
                self._conns.discard(peer)

    def _dec_inflight(self) -> None:
        with self._inflight_cv:
            self._inflight -= 1
            if self._inflight == 0:
                self._inflight_cv.notify_all()

    def _dispatch(self, peer: _PeerConn, frame: Frame) -> None:
        with self._inflight_cv:
            if self._closing:
                return peer.send_err(frame.request_id, "UNAVAILABLE", "Server shutting down", retriable=True)
            self._inflight += 1
        d_mono = time.monotonic() + (frame.deadline_ms / 1000.0) if frame.deadline_ms > 0 else None
        try:
            method, params, is_raw = _unpack_request(frame)
        except FrameProtocolError as exc:
            self._dec_inflight()
            return peer.send_err(frame.request_id, "BAD_REQUEST", str(exc))
        ctx = RpcContext(frame.request_id, method, d_mono, frame.is_idempotent)
        with peer.alock:
            peer.active[ctx.request_id] = ctx
            if peer.closed:
                ctx.cancel_event.set()
        try:
            self._pool.submit(self._run_handler, peer, frame.request_id, method, params, is_raw, ctx)
        except RuntimeError:
            with peer.alock:
                peer.active.pop(frame.request_id, None)
            self._dec_inflight()

    def _run_handler(self, peer: _PeerConn, req_id: int, method: str, params: Any, is_raw: bool, ctx: RpcContext) -> None:
        try:
            with self._hlock:
                entry = self._handlers.get(method)
            if entry is None:
                return peer.send_err(req_id, "METHOD_NOT_FOUND", f"Unregistered method: {method!r}")
            if ctx.is_cancelled():
                return peer.send_err(req_id, "CANCELLED", "Cancelled before execution")
            fn, accepts_ctx, raw_bytes = entry
            res = fn(params, ctx) if accepts_ctx else fn(params)
            if ctx.cancel_event.is_set():
                return peer.send_err(req_id, "CANCELLED", "Cancelled during execution")
            if raw_bytes or (is_raw and isinstance(res, (bytes, bytearray, memoryview))):
                body, flags = bytes(b"" if res is None else res), 0
            else:
                body, flags = json.dumps(res, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8"), FLAG_JSON
            peer.send(encode_frame(MsgType.RESPONSE, req_id, body, flags=flags, max_payload_bytes=self._max_bytes))
        except RpcCancelledError as exc:
            peer.send_err(req_id, "CANCELLED", str(exc))
        except RpcApplicationError as exc:
            peer.send_err(req_id, exc.code, exc.message, retriable=exc.retriable, details=exc.details)
        except Exception as exc:
            peer.send_err(req_id, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}", details={"exception_type": type(exc).__name__})
        finally:
            with peer.alock:
                peer.active.pop(req_id, None)
            self._dec_inflight()

    def shutdown(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        with self._inflight_cv:
            self._closing = True
            end = time.monotonic() + max(0.0, timeout)
            while wait and self._inflight > 0 and (rem := end - time.monotonic()) > 0.0:
                self._inflight_cv.wait(timeout=rem)
        self._stop.set()
        if self._listener is not None:
            with contextlib.suppress(OSError):
                self._listener.close()
            self._listener = None
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.0)
            self._accept_thread = None
        with self._conns_lock:
            conns, readers, self._readers = list(self._conns), list(self._readers), []
        for c in conns:
            c.close()
        for r in readers:
            r.join(timeout=1.0)
        self._pool.shutdown(wait=wait)
        if self._unix_path is not None and os.path.exists(self._unix_path):
            with contextlib.suppress(OSError):
                os.unlink(self._unix_path)

    def __enter__(self) -> RpcServer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.shutdown(wait=True)

@dataclasses.dataclass
class _PendingSlot:
    request_id: int
    event: threading.Event = dataclasses.field(default_factory=threading.Event)
    frame: Optional[Frame] = None
    error: Optional[BaseException] = None

    def resolve(self, *, frame: Optional[Frame] = None, error: Optional[BaseException] = None) -> None:
        if not self.event.is_set():
            self.frame, self.error = frame, error
            self.event.set()

class RpcCallFuture:
    def __init__(self, client: RpcClient, slot: _PendingSlot, deadline_mono: Optional[float]) -> None:
        self._client, self._slot, self._deadline = client, slot, deadline_mono
        self._cancelled, self._lock = False, threading.Lock()

    @property
    def request_id(self) -> int: return self._slot.request_id
    def done(self) -> bool: return self._slot.event.is_set()

    def cancel(self) -> bool:
        with self._lock:
            if self._slot.event.is_set(): return False
            self._cancelled = True
        self._client._abort_call(self._slot.request_id, cancelled=True)
        return True

    def result(self, timeout: Optional[float] = None) -> Any:
        eff = self._deadline if timeout is None else (time.monotonic() + max(0.0, float(timeout)) if self._deadline is None else min(self._deadline, time.monotonic() + max(0.0, float(timeout))))
        if not (self._slot.event.wait() if eff is None else self._slot.event.wait(max(0.0, eff - time.monotonic()))):
            self._client._abort_call(self._slot.request_id, cancelled=False)
            raise RpcTimeoutError(f"RPC request {self._slot.request_id} timed out")
        if self._slot.error is not None: raise self._slot.error
        assert self._slot.frame is not None
        if self._slot.frame.msg_type == MsgType.ERROR:
            err = _unpack_err(self._slot.frame)
            if err.code == "CANCELLED" or self._cancelled: raise RpcCancelledError(err.message)
            raise err
        return json.loads(self._slot.frame.payload.decode("utf-8")) if self._slot.frame.is_json else self._slot.frame.payload


class RpcClient:
    """Thread-safe multiplexed RPC client over a single stream socket."""

    def __init__(self, sock: Optional[socket.socket] = None, *, connector: Optional[Callable[[], socket.socket]] = None, default_timeout_s: float = 5.0, retry_policy: Optional[RetryPolicy] = None, keepalive_interval_s: Optional[float] = None, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> None:
        if sock is None and connector is None:
            raise ValueError("Provide either `sock` or `connector`")
        if not (math.isfinite(default_timeout_s) and default_timeout_s > 0.0):
            raise ValueError("default_timeout_s must be finite and > 0")
        self._connector = connector
        self._sock = sock if sock is not None else connector()  # type: ignore[misc]
        _tune_socket(self._sock)
        self._timeout_s, self._retry, self._max_bytes = float(default_timeout_s), retry_policy or RetryPolicy(), max_payload_bytes
        self._wlock, self._slock, self._next_id = threading.Lock(), threading.Lock(), 1
        self._pending: Dict[int, _PendingSlot] = {}
        self._idempotent: Set[str] = set()
        self._conn_error: Optional[BaseException] = None
        self._closed, self._stop = False, threading.Event()
        self._reader = threading.Thread(target=self._reader_loop, name="rpc-cli-reader", daemon=True)
        self._reader.start()
        self._keepalive: Optional[threading.Thread] = None
        if keepalive_interval_s is not None and keepalive_interval_s > 0.0:
            self._keepalive = threading.Thread(target=self._keepalive_loop, args=(float(keepalive_interval_s),), daemon=True)
            self._keepalive.start()

    @classmethod
    def connect_tcp(cls, host: str, port: int, *, connect_timeout_s: float = 2.0, **kwargs: Any) -> RpcClient:
        return cls(connector=lambda: socket.create_connection((host, port), timeout=connect_timeout_s), **kwargs)

    @classmethod
    def connect_unix(cls, unix_path: Union[str, Path], *, connect_timeout_s: float = 2.0, **kwargs: Any) -> RpcClient:
        def _dial() -> socket.socket:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(connect_timeout_s)
            s.connect(str(unix_path))
            return s
        return cls(connector=_dial, **kwargs)

    def mark_idempotent(self, *methods: str) -> None:
        with self._slock:
            self._idempotent.update(methods)

    def _alloc_and_send(self, msg_type: MsgType, payload: bytes, *, deadline_ms: int = 0, flags: int = 0) -> _PendingSlot:
        with self._slock:
            if self._closed:
                raise RpcConnectionError("RpcClient is closed")
            if self._conn_error is not None:
                raise self._conn_error
            rid = self._next_id
            self._next_id += 1
            self._pending[rid] = (slot := _PendingSlot(rid))
        wire = encode_frame(msg_type, slot.request_id, payload, deadline_ms=deadline_ms, flags=flags, max_payload_bytes=self._max_bytes)
        try:
            self._send_raw(wire)
            return slot
        except BaseException:
            with self._slock:
                self._pending.pop(slot.request_id, None)
            raise

    def _send_raw(self, raw: bytes) -> None:
        with self._wlock:
            if self._closed:
                raise RpcConnectionError("RpcClient is closed")
            try:
                self._sock.sendall(raw)
            except OSError as exc:
                err = RpcConnectionError(f"Failed sending RPC frame: {exc}")
                self._on_disconnect(err)
                raise err from exc

    def _abort_call(self, req_id: int, *, cancelled: bool) -> None:
        with self._slock:
            slot = self._pending.pop(req_id, None)
        if cancelled and slot is not None:
            slot.resolve(error=RpcCancelledError(f"Request {req_id} cancelled by caller"))
        with contextlib.suppress(RpcError):
            self._send_raw(encode_frame(MsgType.CANCEL, req_id))

    def _on_disconnect(self, exc: BaseException) -> None:
        with self._slock:
            if self._conn_error is None:
                self._conn_error = exc
            slots, self._pending = list(self._pending.values()), {}
        for s in slots:
            s.resolve(error=exc)

    def _reader_loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    frame = recv_frame(self._sock, max_payload_bytes=self._max_bytes, stop_event=self._stop)
                except (FrameProtocolError, RpcConnectionError) as exc:
                    return self._on_disconnect(exc)
                if frame is None:
                    return self._on_disconnect(RpcConnectionError("Peer closed RPC connection"))
                if frame.msg_type == MsgType.PING:
                    with contextlib.suppress(RpcError):
                        self._send_raw(encode_frame(MsgType.PONG, frame.request_id, frame.payload, max_payload_bytes=self._max_bytes))
                elif frame.msg_type in (MsgType.RESPONSE, MsgType.ERROR, MsgType.PONG):
                    with self._slock:
                        slot = self._pending.pop(frame.request_id, None)
                    if slot is not None:
                        slot.resolve(frame=frame)
        except BaseException as exc:  # pragma: no cover
            self._on_disconnect(RpcConnectionError(f"Reader crashed: {exc}"))

    def _keepalive_loop(self, interval_s: float) -> None:
        while not self._stop.wait(timeout=interval_s):
            with contextlib.suppress(RpcError):
                self.ping(timeout=min(2.0, interval_s))

    def ping(self, payload: bytes = b"keepalive", *, timeout: Optional[float] = None) -> float:
        wait_s, t0 = (self._timeout_s if timeout is None else float(timeout)), time.monotonic()
        slot = self._alloc_and_send(MsgType.PING, bytes(payload))
        if not slot.event.wait(timeout=max(0.0, wait_s)):
            with self._slock:
                self._pending.pop(slot.request_id, None)
            raise RpcTimeoutError("PING timed out waiting for PONG")
        if slot.error is not None:
            raise slot.error
        assert slot.frame is not None
        if slot.frame.msg_type != MsgType.PONG or slot.frame.payload != bytes(payload):
            raise FrameProtocolError("Mismatched PONG payload or frame type")
        return time.monotonic() - t0

    def call_async(self, method: str, params: Any = None, *, timeout: Optional[float] = None, raw_bytes: bool = False, idempotent: Optional[bool] = None) -> RpcCallFuture:
        eff = self._timeout_s if timeout is None else float(timeout)
        if not (math.isfinite(eff) and eff > 0.0):
            raise ValueError("timeout must be finite and > 0")
        with self._slock:
            is_idemp = bool(idempotent) if idempotent is not None else (method in self._idempotent)
        body, flags = _pack_request(method, params, raw_bytes=raw_bytes)
        d_ms = max(1, min((1 << 32) - 1, int(math.ceil(eff * 1000.0))))
        slot = self._alloc_and_send(MsgType.REQUEST, body, deadline_ms=d_ms, flags=flags | (FLAG_IDEMPOTENT if is_idemp else 0))
        return RpcCallFuture(self, slot, time.monotonic() + eff)

    def _try_reconnect(self) -> bool:
        if self._connector is None or self._closed: return False
        with self._wlock:
            if self._closed: return False
            try: nsock = self._connector()
            except OSError: return False
            _tune_socket(nsock)
            old, self._sock = self._sock, nsock
            with contextlib.suppress(OSError): old.close()
            with self._slock: self._conn_error = None
            if not self._reader.is_alive():
                self._reader = threading.Thread(target=self._reader_loop, name="rpc-cli-reader", daemon=True)
                self._reader.start()
            return True

    def call(self, method: str, params: Any = None, *, timeout: Optional[float] = None, raw_bytes: bool = False, idempotent: Optional[bool] = None) -> Any:
        eff = self._timeout_s if timeout is None else float(timeout)
        if not (math.isfinite(eff) and eff > 0.0): raise ValueError("timeout must be finite and > 0")
        with self._slock:
            is_idemp = bool(idempotent) if idempotent is not None else (method in self._idempotent)
        max_att, end, last_exc = (self._retry.max_attempts if is_idemp else 1), time.monotonic() + eff, None
        for att in range(1, max_att + 1):
            if (rem := end - time.monotonic()) <= 0.0:
                if isinstance(last_exc, RpcError): raise last_exc
                raise RpcTimeoutError(f"RPC {method!r} exceeded overall deadline")
            try:
                return self.call_async(method, params, timeout=rem, raw_bytes=raw_bytes, idempotent=is_idemp).result(timeout=rem)
            except RpcRemoteError as exc:
                last_exc = exc
                if not (is_idemp and (exc.retriable or exc.code in self._retry.retryable_codes)) or att >= max_att: raise
            except RpcConnectionError as exc:
                last_exc = exc
                if not is_idemp or att >= max_att or not self._try_reconnect(): raise
            if (rem_after := end - time.monotonic()) <= 0.0: raise last_exc
            time.sleep(min(self._retry.delay_for_attempt(att), rem_after))
        assert last_exc is not None
        raise last_exc

    def close(self) -> None:
        with self._slock:
            if self._closed: return
            self._closed = True
        self._stop.set()
        self._on_disconnect(RpcConnectionError("RpcClient closed"))
        with self._wlock:
            with contextlib.suppress(OSError): self._sock.shutdown(socket.SHUT_WR)
            with contextlib.suppress(OSError): self._sock.close()
        if self._keepalive is not None: self._keepalive.join(timeout=1.0)
        if threading.current_thread() is not self._reader: self._reader.join(timeout=1.0)

    def __enter__(self) -> RpcClient: return self
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: self.close()
