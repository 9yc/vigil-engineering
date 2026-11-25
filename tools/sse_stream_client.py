#!/usr/bin/env python3
"""HTTP Server-Sent Events (SSE) streaming inference client and latency profiler.

Parses W3C/OpenAI-compatible `data:` event streams over standard `urllib.request`,
records time-to-first-token (TTFT), inter-token latencies (ITL), time-per-output-token
(TPOT) and end-to-end request latencies, computes exact linear-interpolated percentiles,
and supports crash-safe resumable JSONL checkpointing across multi-prompt workloads.
"""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def _reject_constant(token):
    raise ValueError(f"nonstandard JSON constant: {token}")


@dataclass(frozen=True)
class StreamSample:
    request_id: str
    status: str
    http_status: int | None
    ttft_ms: float | None
    e2e_ms: float
    tpot_ms: float | None
    itl_ms: tuple
    output_chunks: int
    output_text: str
    finish_reason: str | None
    error: str | None = None

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "status": self.status,
            "http_status": self.http_status,
            "ttft_ms": self.ttft_ms,
            "e2e_ms": self.e2e_ms,
            "tpot_ms": self.tpot_ms,
            "itl_ms": list(self.itl_ms),
            "output_chunks": self.output_chunks,
            "output_text": self.output_text,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


def iter_sse_events(byte_lines):
    """Yield `(event_type, data_str)` tuples from an iterable of raw SSE lines."""
    event_type = "message"
    data_lines = []
    for raw in byte_lines:
        if isinstance(raw, (bytes, bytearray)):
            line = raw.decode("utf-8", errors="replace")
        else:
            line = str(raw)
        line = line.rstrip("\r\n")
        if not line:
            if data_lines:
                yield event_type, "\n".join(data_lines)
                event_type = "message"
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_type = value or "message"
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield event_type, "\n".join(data_lines)


def _extract_delta_content(payload):
    """Extract `(piece_str, finish_reason)` from OpenAI or generic completion SSE JSON."""
    if not isinstance(payload, dict):
        raise ValueError("SSE JSON event must be an object")
    if "error" in payload and payload["error"]:
        err = payload["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise RuntimeError(f"stream error payload: {msg}")

    if "choices" in payload and isinstance(payload["choices"], list) and payload["choices"]:
        choice = payload["choices"][0]
        if not isinstance(choice, dict):
            return "", None
        finish = choice.get("finish_reason")
        delta = choice.get("delta")
        if isinstance(delta, dict):
            piece = delta.get("content") or ""
            return str(piece), finish
        if "text" in choice and choice["text"] is not None:
            return str(choice["text"]), finish
        return "", finish

    finish = payload.get("stop_reason", payload.get("finish_reason"))
    if payload.get("stop") is True and finish is None:
        finish = "stop"
    piece = payload.get("content", payload.get("token", payload.get("delta", "")))
    if isinstance(piece, dict):
        piece = piece.get("text", "")
    return ("" if piece is None else str(piece)), finish


def _default_urlopen(req, timeout=30.0):
    host = (req.host or "").split(":", 1)[0].lower()
    if host in ("127.0.0.1", "localhost", "::1"):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def stream_Generate_request(
    url,
    payload,
    *,
    request_id="req-0",
    headers=None,
    timeout_s=30.0,
    clock_fn=time.monotonic,
    opener=None,
):
    """Execute one HTTP POST SSE streaming request and measure token latencies."""
    if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("url must be an http:// or https:// endpoint")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON-serializable dict")
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")

    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    req_headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
    open_fn = _default_urlopen if opener is None else opener

    t_start = clock_fn()
    first_token_t = None
    prev_token_t = None
    itl_ms = []
    pieces = []
    finish_reason = None
    http_status = None

    try:
        with open_fn(req, timeout=timeout_s) as resp:
            http_status = getattr(resp, "status", 200)
            for event_type, data_str in iter_sse_events(resp):
                if data_str.strip() == "[DONE]":
                    if finish_reason is None:
                        finish_reason = "done"
                    break
                if event_type == "error":
                    raise RuntimeError(f"SSE error event: {data_str}")
                parsed = json.loads(data_str, parse_constant=_reject_constant)
                piece, reason = _extract_delta_content(parsed)
                if reason is not None:
                    finish_reason = str(reason)
                if piece:
                    now = clock_fn()
                    if first_token_t is None:
                        first_token_t = now
                    else:
                        itl_ms.append(max(0.0, (now - prev_token_t) * 1000.0))
                    prev_token_t = now
                    pieces.append(piece)
    except urllib.error.HTTPError as exc:
        e2e_ms = max(0.0, (clock_fn() - t_start) * 1000.0)
        return StreamSample(
            request_id=str(request_id),
            status="http_error",
            http_status=exc.code,
            ttft_ms=None,
            e2e_ms=e2e_ms,
            tpot_ms=None,
            itl_ms=(),
            output_chunks=0,
            output_text="",
            finish_reason=None,
            error=f"HTTP {exc.code}: {exc.reason}",
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        e2e_ms = max(0.0, (clock_fn() - t_start) * 1000.0)
        return StreamSample(
            request_id=str(request_id),
            status="stream_error",
            http_status=http_status,
            ttft_ms=((first_token_t - t_start) * 1000.0) if first_token_t is not None else None,
            e2e_ms=e2e_ms,
            tpot_ms=None,
            itl_ms=tuple(itl_ms),
            output_chunks=len(pieces),
            output_text="".join(pieces),
            finish_reason=finish_reason,
            error=str(exc),
        )

    t_end = clock_fn()
    e2e_ms = max(0.0, (t_end - t_start) * 1000.0)
    ttft_ms = ((first_token_t - t_start) * 1000.0) if first_token_t is not None else None
    tpot_ms = (math.fsum(itl_ms) / len(itl_ms)) if itl_ms else None
    return StreamSample(
        request_id=str(request_id),
        status="ok",
        http_status=http_status,
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
        tpot_ms=tpot_ms,
        itl_ms=tuple(itl_ms),
        output_chunks=len(pieces),
        output_text="".join(pieces),
        finish_reason=finish_reason,
        error=None,
    )


def percentile(values, q):
    """Compute linear-interpolated percentile `q` in `[0, 100]`."""
    if isinstance(q, bool) or not math.isfinite(q) or not (0.0 <= q <= 100.0):
        raise ValueError("percentile q must be in [0, 100]")
    cleaned = sorted(float(v) for v in values)
    if not cleaned:
        raise ValueError("cannot compute percentile of empty sequence")
    if len(cleaned) == 1:
        return cleaned[0]
    rank = (q / 100.0) * (len(cleaned) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return cleaned[low]
    weight = rank - low
    return cleaned[low] + weight * (cleaned[high] - cleaned[low])


def _metric_summary(values):
    vals = [float(v) for v in values]
    if not vals:
        return None
    return {
        "count": len(vals),
        "mean": math.fsum(vals) / len(vals),
        "min": min(vals),
        "p50": percentile(vals, 50.0),
        "p90": percentile(vals, 90.0),
        "p95": percentile(vals, 95.0),
        "p99": percentile(vals, 99.0),
        "max": max(vals),
    }


def summarize_stream_samples(samples):
    """Aggregate TTFT, ITL, TPOT and E2E latency distributions across samples."""
    rows = list(samples)
    if not rows:
        raise ValueError("samples must be non-empty")
    ok_rows = [r for r in rows if r["status"] == "ok"]
    ttfts = [r["ttft_ms"] for r in ok_rows if r["ttft_ms"] is not None]
    e2es = [r["e2e_ms"] for r in ok_rows]
    tpots = [r["tpot_ms"] for r in ok_rows if r["tpot_ms"] is not None]
    all_itls = [val for r in ok_rows for val in r["itl_ms"]]
    total_chunks = sum(r["output_chunks"] for r in ok_rows)
    total_e2e_s = math.fsum(e2es) / 1000.0

    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    return {
        "total_requests": len(rows),
        "succeeded": len(ok_rows),
        "failed": len(rows) - len(ok_rows),
        "status_counts": by_status,
        "total_output_chunks": total_chunks,
        "mean_chunks_per_ok_request": (total_chunks / len(ok_rows)) if ok_rows else 0.0,
        "sequential_chunk_rate_per_s": (total_chunks / total_e2e_s) if total_e2e_s > 0 else 0.0,
        "ttft_ms": _metric_summary(ttfts),
        "itl_ms": _metric_summary(all_itls),
        "tpot_ms": _metric_summary(tpots),
        "e2e_ms": _metric_summary(e2es),
    }


def load_checkpoint_jsonl(checkpoint_path):
    """Load completed `{request_id -> sample_dict}` from an append-only JSONL file."""
    path = Path(checkpoint_path)
    completed = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped, parse_constant=_reject_constant)
            req_id = record.get("request_id")
            if not isinstance(req_id, str) or not req_id:
                raise ValueError(f"checkpoint line {line_no} missing request_id")
            completed[req_id] = record
    return completed


def run_benchmark_suite(
    url,
    requests_spec,
    *,
    checkpoint_path=None,
    retry_errors=False,
    headers=None,
    timeout_s=30.0,
    clock_fn=time.monotonic,
    opener=None,
):
    """Run a list of `{request_id, payload}` items with resumable JSONL checkpointing."""
    items = list(requests_spec)
    if not items:
        raise ValueError("requests_spec must be non-empty")
    seen_ids = set()
    for item in items:
        req_id = item.get("request_id")
        if not isinstance(req_id, str) or not req_id or req_id in seen_ids:
            raise ValueError(f"duplicate or invalid request_id: {req_id!r}")
        seen_ids.add(req_id)

    prior = load_checkpoint_jsonl(checkpoint_path) if checkpoint_path is not None else {}
    results = []
    ckpt_handle = (
        Path(checkpoint_path).open("a", encoding="utf-8")
        if checkpoint_path is not None
        else None
    )
    try:
        for item in items:
            req_id = item["request_id"]
            cached = prior.get(req_id)
            if cached is not None and (cached.get("status") == "ok" or not retry_errors):
                results.append(cached)
                continue
            sample = stream_Generate_request(
                url,
                item["payload"],
                request_id=req_id,
                headers=headers,
                timeout_s=timeout_s,
                clock_fn=clock_fn,
                opener=opener,
            )
            record = sample.to_dict()
            results.append(record)
            if ckpt_handle is not None:
                ckpt_handle.write(json.dumps(record, allow_nan=False) + "\n")
                ckpt_handle.flush()
                os.fsync(ckpt_handle.fileno())
    finally:
        if ckpt_handle is not None:
            ckpt_handle.close()

    return {
        "results": results,
        "summary": summarize_stream_samples(results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSONL of {request_id, payload}")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        specs = []
        with args.input.open(encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    specs.append(json.loads(raw, parse_constant=_reject_constant))
        out = run_benchmark_suite(
            args.url,
            specs,
            checkpoint_path=args.checkpoint,
            retry_errors=args.retry_errors,
            timeout_s=args.timeout_s,
        )
        encoded = json.dumps(out["summary"], indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as f:
                f.write(encoded)
        else:
            print(encoded, end="")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"sse_stream_client error: {exc}\n")


if __name__ == "__main__":
    main()
