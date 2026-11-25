from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from tools.sse_stream_client import (
    iter_sse_events,
    percentile,
    run_benchmark_suite,
    stream_Generate_request,
    summarize_stream_samples,
)


ROOT = Path(__file__).resolve().parents[1]


class _MockSseHandler(BaseHTTPRequestHandler):
    server_version = "MockSSE/1.0"

    def log_message(self, fmt, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.request_count += 1
        mode = body.get("mode", "openai")

        if mode == "http_503":
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "overloaded"}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        if mode == "corrupt":
            self.wfile.write(b"data: {not-valid-json\n\n")
            self.wfile.flush()
            return

        if mode == "error_event":
            self.wfile.write(b'event: error\ndata: {"message": "aborted"}\n\n')
            self.wfile.flush()
            return

        tokens = body.get("tokens", ["Hello", " ", "world", "!"])
        delay_s = float(body.get("delay_s", 0.005))
        for idx, tok in enumerate(tokens):
            time.sleep(delay_s)
            is_last = idx == len(tokens) - 1
            if mode == "generic":
                frame = json.dumps({"content": tok, "stop": is_last})
            else:
                frame = json.dumps({
                    "choices": [{
                        "delta": {"content": tok},
                        "finish_reason": "stop" if is_last else None,
                    }]
                })
            self.wfile.write(f": keepalive comment\ndata: {frame}\n\n".encode("utf-8"))
            self.wfile.flush()
        if mode != "generic":
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()


class SseStreamClientTests(unittest.TestCase):
    def setUp(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _MockSseHandler)
        self.httpd.request_count = 0
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.httpd.server_address
        self.url = f"http://{host}:{port}/v1/chat/completions"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    def test_sse_frame_parser_comments_and_multiline(self):
        raw_lines = [
            b": comment line\r\n",
            b"event: delta\n",
            b"data: line1\n",
            b"data: line2\n",
            b"\n",
            b"data: [DONE]\n",
        ]
        events = list(iter_sse_events(raw_lines))
        self.assertEqual(events, [("delta", "line1\nline2"), ("message", "[DONE]")])

    def test_openai_and_generic_streaming_ttft_itl_and_errors(self):
        sample = stream_Generate_request(
            self.url,
            {"mode": "openai", "tokens": ["A", "B", "C"], "delay_s": 0.006},
            request_id="r1",
        )
        self.assertEqual(sample.status, "ok")
        self.assertEqual(sample.output_text, "ABC")
        self.assertEqual(sample.output_chunks, 3)
        self.assertEqual(len(sample.itl_ms), 2)
        self.assertIsNotNone(sample.ttft_ms)
        self.assertGreater(sample.ttft_ms, 0.0)
        self.assertAlmostEqual(sample.tpot_ms, sum(sample.itl_ms) / 2.0, places=6)
        self.assertEqual(sample.finish_reason, "stop")

        generic = stream_Generate_request(
            self.url,
            {"mode": "generic", "tokens": ["x", "y"], "delay_s": 0.002},
            request_id="r2",
        )
        self.assertEqual(generic.status, "ok")
        self.assertEqual(generic.output_text, "xy")

        http_err = stream_Generate_request(self.url, {"mode": "http_503"}, request_id="r3")
        self.assertEqual(http_err.status, "http_error")
        self.assertEqual(http_err.http_status, 503)

        corrupt = stream_Generate_request(self.url, {"mode": "corrupt"}, request_id="r4")
        self.assertEqual(corrupt.status, "stream_error")

    def test_resumable_checkpointing_and_cli_execution(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            tmp_dir = Path(tmp)
            ckpt = tmp_dir / "progress.jsonl"
            specs = [
                {"request_id": "job-1", "payload": {"tokens": ["1", "2"], "delay_s": 0.002}},
                {"request_id": "job-2", "payload": {"mode": "http_503"}},
                {"request_id": "job-3", "payload": {"tokens": ["3", "4", "5"], "delay_s": 0.002}},
            ]
            first = run_benchmark_suite(self.url, specs, checkpoint_path=ckpt)
            self.assertEqual(first["summary"]["succeeded"], 2)
            self.assertEqual(first["summary"]["failed"], 1)
            self.assertEqual(self.httpd.request_count, 3)

            # Re-running without retry_errors skips all 3 checkpointed requests!
            second = run_benchmark_suite(self.url, specs, checkpoint_path=ckpt, retry_errors=False)
            self.assertEqual(self.httpd.request_count, 3)
            self.assertEqual(second["summary"]["succeeded"], 2)

            # Fixing job-2 payload and setting retry_errors=True re-runs ONLY job-2!
            specs[1]["payload"] = {"tokens": ["ok"], "delay_s": 0.002}
            third = run_benchmark_suite(self.url, specs, checkpoint_path=ckpt, retry_errors=True)
            self.assertEqual(self.httpd.request_count, 4)
            self.assertEqual(third["summary"]["succeeded"], 3)

            # CLI smoke test + non-destructive --output check
            in_jsonl = tmp_dir / "reqs.jsonl"
            out_json = tmp_dir / "report.json"
            in_jsonl.write_text("\n".join(json.dumps(s) for s in specs) + "\n")
            cli = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "tools/sse_stream_client.py"),
                    "--url",
                    self.url,
                    "--input",
                    str(in_jsonl),
                    "--output",
                    str(out_json),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(cli.returncode, 0, cli.stderr)
            self.assertEqual(json.loads(out_json.read_text())["succeeded"], 3)

            duplicate_cli = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "tools/sse_stream_client.py"),
                    "--url",
                    self.url,
                    "--input",
                    str(in_jsonl),
                    "--output",
                    str(out_json),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertNotEqual(duplicate_cli.returncode, 0)

    def test_percentiles_and_validation(self):
        self.assertAlmostEqual(percentile([10.0, 20.0, 30.0, 40.0], 50.0), 25.0)
        self.assertAlmostEqual(percentile([42.0], 99.0), 42.0)
        with self.assertRaises(ValueError):
            percentile([], 50.0)
        with self.assertRaises(ValueError):
            percentile([1.0], 101.0)
        with self.assertRaises(ValueError):
            summarize_stream_samples([])


if __name__ == "__main__":
    unittest.main()
