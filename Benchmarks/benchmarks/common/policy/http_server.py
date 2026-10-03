"""HTTP policy server — hosts FasterWAM in an isolated process."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

import numpy as np

from .codec import (
    CONTENT_TYPE_BIN,
    CONTENT_TYPE_JSON,
    decode_obs,
    encode_chunk,
    pack_predict_response,
    unpack_predict_request,
)
from .inprocess import LiberoInProcessPolicy, RoboCasaInProcessPolicy, RoboTwinInProcessPolicy


class _PolicyHandler(BaseHTTPRequestHandler):
    server_version = "RealtimePolicyHTTP/1.0"

    @property
    def policy(self):
        return self.server.policy  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        return  # quiet default logging

    def _content_type(self) -> str:
        return (self.headers.get("Content-Type") or CONTENT_TYPE_JSON).split(";")[0].strip().lower()

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _read_json(self) -> dict:
        raw = self._read_body()
        return json.loads(raw.decode("utf-8") or "{}")

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", CONTENT_TYPE_JSON)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", CONTENT_TYPE_BIN)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "benchmark": self.policy.benchmark,
                    "codecs": ["binary", "json"],
                    "default_codec": "binary",
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            if self.path == "/predict":
                ctype = self._content_type()
                if ctype == CONTENT_TYPE_BIN:
                    data = unpack_predict_request(self._read_body())
                    chunk = self.policy.predict(
                        data["obs"], data["instruction"], **data["kwargs"]
                    )
                    self._send_binary(200, pack_predict_response(chunk))
                    return
                data = self._read_json()
                obs = decode_obs(data.get("obs"))
                instruction = str(data.get("instruction", ""))
                chunk = self.policy.predict(obs, instruction, **dict(data.get("kwargs") or {}))
                self._send_json(200, {"chunk": encode_chunk(chunk)})
                return
            if self.path == "/close":
                self.policy.close()
                self._send_json(200, {"ok": True})
                return
            self._send_json(404, {"error": "not found"})
        except Exception as e:
            self._send_json(500, {"error": str(e)})


class PolicyHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, policy: Any):
        super().__init__(server_address, _PolicyHandler)
        self.policy = policy


def load_server_policy(benchmark: str, **load_kwargs: Any) -> Any:
    benchmark = benchmark.lower()
    if benchmark == "libero":
        from benchmarks.dynamic_libero.env import libero_bridge as bridge

        ctx = bridge.load_policy(**load_kwargs)
        return LiberoInProcessPolicy(ctx=ctx, benchmark="libero")
    if benchmark == "robotwin":
        from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge

        ctx = bridge.load_policy(**load_kwargs)
        return RoboTwinInProcessPolicy(ctx=ctx, benchmark="robotwin")
    if benchmark == "robocasa":
        from benchmarks.realtime_robocasa.env import robocasa_bridge as bridge

        ctx = bridge.load_policy(**load_kwargs)
        return RoboCasaInProcessPolicy(ctx=ctx, benchmark="robocasa")
    raise ValueError(f"unknown benchmark {benchmark!r}")


def create_policy_server(policy: Any, host: str = "127.0.0.1", port: int = 0) -> PolicyHTTPServer:
    """Create a bound HTTP server (caller runs ``serve_forever``)."""
    return PolicyHTTPServer((host, int(port)), policy)


def serve_policy(policy: Any, host: str = "127.0.0.1", port: int = 8765) -> PolicyHTTPServer:
    """Start server in a background thread (for tests / embedding)."""
    httpd = create_policy_server(policy, host=host, port=port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd
