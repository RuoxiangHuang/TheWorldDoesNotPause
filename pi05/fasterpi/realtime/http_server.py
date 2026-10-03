"""HTTP policy server compatible with FasterWAM ``realtime_common`` JSON/binary API."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

import numpy as np

from fasterpi.realtime.codec import (
    CONTENT_TYPE_BIN,
    CONTENT_TYPE_JSON,
    decode_obs,
    encode_chunk,
    pack_predict_response,
    unpack_predict_request,
)


def _plain(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)


def _cache_snapshot(policy: Any) -> dict[str, Any]:
    inner = getattr(policy, "policy", None)
    model = getattr(inner, "_model", None) if inner is not None else None
    if model is None:
        model = getattr(policy, "_model", None)
    if model is None:
        return {"status": "unknown"}
    pi: dict[str, Any] = {}
    for controller in getattr(model, "_fasterpi_controllers", None) or []:
        name = getattr(controller, "name", None) or type(controller).__name__
        stats = getattr(controller, "stats", None)
        if callable(stats):
            stats = stats()
        if isinstance(stats, dict):
            pi[str(name)] = {str(k): _plain(v) for k, v in stats.items()}
    if not pi:
        return {"status": "unknown"}
    return {"fasterpi": pi}


class OpenPIHttpPolicy:
    """Wrap :class:`OpenPIRealtimePolicy` for the Real-Time HTTP server."""

    def __init__(self, rt_policy: Any, *, benchmark: str):
        self.rt_policy = rt_policy
        self.benchmark = benchmark

    def predict(self, obs: Any, instruction: str, **kwargs: Any) -> np.ndarray:
        return self.rt_policy.predict(obs, instruction, **kwargs)

    def close(self) -> None:
        self.rt_policy.close()


class _Handler(BaseHTTPRequestHandler):
    server_version = "FasterPI-RealtimeHTTP/1.0"

    @property
    def policy(self) -> OpenPIHttpPolicy:
        return self.server.policy  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        return

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
        if self.path == "/cache_stats":
            self._send_json(200, _cache_snapshot(self.policy.rt_policy))
            return
        if self.path == "/health":
            pol = self.policy
            self._send_json(
                200,
                {
                    "ok": True,
                    "benchmark": pol.benchmark,
                    "label": getattr(pol.rt_policy, "label", "openpi"),
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
                lock = getattr(self.server, "_infer_lock", None)
                if ctype == CONTENT_TYPE_BIN:
                    data = unpack_predict_request(self._read_body())
                    obs = data["obs"]
                    instruction = data["instruction"]
                    kwargs = data["kwargs"]
                    if lock is None:
                        chunk = self.policy.predict(obs, instruction, **kwargs)
                    else:
                        with lock:
                            chunk = self.policy.predict(obs, instruction, **kwargs)
                    self._send_binary(200, pack_predict_response(chunk))
                    return
                data = self._read_json()
                obs = decode_obs(data.get("obs"))
                instruction = str(data.get("instruction", ""))
                extra = dict(data.get("kwargs") or {})
                if lock is None:
                    chunk = self.policy.predict(obs, instruction, **extra)
                else:
                    with lock:
                        chunk = self.policy.predict(obs, instruction, **extra)
                self._send_json(200, {"chunk": encode_chunk(chunk)})
                return
            if self.path == "/close":
                self.policy.close()
                self._send_json(200, {"ok": True})
                return
            self._send_json(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send_json(500, {"error": str(e)})


class PolicyHTTPServer(HTTPServer):
    """Serial HTTP. ThreadingMixIn runs ``/predict`` off the torch.compile
    capture thread and 500s RoboTwin FasterPI with CUDA graph errors.
    Warmup and the first real-obs recapture must share the serve_forever thread.
    Set FASTERPI_HTTP_THREADED=1 only for debug; default evals stay serial.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, policy: OpenPIHttpPolicy):
        super().__init__(server_address, _Handler)
        self.policy = policy
        self._infer_lock = threading.Lock()


def _threaded_requested() -> bool:
    return os.environ.get("FASTERPI_HTTP_THREADED", "").strip() in ("1", "true", "yes")


class ThreadedPolicyHTTPServer(ThreadingMixIn, PolicyHTTPServer):
    daemon_threads = True


def create_server(policy: OpenPIHttpPolicy, host: str = "127.0.0.1", port: int = 8765) -> PolicyHTTPServer:
    if _threaded_requested():
        return ThreadedPolicyHTTPServer((host, int(port)), policy)
    return PolicyHTTPServer((host, int(port)), policy)


def serve_background(policy: OpenPIHttpPolicy, host: str = "127.0.0.1", port: int = 8765) -> PolicyHTTPServer:
    httpd = create_server(policy, host=host, port=port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd
