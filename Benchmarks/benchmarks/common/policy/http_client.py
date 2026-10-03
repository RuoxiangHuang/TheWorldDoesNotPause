"""HTTP PolicyClient (stdlib only; sim and policy may run in separate processes)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np

from .codec import (
    CONTENT_TYPE_BIN,
    CONTENT_TYPE_JSON,
    decode_chunk,
    encode_obs,
    pack_predict_request,
    unpack_predict_response,
)

CodecName = str  # "binary" | "json"


def _resolve_codec(codec: str | None) -> str:
    raw = (codec or os.environ.get("POLICY_HTTP_CODEC") or "binary").strip().lower()
    if raw in ("bin", "binary", "numpy", "realtime-numpy"):
        return "binary"
    if raw in ("json", "legacy"):
        return "json"
    raise ValueError(f"unknown POLICY_HTTP_CODEC/codec {raw!r}; use binary|json")


def _localhost_no_proxy(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _urlopen(req: urllib.request.Request, timeout: float):
    """Open URL; bypass HTTP(S)_PROXY for loopback policy servers."""
    url = req.full_url
    if _localhost_no_proxy(url):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


@dataclass
class HttpPolicyClient:
    """Thin client for the Real-Time policy HTTP server."""

    url: str
    benchmark: str
    timeout_s: float = 600.0
    # None → POLICY_HTTP_CODEC env, else default binary.
    codec: str | None = None

    def __post_init__(self) -> None:
        self.url = self.url.rstrip("/")
        self.codec = _resolve_codec(self.codec)

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/health")

    def cache_stats(self) -> dict[str, Any]:
        return self._request_json("GET", "/cache_stats")

    def predict(self, obs: Any, instruction: str, **kwargs: Any) -> np.ndarray:
        if self.codec == "binary":
            body = pack_predict_request(
                obs=obs,
                instruction=instruction,
                benchmark=self.benchmark,
                kwargs=kwargs,
            )
            raw, _rtt = self._request_raw(
                "POST",
                "/predict",
                body,
                content_type=CONTENT_TYPE_BIN,
                accept=CONTENT_TYPE_BIN,
            )
            return unpack_predict_response(raw)

        payload = {
            "benchmark": self.benchmark,
            "instruction": instruction,
            "obs": encode_obs(obs),
            "kwargs": kwargs,
        }
        resp = self._request_json("POST", "/predict", payload)
        return decode_chunk(resp["chunk"])

    def close(self) -> None:
        try:
            self._request_json("POST", "/close", {})
        except Exception:
            pass

    def _request_json(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        raw, rtt = self._request_raw(
            method,
            path,
            data,
            content_type=CONTENT_TYPE_JSON if data is not None else None,
            accept=CONTENT_TYPE_JSON,
        )
        out = json.loads(raw.decode("utf-8"))
        out["_client_rtt_s"] = rtt
        return out

    def _request_raw(
        self,
        method: str,
        path: str,
        body: bytes | None,
        *,
        content_type: str | None,
        accept: str,
    ) -> tuple[bytes, float]:
        headers = {"Accept": accept}
        if content_type and body is not None:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=body,
            method=method,
            headers=headers,
        )
        t0 = time.perf_counter()
        try:
            with _urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"policy server {path} failed: {e.code} {detail}") from e
        return raw, time.perf_counter() - t0
