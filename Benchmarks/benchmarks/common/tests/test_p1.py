"""Unit tests for common P1 (protocol, executor, policy codec/HTTP)."""

from __future__ import annotations

import json
import threading
import urllib.request

import numpy as np
import pytest

from benchmarks.common.executor import RealtimeExecutor
from benchmarks.common.policy.codec import (
    decode_chunk,
    decode_obs,
    encode_chunk,
    encode_obs,
    pack_predict_request,
    pack_predict_response,
    unpack_predict_request,
    unpack_predict_response,
)
from benchmarks.common.protocol import (
    RealtimeBackend,
    blind_window_ratio,
    effective_replan_steps,
)
from benchmarks.common.thinking import apply_thinking_async


class _MockDriver:
    def __init__(self):
        self.ticks = 0.0
        self.escaped = False

    def tick(self, dt: float = 1.0):
        self.ticks += float(dt)


def test_blind_window_ratio():
    assert blind_window_ratio(0.0, 10) == 0.0
    assert blind_window_ratio(4.0, 10) == pytest.approx(4.0 / 14.0)
    assert blind_window_ratio(float("inf"), 10) is None


def test_effective_replan_steps_locked():
    assert effective_replan_steps("libero", None) == 10
    assert effective_replan_steps("robotwin", None) == 8
    with pytest.raises(ValueError):
        effective_replan_steps("libero", 8, allow_override=False)
    assert effective_replan_steps("libero", 8, allow_override=True) == 8


def test_codec_roundtrip():
    obs = {"a": np.ones((2, 3), dtype=np.float32), "b": {"c": np.array([1, 2])}}
    back = decode_obs(encode_obs(obs))
    assert np.allclose(back["a"], obs["a"])
    assert np.allclose(back["b"]["c"], obs["b"]["c"])
    chunk = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert np.allclose(decode_chunk(encode_chunk(chunk)), chunk)


def test_binary_codec_roundtrip():
    obs = {
        "video.robot0_agentview_left": np.arange(8, dtype=np.uint8).reshape(2, 2, 2),
        "video.robot0_agentview_right": np.ones((2, 2, 2), dtype=np.uint8),
        "video.robot0_eye_in_hand": np.zeros((2, 2, 2), dtype=np.uint8),
        "state.base_position": np.arange(3, dtype=np.float32),
    }
    raw = pack_predict_request(obs=obs, instruction="pick mug", benchmark="robocasa")
    back = unpack_predict_request(raw)
    assert back["instruction"] == "pick mug"
    assert back["benchmark"] == "robocasa"
    assert np.array_equal(back["obs"]["video.robot0_agentview_left"], obs["video.robot0_agentview_left"])
    assert np.allclose(back["obs"]["state.base_position"], obs["state.base_position"])
    chunk = np.arange(24, dtype=np.float32).reshape(2, 12)
    assert np.allclose(unpack_predict_response(pack_predict_response(chunk)), chunk)


def test_async_thinking_executes_stale():
    drv = _MockDriver()
    executed = []

    def step_fn(a):
        executed.append(a)

    stats = apply_thinking_async(drv, 3.0, [10, 11, 12, 13], step_fn)
    assert executed == [10, 11, 12]
    assert stats["stale_executed"] == 3
    assert stats["stale_hold_ticks"] == pytest.approx(0.0)
    assert drv.ticks == pytest.approx(0.0)


def test_async_thinking_hold_when_stale_empty():
    drv = _MockDriver()
    stats = apply_thinking_async(drv, 2.5, [], lambda a: None)
    assert stats["stale_executed"] == 0
    assert stats["stale_hold_ticks"] == pytest.approx(2.5)
    assert drv.ticks == pytest.approx(2.5)


def test_executor_freeze_vs_async():
    drv = _MockDriver()
    exec_f = RealtimeExecutor(replan_steps=4, backend=RealtimeBackend.FREEZE)
    exec_f.on_replan([0, 1, 2, 3, 4, 5, 6, 7], drv, 2.0)
    assert exec_f.stale_executed_total == 0
    assert drv.ticks == pytest.approx(2.0)

    drv2 = _MockDriver()
    exec_a = RealtimeExecutor(replan_steps=4, backend=RealtimeBackend.ASYNC)
    exec_a.chunk = [0, 1, 2, 3, 4, 5, 6, 7]
    exec_a.idx = 2  # stale tail [2,3,...] but on_replan uses idx at call time
    stale_log = []

    def step_fn(a):
        stale_log.append(a)

    exec_a.on_replan([10, 11, 12, 13], drv2, 3.0, async_step_fn=step_fn)
    assert stale_log == [2, 3, 4]


def test_executor_replans_when_short_chunk_exhausted():
    drv = _MockDriver()
    ex = RealtimeExecutor(replan_steps=10, backend=RealtimeBackend.FREEZE)
    ex.on_replan([0, 1, 2], drv, 0.0)
    assert not ex.needs_replan()
    for _ in range(3):
        ex.next_action()
    assert ex.needs_replan()


class _EchoPolicy:
    benchmark = "test"

    def predict(self, obs, instruction: str, **kwargs):
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    def close(self):
        pass


def test_http_client_bypasses_proxy_for_localhost(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:7890")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7890")
    from benchmarks.common.policy.http_client import _localhost_no_proxy

    assert _localhost_no_proxy("http://127.0.0.1:8765")
    assert _localhost_no_proxy("http://localhost:8765")
    assert not _localhost_no_proxy("http://example.com:8765")


def test_http_policy_roundtrip():
    """Exercise handler + client without relying on a long-lived background server."""
    import http.client
    import json

    policy = _EchoPolicy()
    from benchmarks.common.policy.http_server import PolicyHTTPServer

    httpd = PolicyHTTPServer(("127.0.0.1", 0), policy)
    host, port = httpd.server_address

    def _serve(n: int):
        for _ in range(n):
            httpd.handle_request()

    import threading

    t = threading.Thread(target=_serve, args=(2,), daemon=True)
    t.start()

    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("GET", "/health")
    resp = conn.getresponse()
    assert resp.status == 200
    health = json.loads(resp.read().decode("utf-8"))
    assert health["ok"] is True

    payload = {"benchmark": "test", "instruction": "hi", "obs": encode_obs({"x": 1.0}), "kwargs": {}}
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("POST", "/predict", body=json.dumps(payload), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 200
    chunk = decode_chunk(json.loads(resp.read().decode("utf-8"))["chunk"])
    assert chunk.shape == (2, 2)
    t.join(timeout=2)
    httpd.server_close()
