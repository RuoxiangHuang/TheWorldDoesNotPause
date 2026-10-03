"""Shared helpers for CLIRA probe-2 (cost, age, executor alignment)."""

from __future__ import annotations

from typing import Any

import torch


AGES = (1, 2, 4, 8)


class CudaTimer:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.ms: dict[str, float] = {}

    def span(self, name: str):
        return _Span(self, name)


class _Span:
    def __init__(self, timer: CudaTimer, name: str):
        self.timer = timer
        self.name = name
        self._s = None
        self._e = None

    def __enter__(self):
        if not self.timer.enabled:
            return self
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)
        self._s.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.timer.enabled or exc_type is not None:
            return False
        self._e.record()
        torch.cuda.synchronize()
        self.timer.ms[self.name] = float(self._s.elapsed_time(self._e))
        return False


def clone_kv(cache: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{"k": e["k"].detach().clone(), "v": e["v"].detach().clone()} for e in cache]


def clone_past(past: Any) -> Any:
    if past is None:
        return None
    if torch.is_tensor(past):
        return past.detach().clone()
    if isinstance(past, (list, tuple)):
        seq = [clone_past(x) for x in past]
        return list(seq) if isinstance(past, list) else tuple(seq)
    if isinstance(past, dict):
        return {k: clone_past(v) for k, v in past.items()}
    if hasattr(past, "to_legacy_cache"):
        legacy = clone_past(past.to_legacy_cache())
        ctor = getattr(type(past), "from_legacy_cache", None)
        if callable(ctor):
            try:
                return ctor(legacy)
            except Exception:
                pass
        try:
            new = type(past)()
            from_legacy = getattr(new, "from_legacy_cache", None)
            if callable(from_legacy):
                return type(past).from_legacy_cache(legacy)
        except Exception:
            pass
        return legacy
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        new = type(past)()
        new.key_cache = [t.detach().clone() for t in past.key_cache]
        new.value_cache = [t.detach().clone() for t in past.value_cache]
        for attr in ("_seen_tokens", "seen_tokens"):
            if hasattr(past, attr):
                setattr(new, attr, getattr(past, attr))
        return new
    raise TypeError(f"cannot clone cache type {type(past)}")


def f32(x: Any) -> float | None:
    if x is None:
        return None
    return float(x)
