"""Thinking-phase coupling: freeze vs async stale-chunk execution."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from benchmarks.dynamic_libero.eval.measure_latency import apply_thinking_freeze


def apply_thinking_async(
    driver: Any,
    n_freeze: float,
    stale_actions: list[Any],
    step_fn: Callable[[Any], None],
    *,
    max_ticks: float | None = None,
) -> dict[str, float | int]:
    """Execute stale actions during thinking; hold robot when the tail is empty.

    Each thinking tick:
      1. Optionally execute one stale action via ``step_fn`` (robot + object move).
      2. Otherwise advance the world with ``driver.tick(1)`` (robot hold, object moves).

    Returns counters for logging / metrics.
    """
    n = float(n_freeze)
    if n <= 0.0:
        return {"applied_freeze": 0.0, "stale_executed": 0, "stale_hold_ticks": 0}
    if not math.isfinite(n):
        if max_ticks is None:
            raise ValueError("open-loop async thinking requires max_ticks")
        n = float(max_ticks)

    stale = list(stale_actions)
    stale_executed = 0
    stale_hold = 0.0
    remaining = n

    while remaining > 1e-9:
        dt = min(1.0, remaining)
        if stale:
            step_fn(stale.pop(0))
            stale_executed += 1
            # Object also advances one policy tick per executed action (driver hook).
        else:
            driver.tick(dt)
            stale_hold += dt
        remaining -= dt

    return {
        "applied_freeze": float(n),
        "stale_executed": int(stale_executed),
        "stale_hold_ticks": float(stale_hold),
    }
