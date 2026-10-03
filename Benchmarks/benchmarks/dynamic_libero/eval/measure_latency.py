"""Varying-observation latency measurement for Dynamic-LIBERO."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from ..env import libero_bridge as bridge

# Synthetic open-loop anchor: robot never finishes thinking within a normal episode.
OPEN_LOOP_LATENCY_S = float("inf")


def freeze_ticks_from_latency(latency_s: float, control_freq: float) -> float:
    """Map wall-clock latency to fractional control/policy ticks.

    ``n_freeze = L * f`` (no integer rounding). Sub-tick resolution is required
    so methods with L < 0.5 / f still couple into the world (e.g. 21 ms @ 20 Hz
    → 0.42 ticks instead of collapsing to 0).
    """
    lat = float(latency_s)
    freq = float(control_freq)
    if lat < 0.0:
        raise ValueError(f"latency_s must be >= 0, got {lat}")
    if freq <= 0.0:
        raise ValueError(f"control_freq must be > 0, got {freq}")
    if not math.isfinite(lat):
        return OPEN_LOOP_LATENCY_S
    return max(0.0, lat * freq)


def n_freeze_from_latency(latency_s: float, control_freq: float) -> float:
    """Alias of :func:`freeze_ticks_from_latency` (returns float ticks)."""
    return freeze_ticks_from_latency(latency_s, control_freq)


def freeze_segments(n_freeze: float) -> list[float]:
    """Split fractional freeze into unit ticks + optional remainder (for viz)."""
    n = float(n_freeze)
    if n <= 0.0 or not math.isfinite(n):
        return []
    whole = int(n)
    frac = n - whole
    segs = [1.0] * whole
    if frac > 1e-12:
        segs.append(frac)
    return segs


def apply_thinking_freeze(driver: Any, n_freeze: float, *, max_ticks: float | None = None) -> float:
    """Advance the world during THINKING. Returns applied freeze ticks."""
    n = float(n_freeze)
    if n <= 0.0:
        return 0.0
    if not math.isfinite(n):
        if max_ticks is None:
            raise ValueError("open-loop freeze requires max_ticks")
        n = float(max_ticks)
    if n > 0.0:
        driver.tick(n)
    return n


def measure_latency_varying(
    task,
    ctx: bridge.EpisodeContext,
    *,
    k: int = 8,
    warmup: int = 3,
    bridge_module=None,
) -> tuple[float, list[float]]:
    """Median replan latency with changing observations between timed calls.

    Requires ``k > warmup`` so compile / cache warmups are never folded into the
    reported median.
    """
    if int(k) <= int(warmup):
        raise ValueError(
            f"latency sample count k={k} must be > warmup={warmup} "
            "(need at least one post-warmup timed sample; "
            "compile/cache warmups would otherwise pollute the median)."
        )
    bridge_mod = bridge_module or bridge
    resolution = bridge_mod.get_env_resolution()
    env, desc = bridge_mod.get_libero_env(task, resolution, ctx.cfg.get("seed") if getattr(ctx, "cfg", None) else 0)
    try:
        env.reset()
        init_states = list(ctx.task_suite.get_task_init_states(int(ctx.cfg.EVALUATION.task_id if getattr(ctx, "cfg", None) else 0)))
        obs = env.set_init_state(init_states[0])
        wait = int(ctx.cfg.EVALUATION.get("num_steps_wait", 30)) if getattr(ctx, "cfg", None) else 30
        for _ in range(wait):
            obs, _, _, _ = env.step(bridge_mod.dummy_action())
        lats: list[float] = []
        done = False
        for _ in range(int(k)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            chunk, _, _ = bridge_mod.predict_action_chunk(obs, desc, ctx)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lats.append(time.perf_counter() - t0)
            for a in chunk[: min(5, len(chunk))]:
                obs, _, done, _ = env.step(a.tolist() if hasattr(a, "tolist") else a)
                if done:
                    break
            if done:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    if len(lats) <= int(warmup):
        raise RuntimeError(
            f"collected only {len(lats)} latency samples but warmup={warmup}; "
            "episode ended too early during measurement — increase k or use a longer task."
        )
    use = lats[int(warmup) :]
    med = float(np.median(use))
    return med, lats
