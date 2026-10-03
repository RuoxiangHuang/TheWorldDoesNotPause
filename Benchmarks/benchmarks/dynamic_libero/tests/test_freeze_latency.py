"""CPU unit tests for fractional freeze + latency measurement guards."""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.dynamic_libero.eval.measure_latency import (
    OPEN_LOOP_LATENCY_S,
    apply_thinking_freeze,
    freeze_segments,
    freeze_ticks_from_latency,
    measure_latency_varying,
    n_freeze_from_latency,
)
from benchmarks.dynamic_libero.trajectories import build_trajectory


def test_fractional_freeze_no_round_collapse():
    # Former bug: round(0.021 * 20) == 0 → no coupling for fast stacks.
    ticks = freeze_ticks_from_latency(0.021, 20.0)
    assert abs(ticks - 0.42) < 1e-9
    assert ticks > 0.0


def test_integer_alias_and_open_loop():
    assert abs(n_freeze_from_latency(0.1, 20.0) - 2.0) < 1e-12
    assert not math.isfinite(freeze_ticks_from_latency(OPEN_LOOP_LATENCY_S, 20.0))
    assert freeze_ticks_from_latency(0.0, 20.0) == 0.0


def test_freeze_segments_split():
    assert freeze_segments(0.0) == []
    assert freeze_segments(0.42) == pytest.approx([0.42])
    assert freeze_segments(2.0) == pytest.approx([1.0, 1.0])
    assert freeze_segments(2.3) == pytest.approx([1.0, 1.0, 0.3])


def test_apply_thinking_freeze_fractional_advances_path():
    class _Drv:
        def __init__(self):
            self.traj = build_trajectory("linear", speed=0.01, axis="x", direction=1.0)
            self.traj.reset(np.zeros(3))
            self.ticked = 0.0

        def tick(self, dt: float):
            self.ticked += float(dt)
            self.traj.step(dt)

    drv = _Drv()
    applied = apply_thinking_freeze(drv, 0.42)
    assert abs(applied - 0.42) < 1e-12
    assert abs(drv.ticked - 0.42) < 1e-12
    # 0.42 ticks * 0.01 m/tick
    assert abs(drv.traj.path_length() - 0.0042) < 1e-12


def test_apply_open_loop_requires_max_ticks():
    class _Drv:
        def tick(self, dt: float):
            self.dt = dt

    drv = _Drv()
    with pytest.raises(ValueError):
        apply_thinking_freeze(drv, float("inf"))
    applied = apply_thinking_freeze(drv, float("inf"), max_ticks=50.0)
    assert applied == 50.0
    assert drv.dt == 50.0


def test_measure_latency_rejects_k_le_warmup():
    with pytest.raises(ValueError, match="must be > warmup"):
        measure_latency_varying(None, None, k=3, warmup=3)
    with pytest.raises(ValueError, match="must be > warmup"):
        measure_latency_varying(None, None, k=2, warmup=3)
