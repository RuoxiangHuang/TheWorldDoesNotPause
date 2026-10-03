"""CPU tests for the smooth-turn reaction tracker (no MuJoCo)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.dynamic_libero.eval.turn_reaction import TurnReactionTracker
from benchmarks.dynamic_libero.trajectories import build_trajectory


def test_correction_after_takeover_not_before():
    traj = build_trajectory(
        "smooth_turn",
        speed=0.001,
        event_t=20.0,
        turn_sign=1,
        toward_center=False,
        axis="x",
        direction=1.0,
    )
    traj.reset(np.zeros(3))
    tr = TurnReactionTracker(traj, control_hz=20.0, persist=3, drop_m=0.002)
    n_hat = traj.turn_normal_xy()
    tr.on_replan_obs(10.0)
    tr.on_takeover(12.0)
    assert tr.first_obs_t is None
    assert tr.takeover_t is None

    tr.on_replan_obs(22.0)
    tr.on_takeover(24.0)
    assert tr.first_obs_t == pytest.approx(22.0)
    assert tr.takeover_t == pytest.approx(24.0)

    for i, lat in enumerate([0.020, 0.018, 0.016, 0.014, 0.012]):
        rel = np.array([n_hat[0] * lat, n_hat[1] * lat, 0.0])
        tr.on_step(24.0 + i, rel, dist_m=0.05, released=False)
    assert tr.correction_t is not None
    assert tr.correction_t >= 24.0
    s = tr.summary()
    assert s["turn_event_to_first_obs_s"] == pytest.approx(0.1)
    assert s["turn_first_obs_to_takeover_s"] == pytest.approx(0.1)


def test_summary_nulls_reaction_if_event_not_experienced():
    traj = build_trajectory(
        "smooth_turn",
        speed=0.001,
        event_t=20.0,
        turn_sign=1,
        toward_center=False,
        axis="x",
        direction=1.0,
    )
    traj.reset(np.zeros(3))
    tr = TurnReactionTracker(traj, control_hz=20.0)
    tr.on_replan_obs(22.0, rails_active=False)
    assert tr.first_obs_t is None
    s = tr.summary(event_experienced=False)
    assert s["event_experienced"] is False
    assert s["turn_event_to_first_obs_s"] is None
    assert s["turn_first_obs_to_takeover_s"] is None
    assert s["turn_event_to_correction_s"] is None
