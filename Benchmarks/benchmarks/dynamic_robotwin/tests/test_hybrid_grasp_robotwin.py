"""CPU tests for hybrid grasp transitions (RoboTwin driver)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from benchmarks.common.grasp import RailsPhase
from benchmarks.dynamic_robotwin.env.driver import RealtimeRoboTwinDriver
from benchmarks.dynamic_libero.trajectories import build_trajectory


def _driver(grasp_mode: str = "hybrid_a") -> RealtimeRoboTwinDriver:
    traj = build_trajectory("linear", speed=0.003, axis="x")
    drv = RealtimeRoboTwinDriver(
        MagicMock(),
        traj,
        task_name="place_empty_cup",
        release_radius=0.12,
        grasp_mode=grasp_mode,
    )
    drv.entity = MagicMock()
    drv.rails_active = True
    drv.policy_t = 5.0
    return drv


def test_robotwin_hybrid_engage_and_release(monkeypatch):
    drv = _driver("hybrid_a")
    monkeypatch.setattr(drv, "_enter_engage", MagicMock())
    monkeypatch.setattr(drv, "_release_rails", MagicMock())
    monkeypatch.setattr(drv, "_tcp_dist_and_closing", lambda: (0.20, False))
    drv.maybe_release()
    drv._enter_engage.assert_called_once()
    drv._release_rails.assert_not_called()

    monkeypatch.setattr(drv, "_tcp_dist_and_closing", lambda: (0.10, True))
    drv.maybe_release()
    drv._release_rails.assert_called_once()


def test_robotwin_ghost_uses_release_radius(monkeypatch):
    drv = _driver("ghost")
    released = {"n": 0}

    def _release():
        released["n"] += 1

    monkeypatch.setattr(drv, "_release_rails", _release)
    monkeypatch.setattr(drv, "_tcp_dist_and_closing", lambda: (0.15, False))
    drv.maybe_release()
    assert released["n"] == 0
    monkeypatch.setattr(drv, "_tcp_dist_and_closing", lambda: (0.08, False))
    drv.maybe_release()
    assert released["n"] == 1


def test_robotwin_enter_engage_clears_ghost(monkeypatch):
    drv = _driver("hybrid_b")
    drv.rails_drive = "kinematic"
    monkeypatch.setattr(drv, "_disable_kinematic", MagicMock())
    monkeypatch.setattr(drv, "_disable_ghost_collision_filter", MagicMock())
    drv._enter_engage()
    assert drv.rails_phase == RailsPhase.ENGAGE
    assert drv.engage_t == pytest.approx(5.0)
    assert drv.ghost_sliding is False
    assert drv.rails_drive == "velocity"
