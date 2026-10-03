"""CPU tests for hybrid grasp phase transitions (LIBERO driver)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from benchmarks.common.grasp import RailsPhase
from benchmarks.dynamic_libero.env.driver import RealtimeDriver
from benchmarks.dynamic_libero.trajectories import build_trajectory


class _FakeSim:
    def __init__(self):
        self.data = MagicMock()
        self.forward = MagicMock()


class _FakeBase:
    def __init__(self):
        self.sim = _FakeSim()
        self.objects = ["pick_target"]
        self.fixtures = []
        self.object_body_ids = {"pick_target": 1}
        self._rtl_pre_patched = True
        self._rtl_driver = None


class _FakeEnv:
    def __init__(self):
        self.env = _FakeBase()


def _driver(grasp_mode: str = "hybrid_a", rails_variant: str = "pick") -> RealtimeDriver:
    traj = build_trajectory("linear", speed=0.003, axis="x")
    drv = RealtimeDriver(
        _FakeEnv(),
        traj,
        release_radius=0.06,
        grasp_mode=grasp_mode,
        rails_variant=rails_variant,
    )
    drv.target = "pick_target"
    drv.jname = "pick_target_joint0"
    drv.rails_active = True
    drv.motion_enabled = True
    drv.policy_t = 3.0
    return drv


def test_ghost_release_only_at_release_radius(monkeypatch):
    drv = _driver("ghost")
    monkeypatch.setattr(drv, "_release_rails", MagicMock())
    obs_far = {"pick_target_to_robot0_eef_pos": np.array([0.2, 0.0, 0.0])}
    drv.maybe_release(obs_far)
    assert drv.rails_active
    assert drv.rails_phase == RailsPhase.PURSUIT
    drv._release_rails.assert_not_called()

    obs_near = {"pick_target_to_robot0_eef_pos": np.array([0.04, 0.0, 0.0])}
    drv.maybe_release(obs_near)
    drv._release_rails.assert_called_once()


def test_hybrid_enters_engage_before_release():
    drv = _driver("hybrid_a")
    obs_engage = {"pick_target_to_robot0_eef_pos": np.array([0.10, 0.0, 0.0])}
    drv.maybe_release(obs_engage)
    assert drv.rails_active
    assert drv.rails_phase == RailsPhase.ENGAGE
    assert drv.engage_t == pytest.approx(3.0)


def test_hybrid_release_on_closing_at_engage(monkeypatch):
    drv = _driver("hybrid_a")
    monkeypatch.setattr(drv, "_release_rails", MagicMock())
    obs = {
        "pick_target_to_robot0_eef_pos": np.array([0.10, 0.0, 0.0]),
        "robot0_gripper_qpos": np.array([0.005, -0.005]),
    }
    drv.maybe_release(obs)
    drv._release_rails.assert_called_once()
    assert drv.contact_before_release


def test_hybrid_a_skips_pin_in_engage(monkeypatch):
    drv = _driver("hybrid_a")
    drv.rails_phase = RailsPhase.ENGAGE
    pin_calls = {"n": 0}

    def _pin():
        pin_calls["n"] += 1

    monkeypatch.setattr(drv, "_pin", _pin)
    drv._apply_rails_drive()
    assert pin_calls["n"] == 0


def test_hybrid_b_soft_track_in_engage(monkeypatch):
    drv = _driver("hybrid_b")
    drv.rails_phase = RailsPhase.ENGAGE
    soft_calls = {"n": 0}

    def _soft():
        soft_calls["n"] += 1

    monkeypatch.setattr(drv, "_soft_track", _soft)
    drv._apply_rails_drive()
    assert soft_calls["n"] == 1


def test_place_freeze_only_on_both_when_gripper_closes():
    """Place-only must keep the basket moving; both parks only on a real catch."""
    place = _driver("ghost", rails_variant="place")
    place.place_rails_active = True
    place.place_jname = None
    place.place_freeze = "both"
    near_open = {
        "pick_target_to_robot0_eef_pos": np.array([0.04, 0.0, 0.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }
    place.maybe_release(near_open)
    assert place.place_rails_active is True

    both = _driver("ghost", rails_variant="both")
    both.place_rails_active = True
    both.place_jname = None
    both.place_freeze = "both"
    both.maybe_release(near_open)
    assert both.place_rails_active is True
    near_close = {
        "pick_target_to_robot0_eef_pos": np.array([0.04, 0.0, 0.0]),
        "robot0_gripper_qpos": np.array([0.005, -0.005]),
    }
    both.maybe_release(near_close)
    assert both.place_rails_active is False


def test_place_freeze_all_parks_on_proximity():
    drv = _driver("ghost", rails_variant="place")
    drv.place_rails_active = True
    drv.place_jname = None
    drv.place_freeze = "all"
    drv.maybe_release({"pick_target_to_robot0_eef_pos": np.array([0.04, 0.0, 0.0])})
    assert drv.place_rails_active is False


def test_car_handoff_skips_release_kick(monkeypatch):
    drv = _driver("ghost")
    drv.release_kick = False
    drv.jname = "pick_target_joint0"
    monkeypatch.setattr(drv.base.sim.data, "get_joint_qvel", lambda *_: np.zeros(6))
    set_v = MagicMock()
    monkeypatch.setattr(drv.base.sim.data, "set_joint_qvel", set_v)
    drv._apply_release_velocity()
    assert drv.release_velocity_injected is False
    set_v.assert_not_called()
