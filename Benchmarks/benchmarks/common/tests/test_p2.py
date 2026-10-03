"""Unit tests for P2: release velocity, escape AABB, physics calibration helpers."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.common.workspace import (
    LIBERO_DEFAULT_AABB,
    ROBOTWIN_TASK_AABB,
    parse_aabb,
    resolve_workspace_aabb,
)
from benchmarks.dynamic_libero.trajectories import build_trajectory


def test_linear_velocity_at():
    traj = build_trajectory("linear", speed=0.003, axis="x", direction=1.0)
    traj.reset(np.zeros(3))
    v = traj.velocity_at(0.0, hz=20.0)
    assert np.allclose(v, [0.06, 0.0, 0.0], atol=1e-9)


def test_circle_velocity_tangent():
    traj = build_trajectory("circle", speed=0.003, radius=0.08, omega=0.05)
    traj.reset(np.array([0.1, 0.0, 0.8]))
    v = traj.velocity_at(0.0, hz=20.0)
    # at t=0, tangent is +y, |v| = r*omega*hz
    expected = np.array([0.0, 0.08 * 0.05 * 20.0, 0.0])
    assert np.allclose(v, expected, atol=1e-6)


def test_stop_and_go_velocity_zero_during_pause():
    traj = build_trajectory("stop_and_go", speed=0.01, axis="x", period=10, duty=0.5)
    traj.reset(np.zeros(3))
    for _ in range(6):
        traj.step(1.0)
    v = traj.velocity_at(traj.t, hz=20.0)
    assert np.linalg.norm(v) < 1e-6


def test_parse_and_resolve_aabb():
    aabb = parse_aabb("-0.1,-0.2,0.7,0.1,0.2,1.0")
    assert aabb.shape == (2, 3)
    lib = resolve_workspace_aabb("libero", suite_name="libero_object")
    assert lib is not None
    assert np.allclose(lib, LIBERO_DEFAULT_AABB)
    rt = resolve_workspace_aabb("robotwin", task_name="place_empty_cup")
    assert np.allclose(rt, ROBOTWIN_TASK_AABB["place_empty_cup"])
    assert resolve_workspace_aabb("libero", escape_check=False) is None


def test_libero_arena_aabb_not_floor_for_kitchen_suites():
    from benchmarks.common.workspace import expand_aabb_to_include

    spatial = resolve_workspace_aabb("libero", suite_name="libero_spatial")
    goal = resolve_workspace_aabb("libero", suite_name="libero_goal")
    assert spatial is not None and goal is not None
    # Tabletop / kitchen free joints sit near z=0.90; the old z_hi=0.40 box
    # marked every episode escaped on the first rails tick.
    assert float(spatial[0, 2]) >= 0.5
    assert float(spatial[1, 2]) >= 1.2
    assert np.allclose(spatial, goal)
    kitchen10 = resolve_workspace_aabb("libero", suite_name="libero_10", task_id=2)
    living10 = resolve_workspace_aabb("libero", suite_name="libero_10", task_id=0)
    study10 = resolve_workspace_aabb("libero", suite_name="libero_10", task_id=5)
    assert float(kitchen10[0, 2]) >= 0.5
    assert float(living10[0, 2]) < 0.3
    assert float(study10[0, 2]) >= 0.5
    grown = expand_aabb_to_include(
        np.array([[-0.45, -0.45, -0.02], [0.45, 0.45, 0.40]]),
        [0.0, 0.0, 0.91],
        margin=0.08,
    )
    assert grown[0, 2] <= 0.91 - 0.08
    assert grown[1, 2] >= 0.91 + 0.08
    # Kitchen spawn must already be inside the suite box (not only after expand).
    kitchen_spawn = np.array([0.05, -0.10, 0.91])
    assert np.all(kitchen_spawn >= spatial[0]) and np.all(kitchen_spawn <= spatial[1])
    living_spawn = np.array([0.0, 0.1, 0.42])
    assert np.all(living_spawn >= living10[0]) and np.all(living_spawn <= living10[1])


def test_libero_driver_release_velocity():
    from benchmarks.dynamic_libero.env.driver import RealtimeDriver

    class _Sim:
        class data:
            joint_qvel = {}
            joint_qpos = {}

            @staticmethod
            def get_joint_qvel(name):
                return _Sim.data.joint_qvel[name]

            @staticmethod
            def set_joint_qvel(name, v):
                _Sim.data.joint_qvel[name] = np.asarray(v, dtype=np.float64)

            @staticmethod
            def get_joint_qpos(name):
                return _Sim.data.joint_qpos[name]

            @staticmethod
            def set_joint_qpos(name, q):
                _Sim.data.joint_qpos[name] = np.asarray(q, dtype=np.float64)

        @staticmethod
        def forward():
            pass

    class _Base:
        sim = _Sim()
        obj_of_interest = ["obj"]
        objects_dict = {"obj": type("O", (), {"joints": ["obj_joint"]})()}

        def _pre_action(self, action, policy_step=False):
            pass

    class _Env:
        env = _Base()

    traj = build_trajectory("linear", speed=0.003, axis="y", direction=1.0)
    traj.reset(np.array([0.0, 0.0, 0.9]))
    _Sim.data.joint_qpos["obj_joint"] = np.array([0.0, 0.0, 0.9, 1, 0, 0, 0], dtype=np.float64)
    _Sim.data.joint_qvel["obj_joint"] = np.zeros(6)

    drv = RealtimeDriver(_Env(), traj, control_hz=20.0)
    drv.target = "obj"
    drv.jname = "obj_joint"
    drv.rails_active = True
    drv._release_rails()

    assert drv.released is True
    assert drv.rails_active is False
    qvel = _Sim.data.joint_qvel["obj_joint"]
    assert np.allclose(qvel[:3], [0.0, 0.06, 0.0], atol=1e-9)
    assert np.allclose(qvel[3:6], 0.0)


def test_resolve_physics_per_tick_explicit():
    from benchmarks.dynamic_robotwin.env.physics import resolve_physics_per_tick

    ppt, src = resolve_physics_per_tick(None, explicit=15, auto_calibrate=False)
    assert ppt == 15
    assert src == "explicit"

    ppt2, src2 = resolve_physics_per_tick(None, explicit=None, auto_calibrate=False)
    assert ppt2 == 12
    assert src2 == "default"
