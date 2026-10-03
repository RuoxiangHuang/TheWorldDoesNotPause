"""Startup pose must not jump between set_init and begin_policy."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.dynamic_libero.env.driver import RealtimeDriver
from benchmarks.dynamic_libero.env.movers import MoverSpec
from benchmarks.dynamic_libero.trajectories import build_trajectory


class _FakeSimData:
    def __init__(self, qpos: np.ndarray):
        self._qpos = {0: qpos.copy()}
        self._qvel = {0: np.zeros(6)}

    def get_joint_qpos(self, _jname):
        return self._qpos[0].copy()

    def set_joint_qpos(self, _jname, q):
        self._qpos[0] = np.asarray(q, dtype=np.float64).copy()

    def get_joint_qvel(self, _jname):
        return self._qvel[0].copy()

    def set_joint_qvel(self, _jname, v):
        self._qvel[0] = np.asarray(v, dtype=np.float64).copy()


class _FakeSim:
    def __init__(self, qpos: np.ndarray):
        self.data = _FakeSimData(qpos)

    def forward(self):
        return None


class _FakeObj:
    def __init__(self):
        self.joints = ["free_joint"]


class _FakeBase:
    def __init__(self, qpos: np.ndarray):
        self.sim = _FakeSim(qpos)
        self.objects_dict = {"alphabet_soup_1": _FakeObj()}
        self.obj_of_interest = ["alphabet_soup_1"]
        self._rtl_pre_patched = True  # skip MuJoCo pre_action patch

    def _pre_action(self, action, policy_step=False):
        return None


class _FakeEnv:
    def __init__(self, qpos: np.ndarray):
        self.env = _FakeBase(qpos)


def test_no_pose_jump_before_begin_policy():
    q0 = np.array([0.1, -0.2, 0.05, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env = _FakeEnv(q0.copy())
    traj = build_trajectory("linear", speed=0.003, axis="x", direction=1.0)
    mover = MoverSpec(
        target="alphabet_soup_1",
        category="alphabet_soup",
        noun="toy car",
        motion="drive",
        height_offset=0.0,
    )
    driver = RealtimeDriver(env, traj, mover=mover, control_hz=20.0)
    # Bypass _install_pre_action side effects on a real base.
    driver.base = env.env
    env.env._rtl_driver = driver

    driver.on_episode_start()
    q_after_start = env.env.sim.data.get_joint_qpos("free_joint")
    assert np.allclose(q_after_start, q0), "on_episode_start must not teleport"

    # Wait-phase pin must remain a no-op.
    driver._pin()
    q_after_wait_pin = env.env.sim.data.get_joint_qpos("free_joint")
    assert np.allclose(q_after_wait_pin, q0)

    driver.begin_policy()
    q_engage = env.env.sim.data.get_joint_qpos("free_joint")
    # t=0, height_offset=0 → position and orientation unchanged.
    assert np.allclose(q_engage[:3], q0[:3])
    assert np.allclose(q_engage[3:7], q0[3:7])


def test_height_offset_baked_only_at_begin_policy():
    q0 = np.array([0.0, 0.0, 0.04, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env = _FakeEnv(q0.copy())
    traj = build_trajectory("linear", speed=0.003, axis="x", direction=1.0)
    mover = MoverSpec(
        target="alphabet_soup_1",
        category="alphabet_soup",
        noun="toy car",
        height_offset=0.012,
    )
    driver = RealtimeDriver(env, traj, mover=mover)
    driver.base = env.env
    env.env._rtl_driver = driver

    driver.on_episode_start()
    assert np.allclose(env.env.sim.data.get_joint_qpos("j")[:3], q0[:3])

    driver.begin_policy()
    q1 = env.env.sim.data.get_joint_qpos("j")
    assert abs(float(q1[2]) - (0.04 + 0.012)) < 1e-9
    assert driver._height_baked is True
    # Second engage must not double-apply.
    driver.begin_policy()
    q2 = env.env.sim.data.get_joint_qpos("j")
    assert abs(float(q2[2]) - (0.04 + 0.012)) < 1e-9


def test_heading_starts_after_first_step():
    q0 = np.array([0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env = _FakeEnv(q0.copy())
    traj = build_trajectory("linear", speed=0.003, axis="y", direction=1.0)
    mover = MoverSpec(
        target="alphabet_soup_1",
        category="alphabet_soup",
        noun="toy car",
        motion="drive",
        height_offset=0.0,
    )
    driver = RealtimeDriver(env, traj, mover=mover)
    driver.base = env.env
    env.env._rtl_driver = driver
    driver.on_episode_start()
    driver.begin_policy()
    q_t0 = env.env.sim.data.get_joint_qpos("j")
    assert np.allclose(q_t0[3:7], q0[3:7]), "no yaw snap at engage"

    driver.on_control_step()  # advances t, pre_action would pin in real env
    driver._pin()
    q_t1 = env.env.sim.data.get_joint_qpos("j")
    assert not np.allclose(q_t1[3:7], q0[3:7]), "heading aligns after motion"


def test_hold_pins_without_advancing():
    from benchmarks.common.scene import scene_from_preset

    q0 = np.array([0.1, -0.2, 0.05, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env = _FakeEnv(q0.copy())
    traj = build_trajectory("linear", speed=0.003, axis="x", direction=1.0)
    mover = MoverSpec(
        target="alphabet_soup_1",
        category="alphabet_soup",
        noun="toy car",
        motion="drive",
        height_offset=0.0,
    )
    hold = scene_from_preset("apparatus_hold", speed=0.003)
    driver = RealtimeDriver(env, traj, mover=mover, control_hz=20.0, scene=hold)
    driver.base = env.env
    env.env._rtl_driver = driver
    driver.on_episode_start()
    driver.begin_policy()
    t0 = float(driver.trajectory.t)
    q_before = env.env.sim.data.get_joint_qpos("free_joint").copy()
    driver.tick(8.0)
    driver.on_control_step()
    assert float(driver.trajectory.t) == pytest.approx(t0)
    assert driver.rails_active is True
    q_after = env.env.sim.data.get_joint_qpos("free_joint")
    assert np.allclose(q_after[:3], q_before[:3])


def test_grasp_before_event_is_not_experienced():
    q0 = np.array([0.1, -0.2, 0.05, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env = _FakeEnv(q0.copy())
    traj = build_trajectory(
        "smooth_turn",
        speed=0.001,
        event_t=40.0,
        turn_sign=1,
        toward_center=False,
        axis="x",
        direction=1.0,
        event_seed=1,
    )
    mover = MoverSpec(
        target="alphabet_soup_1",
        category="alphabet_soup",
        noun="toy car",
        motion="drive",
        height_offset=0.0,
    )
    driver = RealtimeDriver(env, traj, mover=mover, control_hz=20.0)
    driver.base = env.env
    env.env._rtl_driver = driver
    driver.on_episode_start()
    driver.begin_policy()
    t_before = float(traj.t)
    for _ in range(10):
        driver.on_control_step()
    assert driver.event_experienced is False
    driver.rails_active = False
    driver.released = True
    for _ in range(40):
        driver.on_control_step()
        driver.tick(1.0)
    assert driver.event_experienced is False
    assert float(traj.t) == pytest.approx(t_before + 10.0)


def test_still_on_rails_at_event_is_experienced():
    q0 = np.array([0.1, -0.2, 0.05, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env = _FakeEnv(q0.copy())
    traj = build_trajectory(
        "smooth_turn",
        speed=0.001,
        event_t=8.0,
        turn_sign=1,
        toward_center=False,
        axis="x",
        direction=1.0,
        event_seed=1,
    )
    mover = MoverSpec(
        target="alphabet_soup_1",
        category="alphabet_soup",
        noun="toy car",
        motion="drive",
        height_offset=0.0,
    )
    driver = RealtimeDriver(env, traj, mover=mover, control_hz=20.0)
    driver.base = env.env
    env.env._rtl_driver = driver
    driver.on_episode_start()
    driver.begin_policy()
    for _ in range(10):
        driver.on_control_step()
    assert driver.event_experienced is True
    assert driver.episode_stats()["event_experienced"] is True
