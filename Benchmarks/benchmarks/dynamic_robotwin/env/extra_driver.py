"""Native extra-task driver: step task ODE during freeze ticks, no rails."""

from __future__ import annotations

import types
from typing import Any, Optional

import numpy as np
import sapien

from benchmarks.common.outcomes import AuxLog


def _as_entity(target: Any):
    if target is None:
        return None
    if hasattr(target, "actor") and hasattr(target.actor, "set_pose"):
        return target.actor
    if hasattr(target, "set_pose"):
        return target
    return None


def _set_pose(actor, pose: sapien.Pose) -> None:
    ent = _as_entity(actor)
    if ent is None:
        return
    try:
        ent.set_pose(pose)
    except Exception:
        pass


class ExtraNativeDriver:
    """Duck-types enough of ``RealtimeRoboTwinDriver`` for extra tasks.

    Thinking and execution both advance the task-native motion (ballistic
    flight, hops, roll, steered car, or PhysX scatter). The robot holds its
    joints while the world moves. A kinematic weld is an auxiliary
    ``script_weld`` event for intercepts. It is not rails release and it is
    not success; success stays ``check_success``.
    """

    def __init__(
        self,
        task_env,
        task_name: str,
        *,
        policy_hz: float = 20.0,
        physics_per_tick: int = 12,
        catch_radius: float = 0.12,
        gripper_close: float = 0.50,
    ):
        self.env = task_env
        self.task_name = str(task_name)
        self.policy_hz = float(policy_hz)
        self.physics_per_tick = int(physics_per_tick)
        self.physics_per_tick_source = "explicit"
        self.catch_radius = float(catch_radius)
        self.gripper_close = float(gripper_close)
        self.motion_enabled = False
        self.escaped = False
        self.released = False
        self.caught = False
        self.drift_m = 0.0
        self.policy_t = 0.0
        self.min_tcp_dist = float("inf")
        self.last_grip = 1.0
        self.contact_events: list[dict[str, Any]] = []
        self.trajectory_switches: list[dict[str, Any]] = []
        self.aux = AuxLog()
        self._rel: Optional[sapien.Pose] = None
        self._orig_take_action = None
        self._orig_scene_step = None

    def set_physics_per_tick(self, ppt: int, source: str = "explicit") -> None:
        self.physics_per_tick = max(int(ppt), 1)
        self.physics_per_tick_source = str(source)

    def on_episode_start(self) -> None:
        env = self.env
        bird = getattr(env, "object", None)
        if hasattr(env, "reset_flight"):
            env.reset_flight()
            if bird is not None and hasattr(env, "flight_pose"):
                _set_pose(bird, env.flight_pose())
        self.motion_enabled = False
        self.escaped = False
        self.released = False
        self.caught = False
        self.drift_m = 0.0
        self.policy_t = 0.0
        self.min_tcp_dist = float("inf")
        self.last_grip = 1.0
        self._rel = None
        if hasattr(self, "aux"):
            self.aux.events.clear()
        self._hook_take_action()
        self._install_scene_step_pin()

    def begin_policy(self) -> None:
        self.motion_enabled = True

    def _phys_dt(self) -> float:
        try:
            return float(self.env.scene.get_timestep())
        except Exception:
            return 1.0 / (max(self.policy_hz, 1e-6) * max(1, self.physics_per_tick))

    def _hold_robot(self) -> None:
        robot = self.env.robot
        left = np.asarray(robot.get_left_arm_jointState(), dtype=np.float64)
        right = np.asarray(robot.get_right_arm_jointState(), dtype=np.float64)
        zvel = np.zeros_like(left[:-1])
        robot.set_arm_joints(left[:-1], zvel, "left")
        robot.set_arm_joints(right[:-1], np.zeros_like(right[:-1]), "right")
        try:
            robot.set_gripper(float(left[-1]), "left")
            robot.set_gripper(float(right[-1]), "right")
        except Exception:
            pass

    def _tcp(self) -> np.ndarray:
        return np.asarray(self.env.robot.get_left_tcp_pose(), dtype=np.float64)[:3]

    def _ee_pose(self) -> sapien.Pose:
        p = np.asarray(self.env.get_arm_pose("left"), dtype=np.float64)
        return sapien.Pose(p[:3].tolist(), p[3:].tolist())

    def _weld_enabled(self) -> bool:
        return getattr(self.env, "extra_motion_kind", "kinematic") != "physx"

    def _publish_pose(self) -> None:
        env = self.env
        if getattr(env, "extra_motion_kind", "") == "physx":
            pin = getattr(env, "pin_extra_physics", None)
            if callable(pin):
                pin()
            return
        apply = getattr(env, "apply_extra_visuals", None)
        if callable(apply):
            apply()
            return
        bird = getattr(env, "object", None)
        if bird is not None and hasattr(env, "flight_pose"):
            _set_pose(bird, env.flight_pose())

    def _maybe_weld(self) -> None:
        if not self._weld_enabled():
            return
        if self.caught:
            bird = getattr(self.env, "object", None)
            if bird is not None and self._rel is not None:
                _set_pose(bird, self._ee_pose() * self._rel)
            return
        env = self.env
        bird = getattr(env, "object", None)
        if bird is None:
            return
        try:
            p = np.asarray(bird.get_pose().p, dtype=np.float64)
        except Exception:
            return
        tcp = self._tcp()
        dist = float(np.linalg.norm(p - tcp))
        if dist < self.min_tcp_dist:
            self.min_tcp_dist = dist
        try:
            left = np.asarray(env.robot.get_left_arm_jointState(), dtype=np.float64)
            self.last_grip = float(left[-1])
        except Exception:
            left = None
        if dist > self.catch_radius:
            return
        if left is None:
            return
        if float(left[-1]) > self.gripper_close:
            return
        ee = self._ee_pose()
        try:
            self._rel = ee.inv() * bird.get_pose()
        except Exception:
            self._rel = sapien.Pose()
        self.caught = True
        self.aux.record("script_weld", t=float(self.policy_t), radius=float(self.catch_radius))
        _set_pose(bird, ee * self._rel)

    def _step_object(self, dt_s: float) -> None:
        env = self.env
        bird = getattr(env, "object", None)
        if self.caught:
            self._maybe_weld()
            return
        if hasattr(env, "step_flight"):
            env.step_flight(float(dt_s))
            self._publish_pose()
        self._maybe_weld()

    def tick(self, dt: float = 1.0) -> None:
        if not self.motion_enabled:
            return
        dt = float(dt)
        if dt <= 0.0:
            return
        self._hold_robot()
        phys_dt = self._phys_dt()
        n_phys = max(1, int(round(dt * self.physics_per_tick)))
        for _ in range(n_phys):
            self._step_object(phys_dt)
            self.env.scene.step()
            if self.caught:
                self._maybe_weld()
            else:
                self._publish_pose()
        self.policy_t += dt
        try:
            self.env._update_render()
        except Exception:
            pass
        try:
            if bool(self.env.check_success()):
                self.env.eval_success = True
        except Exception:
            pass

    def on_policy_action(self) -> None:
        if not self.motion_enabled:
            return
        phys_dt = self._phys_dt()
        n_phys = max(1, int(self.physics_per_tick))
        for _ in range(n_phys):
            self._step_object(phys_dt)
            if self.caught:
                self._maybe_weld()
        self.policy_t += 1.0
        try:
            if bool(self.env.check_success()):
                self.env.eval_success = True
        except Exception:
            pass

    def _install_scene_step_pin(self) -> None:
        scene = self.env.scene
        if getattr(scene, "_rtl_extra_step_hooked", False):
            scene._rtl_tw_driver = self
            return
        orig = scene.step

        def step_with_pin(*args, **kwargs):
            drv = getattr(scene, "_rtl_tw_driver", None)
            out = orig(*args, **kwargs)
            if drv is not None:
                if drv.caught:
                    drv._maybe_weld()
                else:
                    drv._publish_pose()
            return out

        scene.step = step_with_pin
        scene._rtl_extra_step_hooked = True
        scene._rtl_tw_driver = self
        self._orig_scene_step = orig

    def _hook_take_action(self) -> None:
        env = self.env
        if getattr(env, "_rtl_extra_hooked", False):
            env._rtl_tw_driver = self
            return
        orig = env.take_action

        def take_action(self_env, action, action_type="qpos"):
            out = orig(action, action_type=action_type)
            drv = getattr(self_env, "_rtl_tw_driver", None)
            if drv is not None:
                drv.on_policy_action()
            return out

        env.take_action = types.MethodType(take_action, env)
        env._rtl_extra_hooked = True
        env._rtl_tw_driver = self
        self._orig_take_action = orig
