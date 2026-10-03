"""RealtimeDriver: pin a free-joint object onto a Trajectory, then hand off to physics."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Optional

import numpy as np

from benchmarks.common.grasp import (
    DEFAULT_ENGAGE_ALPHA,
    DEFAULT_SOFT_KP,
    RailsPhase,
    engage_radius,
    grasp_stats,
    parse_grasp_mode,
    parse_place_freeze,
    should_inject_release_velocity,
    soft_kp,
)
from benchmarks.common.outcomes import AuxLog, GraspHoldTracker
from benchmarks.common.scene import SceneCondition, resolve_scene_condition
from benchmarks.common.workspace import expand_aabb_to_include

from ..trajectories.base import Trajectory

if TYPE_CHECKING:
    from .movers import MoverSpec
    from .pusher import PusherCar


def _yaw_quat_wxyz(forward_xy: np.ndarray) -> np.ndarray:
    """MuJoCo wxyz quaternion: yaw so local +X aligns with ``forward_xy``."""
    x, y = float(forward_xy[0]), float(forward_xy[1])
    n = (x * x + y * y) ** 0.5
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    yaw = float(np.arctan2(y, x))
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _axis_angle_quat_wxyz(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-9 or abs(angle) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / n
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


class RealtimeDriver:
    """Drive a LIBERO pick target until the eef is close.

    Default (no pusher): pin the free joint onto ``trajectory`` (ghost rails).
    With ``pusher``: a visible toy car rides the rails and the original mesh is
    seated on the car bumper — same cm/tick, no mesh swap, no language rewrite.
    """

    def __init__(
        self,
        env,
        trajectory: Trajectory,
        release_radius: float = 0.06,
        workspace_aabb: Optional[np.ndarray] = None,
        control_hz: float = 20.0,
        align_heading: bool = False,
        height_offset: float = 0.0,
        mover: Optional["MoverSpec"] = None,
        grasp_mode: str = "ghost",
        engage_alpha: float = DEFAULT_ENGAGE_ALPHA,
        pusher: Optional["PusherCar"] = None,
        place_pusher: Optional["PusherCar"] = None,
        rails_variant: str = "pick",
        place_target_name: Optional[str] = None,
        clear_path: bool = True,
        place_freeze: str | bool = "both",
        release_kick: Optional[bool] = None,
        scene: Optional[SceneCondition] = None,
    ):
        self.env = env
        self.base = env.env  # robosuite BDDLBaseDomain
        self.trajectory = trajectory
        self.scene = scene or resolve_scene_condition(
            speed=float(getattr(trajectory, "speed", 0.0) or 0.0),
            catalog_job=False,
        )
        self.release_radius = float(release_radius)
        self.grasp_mode = parse_grasp_mode(grasp_mode)
        self.engage_alpha = float(engage_alpha)
        self.r_engage = engage_radius(self.release_radius, alpha=self.engage_alpha)
        self.control_hz = float(control_hz)
        self.mover = mover
        self.pusher = pusher
        self.place_pusher = place_pusher
        self.place_traj = None
        self.place_target: Optional[str] = None
        self.place_jname: Optional[str] = None
        self.place_quat: Optional[np.ndarray] = None
        self.place_rails_active = False
        self._place_spawn: Optional[np.ndarray] = None
        self.path_cleared = 0
        from .dynamic_tasks import parse_rails_variant

        self.rails_variant = parse_rails_variant(rails_variant, default="pick")
        self.pick_motion = self.rails_variant in ("pick", "both")
        self.place_motion = self.rails_variant in ("place", "both")
        self.place_target_name = place_target_name
        self.clear_path = bool(clear_path)
        # Park the moving receptacle only after a latency-sensitive catch.
        # ``both`` (default): place-only keeps the basket moving (FastWAM was
        # already ~0 there); dual-motion parks the basket once the gripper
        # closes on the moving pick object. ``all`` restores the old "stop as
        # soon as EEF is close" behaviour that also let FastWAM hit ceiling.
        self.place_freeze = parse_place_freeze(place_freeze)
        # Ghost-pin leftover: inject trajectory velocity on unpin. Toy-car
        # seating already holds the mesh; kicking it out of the gripper hurts SR.
        if release_kick is None:
            self.release_kick = self.pusher is None and self.place_pusher is None
        else:
            self.release_kick = bool(release_kick)
        # When True, yaw follows planar velocity (toy-car / heading cues).
        if mover is not None:
            self.align_heading = mover.motion == "drive"
            self.height_offset = float(mover.height_offset)
            self.motion = str(mover.motion)
            self.roll_radius = float(mover.roll_radius)
        else:
            self.align_heading = bool(align_heading)
            self.height_offset = float(height_offset)
            self.motion = "drive" if align_heading else "pin"
            self.roll_radius = 0.025
        # Conveyor motive: original mesh on a belt — do not yaw-snap the object.
        if self.motion == "conveyor":
            self.align_heading = False
            self.height_offset = 0.0
        # Pusher motive: car rides the rails; pick mesh stays original (no yaw).
        if self.pusher is not None:
            self.align_heading = False
            self.height_offset = 0.0
            self.motion = "carrier" if getattr(self.pusher, "is_carrier", False) else "pusher"
        elif self.place_motion:
            self.align_heading = False
            self.height_offset = 0.0
            self.motion = "place"
        if self.rails_variant == "both" and not str(self.motion).startswith("carrier"):
            self.motion = "both"
        self._spawn_xyz: Optional[np.ndarray] = None
        # workspace_aabb: optional [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        self.workspace_aabb = None if workspace_aabb is None else np.asarray(workspace_aabb, dtype=np.float64)
        self.target: Optional[str] = None
        self.jname: Optional[str] = None
        self.rails_active = False
        self.released = False
        self.escaped = False
        self.started = False
        self.base_quat: Optional[np.ndarray] = None
        # Motion disabled during num_steps_wait; enable when policy takes over.
        self.motion_enabled = False
        # height_offset baked into trajectory.base_pos once at begin_policy.
        self._height_baked = False
        self.rails_phase = RailsPhase.PURSUIT
        self.policy_t = 0.0
        self.engage_t: float | None = None
        self.release_t: float | None = None
        self.contact_before_release = False
        self.release_velocity_injected = False
        self.aux = AuxLog()
        self.grasp_hold = GraspHoldTracker(hold_radius_m=float(self.release_radius))
        self.place_freeze_t: float | None = None
        # True iff the pick object was still on rails when a world-clock event
        # started. Grasp-before-event episodes keep full BDDL results but are
        # excluded from reaction-time stats.
        self.event_experienced = False
        self._install_pre_action()

    def _resolve_target(self):
        base = self.base
        if self.mover is not None:
            name = self.mover.target
            obj = base.objects_dict.get(name)
            if obj is None or not getattr(obj, "joints", None):
                raise RuntimeError(
                    f"Mover target {name!r} not found or has no free joint in scene"
                )
            return name, obj.joints[-1]
        for name in list(base.obj_of_interest) + list(base.objects_dict.keys()):
            obj = base.objects_dict.get(name)
            if obj is not None and getattr(obj, "joints", None):
                return name, obj.joints[-1]
        return None, None

    def _install_pre_action(self):
        base = self.base
        if getattr(base, "_rtl_pre_patched", False):
            base._rtl_driver = self
            return
        orig_pre = base._pre_action.__func__

        def _pre_action(self_env, action, policy_step=False):
            orig_pre(self_env, action, policy_step=policy_step)
            drv = getattr(self_env, "_rtl_driver", None)
            if drv is not None:
                drv._apply_rails_drive()

        base._pre_action = types.MethodType(_pre_action, base)
        base._rtl_pre_patched = True
        base._rtl_driver = self

    def _heading_quat(self) -> np.ndarray:
        v = self.trajectory.velocity_at(self.trajectory.t, hz=self.control_hz)
        fwd = np.asarray(v[:2], dtype=np.float64)
        if float(np.linalg.norm(fwd)) < 1e-6:
            p0 = self.trajectory.xyz_at(self.trajectory.t)
            p1 = self.trajectory.xyz_at(self.trajectory.t + 1.0)
            fwd = (p1 - p0)[:2]
        return _yaw_quat_wxyz(fwd)

    def _roll_quat(self) -> np.ndarray:
        if self.base_quat is None:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        v = self.trajectory.velocity_at(self.trajectory.t, hz=self.control_hz)
        fwd = np.asarray(v[:2], dtype=np.float64)
        if float(np.linalg.norm(fwd)) < 1e-6:
            p0 = self.trajectory.xyz_at(self.trajectory.t)
            p1 = self.trajectory.xyz_at(self.trajectory.t + 1.0)
            fwd = (p1 - p0)[:2]
        n = float(np.linalg.norm(fwd))
        if n < 1e-9:
            return self.base_quat.copy()
        fwd = fwd / n
        # Roll about axis perpendicular to planar velocity (wheel-like spin).
        axis = np.array([-fwd[1], fwd[0], 0.0], dtype=np.float64)
        dist = float(self.trajectory.path_length())
        angle = dist / max(self.roll_radius, 1e-6)
        spin = _axis_angle_quat_wxyz(axis, angle)
        return _quat_mul_wxyz(spin, self.base_quat)

    def _orientation_quat(self) -> np.ndarray:
        if self.motion == "roll":
            return self._roll_quat()
        # Defer heading snap until the object has actually moved along the
        # trajectory; applying yaw at t=0 caused a one-frame jolt at startup.
        if (
            self.align_heading
            and self.motion_enabled
            and float(self.trajectory.t) > 1e-9
        ):
            return self._heading_quat()
        if self.base_quat is not None:
            return self.base_quat.copy()
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _bake_height_offset(self) -> None:
        """Fold clearance into trajectory origin once (avoid per-pin z pops)."""
        if self._height_baked or abs(self.height_offset) < 1e-12:
            self._height_baked = True
            return
        if self.trajectory.base_pos is None:
            return
        bp = np.asarray(self.trajectory.base_pos, dtype=np.float64).reshape(3).copy()
        bp[2] += float(self.height_offset)
        self.trajectory.base_pos = bp
        self._height_baked = True

    def _bake_settled_height(self) -> None:
        """Keep rails z at the post-wait contact height.

        LIBERO floor inits often spawn with the mesh slightly *through* the
        plane. Physics during ``num_steps_wait`` pops the object up; pinning
        the raw spawn z at ``begin_policy`` looks like the can dropping into
        the ground. Only raise, never lower. Carrier keeps its layout lift.
        """
        if getattr(self.pusher, "is_carrier", False):
            # Relayout after wait-period settle using collision zmin, not XML site.
            if self.jname is not None:
                q = self.base.sim.data.get_joint_qpos(self.jname).copy()
                self.trajectory.base_pos = np.asarray(q[:3], dtype=np.float64).copy()
                self.base_quat = q[3:7].copy() if q.shape[0] >= 7 else self.base_quat
                self.pusher.layout_for_driver(self)
                if self.clear_path:
                    self._clear_car_paths()
            return
        if (
            self.pick_motion
            and self.jname is not None
            and self.trajectory.base_pos is not None
        ):
            z = float(self.base.sim.data.get_joint_qpos(self.jname)[2])
            bp = np.asarray(self.trajectory.base_pos, dtype=np.float64).reshape(3).copy()
            if z > float(bp[2]) + 1e-4:
                bp[2] = z
                self.trajectory.base_pos = bp
        if (
            self.place_jname is not None
            and self.place_traj is not None
            and getattr(self.place_traj, "base_pos", None) is not None
        ):
            z = float(self.base.sim.data.get_joint_qpos(self.place_jname)[2])
            bp = np.asarray(self.place_traj.base_pos, dtype=np.float64).reshape(3).copy()
            if z > float(bp[2]) + 1e-4:
                bp[2] = z
                self.place_traj.base_pos = bp

    def _eef_dist(self, obs: dict) -> float | None:
        if self.target is None:
            return None
        rel = obs.get(f"{self.target}_to_robot0_eef_pos")
        if rel is None:
            return None
        return float(np.linalg.norm(np.asarray(rel, dtype=np.float64)))

    def _gripper_closing(self, obs: dict) -> bool:
        """Best-effort: narrow gripper width ⇒ closing/grasping."""
        gq = obs.get("robot0_gripper_qpos")
        if gq is None:
            try:
                gq = self.base.sim.data.get_joint_qpos("robot0_gripper0_finger_joint1")
            except Exception:
                return False
        gq = np.asarray(gq, dtype=np.float64).reshape(-1)
        if gq.size == 0:
            return False
        if gq.size >= 2:
            width = float(abs(gq[0] - gq[1]))
        else:
            width = float(abs(gq[0]))
        return width < 0.02

    def _enter_engage(self) -> None:
        if self.rails_phase != RailsPhase.PURSUIT:
            return
        self.rails_phase = RailsPhase.ENGAGE
        self.engage_t = float(self.policy_t)

    def _soft_track(self) -> None:
        """hybrid_b engage: velocity drive toward trajectory with decaying kp."""
        if self.jname is None or not self.rails_active:
            return
        xyz_des, _ = self.trajectory.pose_at()
        q = self.base.sim.data.get_joint_qpos(self.jname).copy()
        cur = np.asarray(q[:3], dtype=np.float64)
        dist = float(np.linalg.norm(cur - np.asarray(xyz_des, dtype=np.float64)))
        kp = soft_kp(dist, self.r_engage, kp_max=DEFAULT_SOFT_KP)
        v_traj = self.trajectory.velocity_at(self.trajectory.t, hz=self.control_hz)
        v = np.asarray(v_traj, dtype=np.float64).reshape(3) + kp * (
            np.asarray(xyz_des, dtype=np.float64).reshape(3) - cur
        )
        qvel = self.base.sim.data.get_joint_qvel(self.jname).copy()
        qvel[:3] = v
        qvel[3:6] = 0.0
        self.base.sim.data.set_joint_qvel(self.jname, qvel)

    def _apply_rails_drive(self) -> None:
        """Dispatch pin vs pusher/carrier vs place-car vs free."""
        if not self.motion_enabled:
            return
        if self.pusher is not None:
            if self.rails_active:
                self.pusher.sync_drive(self)
        elif self.rails_active and self.jname is not None:
            if self.grasp_mode == "ghost" or self.rails_phase == RailsPhase.PURSUIT:
                self._pin()
            elif self.rails_phase == RailsPhase.ENGAGE and self.grasp_mode == "hybrid_b":
                self._soft_track()
        if self.place_pusher is not None and self.place_rails_active:
            self.place_pusher.sync_drive(self)

    def _pin(self):
        if not self.rails_active or self.jname is None:
            return
        # Keep pruned_init untouched during num_steps_wait so the scene does not
        # teleport (height/yaw) on the first frame after set_init_state.
        if not self.motion_enabled:
            return
        xyz, _ = self.trajectory.pose_at()
        q = self.base.sim.data.get_joint_qpos(self.jname).copy()
        q[:3] = xyz
        if q.shape[0] >= 7:
            q[3:7] = self._orientation_quat()
        self.base.sim.data.set_joint_qpos(self.jname, q)
        self.base.sim.data.set_joint_qvel(self.jname, np.zeros(6))

    def on_episode_start(self):
        self.base = self.env.env
        self.base._rtl_driver = self
        self.target, self.jname = self._resolve_target()
        if self.target is None:
            raise RuntimeError("No movable free-joint target; use a pick&place suite.")
        q = self.base.sim.data.get_joint_qpos(self.jname).copy()
        self.base_quat = q[3:7].copy() if q.shape[0] >= 7 else None
        self._spawn_xyz = np.asarray(q[:3], dtype=np.float64).copy()
        event_seed = getattr(self.trajectory, "event_seed", None)
        rng = None if event_seed is None else np.random.default_rng(int(event_seed))
        self.trajectory.reset(base_pos=q[:3], base_quat=self.base_quat, rng=rng)
        if self.pusher is not None and self.pick_motion and self.scene.modifies_layout():
            if getattr(self.pusher, "is_carrier", False):
                # Wait first (official settle). Lift onto the bed at begin_policy
                # using collision zmin, otherwise XML bottom_site floats the bowl.
                self.pusher.cfg.pos[:] = [0.0, 0.0, -1.0]
                self.pusher._apply_pose()
            else:
                self.pusher.layout_for_driver(self)
        elif self.pusher is not None:
            self.pusher.cfg.pos[:] = [0.0, 0.0, -1.0]
            self.pusher._apply_pose()
        self.place_rails_active = False
        self.place_target = None
        self.place_jname = None
        self.place_traj = None
        if self.place_motion and self.place_pusher is not None and self.scene.modifies_layout():
            from .pusher import resolve_place_target
            from ..trajectories import clone_trajectory

            pname, pj = resolve_place_target(self.base, self.place_target_name)
            self.place_target, self.place_jname = pname, pj
            if pj is not None:
                pq = self.base.sim.data.get_joint_qpos(pj).copy()
                self.place_quat = pq[3:7].copy() if pq.shape[0] >= 7 else None
                self._place_spawn = np.asarray(pq[:3], dtype=np.float64).copy()
                self.place_traj = clone_trajectory(self.trajectory)
                self.place_traj.reset(base_pos=pq[:3], base_quat=self.place_quat)
                self.place_rails_active = bool(
                    self.place_motion and self.scene.constrains_rails()
                )
                self.place_pusher.bind_and_layout(
                    self,
                    jname=pj,
                    trajectory=self.place_traj,
                    target_name=str(pname),
                    base_quat=self.place_quat,
                )
            else:
                self.place_pusher.cfg.pos[:] = [0.0, 0.0, -1.0]
                self.place_pusher._apply_pose()
        elif self.place_pusher is not None:
            self.place_pusher.cfg.pos[:] = [0.0, 0.0, -1.0]
            self.place_pusher._apply_pose()
        self._deflect_linear_from_sibling()
        self.path_cleared = 0
        deferred_carrier = (
            self.pusher is not None
            and getattr(self.pusher, "is_carrier", False)
            and self.pick_motion
        )
        if (
            self.clear_path
            and self.scene.modifies_layout()
            and not deferred_carrier
            and (self.pusher is not None or self.place_pusher is not None)
        ):
            self._clear_car_paths()
        if self.workspace_aabb is not None:
            # Never mark escaped because the spawn itself was outside a
            # floor-height box (kitchen / study / living-room tables).
            self.workspace_aabb = expand_aabb_to_include(self.workspace_aabb, q[:3])
        self.rails_active = bool(self.pick_motion) and self.scene.constrains_rails()
        self.released = False
        self.escaped = False
        self.started = False
        self.motion_enabled = False
        self._height_baked = False
        self.rails_phase = RailsPhase.PURSUIT
        self.policy_t = 0.0
        self.engage_t = None
        self.release_t = None
        self.contact_before_release = False
        self.release_velocity_injected = False
        self.place_freeze_t = None
        self.aux = AuxLog()
        self.grasp_hold = GraspHoldTracker(hold_radius_m=float(self.release_radius))
        self.event_experienced = False
        # Do not pin here: pinning applied height_offset + heading and caused a
        # visible one-shot shake right after init.
        self.base.sim.forward()

    def _place_xy(self) -> Optional[np.ndarray]:
        jname = self.place_jname
        if jname is None:
            from .pusher import resolve_place_target

            _, jname = resolve_place_target(self.base, self.place_target_name)
        if jname is None:
            return None
        q = self.base.sim.data.get_joint_qpos(jname)
        return np.asarray(q[:2], dtype=np.float64).copy()

    def _deflect_linear_from_sibling(self) -> None:
        """Keep pick/place linear rails from driving through each other.

        Carrier (spatial) plate-miss is handled at lift-on with a 0–90°
        kitchen-quadrant search; do not override that with inward headings
        that can point into the cabinet or stove.
        """
        if getattr(self.pusher, "is_carrier", False):
            return
        from benchmarks.dynamic_libero.trajectories.linear import (
            deflect_linear_heading,
            inward_heading_candidates,
        )

        if self.jname is None:
            return
        pick_xy = np.asarray(
            self.base.sim.data.get_joint_qpos(self.jname)[:2], dtype=np.float64
        )
        place_xy = self._place_xy()
        if place_xy is None:
            return
        changed = False
        if self.pick_motion and self.scene.constrains_rails():
            before = getattr(self.trajectory, "heading_deg", None)
            deflect_linear_heading(
                self.trajectory,
                pick_xy,
                place_xy,
                travel=0.40,
                min_clearance=0.14,
                candidates=inward_heading_candidates(pick_xy),
            )
            changed = changed or getattr(self.trajectory, "heading_deg", None) != before
        if self.place_traj is not None and self.place_motion:
            before = getattr(self.place_traj, "heading_deg", None)
            deflect_linear_heading(
                self.place_traj,
                place_xy,
                pick_xy,
                travel=0.40,
                min_clearance=0.14,
                candidates=inward_heading_candidates(place_xy),
            )
            changed = changed or getattr(self.place_traj, "heading_deg", None) != before
        if not changed:
            return
        if (
            self.pusher is not None
            and self.pick_motion
            and not getattr(self.pusher, "is_carrier", False)
        ):
            self.pusher.layout_for_driver(self)
        if (
            self.place_pusher is not None
            and self.place_jname is not None
            and self.place_traj is not None
        ):
            self.place_pusher.bind_and_layout(
                self,
                jname=self.place_jname,
                trajectory=self.place_traj,
                target_name=str(self.place_target),
                base_quat=self.place_quat,
            )

    def _clear_car_paths(self) -> None:
        from .pusher import clear_path_obstacles

        cars = [c for c in (self.pusher, self.place_pusher) if c is not None]
        if not cars:
            self.path_cleared = 0
            return
        self.path_cleared = int(clear_path_obstacles(self, cars))
        if self.path_cleared:
            self.aux.record("path_cleared", t=0.0, n=int(self.path_cleared))

    def begin_policy(self):
        """Call once after num_steps_wait, before the first replan."""
        self.motion_enabled = True
        self.started = True
        if self.rails_active or self.place_rails_active:
            self._bake_settled_height()
            self._bake_height_offset()
            self._apply_rails_drive()
            self.base.sim.forward()
        if self.jname is not None:
            try:
                self.grasp_hold.set_seated_z(
                    float(self.base.sim.data.get_joint_qpos(self.jname)[2])
                )
            except Exception:
                pass
        self._mark_event_if_due()

    def _event_t(self) -> Optional[float]:
        t = getattr(self.trajectory, "event_t", None)
        if t is None:
            return None
        return float(t)

    def _mark_event_if_due(self) -> None:
        """Record whether the moving object was still on rails at event time.

        After release the object is free: do not keep dragging it through the
        remaining trajectory, and do not count the episode as having seen the
        motion event.
        """
        if self.event_experienced:
            return
        t_ev = self._event_t()
        if t_ev is None:
            return
        if self.rails_active and self.policy_t + 1e-9 >= t_ev:
            self.event_experienced = True

    def tick(self, dt: float = 1.0):
        """Advance trajectory by `dt` ticks (used during THINKING freezes).

        ``dt`` may be fractional so sub-control-period latencies still couple.
        Release is evaluated on the control-step path (``maybe_release`` after
        ``env.step``), not here.
        """
        if not self.motion_enabled:
            return
        dt = float(dt)
        if dt <= 0.0:
            return
        self.started = True
        if self.scene.advances_rails():
            if self.rails_active:
                self.trajectory.step(dt)
            if self.place_rails_active and self.place_traj is not None:
                self.place_traj.step(dt)
        if self.rails_active or self.place_rails_active:
            self._apply_rails_drive()
            self.base.sim.forward()
            self._check_escape()
        self.policy_t += dt
        self._mark_event_if_due()

    def on_control_step(self):
        """Called once per executed robot action (object keeps same speed)."""
        if not self.motion_enabled:
            return
        self.started = True
        if self.scene.advances_rails():
            if self.rails_active:
                self.trajectory.step(1.0)
            if self.place_rails_active and self.place_traj is not None:
                self.place_traj.step(1.0)
        if self.rails_active or self.place_rails_active:
            self._check_escape()
        self.policy_t += 1.0
        self._mark_event_if_due()

    def _check_escape(self):
        if self.workspace_aabb is None:
            return
        lo, hi = self.workspace_aabb[0], self.workspace_aabb[1]

        def _out(xyz) -> bool:
            p = np.asarray(xyz, dtype=np.float64).reshape(3)
            return bool(np.any(p < lo) or np.any(p > hi))

        if self.rails_active:
            if self.jname is not None:
                xyz = self.base.sim.data.get_joint_qpos(self.jname)[:3]
            else:
                xyz = self.trajectory.xyz_at(self.trajectory.t)
            if _out(xyz):
                self.escaped = True
                return
        if self.place_rails_active and self.place_jname is not None:
            pxyz = self.base.sim.data.get_joint_qpos(self.place_jname)[:3]
            if _out(pxyz):
                self.escaped = True

    def _freeze_place_rails(self) -> None:
        """Park the receptacle once the pick object is acquired.

        Place/both failures are almost all timeout-after-grasp: the basket stays
        pinned on rails while the policy tries to insert. BDDL is unchanged.
        """
        if not self.place_rails_active:
            return
        self.place_rails_active = False
        self.place_freeze_t = float(self.policy_t)
        self.aux.record("place_freeze", t=self.place_freeze_t)
        if self.place_jname is None:
            return
        try:
            qvel = self.base.sim.data.get_joint_qvel(self.place_jname).copy()
            qvel[:] = 0.0
            self.base.sim.data.set_joint_qvel(self.place_jname, qvel)
        except Exception:
            return

    def _pick_acquired(self, obs: dict) -> bool:
        dist = self._eef_dist(obs)
        if dist is None:
            return False
        return float(dist) < float(self.release_radius)

    def _should_freeze_place(self, obs: dict) -> bool:
        if self.place_freeze == "off" or not self.place_rails_active:
            return False
        if self.place_freeze == "all":
            return self._pick_acquired(obs)
        # ``both``: only dual-motion, and only once the gripper is closing.
        # Proximity alone parked the basket while FastWAM was still thinking,
        # turning the rest of the episode into static place (SR→1 for off too).
        if not (self.pick_motion and self.place_motion):
            return False
        return self._pick_acquired(obs) and self._gripper_closing(obs)

    def _apply_release_velocity(self) -> None:
        if not self.release_kick:
            self.release_velocity_injected = False
            return
        if self.jname is None or abs(self.trajectory.speed) < 1e-12:
            return
        if not should_inject_release_velocity(
            self.grasp_mode,
            phase=self.rails_phase,
            had_contact=self.contact_before_release,
        ):
            self.release_velocity_injected = False
            return
        v = self.trajectory.velocity_at(self.trajectory.t, hz=self.control_hz)
        qvel = self.base.sim.data.get_joint_qvel(self.jname).copy()
        qvel[:3] = v
        qvel[3:6] = 0.0
        self.base.sim.data.set_joint_qvel(self.jname, qvel)
        self.release_velocity_injected = True
        self.aux.record("release_kick", t=float(self.policy_t))

    def _release_rails(self) -> None:
        if not self.rails_active:
            return
        self._apply_release_velocity()
        self.rails_active = False
        self.released = True
        self.rails_phase = RailsPhase.FREE
        self.release_t = float(self.policy_t)
        self.aux.record("rails_release", t=self.release_t)

    def _update_grasp_hold(self, obs: dict | None) -> None:
        dist = None
        if isinstance(obs, dict):
            dist = self._eef_dist(obs)
        z = None
        if self.jname is not None:
            try:
                z = float(self.base.sim.data.get_joint_qpos(self.jname)[2])
            except Exception:
                z = None
        self.grasp_hold.update(dist_m=dist, object_z=z)

    def maybe_release(self, obs: dict):
        if self._should_freeze_place(obs):
            self._freeze_place_rails()
        if not self.rails_active or self.target is None:
            return
        dist = self._eef_dist(obs)
        if dist is None:
            return
        closing = self._gripper_closing(obs)

        if self.grasp_mode != "ghost":
            if self.rails_phase == RailsPhase.PURSUIT and (
                dist < self.r_engage or (closing and dist < self.r_engage)
            ):
                self._enter_engage()
            if dist < self.r_engage and closing:
                self.contact_before_release = True
            if dist < self.release_radius or (closing and dist < self.r_engage):
                self._release_rails()
            return

        if dist < self.release_radius:
            self._release_rails()

    @property
    def drift_m(self) -> float:
        return float(self.trajectory.path_length())

    @property
    def object_drift_m(self) -> float:
        if self.jname is None or self._spawn_xyz is None:
            return float(self.drift_m)
        try:
            sim = getattr(self.base, "sim", None)
            if sim is None or getattr(sim, "data", None) is None:
                return float(self.drift_m)
            q = np.asarray(sim.data.get_joint_qpos(self.jname)[:3], dtype=np.float64)
        except Exception:
            return float(self.drift_m)
        return float(np.linalg.norm(q - self._spawn_xyz))

    def episode_stats(self) -> dict:
        out = {
            "target": self.target,
            "released": bool(self.released),
            "rails_released": bool(self.released),
            "escaped": bool(self.escaped),
            "motion": self.motion,
            "scene": self.scene.to_dict(),
            "aux_events": list(self.aux.events),
            "place_freeze_t": self.place_freeze_t,
            **self.grasp_hold.snapshot(),
            "align_heading": bool(self.align_heading),
            "height_offset": float(self.height_offset),
            "object_drift_m": float(self.object_drift_m),
            "pusher": self.pusher is not None,
            "place_pusher": self.place_pusher is not None,
            "place_target": self.place_target,
            "place_rails": bool(self.place_rails_active),
            "place_freeze": str(getattr(self, "place_freeze", "both")),
            "release_kick": bool(getattr(self, "release_kick", True)),
            "path_cleared": int(getattr(self, "path_cleared", 0)),
            "rails_variant": getattr(self, "rails_variant", "pick"),
            "pick_motion": bool(getattr(self, "pick_motion", True)),
            "place_motion": bool(getattr(self, "place_motion", False)),
            **grasp_stats(
                grasp_mode=self.grasp_mode,
                rails_phase=self.rails_phase.value,
                engage_t=self.engage_t,
                release_t=self.release_t,
                contact_before_release=self.contact_before_release,
                release_velocity_injected=self.release_velocity_injected,
                engage_radius_m=self.r_engage,
            ),
        }
        if self.mover is not None:
            out["mover_category"] = self.mover.category
        spec_fn = getattr(self.trajectory, "event_spec", None)
        if callable(spec_fn):
            try:
                out.update(spec_fn())
            except Exception:
                pass
        if getattr(self.trajectory, "event_t", None) is not None:
            out["event_experienced"] = bool(self.event_experienced)
        return out


def wrap_env_with_driver(env, driver: RealtimeDriver):
    """Attach driver lifecycle to env.step / set_init_state."""
    orig_step = env.step
    orig_set_init = env.set_init_state
    env._rtl_driver_ref = driver

    def step(action):
        drv = env._rtl_driver_ref
        if drv is not None:
            drv.on_control_step()
        obs, r, done, info = orig_step(action)
        if drv is not None:
            drv.maybe_release(obs)
            drv._update_grasp_hold(obs)
            if drv.escaped and not done:
                if isinstance(info, dict):
                    info = dict(info)
                    info["escaped"] = True
        return obs, r, done, info

    def set_init_state(state):
        obs = orig_set_init(state)
        drv = env._rtl_driver_ref
        if drv is not None:
            drv.on_episode_start()
        return obs

    env.step = step
    env.set_init_state = set_init_state
    return env
