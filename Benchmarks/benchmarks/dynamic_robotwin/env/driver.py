"""RealtimeRoboTwinDriver: trajectory rails + Dynamic-RoboTwin stresses.

Rails prefer PhysX kinematic / velocity drive over per-step ``set_pose``
teleports (see PROTOCOL.md §8 / §9.7). Contact with the robot does not block
the scripted path during rails (protocol ghost sliding), unless explicitly
enabled for a collision-visible demo.
"""

from __future__ import annotations

import types
from typing import Any, List, Optional

import numpy as np

from benchmarks.common.grasp import (
    DEFAULT_ENGAGE_ALPHA,
    RailsPhase,
    engage_radius,
    grasp_stats,
    parse_grasp_mode,
    should_inject_release_velocity,
    soft_kp,
)
from benchmarks.common.outcomes import AuxLog, GraspHoldTracker
from benchmarks.common.scene import SceneCondition, resolve_scene_condition

from benchmarks.dynamic_libero.trajectories.base import Trajectory

from .dynamics import (
    ContactChainConfig,
    ContactChainManager,
    ContactTriggerConfig,
    DynamicConfig,
    OcclusionConfig,
    OcclusionManager,
    apply_trajectory_switch,
)
from .task_registry import get_task_meta, resolve_place_attr, resolve_target_attr

def _as_entity(target: Any):
    """RoboTwin Actor wraps a sapien Entity on `.actor`; Entity is used as-is."""
    if target is None:
        return None
    if hasattr(target, "actor") and hasattr(target.actor, "set_pose"):
        return target.actor
    if hasattr(target, "set_pose"):
        return target
    raise TypeError(f"Unsupported target type for rails: {type(target)}")


def _yaw_quat_wxyz(
    forward_xy: np.ndarray,
    *,
    mesh_forward: str = "x",
) -> np.ndarray:
    """SAPIEN wxyz yaw; ``mesh_forward`` is which local axis is the vehicle nose.

    RoboTwin ``057_toycar`` GLB is Z-up with length along **+Y** (not +X).
    """
    x, y = float(forward_xy[0]), float(forward_xy[1])
    n = (x * x + y * y) ** 0.5
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    # Map mesh forward axis onto world planar velocity.
    if mesh_forward == "y":
        # R_z(yaw) @ [0,1,0] = [-sin, cos]  ⇒  yaw = atan2(-vx, vy)
        yaw = float(np.arctan2(-x, y))
    else:
        # R_z(yaw) @ [1,0,0] = [cos, sin]  ⇒  yaw = atan2(vy, vx)
        yaw = float(np.arctan2(y, x))
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for SAPIEN's wxyz quaternions (left @ right)."""
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64).reshape(4)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64).reshape(4)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a local vector by a SAPIEN wxyz quaternion."""
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    vq = np.array([0.0, *np.asarray(vector, dtype=np.float64).reshape(3)])
    return _quat_multiply_wxyz(
        _quat_multiply_wxyz(q, vq), np.array([q[0], -q[1], -q[2], -q[3]])
    )[1:]


def _planar_local_axis(base_quat: np.ndarray, mesh_forward: str) -> np.ndarray:
    """Local axis whose world XY shadow is a usable driving direction.

    A tilted asset can stand its nominal forward axis vertical, leaving only
    float noise in XY; local Z is then the longest planar axis. Callers must
    agree on this choice or a yaw error computed against a near-vertical axis
    is pure noise.
    """
    local_forward = (
        np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if mesh_forward == "y"
        else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    if float(np.linalg.norm(_rotate_wxyz(base_quat, local_forward)[:2])) < 1e-3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return local_forward


def _heading_quat_wxyz(
    forward_xy: np.ndarray, *, base_quat: np.ndarray, mesh_forward: str
) -> np.ndarray:
    """Yaw an upright base pose so its mesh-forward axis follows XY velocity."""
    local_forward = _planar_local_axis(base_quat, mesh_forward)
    base_forward = _rotate_wxyz(base_quat, local_forward)[:2]
    target = np.asarray(forward_xy, dtype=np.float64).reshape(2)
    if float(np.linalg.norm(target)) < 1e-9 or float(np.linalg.norm(base_forward)) < 1e-9:
        return np.asarray(base_quat, dtype=np.float64).reshape(4).copy()
    yaw = float(np.arctan2(target[1], target[0]) - np.arctan2(base_forward[1], base_forward[0]))
    half = 0.5 * yaw
    return _quat_multiply_wxyz(
        np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64),
        base_quat,
    )


def _zero_velocities(entity) -> None:
    try:
        import sapien

        comp = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if comp is not None:
            comp.set_linear_velocity(np.zeros(3))
            comp.set_angular_velocity(np.zeros(3))
    except Exception:
        pass


def _rigid_dynamic(entity):
    if entity is None:
        return None
    try:
        import sapien

        return entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    except Exception:
        return None


def _rigid_static(entity):
    if entity is None:
        return None
    try:
        import sapien

        return entity.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
    except Exception:
        return None


def _physx_body(entity):
    return _rigid_dynamic(entity) or _rigid_static(entity)


class RealtimeRoboTwinDriver:
    """Pin a RoboTwin pick target onto a Trajectory with proximity release.

    Time unit is the **policy tick** (see PROTOCOL.md), not TOPP physics steps.
    Optional Dynamic-RoboTwin modes: contact_trigger, occlusion, contact_chain,
    and secondary rails for handoff placement targets.
    """

    # Compliant rails gains, only used when ``rails_collide`` trades protocol
    # ghost sliding for real contact (see _compliant_rails_velocity).
    RAILS_COLLIDE_KP = 3.0  # 1/s on position error
    RAILS_COLLIDE_POS_DEADBAND_M = 0.002
    RAILS_COLLIDE_V_SMOOTHING = 0.8  # EMA weight on the previous command
    RAILS_COLLIDE_KYAW = 4.0  # 1/s on heading error
    RAILS_COLLIDE_YAW_DEADBAND_RAD = 0.035  # ~2°
    RAILS_COLLIDE_WMAX = 2.0  # rad/s

    def __init__(
        self,
        task_env,
        trajectory: Trajectory,
        task_name: str,
        release_radius: float = 0.12,
        physics_per_tick: int = 12,
        policy_hz: float = 20.0,
        workspace_aabb: Optional[np.ndarray] = None,
        dynamic: Optional[DynamicConfig] = None,
        align_heading: bool = False,
        heading_mesh_forward: str = "x",
        rails_collide: bool = False,
        grasp_mode: str = "ghost",
        engage_alpha: float = DEFAULT_ENGAGE_ALPHA,
        rails_variant: str = "pick",
        spawn_pushers: bool = False,
        place_attr: Optional[str] = None,
        scene: Optional[SceneCondition] = None,
    ):
        self.env = task_env
        self.trajectory = trajectory
        self.task_name = task_name
        self.target_attr = resolve_target_attr(task_name)
        self.release_radius = float(release_radius)
        self.grasp_mode = parse_grasp_mode(grasp_mode)
        self.engage_alpha = float(engage_alpha)
        self.r_engage = engage_radius(self.release_radius, alpha=self.engage_alpha)
        self.physics_per_tick = int(physics_per_tick)
        self.policy_hz = float(policy_hz)
        self.physics_per_tick_source = "explicit"
        self.workspace_aabb = None if workspace_aabb is None else np.asarray(workspace_aabb, dtype=np.float64)
        self.dynamic = dynamic or DynamicConfig(modes=[])
        self.task_meta = get_task_meta(task_name)
        # Demo toycar: yaw follows planar velocity (primary rails only).
        self.align_heading = bool(align_heading)
        self.heading_mesh_forward = str(heading_mesh_forward or "x")
        # Demo-only opt-out of protocol ghost sliding: keep the rails target's
        # collision groups intact so it can physically push scene props.
        # Benchmark runs must leave this False (see PROTOCOL.md §8.1).
        self.rails_collide = bool(rails_collide)
        from benchmarks.dynamic_libero.env.dynamic_tasks import parse_rails_variant

        self.rails_variant = parse_rails_variant(rails_variant, default="pick")
        self.pick_motion = self.rails_variant in ("pick", "both")
        self.place_motion = self.rails_variant in ("place", "both")
        self.spawn_pushers = bool(spawn_pushers)
        self.place_attr = place_attr if place_attr is not None else resolve_place_attr(task_name)
        self.scene = scene or resolve_scene_condition(
            speed=float(getattr(trajectory, "speed", 0.0) or 0.0),
            catalog_job=bool(spawn_pushers),
        )
        self.aux = AuxLog()
        self.grasp_hold = GraspHoldTracker(hold_radius_m=float(self.release_radius))
        self.pick_rails_active = False
        self.place_rails_active = False
        self._place_entity = None
        self._place_wrapper = None
        self._place_traj = None
        self._place_base_quat = None
        self._place_rails_drive = "pose_fallback"
        self._pushers: list = []

        self.target_wrapper = None
        self.entity = None
        self.base_quat = None
        self.rails_active = False
        self.released = False
        self.escaped = False
        self.motion_enabled = False
        self.rails_phase = RailsPhase.PURSUIT
        self.engage_t: float | None = None
        self.release_t: float | None = None
        self.contact_before_release = False
        self.release_velocity_injected = False
        self._orig_take_action = None
        self._orig_scene_step = None
        self.min_tcp_dist = float("inf")
        self.held_streak = 0
        self.max_held_streak = 0
        self.ever_approached = False
        self.ever_closing_near = False

        # Rails drive bookkeeping (velocity / kinematic / pose_fallback).
        self.rails_drive = "pose_fallback"
        self.ghost_sliding = False
        self.ghost_collision_filter = False
        self._kinematic_entities: set[int] = set()
        self._ghost_saved_groups: dict[int, list[tuple[Any, list[int]]]] = {}
        self._v_cmd: dict[int, np.ndarray] = {}

        # Dynamic-RoboTwin bookkeeping
        self.policy_t = 0.0
        self.contact_events: List[dict[str, Any]] = []
        self.trajectory_switches: List[dict[str, Any]] = []
        self._contact_cfg = (
            self.dynamic.contact_trigger
            if self.dynamic.wants("contact_trigger")
            else ContactTriggerConfig(level="off")
        )
        self._occlusion = OcclusionManager(
            self.dynamic.occlusion
            if self.dynamic.wants("occlusion")
            else OcclusionConfig(enabled=False)
        )
        self._chain = ContactChainManager(
            self.dynamic.contact_chain
            if self.dynamic.wants("contact_chain")
            else ContactChainConfig(enabled=False)
        )
        self._secondary_entity = None
        self._secondary_traj = None
        self._secondary_base_quat = None
        self._secondary_rails_active = False

    def _resolve_target(self):
        wrapper = getattr(self.env, self.target_attr, None)
        if wrapper is None:
            raise RuntimeError(
                f"Task {self.task_name!r} has no attribute {self.target_attr!r} after setup_demo."
            )
        return wrapper, _as_entity(wrapper)

    def _resolve_place(self):
        attr = self.place_attr
        if not attr or not self.place_motion:
            return None, None
        wrapper = getattr(self.env, attr, None)
        if wrapper is None:
            return None, None
        return wrapper, _as_entity(wrapper)

    def _resolve_secondary(self):
        """Receive-side placement target for handoff coordination windows."""
        attr = self.task_meta.get("secondary_attr")
        if not attr or not self.dynamic.secondary_rails:
            return None, None
        wrapper = getattr(self.env, attr, None)
        if wrapper is None:
            return None, None
        return wrapper, _as_entity(wrapper)

    def _phys_dt(self) -> float:
        """Nominal seconds per PhysX step under policy-tick accounting."""
        return 1.0 / (max(self.policy_hz, 1e-6) * max(1, self.physics_per_tick))

    def _choose_rails_drive(self, entity) -> str:
        """Prefer kinematic, then velocity; else pose teleport fallback.

        ``rails_collide`` skips kinematic: an infinite-mass body accumulates
        penetration against props and the solver then ejects them explosively,
        so a contact-visible demo needs a real dynamic body.
        """
        comp = _rigid_dynamic(entity)
        if comp is None:
            return "pose_fallback"
        if not self.rails_collide and hasattr(comp, "set_kinematic") and hasattr(comp, "set_kinematic_target"):
            return "kinematic"
        if hasattr(comp, "set_linear_velocity"):
            return "velocity"
        return "pose_fallback"

    def _iter_collision_shapes(self, entity):
        comp = _physx_body(entity)
        if comp is None:
            return []
        try:
            shapes = comp.get_collision_shapes()
            return list(shapes) if shapes is not None else []
        except Exception:
            try:
                shapes = getattr(comp, "collision_shapes", None)
                return list(shapes) if shapes is not None else []
            except Exception:
                return []

    def _enable_ghost_collision_filter(self, entity) -> bool:
        """Disable rails-target contacts (g0/g1=0) so the path is not contact-blocked.

        Only the scripted target is modified; robot self-collision groups stay
        intact. Release restores the saved groups. If shapes are unavailable,
        returns False and PROTOCOL ghost-sliding still applies at the drive layer.
        """
        if entity is None:
            return False
        eid = id(entity)
        if eid in self._ghost_saved_groups:
            return True
        target_shapes = self._iter_collision_shapes(entity)
        if not target_shapes:
            return False
        saved: list[tuple[Any, list[int]]] = []
        try:
            for shape in target_shapes:
                groups = list(shape.get_collision_groups())
                saved.append((shape, groups))
                # Clear contact type / affinity → no PhysX contacts with any body.
                shape.set_collision_groups([0, 0, groups[2], groups[3]])
            self._ghost_saved_groups[eid] = saved
            return True
        except Exception:
            for shape, groups in saved:
                try:
                    shape.set_collision_groups(groups)
                except Exception:
                    pass
            return False

    def _disable_ghost_collision_filter(self, entity) -> None:
        if entity is None:
            return
        eid = id(entity)
        saved = self._ghost_saved_groups.pop(eid, None)
        if not saved:
            return
        for shape, groups in saved:
            try:
                shape.set_collision_groups(groups)
            except Exception:
                pass

    def _enable_kinematic(self, entity) -> bool:
        comp = _rigid_dynamic(entity)
        if comp is None or not hasattr(comp, "set_kinematic"):
            return False
        try:
            if hasattr(comp, "get_kinematic") and bool(comp.get_kinematic()):
                self._kinematic_entities.add(id(entity))
                return True
            comp.set_kinematic(True)
            self._kinematic_entities.add(id(entity))
            return True
        except Exception:
            return False

    def _disable_kinematic(self, entity) -> None:
        if entity is None or id(entity) not in self._kinematic_entities:
            return
        comp = _rigid_dynamic(entity)
        try:
            if comp is not None and hasattr(comp, "set_kinematic"):
                comp.set_kinematic(False)
        except Exception:
            pass
        self._kinematic_entities.discard(id(entity))

    def _align_chain_with_path(self) -> None:
        """Lay contact-chain props along the path, ahead of the rails target.

        Under ghost sliding the chain fires from ``on_primary_contact`` and its
        geometry is arbitrary, but a contact-visible demo needs the props in
        the way. The wider spacing gives the target a run-up before hop 0
        instead of spawning already in contact.
        """
        v = np.asarray(
            self.trajectory.velocity_at(0.0, hz=self.policy_hz), dtype=np.float64
        ).reshape(3)
        n = float(np.linalg.norm(v[:2]))
        if n < 1e-9:
            return
        self._chain.cfg.offset_dir = (float(v[0] / n), float(v[1] / n), 0.0)
        self._chain.cfg.spacing = max(float(self._chain.cfg.spacing), 0.12)

    def _compliant_rails_velocity(
        self,
        err: np.ndarray,
        traj,
        entity,
        *,
        kp_max: float | None = None,
    ) -> np.ndarray:
        """Path feed-forward + soft correction, for hybrid_b engage or collide demo."""
        try:
            v_ff = np.asarray(
                traj.velocity_at(traj.t, hz=self.policy_hz), dtype=np.float64
            ).reshape(3)
        except Exception:
            v_ff = np.zeros(3, dtype=np.float64)
        err = np.asarray(err, dtype=np.float64).reshape(3)
        kp = float(kp_max if kp_max is not None else self.RAILS_COLLIDE_KP)
        if float(np.linalg.norm(err)) < self.RAILS_COLLIDE_POS_DEADBAND_M:
            correction = np.zeros(3, dtype=np.float64)
        else:
            correction = kp * err
        v = v_ff + correction
        prev = self._v_cmd.get(id(entity))
        if prev is not None:
            a = self.RAILS_COLLIDE_V_SMOOTHING
            v = a * np.asarray(prev, dtype=np.float64).reshape(3) + (1.0 - a) * v
        self._v_cmd[id(entity)] = v.copy()
        return v

    def _heading_angular_velocity(self, cur_pose, desired_quat, dt: float, entity) -> list[float]:
        """Yaw rate that turns a velocity-driven body toward ``desired_quat``.

        Velocity drive never writes orientation, so a dynamic rails target
        would keep its spawn yaw. Roll/pitch stay zeroed to keep wheels down.
        """
        if not self.align_heading or entity is not self.entity or self.base_quat is None:
            return [0.0, 0.0, 0.0]
        local_forward = _planar_local_axis(self.base_quat, self.heading_mesh_forward)
        try:
            cur_q = np.asarray(cur_pose.q, dtype=np.float64).reshape(4)
        except Exception:
            return [0.0, 0.0, 0.0]
        cur_f = _rotate_wxyz(cur_q, local_forward)[:2]
        des_f = _rotate_wxyz(np.asarray(desired_quat, dtype=np.float64).reshape(4), local_forward)[:2]
        if float(np.linalg.norm(cur_f)) < 1e-3 or float(np.linalg.norm(des_f)) < 1e-3:
            return [0.0, 0.0, 0.0]
        err = float(
            np.arctan2(
                cur_f[0] * des_f[1] - cur_f[1] * des_f[0],
                float(np.dot(cur_f, des_f)),
            )
        )
        if self.rails_collide:
            # Same reasoning as the linear channel: a deadbeat yaw rate rings
            # after a bump. Deadband + modest gain settles it instead.
            if abs(err) < self.RAILS_COLLIDE_YAW_DEADBAND_RAD:
                return [0.0, 0.0, 0.0]
            wz = float(
                np.clip(
                    self.RAILS_COLLIDE_KYAW * err,
                    -self.RAILS_COLLIDE_WMAX,
                    self.RAILS_COLLIDE_WMAX,
                )
            )
        else:
            wz = float(np.clip(err / max(dt, 1e-6), -6.0, 6.0))
        return [0.0, 0.0, wz]

    def _drive_entity(self, entity, traj, base_quat, *, mode: str) -> None:
        """Drive one entity toward the current trajectory sample."""
        if entity is None or traj is None:
            return
        # hybrid_a engage: stop ghost drive so the object can be pushed/grasped.
        if (
            entity is self.entity
            and self.grasp_mode != "ghost"
            and self.rails_phase == RailsPhase.ENGAGE
            and self.grasp_mode == "hybrid_a"
        ):
            return
        import sapien

        xyz, quat = traj.pose_at()
        if self.align_heading and entity is self.entity:
            v = traj.velocity_at(traj.t, hz=self.policy_hz)
            fwd = np.asarray(v[:2], dtype=np.float64)
            if float(np.linalg.norm(fwd)) < 1e-6:
                p0 = traj.xyz_at(traj.t)
                p1 = traj.xyz_at(traj.t + 1.0)
                fwd = (p1 - p0)[:2]
            q = _heading_quat_wxyz(
                fwd,
                base_quat=base_quat if base_quat is not None else np.array([1.0, 0.0, 0.0, 0.0]),
                mesh_forward=self.heading_mesh_forward,
            )
        else:
            q = base_quat if base_quat is not None else quat
        # Keep rails height from an upright spawn (avoid re-burying after tip fixes).
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3).copy()
        if self.align_heading and entity is self.entity and self.base_quat is not None:
            # Preserve spawn Z (wheels-on-table) while XY follows the trajectory.
            try:
                xyz[2] = float(np.asarray(entity.get_pose().p, dtype=np.float64).reshape(3)[2])
            except Exception:
                pass
        pose = sapien.Pose(xyz.tolist(), np.asarray(q, dtype=np.float64).reshape(4).tolist())

        if mode == "kinematic":
            comp = _rigid_dynamic(entity)
            if comp is None:
                entity.set_pose(pose)
                _zero_velocities(entity)
                return
            if id(entity) not in self._kinematic_entities:
                if not self._enable_kinematic(entity):
                    # Fall through to velocity within this call.
                    mode = "velocity"
                else:
                    # Snap once so the kinematic body starts on-rail.
                    try:
                        entity.set_pose(pose)
                    except Exception:
                        pass
            if mode == "kinematic":
                try:
                    comp.set_kinematic_target(pose)
                    return
                except Exception:
                    mode = "velocity"

        if mode == "velocity":
            comp = _rigid_dynamic(entity)
            if comp is None:
                entity.set_pose(pose)
                _zero_velocities(entity)
                return
            try:
                cur_pose = entity.get_pose()
                cur = np.asarray(cur_pose.p, dtype=np.float64).reshape(3)
                dt = max(self._phys_dt(), 1e-6)
                err = np.asarray(xyz, dtype=np.float64).reshape(3) - cur
                speed_mps = abs(float(getattr(traj, "speed", 0.0))) * self.policy_hz
                use_compliant = self.rails_collide or (
                    entity is self.entity
                    and self.grasp_mode == "hybrid_b"
                    and self.rails_phase == RailsPhase.ENGAGE
                )
                if use_compliant:
                    dist = float(np.linalg.norm(err))
                    kp = self.RAILS_COLLIDE_KP
                    if (
                        entity is self.entity
                        and self.grasp_mode == "hybrid_b"
                        and self.rails_phase == RailsPhase.ENGAGE
                    ):
                        kp = soft_kp(dist, self.r_engage, kp_max=self.RAILS_COLLIDE_KP)
                    v = self._compliant_rails_velocity(err, traj, entity, kp_max=kp)
                    # Near-nominal clamp so a blocked car is genuinely slowed by
                    # contact instead of building up a corrective shove.
                    vmax = max(1.5 * max(speed_mps, 1e-3), 0.02)
                else:
                    v = err / dt
                    # Soft clamp: avoid huge corrective spikes if a step was skipped.
                    vmax = max(0.5, 8.0 * max(speed_mps, 1e-3))
                n = float(np.linalg.norm(v))
                if n > vmax:
                    v = v * (vmax / n)
                comp.set_linear_velocity(v.tolist())
                comp.set_angular_velocity(self._heading_angular_velocity(cur_pose, q, dt, entity))
                return
            except Exception:
                pass

        # pose_fallback
        entity.set_pose(pose)
        _zero_velocities(entity)

    def _drive_pick(self) -> bool:
        """Pick rails, or legacy ``rails_active`` when this skill is pick-only."""
        if self.pick_rails_active:
            return True
        return bool(self.rails_active and self.entity is not None and not self.place_motion)

    def _apply_rails_drive(self) -> None:
        """Drive pick / place rails targets and bumper cars."""
        if self._drive_pick() and self.entity is not None:
            mode = self.rails_drive
            if (
                self.grasp_mode != "ghost"
                and self.rails_phase == RailsPhase.ENGAGE
                and self.rails_drive == "kinematic"
            ):
                mode = "velocity"
            self._drive_entity(self.entity, self.trajectory, self.base_quat, mode=mode)
        if self.place_rails_active and self._place_entity is not None and self._place_traj is not None:
            self._drive_entity(
                self._place_entity,
                self._place_traj,
                self._place_base_quat,
                mode=self._place_rails_drive,
            )
        if self._secondary_rails_active:
            self._pin_secondary()
        for car in self._pushers:
            try:
                car.sync()
            except Exception:
                pass

    def _pin(self):
        """Drive primary target to the current trajectory sample (rails)."""
        self._apply_rails_drive()

    def _enter_engage(self) -> None:
        if self.rails_phase != RailsPhase.PURSUIT:
            return
        self.rails_phase = RailsPhase.ENGAGE
        self.engage_t = float(self.policy_t)
        self._disable_kinematic(self.entity)
        self._disable_ghost_collision_filter(self.entity)
        self.ghost_sliding = False
        if self.grasp_mode == "hybrid_b" and self.rails_drive == "kinematic":
            self.rails_drive = "velocity"

    def _tcp_dist_and_closing(self) -> tuple[float | None, bool]:
        if self.entity is None:
            return None, False
        try:
            tgt = np.asarray(self.entity.get_pose().p, dtype=np.float64)
            robot = self.env.robot
            left = np.asarray(robot.get_left_tcp_pose()[:3], dtype=np.float64)
            right = np.asarray(robot.get_right_tcp_pose()[:3], dtype=np.float64)
            dist = min(float(np.linalg.norm(tgt - left)), float(np.linalg.norm(tgt - right)))
            closing = bool(robot.is_left_gripper_close() or robot.is_right_gripper_close())
            return dist, closing
        except Exception:
            return None, False

    def _update_grasp_trace(self) -> None:
        """Track approach / closed-gripper hold for fail-stage reporting."""
        dist, closing = self._tcp_dist_and_closing()
        if dist is None:
            return
        if dist < self.min_tcp_dist:
            self.min_tcp_dist = float(dist)
        approach_r = max(0.15, 2.0 * float(self.release_radius))
        hold_r = max(float(self.release_radius), 0.08)
        if dist < approach_r:
            self.ever_approached = True
        if closing and dist < hold_r:
            self.ever_closing_near = True
            self.held_streak += 1
            if self.held_streak > self.max_held_streak:
                self.max_held_streak = int(self.held_streak)
        else:
            self.held_streak = 0

    def grasped_held(self, *, min_streak: int = 10) -> bool:
        """Deprecated gripper-close streak. Prefer grasp_hold.snapshot()."""
        return int(self.max_held_streak) >= int(min_streak)

    def _update_grasp_hold(self) -> None:
        dist, _closing = self._tcp_dist_and_closing()
        z = None
        if self.entity is not None:
            try:
                z = float(self.entity.get_pose().p[2])
            except Exception:
                z = None
        self.grasp_hold.update(dist_m=dist, object_z=z)

    def _pin_secondary(self):
        if (
            not self._secondary_rails_active
            or self._secondary_entity is None
            or self._secondary_traj is None
        ):
            return
        self._drive_entity(
            self._secondary_entity,
            self._secondary_traj,
            self._secondary_base_quat,
            mode=self.rails_drive,
        )

    def _deflect_linear_from_sibling(self, place_ent) -> None:
        """Keep planar linear rails from driving one actor through the other."""
        from benchmarks.dynamic_libero.trajectories.linear import (
            deflect_linear_heading,
            inward_heading_candidates,
        )

        try:
            pk = np.asarray(self.entity.get_pose().p, dtype=np.float64).reshape(3)[:2]
            pl = np.asarray(place_ent.get_pose().p, dtype=np.float64).reshape(3)[:2]
        except Exception:
            return
        if self.pick_rails_active:
            deflect_linear_heading(
                self.trajectory,
                pk,
                pl,
                travel=0.40,
                min_clearance=0.14,
                candidates=inward_heading_candidates(pk),
            )
        if self._place_traj is not None:
            deflect_linear_heading(
                self._place_traj,
                pl,
                pk,
                travel=0.40,
                min_clearance=0.14,
                candidates=inward_heading_candidates(pl),
            )

    def on_episode_start(self):
        self.target_wrapper, self.entity = self._resolve_target()
        pose = self.entity.get_pose()
        p = np.asarray(pose.p, dtype=np.float64).reshape(3)
        q = np.asarray(pose.q, dtype=np.float64).reshape(4)
        self.base_quat = q.copy()
        moving = self.scene.constrains_rails()
        self.trajectory.reset(base_pos=p, base_quat=q)
        if moving:
            from .pusher import orient_curve_inward

            orient_curve_inward(self.trajectory)
        self.pick_rails_active = bool(self.pick_motion and moving)
        self.released = False
        self.escaped = False
        self.motion_enabled = False
        self.policy_t = 0.0
        self.rails_phase = RailsPhase.PURSUIT
        self.engage_t = None
        self.release_t = None
        self.contact_before_release = False
        self.release_velocity_injected = False
        self.contact_events = []
        self.trajectory_switches = []
        self.min_tcp_dist = float("inf")
        self.held_streak = 0
        self.max_held_streak = 0
        self.ever_approached = False
        self.ever_closing_near = False
        self.aux = AuxLog()
        self.grasp_hold = GraspHoldTracker(hold_radius_m=float(self.release_radius))
        self._kinematic_entities.clear()
        self._ghost_saved_groups.clear()
        self._v_cmd.clear()
        self.ghost_collision_filter = False
        self._pushers = []
        self._place_entity = None
        self._place_wrapper = None
        self._place_traj = None
        self._place_base_quat = None
        self.place_rails_active = False

        place_wrap, place_ent = self._resolve_place()
        if place_ent is not None and self.place_motion and moving:
            from .pusher import clone_trajectory, orient_curve_inward

            pp = np.asarray(place_ent.get_pose().p, dtype=np.float64).reshape(3)
            pq = np.asarray(place_ent.get_pose().q, dtype=np.float64).reshape(4)
            self._place_traj = clone_trajectory(self.trajectory)
            self._place_traj.reset(base_pos=pp, base_quat=pq)
            orient_curve_inward(self._place_traj)
            self._place_entity = place_ent
            self._place_wrapper = place_wrap
            self._place_base_quat = pq.copy()
            self.place_rails_active = True
            self._place_rails_drive = self._choose_rails_drive(place_ent)
        elif place_ent is not None:
            self._place_entity = place_ent
            self._place_wrapper = place_wrap

        if moving and place_ent is not None:
            self._deflect_linear_from_sibling(place_ent)

        self.rails_active = bool(self.pick_rails_active or self.place_rails_active)

        if self.pick_rails_active:
            self.rails_drive = self._choose_rails_drive(self.entity)
            self.ghost_sliding = not self.rails_collide
            if not self.rails_collide:
                filt = self._enable_ghost_collision_filter(self.entity)
                self.ghost_collision_filter = bool(filt)
            if self.rails_drive == "kinematic":
                self._enable_kinematic(self.entity)
        else:
            self.rails_drive = "pose_fallback"
            self.ghost_sliding = (not self.rails_collide) and self.place_rails_active

        if self.place_rails_active and self._place_entity is not None:
            if not self.rails_collide:
                if self._enable_ghost_collision_filter(self._place_entity):
                    self.ghost_collision_filter = True
            if self._place_rails_drive == "kinematic":
                self._enable_kinematic(self._place_entity)

        # Secondary placement target rails (handoff receive-side window).
        self._secondary_entity = None
        self._secondary_traj = None
        self._secondary_base_quat = None
        self._secondary_rails_active = False
        _, sec_ent = self._resolve_secondary()
        if sec_ent is not None and self.rails_active:
            from benchmarks.dynamic_libero.trajectories import build_trajectory

            sp = np.asarray(sec_ent.get_pose().p, dtype=np.float64).reshape(3)
            sq = np.asarray(sec_ent.get_pose().q, dtype=np.float64).reshape(4)
            # Opposite axis drift vs primary so the coordination window closes.
            axis = getattr(self.trajectory, "axis_name", "y")
            other = "x" if axis == "y" else "y"
            self._secondary_traj = build_trajectory(
                "linear",
                speed=float(self.trajectory.speed) * 0.75,
                axis=other,
                toward_center=True,
            )
            self._secondary_traj.reset(base_pos=sp, base_quat=sq)
            self._secondary_entity = sec_ent
            self._secondary_base_quat = sq.copy()
            self._secondary_rails_active = True
            if self.ghost_sliding and not self.rails_collide:
                if self._enable_ghost_collision_filter(sec_ent):
                    self.ghost_collision_filter = True
            if self.rails_drive == "kinematic":
                self._enable_kinematic(sec_ent)

        if self.spawn_pushers and self.rails_active and self.scene.installs_visual_actor():
            self._install_pushers()

        self._pin()
        self._pin_secondary()
        try:
            self._occlusion.setup(self.env, p)
        except Exception:
            pass
        try:
            if self.dynamic.wants("contact_chain"):
                if self.rails_collide:
                    self._align_chain_with_path()
                self._chain.setup(self.env, p)
        except Exception:
            pass
        self._install_take_action_hook()
        self._install_scene_step_pin()

    def _table_z(self) -> float:
        from .pusher import TABLE_Z

        bias = float(getattr(self.env, "table_z_bias", 0.0) or 0.0)
        return float(TABLE_Z) + bias

    def _install_pushers(self) -> None:
        from .pusher import (
            RoboTwinPusherCar,
            measure_planar_footprint,
            pusher_object_radius,
            spawn_bumper_car,
        )

        self._pushers = []
        table_z = self._table_z()
        hz = float(self.policy_hz)
        if self.pick_rails_active:
            ent = spawn_bumper_car(self.env, name="rtl_tw_pusher", palette="red")
            car = RoboTwinPusherCar(ent, palette="red", target_name=str(self.target_attr))
            car.bind(
                self.trajectory,
                table_z=table_z,
                hz=hz,
                obj_radius=pusher_object_radius(
                    self.target_wrapper,
                    self.target_attr,
                    entity=self.entity,
                ),
                footprint=measure_planar_footprint(self.target_wrapper, entity=self.entity),
            )
            self._pushers.append(car)
        if self.place_rails_active and self._place_traj is not None:
            ent = spawn_bumper_car(self.env, name="rtl_tw_place_pusher", palette="blue")
            car = RoboTwinPusherCar(ent, palette="blue", target_name=str(self.place_attr or "coaster"))
            car.bind(
                self._place_traj,
                table_z=table_z,
                hz=hz,
                obj_radius=pusher_object_radius(
                    self._place_wrapper,
                    self.place_attr or "coaster",
                    entity=self._place_entity,
                ),
                footprint=measure_planar_footprint(
                    self._place_wrapper, entity=self._place_entity
                ),
            )
            self._pushers.append(car)

    def begin_policy(self):
        self.motion_enabled = True
        if self.entity is not None:
            try:
                self.grasp_hold.set_seated_z(float(self.entity.get_pose().p[2]))
            except Exception:
                pass

    def _install_scene_step_pin(self):
        """Re-drive every physics step so TOPP internals cannot drop the target."""
        scene = self.env.scene
        if getattr(scene, "_rtl_tw_step_hooked", False):
            scene._rtl_tw_driver = self
            return
        orig = scene.step

        def step_with_drive(*args, **kwargs):
            drv = getattr(scene, "_rtl_tw_driver", None)
            if drv is not None and drv.rails_active:
                drv._apply_rails_drive()
            return orig(*args, **kwargs)

        scene.step = step_with_drive
        scene._rtl_tw_step_hooked = True
        scene._rtl_tw_driver = self
        self._orig_scene_step = orig

    def _hold_robot(self):
        """Keep current joint drive targets so the arms do not sag during freeze."""
        robot = self.env.robot
        left = np.asarray(robot.get_left_arm_jointState(), dtype=np.float64)
        right = np.asarray(robot.get_right_arm_jointState(), dtype=np.float64)
        left_arm, left_g = left[:-1], float(left[-1])
        right_arm, right_g = right[:-1], float(right[-1])
        zvel = np.zeros_like(left_arm)
        robot.set_arm_joints(left_arm, zvel, "left")
        robot.set_arm_joints(right_arm, zvel, "right")
        try:
            robot.set_gripper(left_g, "left")
            robot.set_gripper(right_g, "right")
        except Exception:
            pass

    def _target_xyz(self) -> Optional[np.ndarray]:
        if self.entity is None:
            return None
        try:
            return np.asarray(self.entity.get_pose().p, dtype=np.float64).reshape(3)
        except Exception:
            return None

    def _advance_occlusion(self, dt: float) -> None:
        xyz = self._target_xyz()
        if xyz is None:
            return
        try:
            self._occlusion.tick(self.policy_t, xyz, dt=dt)
        except Exception:
            pass

    def tick(self, dt: float = 1.0):
        """Advance trajectory by `dt` policy ticks and step physics (THINKING).

        ``dt`` may be fractional. Physics substeps scale as
        ``round(dt * physics_per_tick)`` so a 21 ms latency at 20 Hz
        (0.42 ticks) still advances the world instead of collapsing to zero.
        """
        if not self.motion_enabled:
            return
        dt = float(dt)
        if dt <= 0.0:
            return
        if self.scene.advances_rails():
            if self._drive_pick():
                self.trajectory.step(dt)
            if self.place_rails_active and self._place_traj is not None:
                self._place_traj.step(dt)
            if self._secondary_rails_active and self._secondary_traj is not None:
                self._secondary_traj.step(dt)
        if self.rails_active:
            self._pin()
            self._check_escape()
        if self._secondary_rails_active:
            self._pin_secondary()
        self.policy_t += dt
        if self.scene.advances_rails():
            self._maybe_contact_trigger()
        self._advance_occlusion(dt)
        if self.scene.advances_rails():
            self._advance_contact_chain(dt)
        self._hold_robot()
        n_phys = max(1, int(round(dt * self.physics_per_tick)))
        for _ in range(n_phys):
            # Drive is applied inside the scene.step hook (same as take_action).
            self.env.scene.step()
        try:
            self.env._update_render()
        except Exception:
            pass

    def on_policy_action(self):
        """Called once after each take_action (execution segment)."""
        if not self.motion_enabled:
            return
        if self.scene.advances_rails():
            if self._drive_pick():
                self.trajectory.step(1.0)
            if self.place_rails_active and self._place_traj is not None:
                self._place_traj.step(1.0)
            if self._secondary_rails_active and self._secondary_traj is not None:
                self._secondary_traj.step(1.0)
        if self.rails_active:
            self._pin()
            self._check_escape()
        if self._secondary_rails_active:
            self._pin_secondary()
        self.policy_t += 1.0
        if self.scene.advances_rails():
            self._maybe_contact_trigger()
        self._advance_occlusion(1.0)
        if self.scene.advances_rails():
            self._advance_contact_chain(1.0)
        self.maybe_release()
        self._update_grasp_trace()
        self._update_grasp_hold()

    def _advance_contact_chain(self, dt: float) -> None:
        try:
            self._chain.tick(self.policy_t, dt=dt)
        except Exception:
            pass

    def _probe_contact(self) -> Optional[dict[str, Any]]:
        """TCP / gripper heuristic for either-arm contact with the primary target."""
        if self.entity is None:
            return None
        try:
            tgt = np.asarray(self.entity.get_pose().p, dtype=np.float64)
            robot = self.env.robot
            left = np.asarray(robot.get_left_tcp_pose()[:3], dtype=np.float64)
            right = np.asarray(robot.get_right_tcp_pose()[:3], dtype=np.float64)
            d_left = float(np.linalg.norm(tgt - left))
            d_right = float(np.linalg.norm(tgt - right))
            dist = min(d_left, d_right)
            arm = "left" if d_left <= d_right else "right"
        except Exception:
            return None
        radius = float(self._contact_cfg.contact_radius)
        try:
            closing = bool(robot.is_left_gripper_close() or robot.is_right_gripper_close())
        except Exception:
            closing = False
        near = dist < radius
        grasp_near = closing and dist < max(radius * 1.25, radius + 0.03)
        if not (near or grasp_near):
            return None
        return {
            "arm": arm,
            "dist": dist,
            "closing": closing,
            "t": float(self.policy_t),
            "kind": "tcp" if near else "gripper",
        }

    def _maybe_contact_trigger(self) -> None:
        if not self.rails_active:
            return
        if not self._contact_cfg.enabled():
            return
        if len(self.trajectory_switches) >= int(self._contact_cfg.max_switches):
            return
        event = self._probe_contact()
        if event is None:
            return
        self.contact_events.append(event)
        mode = apply_trajectory_switch(
            self.trajectory,
            self._contact_cfg.level,
            switch_mode=self._contact_cfg.switch_mode,
        )
        self.trajectory_switches.append(
            {
                "t": event["t"],
                "arm": event["arm"],
                "mode": mode,
                "level": self._contact_cfg.level,
                "speed_after": float(self.trajectory.speed),
            }
        )
        self.aux.record(
            "trajectory_switch",
            t=float(event["t"]),
            mode=mode,
            level=self._contact_cfg.level,
        )
        self._pin()
        try:
            self._chain.on_primary_contact(t_ticks=float(self.policy_t))
        except Exception:
            pass

    def _check_escape(self):
        if self.workspace_aabb is None or not self.rails_active:
            return
        lo, hi = self.workspace_aabb[0], self.workspace_aabb[1]

        def _out(traj) -> bool:
            if traj is None:
                return False
            xyz = traj.xyz_at(traj.t)
            return bool(np.any(xyz < lo) or np.any(xyz > hi))

        if self.pick_rails_active and _out(self.trajectory):
            self.escaped = True
        if self.place_rails_active and _out(self._place_traj):
            self.escaped = True

    def set_physics_per_tick(self, value: int, source: str = "explicit") -> None:
        self.physics_per_tick = max(1, int(value))
        self.physics_per_tick_source = str(source)

    def _apply_release_velocity(self) -> None:
        """Inject trajectory linear velocity when rails release."""
        if self.entity is None or abs(self.trajectory.speed) < 1e-12:
            self.release_velocity_injected = False
            return
        if not should_inject_release_velocity(
            self.grasp_mode,
            phase=self.rails_phase,
            had_contact=self.contact_before_release,
        ):
            self.release_velocity_injected = False
            return
        try:
            import sapien

            comp = self.entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if comp is None:
                self.release_velocity_injected = False
                return
            v = self.trajectory.velocity_at(self.trajectory.t, hz=self.policy_hz)
            comp.set_linear_velocity(v.tolist())
            comp.set_angular_velocity([0.0, 0.0, 0.0])
            self.release_velocity_injected = True
        except Exception:
            self.release_velocity_injected = False

    def _release_rails(self) -> None:
        if not self.rails_active:
            return
        self._disable_kinematic(self.entity)
        if self._secondary_entity is not None:
            self._disable_kinematic(self._secondary_entity)
        self._apply_release_velocity()
        self._disable_ghost_collision_filter(self.entity)
        if self._secondary_entity is not None:
            self._disable_ghost_collision_filter(self._secondary_entity)
        if self._place_entity is not None:
            self._disable_kinematic(self._place_entity)
            self._disable_ghost_collision_filter(self._place_entity)
        self.rails_active = False
        self.pick_rails_active = False
        self.place_rails_active = False
        self.released = True
        self.rails_phase = RailsPhase.FREE
        self.release_t = float(self.policy_t)
        self.aux.record("rails_release", t=self.release_t)
        self._secondary_rails_active = False
        self._pushers = []

    def maybe_release(self):
        """Release rails when a gripper TCP is close enough to grasp."""
        if not self.rails_active or self.entity is None:
            return
        dist, closing = self._tcp_dist_and_closing()
        if dist is None:
            return

        if self.grasp_mode != "ghost":
            if self.rails_phase == RailsPhase.PURSUIT and (
                dist < self.r_engage or (closing and dist < self.r_engage)
            ):
                self._enter_engage()
            if dist < self.r_engage and closing:
                self.contact_before_release = True
            if dist < self.release_radius or (closing and dist < self.r_engage):
                if self._contact_cfg.enabled() and not self.trajectory_switches:
                    ev = self._probe_contact()
                    if ev is not None:
                        self.contact_events.append(ev)
                        try:
                            self._chain.on_primary_contact(t_ticks=float(self.policy_t))
                        except Exception:
                            pass
                self._release_rails()
            return

        if dist < self.release_radius:
            if self._contact_cfg.enabled() and not self.trajectory_switches:
                ev = self._probe_contact()
                if ev is not None:
                    self.contact_events.append(ev)
                    try:
                        self._chain.on_primary_contact(t_ticks=float(self.policy_t))
                    except Exception:
                        pass
            self._release_rails()
            return
        if closing and dist < max(self.release_radius * 1.5, self.release_radius + 0.04):
            self._release_rails()

    @property
    def drift_m(self) -> float:
        return float(self.trajectory.path_length())

    @property
    def occlusion_active_ticks(self) -> float:
        return float(self._occlusion.active_ticks)

    def episode_dynamics_stats(self) -> dict[str, Any]:
        chain_stats = self._chain.stats()
        return {
            "dynamic_modes": list(self.dynamic.modes),
            "contact_switch_level": self._contact_cfg.level,
            "contact_events": list(self.contact_events),
            "trajectory_switches": list(self.trajectory_switches),
            "occlusion_active_ticks": self.occlusion_active_ticks,
            "occlusion_enabled": bool(self._occlusion.cfg.enabled),
            **chain_stats,
            "secondary_rails": bool(self._secondary_traj is not None),
            "secondary_rails_requested": bool(self.dynamic.secondary_rails),
            "bimanual": bool(self.task_meta.get("bimanual")),
            "stress": self.task_meta.get("stress"),
            "rails_drive": str(self.rails_drive),
            "ghost_sliding": bool(self.ghost_sliding),
            "ghost_collision_filter": bool(self.ghost_collision_filter),
            "rails_collide": bool(self.rails_collide),
            "rails_variant": str(self.rails_variant),
            "pick_rails": bool(self.pick_rails_active),
            "place_rails": bool(self.place_rails_active),
            "n_pushers": int(len(self._pushers)),
            "scene": self.scene.to_dict(),
            "aux_events": list(self.aux.events),
            "rails_released": bool(self.released),
            **self.grasp_hold.snapshot(),
            "min_tcp_dist": (
                None if self.min_tcp_dist == float("inf") else float(self.min_tcp_dist)
            ),
            "max_held_streak": int(self.max_held_streak),
            "ever_approached": bool(self.ever_approached or self.released),
            "ever_closing_near": bool(self.ever_closing_near),
            "grasped_held": bool(self.grasped_held()),
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

    def _install_take_action_hook(self):
        env = self.env
        if getattr(env, "_rtl_tw_hooked", False):
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
        env._rtl_tw_hooked = True
        env._rtl_tw_driver = self
        self._orig_take_action = orig


def wrap_env_with_driver(task_env, driver: RealtimeRoboTwinDriver):
    """Attach driver; call `driver.on_episode_start()` after setup_demo."""
    task_env._rtl_tw_driver = driver
    return task_env
