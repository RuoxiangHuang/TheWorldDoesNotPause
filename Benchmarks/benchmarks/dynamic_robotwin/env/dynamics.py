"""RoboTwin-specific dynamic interventions (contact / occlusion / contact-chain).

Shared trajectory families and latency coupling stay in common /
dynamic_libero; this module only implements Dynamic-RoboTwin stresses from
the paper (coordination windows under occlusion and contact).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import numpy as np

from .task_registry import CONTACT_SWITCH_LEVELS

# Approximate head-camera look-from used when camera pose is unavailable.
_DEFAULT_CAMERA_XYZ = np.array([0.0, -0.45, 1.35], dtype=np.float64)


@dataclass
class ContactTriggerConfig:
    """Contact-triggered trajectory switch (paper: target motion changes on contact)."""

    level: str = "medium"  # off | mild | medium | strong
    contact_radius: float = 0.18
    max_switches: int = 1
    # Mode applied per switch; level selects intensity multipliers.
    switch_mode: str = "auto"  # auto | reverse | accelerate | redirect

    def enabled(self) -> bool:
        return str(self.level).lower() not in ("off", "none", "0", "")

    @classmethod
    def from_level(
        cls,
        level: str = "medium",
        *,
        contact_radius: float = 0.18,
        max_switches: int = 1,
        switch_mode: str = "auto",
    ) -> "ContactTriggerConfig":
        lv = str(level).lower()
        if lv not in CONTACT_SWITCH_LEVELS:
            raise ValueError(
                f"contact_switch level must be one of {CONTACT_SWITCH_LEVELS}, got {level!r}"
            )
        return cls(
            level=lv,
            contact_radius=float(contact_radius),
            max_switches=int(max_switches),
            switch_mode=str(switch_mode),
        )


@dataclass
class OcclusionConfig:
    """Transient visual occlusion (does not affect check_success physics)."""

    enabled: bool = True
    duty: float = 0.35  # fraction of each period occluder is in view
    period_ticks: float = 40.0
    half_size: Sequence[float] = (0.06, 0.02, 0.08)
    # Offset along camera→target at which the box sits when active.
    depth_frac: float = 0.45

    @classmethod
    def from_args(
        cls,
        enabled: bool = True,
        duty: float = 0.35,
        period_ticks: float = 40.0,
        half_size: Sequence[float] | None = None,
    ) -> "OcclusionConfig":
        return cls(
            enabled=bool(enabled),
            duty=float(np.clip(duty, 0.0, 1.0)),
            period_ticks=max(1.0, float(period_ticks)),
            half_size=tuple(half_size) if half_size is not None else (0.06, 0.02, 0.08),
        )


@dataclass
class ContactChainConfig:
    """Multi-object contact-chain: primary contact propagates through aux props.

    Aux boxes are placed in a line. The first hop fires on primary contact; later
    hops fire when neighbors enter ``propagate_radius`` or after ``hop_delay_ticks``
    (timer fallback keeps the chain observable under soft PhysX contacts).
    """

    enabled: bool = True
    n_aux: int = 3
    impulse: float = 0.35  # m/s linear kick magnitude
    spacing: float = 0.08  # metres between consecutive aux props
    propagate_radius: float = 0.10
    hop_delay_ticks: float = 2.0
    # Unit direction of the chain axis in world XY (z ignored for placement).
    offset_dir: Sequence[float] = (1.0, 0.0, 0.0)

    @classmethod
    def from_args(
        cls,
        enabled: bool = True,
        n_aux: int = 3,
        impulse: float = 0.35,
        spacing: float = 0.08,
        propagate_radius: float = 0.10,
        hop_delay_ticks: float = 2.0,
    ) -> "ContactChainConfig":
        return cls(
            enabled=bool(enabled),
            n_aux=max(0, int(n_aux)),
            impulse=float(impulse),
            spacing=float(spacing),
            propagate_radius=float(propagate_radius),
            hop_delay_ticks=float(hop_delay_ticks),
        )


@dataclass
class DynamicConfig:
    """Aggregate Dynamic-RoboTwin mode flags for one episode."""

    modes: List[str] = field(default_factory=list)
    contact_trigger: ContactTriggerConfig = field(default_factory=ContactTriggerConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    contact_chain: ContactChainConfig = field(default_factory=ContactChainConfig)
    # Second kinematic rail on receive-side placement target (handoff).
    secondary_rails: bool = False

    def wants(self, mode: str) -> bool:
        return mode in self.modes


def resolve_secondary_rails(
    task_name: str,
    secondary_rails: bool | str | None = None,
    *,
    modes: Optional[Sequence[str]] = None,
) -> bool:
    """Resolve receive-side rails flag.

    - ``True`` / ``\"on\"`` → always on
    - ``False`` / ``\"off\"`` → always off
    - ``None`` / ``\"auto\"`` → on when task has ``stress=handoff`` and
      ``secondary_attr``, and handoff mode is active (or modes unset).
    """
    if isinstance(secondary_rails, str):
        key = secondary_rails.strip().lower()
        if key in ("on", "true", "1", "yes"):
            return True
        if key in ("off", "false", "0", "no"):
            return False
        if key not in ("auto", ""):
            raise ValueError(
                f"secondary_rails must be auto/on/off, got {secondary_rails!r}"
            )
        secondary_rails = None
    if secondary_rails is True:
        return True
    if secondary_rails is False:
        return False
    from .task_registry import get_task_meta

    meta = get_task_meta(task_name)
    if meta.get("stress") != "handoff" or not meta.get("secondary_attr"):
        return False
    if modes is not None and "handoff" not in modes and "all" not in list(modes):
        # Handoff stress tasks still auto-enable when modes omitted handoff
        # but stress is handoff — treat recommended.
        pass
    return True


def build_dynamic_config(
    modes: Optional[Sequence[str]] = None,
    *,
    contact_switch: str = "medium",
    occlusion: bool | None = None,
    occlusion_duty: float = 0.35,
    occlusion_period: float = 40.0,
    contact_chain: bool | None = None,
    contact_chain_n_aux: int = 3,
    contact_chain_impulse: float = 0.35,
    contact_radius: float = 0.18,
    secondary_rails: bool = False,
) -> DynamicConfig:
    """Construct ``DynamicConfig`` from CLI / eval flags.

    ``modes`` selects which stresses run. Dedicated booleans override membership
    for ``occlusion`` / ``contact_chain``. ``contact_switch=off`` disables
    contact_trigger even if listed in modes.
    """
    from .task_registry import parse_dynamic_modes

    mode_list = parse_dynamic_modes(modes)
    if occlusion is True and "occlusion" not in mode_list:
        mode_list.append("occlusion")
    if occlusion is False:
        mode_list = [m for m in mode_list if m != "occlusion"]
    if contact_chain is True and "contact_chain" not in mode_list:
        mode_list.append("contact_chain")
    if contact_chain is False:
        mode_list = [m for m in mode_list if m != "contact_chain"]
    if str(contact_switch).lower() in ("off", "none"):
        mode_list = [m for m in mode_list if m != "contact_trigger"]

    ct = ContactTriggerConfig.from_level(
        "off" if "contact_trigger" not in mode_list else contact_switch,
        contact_radius=contact_radius,
    )
    occ = OcclusionConfig.from_args(
        enabled="occlusion" in mode_list,
        duty=occlusion_duty,
        period_ticks=occlusion_period,
    )
    chain = ContactChainConfig.from_args(
        enabled="contact_chain" in mode_list,
        n_aux=contact_chain_n_aux,
        impulse=contact_chain_impulse,
    )
    return DynamicConfig(
        modes=list(mode_list),
        contact_trigger=ct,
        occlusion=occ,
        contact_chain=chain,
        secondary_rails=bool(secondary_rails),
    )


def resolve_contact_intensity(level: str) -> dict[str, Any]:
    """Map contact_switch level → concrete switch parameters."""
    lv = str(level).lower()
    table = {
        "off": {"speed_scale": 1.0, "reverse": False, "redirect": False, "mode": "none"},
        "mild": {"speed_scale": 1.0, "reverse": True, "redirect": False, "mode": "reverse"},
        "medium": {"speed_scale": 1.5, "reverse": True, "redirect": False, "mode": "accelerate"},
        "strong": {"speed_scale": 2.0, "reverse": True, "redirect": True, "mode": "redirect"},
    }
    if lv not in table:
        raise ValueError(f"unknown contact_switch level {level!r}")
    return dict(table[lv])


def apply_trajectory_switch(trajectory, level: str, switch_mode: str = "auto") -> str:
    """Mutate ``trajectory`` in-place; return applied mode name.

    Compatible with shared Linear/Sine/Circle trajectories from dynamic_libero.
    Rebases ``base_pos`` to the current pose so the path is continuous.
    """
    intensity = resolve_contact_intensity(level)
    if intensity["mode"] == "none":
        return "none"

    mode = switch_mode if switch_mode != "auto" else intensity["mode"]
    # Continuity: freeze current XYZ as new base and zero path time.
    try:
        cur = np.asarray(trajectory.xyz_at(trajectory.t), dtype=np.float64).reshape(3)
        trajectory.base_pos = cur.copy()
        trajectory.t = 0.0
    except Exception:
        pass

    if mode in ("reverse", "accelerate", "redirect") or intensity["reverse"]:
        if hasattr(trajectory, "direction"):
            trajectory.direction = -float(np.sign(trajectory.direction) or 1.0)
            rebuild = getattr(trajectory, "_rebuild_axis", None)
            if callable(rebuild):
                rebuild()
            elif hasattr(trajectory, "_axis") and hasattr(trajectory, "axis_name"):
                axes = {
                    "x": np.array([1.0, 0.0, 0.0]),
                    "y": np.array([0.0, 1.0, 0.0]),
                    "z": np.array([0.0, 0.0, 1.0]),
                }
                an = getattr(trajectory, "axis_name", "y")
                if an in axes:
                    trajectory._axis = axes[an] * trajectory.direction
        elif hasattr(trajectory, "omega"):
            trajectory.omega = -float(trajectory.omega)

    if mode in ("accelerate", "redirect") or intensity["speed_scale"] != 1.0:
        trajectory.speed = float(trajectory.speed) * float(intensity["speed_scale"])

    if mode == "redirect" or intensity["redirect"]:
        hd = getattr(trajectory, "heading_deg", None)
        if hd is not None:
            trajectory.heading_deg = float(hd) + 90.0
            rebuild = getattr(trajectory, "_rebuild_axis", None)
            if callable(rebuild):
                rebuild()
        elif hasattr(trajectory, "axis_name"):
            swap = {"x": "y", "y": "x", "z": "x"}
            new_axis = swap.get(trajectory.axis_name, "y")
            trajectory.axis_name = new_axis
            axes = {
                "x": np.array([1.0, 0.0, 0.0]),
                "y": np.array([0.0, 1.0, 0.0]),
                "z": np.array([0.0, 0.0, 1.0]),
            }
            direction = float(getattr(trajectory, "direction", 1.0) or 1.0)
            trajectory._axis = axes[new_axis] * direction

    return mode


def occlusion_is_active(t_ticks: float, period_ticks: float, duty: float) -> bool:
    """Duty-cycle helper (pure; unit-testable without SAPIEN)."""
    if duty <= 0.0 or period_ticks <= 0.0:
        return False
    if duty >= 1.0:
        return True
    phase = float(t_ticks) % float(period_ticks)
    return phase < float(duty) * float(period_ticks)


class OcclusionManager:
    """Kinematic box that intermittently blocks the head-camera line of sight."""

    def __init__(self, cfg: OcclusionConfig):
        self.cfg = cfg
        self.entity = None
        self.active_ticks = 0.0
        self._park_pose = np.array([0.0, 0.0, -5.0], dtype=np.float64)
        self._camera_xyz = _DEFAULT_CAMERA_XYZ.copy()
        self._built = False

    def setup(self, task_env, target_xyz: np.ndarray) -> None:
        if not self.cfg.enabled:
            return
        self._resolve_camera(task_env)
        self._build_box(task_env)
        self._place(target_xyz, active=False)
        self.active_ticks = 0.0

    def _resolve_camera(self, task_env) -> None:
        self._camera_xyz = _DEFAULT_CAMERA_XYZ.copy()
        try:
            cams = getattr(task_env, "cameras", None)
            if cams is None:
                return
            # RoboTwin Camera wrapper may expose head camera entity / pose.
            for name in ("head_camera", "head", "camera"):
                cam = getattr(cams, name, None)
                if cam is None:
                    continue
                if hasattr(cam, "get_pose"):
                    self._camera_xyz = np.asarray(cam.get_pose().p, dtype=np.float64).reshape(3)
                    return
                if hasattr(cam, "get_intrinsic_and_extrinsic"):
                    pass
            # Fallback: first camera config dict with position.
            cfg = getattr(cams, "camera_config", None) or getattr(cams, "config", None)
            if isinstance(cfg, dict):
                head = cfg.get("head_camera") or cfg.get("head")
                if isinstance(head, dict) and "position" in head:
                    self._camera_xyz = np.asarray(head["position"], dtype=np.float64).reshape(3)
        except Exception:
            pass

    def _build_box(self, task_env) -> None:
        if self._built:
            return
        import sapien

        scene = task_env.scene
        hs = [float(x) for x in self.cfg.half_size]
        builder = scene.create_actor_builder()
        # Visual only — no collision so grasp physics / success are unaffected.
        try:
            mat = sapien.render.RenderMaterial()
            mat.set_base_color([0.15, 0.15, 0.15, 1.0])
            builder.add_box_visual(half_size=hs, material=mat)
        except Exception:
            builder.add_box_visual(half_size=hs)
        try:
            self.entity = builder.build_kinematic(name="rtl_tw_occlusion")
        except Exception:
            # Older SAPIEN: build() then lock.
            self.entity = builder.build(name="rtl_tw_occlusion")
        self._built = True

    def _place(self, target_xyz: np.ndarray, active: bool) -> None:
        if self.entity is None:
            return
        import sapien

        if not active:
            self.entity.set_pose(sapien.Pose(self._park_pose.tolist()))
            return
        cam = self._camera_xyz
        tgt = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        mid = cam + float(self.cfg.depth_frac) * (tgt - cam)
        # Lift slightly so the box sits in the optical path above the table.
        mid = mid.copy()
        mid[2] = max(mid[2], float(tgt[2]) + 0.05)
        self.entity.set_pose(sapien.Pose(mid.tolist()))

    def tick(self, t_ticks: float, target_xyz: np.ndarray, dt: float = 1.0) -> bool:
        if not self.cfg.enabled or self.entity is None:
            return False
        active = occlusion_is_active(t_ticks, self.cfg.period_ticks, self.cfg.duty)
        self._place(target_xyz, active=active)
        if active:
            self.active_ticks += float(dt)
        return active


class ContactChainManager:
    """Multi-object contact-chain with hop propagation along aux props."""

    def __init__(self, cfg: ContactChainConfig):
        self.cfg = cfg
        self.entities: List[Any] = []
        self.positions: List[np.ndarray] = []  # CPU fallback when entities lack poses
        self.primary_fired = False
        self.fired: set[int] = set()
        self.fire_times: dict[int, float] = {}
        self.hop_events: List[dict[str, Any]] = []
        self.n_impulses = 0

    def setup(self, task_env, target_xyz: np.ndarray) -> None:
        self.entities = []
        self.positions = []
        self.primary_fired = False
        self.fired = set()
        self.fire_times = {}
        self.hop_events = []
        self.n_impulses = 0
        if not self.cfg.enabled or self.cfg.n_aux <= 0:
            return

        base = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        direction = np.asarray(self.cfg.offset_dir, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction[:2]))
        if norm < 1e-9:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            direction = direction / max(norm, 1e-9)
            direction[2] = 0.0

        try:
            import sapien

            scene = task_env.scene
        except Exception:
            scene = None
            sapien = None  # type: ignore

        for i in range(self.cfg.n_aux):
            pos = base + direction * float(self.cfg.spacing) * float(i + 1)
            pos[2] = max(float(pos[2]), float(base[2]))
            self.positions.append(pos.copy())
            if scene is None or sapien is None:
                self.entities.append(None)
                continue
            builder = scene.create_actor_builder()
            hs = [0.025, 0.025, 0.025]
            # Slight color ramp so hops are visually distinct.
            t = (i + 1) / max(self.cfg.n_aux, 1)
            color = [0.9, 0.55 - 0.25 * t, 0.15 + 0.4 * t, 1.0]
            try:
                mat = sapien.render.RenderMaterial()
                mat.set_base_color(color)
                builder.add_box_visual(half_size=hs, material=mat)
            except Exception:
                builder.add_box_visual(half_size=hs)
            builder.add_box_collision(half_size=hs)
            try:
                ent = builder.build(name=f"rtl_tw_chain_{i}")
            except Exception:
                try:
                    ent = builder.build_kinematic(name=f"rtl_tw_chain_{i}")
                except Exception:
                    ent = None
            if ent is not None:
                ent.set_pose(sapien.Pose(pos.tolist()))
            self.entities.append(ent)

    def _entity_xyz(self, idx: int) -> Optional[np.ndarray]:
        if idx < 0 or idx >= len(self.positions):
            return None
        ent = self.entities[idx] if idx < len(self.entities) else None
        if ent is not None:
            try:
                p = np.asarray(ent.get_pose().p, dtype=np.float64).reshape(3)
                self.positions[idx] = p
                return p
            except Exception:
                pass
        return np.asarray(self.positions[idx], dtype=np.float64).reshape(3)

    def _kick(self, idx: int, t_ticks: float, *, source: str) -> None:
        if idx in self.fired or idx < 0 or idx >= len(self.positions):
            return
        direction = np.asarray(self.cfg.offset_dir, dtype=np.float64).reshape(3)
        xy = direction[:2]
        n = float(np.linalg.norm(xy))
        if n < 1e-9:
            kick_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            kick_dir = np.array([xy[0] / n, xy[1] / n, 0.15], dtype=np.float64)
        v = kick_dir * float(self.cfg.impulse)
        # CPU bookkeeping: advance cached pose so proximity can propagate without PhysX.
        self.positions[idx] = self.positions[idx] + v * 0.05
        ent = self.entities[idx] if idx < len(self.entities) else None
        if ent is not None:
            try:
                import sapien

                comp = ent.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                if comp is not None:
                    comp.set_linear_velocity(v.tolist())
            except Exception:
                pass
        self.fired.add(idx)
        self.fire_times[idx] = float(t_ticks)
        self.n_impulses += 1
        self.hop_events.append(
            {
                "hop": int(idx),
                "t": float(t_ticks),
                "source": str(source),
                "impulse": float(self.cfg.impulse),
            }
        )

    def on_primary_contact(self, t_ticks: float = 0.0) -> None:
        if self.primary_fired or not self.cfg.enabled:
            return
        self.primary_fired = True
        if not self.positions:
            return
        self._kick(0, float(t_ticks), source="primary")

    def tick(self, t_ticks: float, dt: float = 1.0) -> int:
        """Propagate hops; return number of new impulses this call."""
        del dt  # reserved for future continuous integration
        if not self.cfg.enabled or not self.primary_fired:
            return 0
        before = self.n_impulses
        n = len(self.positions)
        for i in range(n - 1):
            if i not in self.fired or (i + 1) in self.fired:
                continue
            a = self._entity_xyz(i)
            b = self._entity_xyz(i + 1)
            reason = None
            if a is not None and b is not None:
                if float(np.linalg.norm(a - b)) <= float(self.cfg.propagate_radius):
                    reason = "proximity"
            t0 = self.fire_times.get(i)
            if reason is None and t0 is not None:
                if float(t_ticks) - float(t0) >= float(self.cfg.hop_delay_ticks):
                    reason = "delay"
            if reason is not None:
                self._kick(i + 1, float(t_ticks), source=reason)
        return int(self.n_impulses - before)

    def stats(self) -> dict[str, Any]:
        return {
            "contact_chain_impulses": int(self.n_impulses),
            "contact_chain_fired": bool(self.primary_fired),
            "contact_chain_hops": int(len(self.fired)),
            "contact_chain_events": list(self.hop_events),
            "contact_chain_n_aux": int(self.cfg.n_aux),
        }
