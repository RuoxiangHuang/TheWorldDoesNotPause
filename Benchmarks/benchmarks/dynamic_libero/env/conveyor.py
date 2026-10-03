"""Visible conveyor belt for the ``conveyor`` motion motive.

Injects a **fixed** (0-DoF) belt body into the MuJoCo XML via robosuite's
``set_xml_processor``, so ``nq``/``nv`` stay compatible with ``.pruned_init``.

Narrative: the belt stays planted on the table; only surface stripes scroll.
The rails target is lifted onto the deck and slides along the belt axis.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

CONVEYOR_BODY = "rtl_conveyor"
_N_STRIPES = 8
# Deck half-height (m). Keep thin so the lift stays small.
DECK_HALF_H = 0.008
# Extra clearance above deck top after seating (metres).
SIT_CLEARANCE = 0.015
# Minimum belt width so typical LIBERO cans/boxes do not clip side rails.
MIN_BELT_WIDTH = 0.10


def _yaw_quat_wxyz(forward_xy: np.ndarray) -> np.ndarray:
    x, y = float(forward_xy[0]), float(forward_xy[1])
    n = (x * x + y * y) ** 0.5
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    yaw = float(np.arctan2(y, x))
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


@dataclass
class ConveyorConfig:
    """Mutable layout used by the XML processor and runtime updater."""

    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, DECK_HALF_H]))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    half_length: float = 0.35  # along motion axis
    half_width: float = 0.055
    half_height: float = DECK_HALF_H
    stripe_phase: float = 0.0  # metres along belt for scroll animation
    installed: bool = False
    # World-frame unit direction of belt travel (+local X after quat).
    forward_xy: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))


def inject_conveyor_xml(xml: str, cfg: ConveyorConfig) -> str:
    """Insert / replace the fixed conveyor body in a MuJoCo XML string."""
    root = ET.fromstring(xml)
    wb = root.find("worldbody")
    if wb is None:
        raise RuntimeError("MuJoCo XML missing worldbody")
    for body in list(wb.findall("body")):
        if body.get("name") == CONVEYOR_BODY:
            wb.remove(body)

    pos = np.asarray(cfg.pos, dtype=np.float64).reshape(3)
    quat = np.asarray(cfg.quat, dtype=np.float64).reshape(4)
    body = ET.SubElement(
        wb,
        "body",
        name=CONVEYOR_BODY,
        pos=f"{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}",
        quat=f"{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}",
    )
    L, W, H = float(cfg.half_length), float(cfg.half_width), float(cfg.half_height)
    # Deck (visual only — contype=0 so we do not fight rails pinning).
    ET.SubElement(
        body,
        "geom",
        name=f"{CONVEYOR_BODY}_deck",
        type="box",
        size=f"{L:.5f} {W:.5f} {H:.5f}",
        rgba="0.14 0.14 0.18 1",
        contype="0",
        conaffinity="0",
        group="1",
    )
    # Side rails — short lips so slight overhang does not read as mesh clipping.
    for side, sy in (("neg", -W), ("pos", W)):
        ET.SubElement(
            body,
            "geom",
            name=f"{CONVEYOR_BODY}_rail_{side}",
            type="box",
            size=f"{L:.5f} {0.004:.5f} {H * 1.2:.5f}",
            pos=f"0 {sy:.5f} {H * 0.6:.5f}",
            rgba="0.32 0.32 0.36 1",
            contype="0",
            conaffinity="0",
            group="1",
        )
    # End rollers
    for end, sx in (("a", -L), ("b", L)):
        ET.SubElement(
            body,
            "geom",
            name=f"{CONVEYOR_BODY}_roller_{end}",
            type="cylinder",
            size=f"{H * 1.5:.5f} {W * 0.92:.5f}",
            pos=f"{sx:.5f} 0 0",
            quat="0.7071 0.7071 0 0",
            rgba="0.42 0.42 0.48 1",
            contype="0",
            conaffinity="0",
            group="1",
        )
    # Motion stripes (local +X = belt forward)
    stripe_half = min(0.012, L * 0.05)
    for i in range(_N_STRIPES):
        ET.SubElement(
            body,
            "geom",
            name=f"{CONVEYOR_BODY}_stripe_{i}",
            type="box",
            size=f"{stripe_half:.5f} {W * 0.85:.5f} {H * 1.15:.5f}",
            pos="0 0 0",
            rgba="0.90 0.58 0.10 1",
            contype="0",
            conaffinity="0",
            group="1",
        )
    return ET.tostring(root, encoding="utf8").decode("utf8")


class ConveyorBelt:
    """Install a stationary conveyor; scroll stripes; seat the rails object on it."""

    def __init__(self, cfg: Optional[ConveyorConfig] = None):
        self.cfg = cfg or ConveyorConfig()
        self._env = None
        self.object_lift: float = 0.0  # applied to rails target free-joint z

    @property
    def deck_top_z(self) -> float:
        return float(self.cfg.pos[2]) + float(self.cfg.half_height)

    def install(self, env) -> None:
        """Attach XML processor and hard-reset so the belt enters the model."""
        base = env.env
        cfg = self.cfg

        def _processor(xml: str) -> str:
            return inject_conveyor_xml(xml, cfg)

        base.set_xml_processor(_processor)
        was_det = bool(getattr(base, "deterministic_reset", False))
        base.deterministic_reset = False
        if hasattr(env, "reset"):
            env.reset()
        else:
            base.reset()
        base.deterministic_reset = was_det
        self._env = env
        self.cfg.installed = True
        if CONVEYOR_BODY not in base.sim.model.body_names:
            raise RuntimeError("Conveyor body missing after XML install")

    def layout(
        self,
        *,
        object_xyz: Sequence[float],
        axis: str = "x",
        direction: float = 1.0,
        travel: float = 0.35,
        width: float = 0.08,
        bottom_offset_z: float = -0.03,
    ) -> float:
        """Plant a **stationary** belt along the travel corridor.

        Object starts near the trailing end and moves toward the leading end.
        Returns the free-joint z lift so the object's bottom sits on the deck.
        """
        if self._env is None:
            raise RuntimeError("Call install() before layout()")
        axis = axis.strip().lower()
        if axis not in ("x", "y"):
            axis = "x"
        fwd = np.array([1.0, 0.0] if axis == "x" else [0.0, 1.0], dtype=np.float64)
        if float(direction) < 0:
            fwd = -fwd
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
        self.cfg.forward_xy = fwd.copy()

        obj = np.asarray(object_xyz, dtype=np.float64).reshape(3)
        travel = max(float(travel), 0.20)
        length = travel + 0.28
        self.cfg.half_length = 0.5 * length
        self.cfg.half_width = 0.5 * float(width)
        self.cfg.half_height = DECK_HALF_H
        self.cfg.quat = _yaw_quat_wxyz(fwd)

        # Start near trailing end; belt extends forward along travel.
        trailing = 0.10
        center_xy = obj[:2] + fwd * (0.5 * length - trailing)
        table_z = 0.0
        self.cfg.pos = np.array(
            [float(center_xy[0]), float(center_xy[1]), table_z + DECK_HALF_H],
            dtype=np.float64,
        )
        self.cfg.stripe_phase = 0.0
        self._apply_pose()
        self._apply_stripes()

        # bottom_world ≈ joint_z + bottom_offset_z (upright objects).
        bottom_z = float(obj[2]) + float(bottom_offset_z)
        self.object_lift = max(0.0, self.deck_top_z + SIT_CLEARANCE - bottom_z)
        return float(self.object_lift)

    def seat_object(self, driver: Any) -> None:
        """Raise rails target onto the deck immediately (before motion starts)."""
        if driver.jname is None or driver.trajectory.base_pos is None:
            return
        lift = float(self.object_lift)
        if lift < 1e-6:
            return
        driver.height_offset = lift
        if not getattr(driver, "_height_baked", False):
            bp = np.asarray(driver.trajectory.base_pos, dtype=np.float64).reshape(3).copy()
            bp[2] += lift
            driver.trajectory.base_pos = bp
            driver._height_baked = True
        q = driver.base.sim.data.get_joint_qpos(driver.jname).copy()
        q[2] = float(driver.trajectory.base_pos[2])
        driver.base.sim.data.set_joint_qpos(driver.jname, q)
        driver.base.sim.data.set_joint_qvel(driver.jname, np.zeros(6))
        driver.base.sim.forward()

    def sync_scroll(self, path_length_m: float) -> None:
        """Scroll stripes with rails path length (belt surface moves; body stays)."""
        self.cfg.stripe_phase = float(path_length_m)
        self._apply_stripes()

    def _body_id(self) -> int:
        return int(self._env.env.sim.model.body_name2id(CONVEYOR_BODY))

    def _apply_pose(self) -> None:
        base = self._env.env
        bid = self._body_id()
        base.sim.model.body_pos[bid][:] = np.asarray(self.cfg.pos, dtype=np.float64)
        base.sim.model.body_quat[bid][:] = np.asarray(self.cfg.quat, dtype=np.float64)
        base.sim.forward()

    def _apply_stripes(self) -> None:
        base = self._env.env
        model = base.sim.model
        L = float(self.cfg.half_length)
        spacing = (2.0 * L) / _N_STRIPES
        phase = float(self.cfg.stripe_phase) % max(spacing, 1e-6)
        for i in range(_N_STRIPES):
            name = f"{CONVEYOR_BODY}_stripe_{i}"
            try:
                gid = int(model.geom_name2id(name))
            except Exception:
                continue
            x = -L + spacing * 0.5 + i * spacing + phase
            while x > L:
                x -= 2.0 * L
            while x < -L:
                x += 2.0 * L
            # Slightly above deck to avoid z-fight with the deck geom.
            model.geom_pos[gid][:] = [x, 0.0, float(self.cfg.half_height) * 0.15]
        base.sim.forward()


def layout_conveyor_for_driver(
    belt: ConveyorBelt,
    driver: Any,
    trajectory: Any,
    *,
    horizon_steps: float = 160.0,
) -> float:
    """Place a stationary belt covering the planned travel; seat the object.

    Returns the applied object lift (metres).
    """
    if driver.jname is None:
        return 0.0
    q = driver.base.sim.data.get_joint_qpos(driver.jname).copy()
    axis = str(getattr(trajectory, "axis_name", None) or "x")
    direction = float(getattr(trajectory, "direction", 1.0) or 1.0)
    speed = abs(float(getattr(trajectory, "speed", 0.003)))
    travel = max(0.25, speed * float(horizon_steps))

    bottom_z = -0.03
    width = MIN_BELT_WIDTH
    if driver.target is not None:
        obj = driver.base.objects_dict.get(driver.target)
        if obj is not None:
            bo = getattr(obj, "bottom_offset", None)
            if bo is not None:
                bottom_z = float(np.asarray(bo).reshape(-1)[-1])
            hr = getattr(obj, "horizontal_radius", None)
            if hr is not None:
                width = float(max(MIN_BELT_WIDTH, 2.8 * float(hr)))

    # Prefer a travel direction that keeps the corridor clear of other free objects.
    axis, direction = _pick_clear_corridor(
        driver, q[:2], axis=axis, direction=direction, travel=travel, width=width
    )
    # Keep trajectory consistent with the chosen corridor.
    if hasattr(trajectory, "axis_name"):
        trajectory.axis_name = axis
    if hasattr(trajectory, "direction"):
        trajectory.direction = float(direction)
    if hasattr(trajectory, "_axis"):
        from benchmarks.dynamic_libero.trajectories.linear import _AXES

        if axis in _AXES:
            trajectory._axis = _AXES[axis] * float(np.sign(direction) or 1.0)

    lift = belt.layout(
        object_xyz=q[:3],
        axis=axis,
        direction=direction,
        travel=travel,
        width=width,
        bottom_offset_z=bottom_z,
    )
    belt.seat_object(driver)
    _seat_neighbors_on_belt(belt, driver)
    return float(lift)


def _pick_clear_corridor(
    driver: Any,
    start_xy: np.ndarray,
    *,
    axis: str,
    direction: float,
    travel: float,
    width: float,
) -> tuple[str, float]:
    """Score ±x/±y corridors; keep caller axis if already clear enough."""
    candidates = [("x", 1.0), ("x", -1.0), ("y", 1.0), ("y", -1.0)]
    best = (axis, float(np.sign(direction) or 1.0))
    best_score = 10**9
    half_w = 0.5 * float(width)
    for ax, d in candidates:
        fwd = np.array([1.0, 0.0] if ax == "x" else [0.0, 1.0]) * d
        score = 0
        for name, obj in driver.base.objects_dict.items():
            if name == driver.target or not getattr(obj, "joints", None):
                continue
            p = driver.base.sim.data.get_joint_qpos(obj.joints[-1])[:2]
            rel = p - start_xy
            along = float(np.dot(rel, fwd))
            lat = float(np.linalg.norm(rel - along * fwd))
            if 0.0 < along < float(travel) + 0.15 and lat < half_w + 0.03:
                score += 1
        # Prefer the caller's axis/dir on ties.
        tie = 0 if (ax == axis and d == float(np.sign(direction) or 1.0)) else 0.1
        if score + tie < best_score:
            best_score = score + tie
            best = (ax, d)
    return best


def _seat_neighbors_on_belt(belt: ConveyorBelt, driver: Any) -> None:
    """Lift other free objects that sit on the belt footprint so they do not clip."""
    center = np.asarray(belt.cfg.pos[:2], dtype=np.float64)
    quat = np.asarray(belt.cfg.quat, dtype=np.float64)
    # Local +X from yaw quat (wxyz).
    w, x, y, z = quat
    # yaw-only: forward = (cos yaw, sin yaw)
    yaw = 2.0 * float(np.arctan2(z, w))
    fwd = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    left = np.array([-fwd[1], fwd[0]], dtype=np.float64)
    half_l = float(belt.cfg.half_length)
    half_w = float(belt.cfg.half_width)
    deck_top = belt.deck_top_z

    for name, obj in driver.base.objects_dict.items():
        if name == driver.target or not getattr(obj, "joints", None):
            continue
        jname = obj.joints[-1]
        q = driver.base.sim.data.get_joint_qpos(jname).copy()
        rel = q[:2] - center
        along = float(np.dot(rel, fwd))
        lat = float(np.dot(rel, left))
        if abs(along) > half_l or abs(lat) > half_w:
            continue
        bo = getattr(obj, "bottom_offset", None)
        bottom_off = float(np.asarray(bo).reshape(-1)[-1]) if bo is not None else -0.03
        bottom_z = float(q[2]) + bottom_off
        need = deck_top + SIT_CLEARANCE - bottom_z
        if need <= 1e-4:
            continue
        q[2] = float(q[2]) + need
        driver.base.sim.data.set_joint_qpos(jname, q)
        driver.base.sim.data.set_joint_qvel(jname, np.zeros(6))
    driver.base.sim.forward()
