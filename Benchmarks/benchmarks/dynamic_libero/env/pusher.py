"""Visible toy-car actors for Dynamic-LIBERO (``pusher`` / ``carrier`` motives).

The pick (and optional place) meshes stay original LIBERO assets. 0-DoF car
bodies are injected so ``nq``/``nv`` stay ``.pruned_init`` compatible.

- ``bumper``: car rides rails behind the object and seats it on the bumper.
- ``carrier``: larger pickup with a rear flatbed; the object sits on the bed.
A second bumper car can push ``basket_1`` so pick and place targets both move.
Visual geoms stay ``contype=0``; distractors on the travel corridor are nudged
aside at episode start instead of giving the car a collision hull.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

PUSHER_BODY = "rtl_pusher"
PLACE_PUSHER_BODY = "rtl_place_pusher"

HALF_LENGTH = 0.055
HALF_WIDTH = 0.032
HALF_HEIGHT = 0.016
WHEEL_RADIUS = 0.012
# Bumper geom is centered at half_length + bumper_half, so its nose is
# 2*bumper_half past the chassis. A zero gap plus a standoff that ignored
# that nose put 16 mm of bumper inside the object. Collision is contype=0,
# so MuJoCo never separates them.
CONTACT_GAP = 0.003
BUMPER_HALF = 0.008
DEFAULT_OBJECT_RADIUS = 0.025

CARRIER_HALF_LENGTH = 0.095
CARRIER_HALF_WIDTH = 0.050
CARRIER_HALF_HEIGHT = 0.018
CARRIER_WHEEL_RADIUS = 0.015
BED_HALF_H = 0.006

BASKET_RADIUS = 0.085
PLATE_RADIUS = 0.080
PLACE_HALF_LENGTH = 0.070
PLACE_HALF_WIDTH = 0.042
# Lateral padding when sliding distractors off a car's travel corridor (m).
PATH_CLEAR_MARGIN = 0.050
PATH_CLEAR_HORIZON = 0.55

PALETTES: dict[str, dict[str, str]] = {
    "red": {
        "chassis": "0.90 0.12 0.10 1",
        "hood": "0.95 0.78 0.10 1",
        "cabin": "0.20 0.48 0.82 1",
        "bumper": "0.98 0.92 0.15 1",
        "bed": "0.35 0.35 0.38 1",
    },
    "blue": {
        "chassis": "0.12 0.38 0.82 1",
        "hood": "0.55 0.78 0.95 1",
        "cabin": "0.10 0.18 0.32 1",
        "bumper": "0.95 0.85 0.20 1",
        "bed": "0.25 0.30 0.40 1",
    },
}


def _yaw_quat_wxyz(forward_xy: np.ndarray) -> np.ndarray:
    x, y = float(forward_xy[0]), float(forward_xy[1])
    n = (x * x + y * y) ** 0.5
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    yaw = float(np.arctan2(y, x))
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def bumper_half_for(half_length: float) -> float:
    """Matches ``_add_bumper_visuals``: at least 8 mm, else 14% of the chassis."""
    return max(float(BUMPER_HALF), float(half_length) * 0.14)


def nose_extent(half_length: float) -> float:
    """Distance from the car origin to the bumper face."""
    return float(half_length) + 2.0 * bumper_half_for(half_length)


def _unit_xy(v: np.ndarray) -> np.ndarray:
    fwd = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(fwd))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return fwd / n


def forward_xy_from_traj(traj: Any, hz: float = 20.0) -> np.ndarray:
    v = np.asarray(traj.velocity_at(traj.t, hz=float(hz)), dtype=np.float64).reshape(3)
    fwd = v[:2].copy()
    if float(np.linalg.norm(fwd)) < 1e-9:
        p0 = np.asarray(traj.xyz_at(traj.t), dtype=np.float64).reshape(3)
        p1 = np.asarray(traj.xyz_at(traj.t + 1.0), dtype=np.float64).reshape(3)
        fwd = (p1 - p0)[:2]
    return _unit_xy(fwd)


def forward_xy_from_driver(driver: Any) -> np.ndarray:
    hz = float(getattr(driver, "control_hz", 20.0) or 20.0)
    return forward_xy_from_traj(driver.trajectory, hz=hz)


def object_horizontal_radius(driver: Any, target: Optional[str] = None) -> float:
    name = target if target is not None else getattr(driver, "target", None)
    if name and "basket" in str(name):
        return float(BASKET_RADIUS)
    if name and "plate" in str(name):
        return float(PLATE_RADIUS)
    xy_r = object_collision_xy_radius(driver, name)
    if xy_r is not None:
        return float(max(0.020, min(0.12, xy_r)))
    obj = driver.base.objects_dict.get(name) if name else None
    hr = getattr(obj, "horizontal_radius", None) if obj is not None else None
    if hr is None:
        return DEFAULT_OBJECT_RADIUS
    return float(max(0.012, min(0.10, float(hr))))


def object_bottom_offset_z(driver: Any, target: Optional[str] = None) -> float:
    name = target if target is not None else getattr(driver, "target", None)
    obj = driver.base.objects_dict.get(name) if name else None
    bo = getattr(obj, "bottom_offset", None) if obj is not None else None
    if bo is None:
        return -0.03
    return float(np.asarray(bo, dtype=np.float64).reshape(-1)[-1])


def _geom_name(model: Any, gid: int) -> str:
    names = getattr(model, "geom_names", None)
    if names is not None:
        return str(names[gid])
    fn = getattr(model, "geom_id2name", None)
    if fn is None:
        return ""
    got = fn(gid)
    return "" if got is None else str(got)


def oriented_box_zmin(pos: np.ndarray, xmat: np.ndarray, size: np.ndarray) -> float:
    """World-frame lowest point of an oriented box geom."""
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    mat = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    size = np.asarray(size, dtype=np.float64).reshape(3)
    extent = (
        abs(float(mat[2, 0])) * float(size[0])
        + abs(float(mat[2, 1])) * float(size[1])
        + abs(float(mat[2, 2])) * float(size[2])
    )
    return float(pos[2] - extent)


def _geom_world_zmin(sim: Any, gid: int) -> float:
    pos = np.asarray(sim.data.geom_xpos[gid], dtype=np.float64)
    gtype = int(sim.model.geom_type[gid])
    size = np.asarray(sim.model.geom_size[gid], dtype=np.float64)
    # 2=sphere, 3=capsule, 5=cylinder, 6=box, 7=mesh
    if gtype == 6:
        mat = np.asarray(sim.data.geom_xmat[gid], dtype=np.float64)
        return oriented_box_zmin(pos, mat, size)
    if gtype == 2:
        return float(pos[2] - float(size[0]))
    if gtype == 5:
        mat = np.asarray(sim.data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        return float(
            pos[2]
            - abs(float(mat[2, 2])) * float(size[1])
            - abs(float(mat[2, 0])) * float(size[0])
            - abs(float(mat[2, 1])) * float(size[0])
        )
    rbound = float(getattr(sim.model, "geom_rbound")[gid])
    return float(pos[2] - rbound)


def object_collision_zmin(driver: Any, target: Optional[str] = None) -> Optional[float]:
    """Lowest world-z of the object's *collision* geoms (not XML bottom_site).

    Scanned LIBERO meshes (akita bowl) have a conservative ``bottom_site`` that
    sits centimetres below the visible bowl. Using that site as the cargo
    datum leaves the mesh floating above the flatbed.
    """
    name = target if target is not None else getattr(driver, "target", None)
    if not name:
        return None
    sim = getattr(getattr(driver, "base", None), "sim", None)
    if sim is None or getattr(sim, "model", None) is None:
        return None
    model = sim.model
    try:
        ngeom = int(model.ngeom)
        sim.forward()
    except Exception:
        return None
    prefix = str(name)
    zmin: Optional[float] = None
    for gid in range(ngeom):
        gname = _geom_name(model, gid)
        if not gname or not (gname == prefix or gname.startswith(prefix + "_")):
            continue
        try:
            group = int(model.geom_group[gid])
        except Exception:
            group = 0
        if group != 0:
            continue
        z = _geom_world_zmin(sim, gid)
        zmin = z if zmin is None else min(zmin, z)
    return zmin


def object_collision_xy_radius(driver: Any, target: Optional[str] = None) -> Optional[float]:
    """Planar radius from joint xy to the farthest collision geom."""
    name = target if target is not None else getattr(driver, "target", None)
    if not name:
        return None
    sim = getattr(getattr(driver, "base", None), "sim", None)
    if sim is None or getattr(sim, "model", None) is None:
        return None
    obj = driver.base.objects_dict.get(name)
    if obj is None or not getattr(obj, "joints", None):
        return None
    try:
        ngeom = int(sim.model.ngeom)
        q = np.asarray(sim.data.get_joint_qpos(obj.joints[-1])[:2], dtype=np.float64)
        sim.forward()
    except Exception:
        return None
    prefix = str(name)
    rmax: Optional[float] = None
    for gid in range(ngeom):
        gname = _geom_name(sim.model, gid)
        if not gname or not (gname == prefix or gname.startswith(prefix + "_")):
            continue
        p = np.asarray(sim.data.geom_xpos[gid][:2], dtype=np.float64)
        gtype = int(sim.model.geom_type[gid])
        size = np.asarray(sim.model.geom_size[gid], dtype=np.float64)
        # Include visual meshes (group != 0). A collision-only radius leaves
        # the visible shell inside the bumper, and contype=0 will not push it out.
        if gtype == 6:
            extra = float(np.hypot(float(size[0]), float(size[1])))
        elif gtype == 7:
            extra = float(sim.model.geom_rbound[gid])
        else:
            extra = float(size[0])
        d = float(np.linalg.norm(p - q)) + extra
        rmax = d if rmax is None else max(rmax, d)
    return rmax


def object_support_z(driver: Any, target: Optional[str] = None) -> float:
    """Surface the object is resting on: collision zmin, else joint+bottom_site."""
    zmin = object_collision_zmin(driver, target)
    if zmin is not None:
        return float(zmin)
    name = target if target is not None else getattr(driver, "target", None)
    jname = None
    obj = driver.base.objects_dict.get(name) if name else None
    if obj is not None and getattr(obj, "joints", None):
        jname = obj.joints[-1]
    elif getattr(driver, "jname", None):
        jname = driver.jname
    if jname is None:
        return 0.0
    qz = float(driver.base.sim.data.get_joint_qpos(jname)[2])
    return qz + object_bottom_offset_z(driver, target)


# Kitchen fixtures the visual-only carrier must not spawn inside.
_FIXTURE_NAME_TOKENS = (
    "cabinet",
    "stove",
    "drawer",
    "microwave",
    "oven",
    "sink",
    "shelf",
)
# Bowl-on-cookie-box / ramekin is a few centimetres; drawer/cabinet are ~20 cm.
_CARRIER_TABLE_SLACK = 0.015
# Ignore 5 mm of AABB kiss so a car parked in front of the cabinet is allowed.
_FIXTURE_OVERLAP_MARGIN = -0.005


def table_support_z(driver: Any, target_name: Optional[str] = None) -> float:
    """Table plane for a carrier: plate/basket, not a stacked/in-drawer cargo."""
    names: list[str] = []
    for n in (
        getattr(driver, "place_target", None),
        getattr(driver, "place_target_name", None),
        "plate_1",
        "basket_1",
    ):
        if n and str(n) not in names:
            names.append(str(n))
    objs = getattr(getattr(driver, "base", None), "objects_dict", None) or {}
    for n in names:
        if n in objs:
            return float(object_support_z(driver, n))
    return float(object_support_z(driver, target_name))


def _body_name(model: Any, bid: int) -> str:
    names = getattr(model, "body_names", None)
    if names is not None:
        return str(names[bid])
    fn = getattr(model, "body_id2name", None)
    if fn is None:
        return ""
    got = fn(bid)
    return "" if got is None else str(got)


def _geom_world_aabb(sim: Any, gid: int) -> tuple[np.ndarray, np.ndarray]:
    """Tight world AABB of one geom (box/sphere/cylinder; else rbound)."""
    pos = np.asarray(sim.data.geom_xpos[gid], dtype=np.float64).reshape(3)
    gtype = int(sim.model.geom_type[gid])
    size = np.asarray(sim.model.geom_size[gid], dtype=np.float64)
    mat = np.asarray(sim.data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    if gtype == 6:  # box
        extent = np.abs(mat) @ size[:3]
        return pos - extent, pos + extent
    if gtype == 2:  # sphere
        r = float(size[0])
        return pos - r, pos + r
    if gtype in (3, 5):  # capsule / cylinder: radius, half-length along local z
        r = float(size[0])
        hl = float(size[1]) if size.size > 1 else r
        axis = mat[:, 2]
        radial = np.sqrt(np.maximum(0.0, 1.0 - axis * axis))
        extent = np.abs(axis) * hl + r * radial
        return pos - extent, pos + extent
    r = float(sim.model.geom_rbound[gid])
    return pos - r, pos + r


def fixture_aabbs(sim: Any) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """World AABBs of cabinet / stove / drawer collision geoms."""
    if sim is None or getattr(sim, "model", None) is None:
        return []
    model = sim.model
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    try:
        nbody = int(model.nbody)
        ngeom = int(model.ngeom)
    except Exception:
        return []
    for bid in range(nbody):
        name = _body_name(model, bid)
        key = name.lower()
        if not any(tok in key for tok in _FIXTURE_NAME_TOKENS):
            continue
        lo = None
        hi = None
        for gid in range(ngeom):
            if int(model.geom_bodyid[gid]) != bid:
                continue
            glo, ghi = _geom_world_aabb(sim, gid)
            lo = glo if lo is None else np.minimum(lo, glo)
            hi = ghi if hi is None else np.maximum(hi, ghi)
        if lo is None or hi is None:
            continue
        out.append((name, lo, hi))
    return out


def car_world_aabb(car: "PusherCar") -> tuple[np.ndarray, np.ndarray]:
    """Oriented car footprint as a world AABB (yaw applied in XY)."""
    p = np.asarray(car.cfg.pos, dtype=np.float64).reshape(3)
    fwd = _unit_xy(car.cfg.forward_xy)
    left = np.array([-fwd[1], fwd[0]], dtype=np.float64)
    hl = float(car.cfg.half_length)
    hw = float(car.cfg.half_width)
    hz = float(car.cfg.half_height) + float(car.cfg.wheel_radius)
    xs: list[float] = []
    ys: list[float] = []
    for sx in (-hl, hl):
        for sy in (-hw, hw):
            xy = p[:2] + sx * fwd + sy * left
            xs.append(float(xy[0]))
            ys.append(float(xy[1]))
    lo = np.array([min(xs), min(ys), float(p[2] - hz)], dtype=np.float64)
    hi = np.array([max(xs), max(ys), float(p[2] + hz)], dtype=np.float64)
    return lo, hi


def aabb_overlap(
    a: tuple[np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray],
    *,
    margin: float = 0.0,
) -> bool:
    alo, ahi = a
    blo, bhi = b
    return bool(np.all(ahi + margin > blo) and np.all(bhi + margin > alo))


def car_clips_fixtures(
    car: "PusherCar",
    fixtures: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    margin: float = _FIXTURE_OVERLAP_MARGIN,
) -> list[str]:
    ca = car_world_aabb(car)
    hit: list[str] = []
    for name, lo, hi in fixtures:
        if aabb_overlap(ca, (lo, hi), margin=margin):
            hit.append(str(name))
    return hit


def _trial_car_pose(car: "PusherCar", rails_xy: np.ndarray, fwd: np.ndarray, table_z: float) -> None:
    car.cfg.forward_xy = _unit_xy(fwd)
    car.cfg.quat = _yaw_quat_wxyz(car.cfg.forward_xy)
    car.cfg.pos[2] = float(table_z) + float(car.cfg.half_height) + float(car.cfg.wheel_radius)
    cxy = car._car_xy(np.asarray(rails_xy, dtype=np.float64).reshape(2), car.cfg.forward_xy)
    car.cfg.pos[0] = float(cxy[0])
    car.cfg.pos[1] = float(cxy[1])


def _xy_separation(
    a: tuple[np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray],
) -> float:
    """Planar gap between AABBs. 0 = overlapping, else axis-aligned clearance."""
    alo, ahi = a
    blo, bhi = b
    dx = max(0.0, float(alo[0] - bhi[0]), float(blo[0] - ahi[0]))
    dy = max(0.0, float(alo[1] - bhi[1]), float(blo[1] - ahi[1]))
    if dx == 0.0 and dy == 0.0:
        return 0.0
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return float(np.hypot(dx, dy))


def propose_tabletop_xy(
    car: "PusherCar",
    *,
    current_xy: np.ndarray,
    fwd: np.ndarray,
    table_z: float,
    fixtures: Sequence[tuple[str, np.ndarray, np.ndarray]],
    plate_xy: Optional[np.ndarray] = None,
    plate_radius: float = PLATE_RADIUS,
    workspace_aabb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pick a table XY whose carrier footprint misses kitchen fixtures."""
    current = np.asarray(current_xy, dtype=np.float64).reshape(2)
    _trial_car_pose(car, current, fwd, table_z)
    if not car_clips_fixtures(car, fixtures):
        return current

    # Open kitchen table: in front of the cabinet (y ≳ 0) and right of the stove.
    xs = np.linspace(-0.06, 0.10, 6)
    ys = np.linspace(0.04, 0.18, 5)
    plate = None if plate_xy is None else np.asarray(plate_xy, dtype=np.float64).reshape(2)
    best = None
    best_score = 1e9
    lo_w = hi_w = None
    if workspace_aabb is not None:
        box = np.asarray(workspace_aabb, dtype=np.float64)
        lo_w, hi_w = box[0, :2], box[1, :2]
    for x in xs:
        for y in ys:
            cand = np.array([float(x), float(y)], dtype=np.float64)
            if lo_w is not None:
                if cand[0] < lo_w[0] + 0.08 or cand[0] > hi_w[0] - 0.08:
                    continue
                if cand[1] < lo_w[1] + 0.08 or cand[1] > hi_w[1] - 0.08:
                    continue
            _trial_car_pose(car, cand, fwd, table_z)
            if car_clips_fixtures(car, fixtures):
                continue
            ca = car_world_aabb(car)
            clearance = min((_xy_separation(ca, (lo, hi)) for _, lo, hi in fixtures), default=1.0)
            if clearance < 0.02:
                continue
            score = -3.0 * float(clearance) + 0.2 * float(np.linalg.norm(cand - current))
            if plate is not None:
                dplate = float(np.linalg.norm(cand - plate))
                need = float(plate_radius) + float(car.cfg.half_width) + 0.02
                if dplate < need:
                    score += (need - dplate) * 4.0
            if score < best_score:
                best_score = score
                best = cand
    return current if best is None else best


def _drop_cargo_to_table(driver: Any, jname: str, target_name: str, table_z: float) -> np.ndarray:
    """Lower a stacked / in-drawer mesh until its collision bottom meets the table."""
    q = driver.base.sim.data.get_joint_qpos(jname).copy()
    zmin = object_collision_zmin(driver, target_name)
    if zmin is None:
        zmin = object_support_z(driver, target_name)
    dz = float(zmin) - float(table_z)
    if dz > _CARRIER_TABLE_SLACK:
        q[2] = float(q[2]) - dz
        driver.base.sim.data.set_joint_qpos(jname, q)
        driver.base.sim.data.set_joint_qvel(jname, np.zeros(6))
        driver.base.sim.forward()
        q = driver.base.sim.data.get_joint_qpos(jname).copy()
    return q


def home_carrier_to_tabletop(
    car: "PusherCar",
    driver: Any,
    *,
    jname: str,
    trajectory: Any,
    target_name: str,
    fwd: np.ndarray,
    table_z: float,
) -> np.ndarray:
    """Seat a fixture/stacked spawn onto the table before the car is laid out.

    Spatial t04 (drawer), t07 (stove), t09 (cabinet top) spawn the bowl inside
    furniture. Using the cargo support z plants the visual-only pickup *in* the
    cabinet. Drop to the plate's table plane and, if the footprint still hits a
    fixture, slide onto open tabletop.
    """
    q = _drop_cargo_to_table(driver, jname, target_name, table_z)
    sim = getattr(getattr(driver, "base", None), "sim", None)
    fixtures = fixture_aabbs(sim)
    plate_xy = None
    for n in (
        getattr(driver, "place_target", None),
        getattr(driver, "place_target_name", None),
        "plate_1",
    ):
        if not n:
            continue
        obj = driver.base.objects_dict.get(str(n))
        if obj is None or not getattr(obj, "joints", None):
            continue
        plate_xy = np.asarray(
            driver.base.sim.data.get_joint_qpos(obj.joints[-1])[:2], dtype=np.float64
        )
        break
    parked = propose_tabletop_xy(
        car,
        current_xy=q[:2],
        fwd=fwd,
        table_z=table_z,
        fixtures=fixtures,
        plate_xy=plate_xy,
        workspace_aabb=getattr(driver, "workspace_aabb", None),
    )
    if float(np.linalg.norm(parked - q[:2])) > 1e-4:
        q[0] = float(parked[0])
        q[1] = float(parked[1])
        driver.base.sim.data.set_joint_qpos(jname, q)
        driver.base.sim.data.set_joint_qvel(jname, np.zeros(6))
        driver.base.sim.forward()
        q = driver.base.sim.data.get_joint_qpos(jname).copy()
    if getattr(trajectory, "base_pos", None) is not None:
        bp = np.asarray(trajectory.base_pos, dtype=np.float64).reshape(3).copy()
        bp[:3] = q[:3]
        trajectory.base_pos = bp
    if plate_xy is not None:
        from benchmarks.dynamic_libero.trajectories.linear import deflect_linear_heading

        deflect_linear_heading(
            trajectory,
            q[:2],
            plate_xy,
            travel=0.42,
            min_clearance=float(PLATE_RADIUS) + float(car.obj_radius) + 0.04,
            candidates=list(range(0, 91, 5)),
        )
    return q


def bumper_contact_xy(*, rails_xy: np.ndarray, forward_xy: np.ndarray) -> np.ndarray:
    return np.asarray(rails_xy, dtype=np.float64).reshape(2).copy()


def car_xy_behind(
    *,
    rails_xy: np.ndarray,
    forward_xy: np.ndarray,
    half_length: float,
    object_radius: float,
    gap: float = CONTACT_GAP,
    nose: float | None = None,
) -> np.ndarray:
    front = float(half_length if nose is None else nose)
    standoff = front + float(object_radius) + float(gap)
    rails = np.asarray(rails_xy, dtype=np.float64).reshape(2)
    fwd = _unit_xy(forward_xy)
    return rails - fwd * standoff


def car_xy_for_cargo(
    *,
    rails_xy: np.ndarray,
    forward_xy: np.ndarray,
    cargo_local_xy: np.ndarray,
) -> np.ndarray:
    rails = np.asarray(rails_xy, dtype=np.float64).reshape(2)
    fwd = _unit_xy(forward_xy)
    left = np.array([-fwd[1], fwd[0]], dtype=np.float64)
    loc = np.asarray(cargo_local_xy, dtype=np.float64).reshape(2)
    offset = loc[0] * fwd + loc[1] * left
    return rails - offset


@dataclass
class PusherConfig:
    body_name: str = PUSHER_BODY
    style: str = "bumper"
    palette: str = "red"
    pos: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, HALF_HEIGHT + WHEEL_RADIUS])
    )
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    half_length: float = HALF_LENGTH
    half_width: float = HALF_WIDTH
    half_height: float = HALF_HEIGHT
    wheel_radius: float = WHEEL_RADIUS
    cargo_local_xy: np.ndarray = field(default_factory=lambda: np.array([-0.038, 0.0]))
    installed: bool = False
    forward_xy: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))


def bumper_config(**kwargs: Any) -> PusherConfig:
    kw: dict[str, Any] = dict(body_name=PUSHER_BODY, style="bumper", palette="red")
    kw.update(kwargs)
    return PusherConfig(**kw)


def carrier_config(**kwargs: Any) -> PusherConfig:
    kw: dict[str, Any] = dict(
        body_name=PUSHER_BODY,
        style="carrier",
        palette="red",
        half_length=CARRIER_HALF_LENGTH,
        half_width=CARRIER_HALF_WIDTH,
        half_height=CARRIER_HALF_HEIGHT,
        wheel_radius=CARRIER_WHEEL_RADIUS,
        cargo_local_xy=np.array([-0.038, 0.0]),
    )
    kw.update(kwargs)
    return PusherConfig(**kw)


def place_bumper_config(**kwargs: Any) -> PusherConfig:
    kw: dict[str, Any] = dict(
        body_name=PLACE_PUSHER_BODY,
        style="bumper",
        palette="blue",
        half_length=PLACE_HALF_LENGTH,
        half_width=PLACE_HALF_WIDTH,
        half_height=HALF_HEIGHT,
        wheel_radius=WHEEL_RADIUS,
    )
    kw.update(kwargs)
    return PusherConfig(**kw)


def _add_wheels(body: ET.Element, cfg: PusherConfig, vis: dict[str, str]) -> None:
    L, W, H = float(cfg.half_length), float(cfg.half_width), float(cfg.half_height)
    wr = float(cfg.wheel_radius)
    # Keep the tire inside the chassis so the sidewall cannot enter a wide mesh.
    wx, wy = L * 0.55, W * 0.62
    prefix = str(cfg.body_name)
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        ET.SubElement(
            body,
            "geom",
            name=f"{prefix}_wheel_{i}",
            type="cylinder",
            size=f"{wr:.5f} {0.008:.5f}",
            pos=f"{sx * wx:.5f} {sy * wy:.5f} {-H:.5f}",
            quat="0.7071 0.7071 0 0",
            rgba="0.08 0.08 0.08 1",
            **vis,
        )


def _add_bumper_visuals(body: ET.Element, cfg: PusherConfig, vis: dict[str, str]) -> None:
    pal = PALETTES.get(cfg.palette, PALETTES["red"])
    L, W, H = float(cfg.half_length), float(cfg.half_width), float(cfg.half_height)
    p = str(cfg.body_name)
    ET.SubElement(
        body, "geom", name=f"{p}_chassis", type="box",
        size=f"{L:.5f} {W:.5f} {H:.5f}", rgba=pal["chassis"], **vis,
    )
    hood_l = L * 0.38
    ET.SubElement(
        body, "geom", name=f"{p}_hood", type="box",
        size=f"{hood_l:.5f} {W * 0.92:.5f} {H * 0.35:.5f}",
        pos=f"{L - hood_l:.5f} 0 {H + H * 0.35:.5f}", rgba=pal["hood"], **vis,
    )
    cabin_l, cabin_w, cabin_h = L * 0.32, W * 0.72, H * 1.15
    ET.SubElement(
        body, "geom", name=f"{p}_cabin", type="box",
        size=f"{cabin_l:.5f} {cabin_w:.5f} {cabin_h:.5f}",
        pos=f"{-L * 0.28:.5f} 0 {H + cabin_h:.5f}", rgba=pal["cabin"], **vis,
    )
    bump_l = bumper_half_for(L)
    ET.SubElement(
        body, "geom", name=f"{p}_bumper", type="box",
        size=f"{bump_l:.5f} {W * 1.02:.5f} {H * 0.55:.5f}",
        pos=f"{L + bump_l:.5f} 0 {-H * 0.15:.5f}", rgba=pal["bumper"], **vis,
    )
    for side in (-1.0, 1.0):
        ET.SubElement(
            body, "geom", name=f"{p}_lamp_{'l' if side < 0 else 'r'}", type="box",
            size=f"{0.004:.5f} {0.007:.5f} {0.004:.5f}",
            pos=f"{L + 2.0 * bump_l - 0.006:.5f} {side * W * 0.55:.5f} {-H * 0.05:.5f}",
            rgba="0.95 0.95 0.82 1", **vis,
        )
    _add_wheels(body, cfg, vis)


def _add_carrier_visuals(body: ET.Element, cfg: PusherConfig, vis: dict[str, str]) -> None:
    pal = PALETTES.get(cfg.palette, PALETTES["red"])
    L, W, H = float(cfg.half_length), float(cfg.half_width), float(cfg.half_height)
    p = str(cfg.body_name)
    ET.SubElement(
        body, "geom", name=f"{p}_chassis", type="box",
        size=f"{L:.5f} {W:.5f} {H:.5f}", rgba=pal["chassis"], **vis,
    )
    cab_l, cab_w, cab_h = L * 0.28, W * 0.82, H * 1.35
    cab_x = L - cab_l - 0.004
    ET.SubElement(
        body, "geom", name=f"{p}_cabin", type="box",
        size=f"{cab_l:.5f} {cab_w:.5f} {cab_h:.5f}",
        pos=f"{cab_x:.5f} 0 {H + cab_h:.5f}", rgba=pal["cabin"], **vis,
    )
    hood_l = L * 0.16
    ET.SubElement(
        body, "geom", name=f"{p}_hood", type="box",
        size=f"{hood_l:.5f} {W * 0.88:.5f} {H * 0.40:.5f}",
        pos=f"{L - hood_l:.5f} 0 {H + H * 0.40:.5f}", rgba=pal["hood"], **vis,
    )
    bump_l = 0.010
    ET.SubElement(
        body, "geom", name=f"{p}_bumper", type="box",
        size=f"{bump_l:.5f} {W * 1.05:.5f} {H * 0.65:.5f}",
        pos=f"{L + bump_l:.5f} 0 {-H * 0.05:.5f}", rgba=pal["bumper"], **vis,
    )
    bed_l = L * 0.52
    bed_x = -L + bed_l
    bh = BED_HALF_H
    ET.SubElement(
        body, "geom", name=f"{p}_bed", type="box",
        size=f"{bed_l:.5f} {W * 0.92:.5f} {bh:.5f}",
        pos=f"{bed_x:.5f} 0 {H + bh:.5f}", rgba=pal["bed"], **vis,
    )
    wall_h = 0.012
    for side, sy in (("l", W * 0.90), ("r", -W * 0.90)):
        ET.SubElement(
            body, "geom", name=f"{p}_rail_{side}", type="box",
            size=f"{bed_l:.5f} {0.005:.5f} {wall_h:.5f}",
            pos=f"{bed_x:.5f} {sy:.5f} {H + bh + wall_h:.5f}",
            rgba=pal["bumper"], **vis,
        )
    ET.SubElement(
        body, "geom", name=f"{p}_tail", type="box",
        size=f"{0.005:.5f} {W * 0.92:.5f} {wall_h:.5f}",
        pos=f"{-L + 0.005:.5f} 0 {H + bh + wall_h:.5f}",
        rgba=pal["bumper"], **vis,
    )
    _add_wheels(body, cfg, vis)


def inject_pusher_xml(xml: str, cfg: PusherConfig) -> str:
    root = ET.fromstring(xml)
    wb = root.find("worldbody")
    if wb is None:
        raise RuntimeError("MuJoCo XML missing worldbody")
    name = str(cfg.body_name)
    for body in list(wb.findall("body")):
        if body.get("name") == name:
            wb.remove(body)
    pos = np.asarray(cfg.pos, dtype=np.float64).reshape(3)
    quat = np.asarray(cfg.quat, dtype=np.float64).reshape(4)
    body = ET.SubElement(
        wb, "body", name=name,
        pos=f"{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}",
        quat=f"{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}",
    )
    vis = {"contype": "0", "conaffinity": "0", "group": "1"}
    if str(cfg.style) == "carrier":
        _add_carrier_visuals(body, cfg, vis)
    else:
        _add_bumper_visuals(body, cfg, vis)
    return ET.tostring(root, encoding="utf8").decode("utf8")


class PusherCar:
    """One kinematic car bound to a free joint + trajectory."""

    def __init__(self, cfg: Optional[PusherConfig] = None):
        self.cfg = cfg or bumper_config()
        self._env = None
        self.obj_radius: float = DEFAULT_OBJECT_RADIUS
        self.gap: float = CONTACT_GAP
        self.table_z: float = 0.0
        self.lift_z: float = 0.0
        self._jname: Optional[str] = None
        self._traj: Any = None
        self._target_name: Optional[str] = None
        self._base_quat: Optional[np.ndarray] = None

    @property
    def standoff(self) -> float:
        return nose_extent(self.cfg.half_length) + float(self.obj_radius) + float(self.gap)

    @property
    def is_carrier(self) -> bool:
        return str(self.cfg.style) == "carrier"

    def install(self, env) -> None:
        install_cars(env, [self])

    def bind_and_layout(
        self,
        driver: Any,
        *,
        jname: str,
        trajectory: Any,
        target_name: str,
        base_quat: Optional[np.ndarray],
    ) -> None:
        self._jname = jname
        self._traj = trajectory
        self._target_name = target_name
        self._base_quat = (
            None if base_quat is None else np.asarray(base_quat, dtype=np.float64).copy()
        )
        q = driver.base.sim.data.get_joint_qpos(jname).copy()
        self.obj_radius = object_horizontal_radius(driver, target_name)
        hz = float(getattr(driver, "control_hz", 20.0) or 20.0)
        fwd = forward_xy_from_traj(trajectory, hz=hz)
        self.cfg.forward_xy = fwd.copy()
        self.cfg.quat = _yaw_quat_wxyz(fwd)
        if self.is_carrier:
            self.table_z = max(0.0, float(table_support_z(driver, target_name)))
            q = home_carrier_to_tabletop(
                self,
                driver,
                jname=jname,
                trajectory=trajectory,
                target_name=target_name,
                fwd=fwd,
                table_z=self.table_z,
            )
            fwd = forward_xy_from_traj(trajectory, hz=hz)
            self.cfg.forward_xy = fwd.copy()
            self.cfg.quat = _yaw_quat_wxyz(fwd)
        else:
            self.table_z = max(0.0, float(object_support_z(driver, target_name)))
        self.cfg.pos[2] = self.table_z + float(self.cfg.half_height) + float(self.cfg.wheel_radius)
        rails_xy = q[:2]
        cxy = self._car_xy(rails_xy, fwd)
        self.cfg.pos[0] = float(cxy[0])
        self.cfg.pos[1] = float(cxy[1])
        if self.is_carrier:
            bed_top = float(self.cfg.pos[2]) + float(self.cfg.half_height) + 2.0 * BED_HALF_H
            self.lift_z = max(0.0, bed_top - self.table_z)
        else:
            self.lift_z = 0.0
        self._apply_pose()
        self._seat(
            driver,
            jname=jname,
            rails_xy=rails_xy,
            rails_z=float(q[2]),
            base_quat=self._base_quat,
        )

    def layout_for_driver(self, driver: Any) -> None:
        if driver.jname is None:
            return
        self.bind_and_layout(
            driver,
            jname=driver.jname,
            trajectory=driver.trajectory,
            target_name=str(driver.target),
            base_quat=driver.base_quat,
        )

    def sync_drive(self, driver: Any) -> None:
        jname = self._jname or driver.jname
        traj = self._traj or driver.trajectory
        if jname is None or traj is None:
            return
        xyz, _ = traj.pose_at()
        rails_xy = np.asarray(xyz, dtype=np.float64).reshape(3)[:2]
        hz = float(getattr(driver, "control_hz", 20.0) or 20.0)
        fwd = forward_xy_from_traj(traj, hz=hz)
        self.cfg.forward_xy = fwd.copy()
        self.cfg.quat = _yaw_quat_wxyz(fwd)
        cxy = self._car_xy(rails_xy, fwd)
        self.cfg.pos[0] = float(cxy[0])
        self.cfg.pos[1] = float(cxy[1])
        self._apply_pose()
        quat = self._base_quat if self._base_quat is not None else getattr(driver, "base_quat", None)
        self._seat(
            driver,
            jname=jname,
            rails_xy=rails_xy,
            rails_z=float(xyz[2]),
            base_quat=quat,
        )

    def _car_xy(self, rails_xy: np.ndarray, fwd: np.ndarray) -> np.ndarray:
        if self.is_carrier:
            return car_xy_for_cargo(
                rails_xy=rails_xy,
                forward_xy=fwd,
                cargo_local_xy=self.cfg.cargo_local_xy,
            )
        return car_xy_behind(
            rails_xy=rails_xy,
            forward_xy=fwd,
            half_length=self.cfg.half_length,
            object_radius=self.obj_radius,
            gap=self.gap,
            nose=nose_extent(self.cfg.half_length),
        )

    def _seat(
        self,
        driver: Any,
        *,
        jname: str,
        rails_xy: Sequence[float],
        rails_z: float,
        base_quat: Optional[np.ndarray],
    ) -> None:
        q = driver.base.sim.data.get_joint_qpos(jname).copy()
        q[0] = float(rails_xy[0])
        q[1] = float(rails_xy[1])
        q[2] = float(rails_z) + float(self.lift_z)
        if q.shape[0] >= 7 and base_quat is not None:
            q[3:7] = np.asarray(base_quat, dtype=np.float64).reshape(4)
        driver.base.sim.data.set_joint_qpos(jname, q)
        driver.base.sim.data.set_joint_qvel(jname, np.zeros(6))

    def _body_id(self) -> Optional[int]:
        if self._env is None:
            return None
        name = str(self.cfg.body_name)
        model = self._env.env.sim.model
        try:
            return int(model.body_name2id(name))
        except Exception:
            names = list(getattr(model, "body_names", []))
            if name not in names:
                return None
            return int(names.index(name))

    def _apply_pose(self) -> None:
        bid = self._body_id()
        if bid is None:
            return
        base = self._env.env
        base.sim.model.body_pos[bid][:] = np.asarray(self.cfg.pos, dtype=np.float64)
        base.sim.model.body_quat[bid][:] = np.asarray(self.cfg.quat, dtype=np.float64)
        base.sim.forward()


def install_cars(env, cars: Sequence[PusherCar]) -> None:
    cars = list(cars)
    if not cars:
        return
    base = env.env

    def _processor(xml: str) -> str:
        out = xml
        for car in cars:
            out = inject_pusher_xml(out, car.cfg)
        return out

    base.set_xml_processor(_processor)
    was_det = bool(getattr(base, "deterministic_reset", False))
    base.deterministic_reset = False
    if hasattr(env, "reset"):
        env.reset()
    else:
        base.reset()
    base.deterministic_reset = was_det
    names = list(getattr(base.sim.model, "body_names", []))
    for car in cars:
        car._env = env
        car.cfg.installed = True
        if car.cfg.body_name not in names:
            raise RuntimeError(f"Car body {car.cfg.body_name!r} missing after XML install")


def make_pusher_scene(
    motive: str,
    variant: str = "pick",
) -> tuple[Optional[PusherCar], Optional[PusherCar]]:
    """Cars for a rails variant. Carrier + pick = bowl on flatbed only."""
    from benchmarks.dynamic_libero.env.dynamic_tasks import parse_rails_variant

    if motive == "carrier":
        var = parse_rails_variant(variant, default="pick")
        pick = PusherCar(carrier_config()) if var in ("pick", "both") else None
        place = PusherCar(place_bumper_config()) if var in ("place", "both") else None
        return pick, place
    var = parse_rails_variant(variant, default="pick")
    pick = PusherCar(bumper_config()) if var in ("pick", "both") else None
    place = PusherCar(place_bumper_config()) if var in ("place", "both") else None
    return pick, place


def resolve_place_target(
    base: Any,
    name: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    candidates: list[str] = []
    if name:
        candidates.append(str(name))
    candidates.extend(("basket_1", "basket", "plate_1"))
    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        obj = base.objects_dict.get(cand)
        if obj is not None and getattr(obj, "joints", None):
            return cand, obj.joints[-1]
    return None, None


def corridor_lateral_delta(
    xy: np.ndarray,
    *,
    start_xy: np.ndarray,
    forward_xy: np.ndarray,
    travel: float,
    rear: float,
    half_width: float,
) -> Optional[np.ndarray]:
    """Return a planar delta that slides ``xy`` off the travel stadium, or None."""
    start = np.asarray(start_xy, dtype=np.float64).reshape(2)
    pt = np.asarray(xy, dtype=np.float64).reshape(2)
    fwd = _unit_xy(forward_xy)
    left = np.array([-fwd[1], fwd[0]], dtype=np.float64)
    rel = pt - start
    along = float(np.dot(rel, fwd))
    lat = float(np.dot(rel, left))
    if along < -float(rear) or along > float(travel):
        return None
    need = float(half_width)
    if abs(lat) >= need:
        return None
    side = 1.0 if lat >= 0.0 else -1.0
    return left * side * (need - abs(lat) + 0.01)


def _nearest_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ab = b - a
    l2 = float(np.dot(ab, ab))
    if l2 < 1e-12:
        rel = p - a
        return a.copy(), rel
    t = float(np.clip(np.dot(p - a, ab) / l2, 0.0, 1.0))
    q = a + t * ab
    return q, p - q


def corridor_delta_polyline(
    xy: np.ndarray,
    verts: np.ndarray,
    *,
    half_width: float,
    rear: float = 0.12,
    min_along: float = -1e9,
) -> Optional[np.ndarray]:
    """Push ``xy`` off a piecewise-linear corridor (plus a rear bumper pocket)."""
    pts = [np.asarray(v, dtype=np.float64).reshape(2) for v in verts]
    if len(pts) < 2:
        return None
    origin = pts[0].copy()
    d0 = pts[1] - pts[0]
    n0 = float(np.linalg.norm(d0))
    fwd0 = d0 / n0 if n0 > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)
    p = np.asarray(xy, dtype=np.float64).reshape(2)
    if float(np.dot(p - origin, fwd0)) < float(min_along):
        return None
    if n0 > 1e-9:
        pts[0] = pts[0] - fwd0 * float(rear)
    best_dist = 1e9
    best_rel = np.zeros(2, dtype=np.float64)
    for a, b in zip(pts[:-1], pts[1:]):
        q, rel = _nearest_on_segment(p, a, b)
        dist = float(np.linalg.norm(rel))
        if dist < best_dist:
            best_dist = dist
            if dist < 1e-9:
                seg = b - a
                left = np.array([-seg[1], seg[0]], dtype=np.float64)
                ln = float(np.linalg.norm(left))
                best_rel = left / ln if ln > 1e-9 else np.array([0.0, 1.0])
            else:
                best_rel = rel
    need = float(half_width)
    if best_dist >= need:
        return None
    nrm = float(np.linalg.norm(best_rel))
    push_dir = best_rel / nrm if nrm > 1e-9 else np.array([0.0, 1.0], dtype=np.float64)
    return push_dir * (need - best_dist + 0.01)


def path_verts_xy(traj: Any, *, horizon: float = PATH_CLEAR_HORIZON, n: int = 20) -> Optional[np.ndarray]:
    if traj is None or getattr(traj, "base_pos", None) is None:
        return None
    wps = getattr(traj, "rel_waypoints", None)
    base = np.asarray(traj.base_pos, dtype=np.float64).reshape(3)
    if wps:
        return np.stack(
            [base[:2] + np.asarray(w, dtype=np.float64).reshape(3)[:2] for w in wps],
            axis=0,
        )
    speed = abs(float(getattr(traj, "speed", 0.0) or 0.0))
    tmax = float(horizon) / max(speed, 1e-6)
    ts = np.linspace(0.0, tmax, max(2, int(n)))
    return np.stack(
        [np.asarray(traj.xyz_at(float(t)), dtype=np.float64).reshape(3)[:2] for t in ts],
        axis=0,
    )


def _sweep_for_car(car: PusherCar, horizon: float = PATH_CLEAR_HORIZON) -> Optional[dict[str, Any]]:
    verts = path_verts_xy(car._traj, horizon=horizon)
    if verts is None or len(verts) < 2:
        return None
    if car.is_carrier:
        rear = 0.02
        # Include the car footprint under the cargo (cookie box / ramekin the
        # bowl was sitting on) but not objects clearly behind the spawn.
        rear_span = float(car.cfg.half_length) - abs(float(car.cfg.cargo_local_xy[0]))
        min_along = -max(0.05, rear_span)
    else:
        rear = float(car.standoff) + float(car.cfg.half_length) + 0.02
        min_along = -1e9
    half_w = max(float(car.cfg.half_width), float(car.obj_radius)) + float(PATH_CLEAR_MARGIN)
    return {"verts": verts, "rear": rear, "half_w": half_w, "min_along": min_along}


def clear_path_obstacles(driver: Any, cars: Sequence[PusherCar]) -> int:
    """Slide non-rails free objects off each car's travel corridor.

    Only the rails cargo is left in the corridor. The BDDL receptacle
    (``place_target`` / ``place_target_name``) is **always** protected, even
    when it is stationary — sliding the spatial plate off the table makes
    ``bowl-on-plate`` unachievable. Cars stay visual-only (``contype=0``) so
    they may pass over a still plate without a collision hull. The corridor
    follows the full polyline.
    """
    sweeps = [s for s in (_sweep_for_car(c) for c in cars if c is not None) if s is not None]
    if not sweeps:
        return 0
    protected: set[str] = set()
    if getattr(driver, "pick_motion", True) and getattr(driver, "target", None):
        protected.add(str(driver.target))
    for n in (getattr(driver, "place_target", None), getattr(driver, "place_target_name", None)):
        if n:
            protected.add(str(n))
    aabb = getattr(driver, "workspace_aabb", None)
    n_moved = 0
    for name, obj in list(driver.base.objects_dict.items()):
        if name in protected or not getattr(obj, "joints", None):
            continue
        jname = obj.joints[-1]
        q = driver.base.sim.data.get_joint_qpos(jname).copy()
        xy = q[:2].copy()
        delta = np.zeros(2, dtype=np.float64)
        hr = object_horizontal_radius(driver, name)
        for sw in sweeps:
            d = corridor_delta_polyline(
                xy + delta,
                sw["verts"],
                half_width=sw["half_w"] + hr,
                rear=sw["rear"],
                min_along=float(sw.get("min_along", -1e9)),
            )
            if d is not None:
                delta = delta + d
        if float(np.linalg.norm(delta)) < 1e-6:
            continue
        q[0] = float(xy[0] + delta[0])
        q[1] = float(xy[1] + delta[1])
        if aabb is not None:
            lo, hi = np.asarray(aabb[0], dtype=np.float64), np.asarray(aabb[1], dtype=np.float64)
            q[0] = float(np.clip(q[0], lo[0] + 0.04, hi[0] - 0.04))
            q[1] = float(np.clip(q[1], lo[1] + 0.04, hi[1] - 0.04))
        driver.base.sim.data.set_joint_qpos(jname, q)
        driver.base.sim.data.set_joint_qvel(jname, np.zeros(6))
        n_moved += 1
    if n_moved:
        driver.base.sim.forward()
    return n_moved


def layout_pusher_for_driver(car: PusherCar, driver: Any) -> None:
    car.layout_for_driver(driver)


