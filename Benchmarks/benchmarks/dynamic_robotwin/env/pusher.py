"""Visible bumper cars for Dynamic-RoboTwin (LIBERO ``pusher`` analogue).

The original cup / coaster meshes stay. Each car is a kinematic SAPIEN actor
with visual-only collision groups; it rides the rails *behind* the object so
the bumper appears to push it. Official eval still defaults to ghost rails
with no extra actor.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

HALF_LENGTH = 0.055
HALF_WIDTH = 0.032
HALF_HEIGHT = 0.016
WHEEL_RADIUS = 0.012
BUMPER_HALF = 0.008
# Bumper box is centered on the chassis front face, so the nose is one
# half-thickness past the chassis. Placement must use this nose, not HALF_LENGTH.
FRONT_EXTENT = HALF_LENGTH + BUMPER_HALF
# Skin past the measured footprint. Zero lets coplanar faces z-fight and lets a
# slightly small radius sink the bumper into the mesh. Collision groups on the
# car are disabled, so this gap is the only thing that keeps the meshes apart.
CONTACT_GAP = 0.003
RADIUS_MARGIN = 1.0
DEFAULT_OBJECT_RADIUS = 0.037  # 021_cup wall at bumper height (~3.7 cm)
COASTER_RADIUS = 0.052  # 019_coaster visual disc radius
TABLE_Z = 0.742

PALETTES: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "red": {
        "chassis": (0.90, 0.12, 0.10, 1.0),
        "hood": (0.95, 0.78, 0.10, 1.0),
        "cabin": (0.20, 0.48, 0.82, 1.0),
        "bumper": (0.98, 0.92, 0.15, 1.0),
        "wheel": (0.12, 0.12, 0.12, 1.0),
    },
    "blue": {
        "chassis": (0.12, 0.38, 0.82, 1.0),
        "hood": (0.55, 0.78, 0.95, 1.0),
        "cabin": (0.10, 0.18, 0.32, 1.0),
        "bumper": (0.95, 0.85, 0.20, 1.0),
        "wheel": (0.12, 0.12, 0.12, 1.0),
    },
}


def _unit_xy(v: np.ndarray) -> np.ndarray:
    fwd = np.asarray(v, dtype=np.float64).reshape(-1)[:2]
    n = float(np.linalg.norm(fwd))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return fwd / n


def _yaw_quat_wxyz(forward_xy: np.ndarray) -> np.ndarray:
    x, y = float(forward_xy[0]), float(forward_xy[1])
    n = (x * x + y * y) ** 0.5
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    yaw = float(np.arctan2(y, x))
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def forward_xy_from_traj(traj: Any, hz: float = 20.0) -> np.ndarray:
    v = np.asarray(traj.velocity_at(traj.t, hz=float(hz)), dtype=np.float64).reshape(3)
    fwd = v[:2].copy()
    if float(np.linalg.norm(fwd)) < 1e-9:
        p0 = np.asarray(traj.xyz_at(traj.t), dtype=np.float64).reshape(3)
        p1 = np.asarray(traj.xyz_at(traj.t + 1.0), dtype=np.float64).reshape(3)
        fwd = (p1 - p0)[:2]
    return _unit_xy(fwd)


def object_xy_radius(name: str | None, default: float | None = DEFAULT_OBJECT_RADIUS) -> float | None:
    """Named visual planar radius, or ``default`` (None = no cap, use AABB)."""
    key = str(name or "").lower()
    # Exact keys first so ``target`` (mouse pad) does not match ``target_object``.
    exact = {
        "target": 0.065,  # place_mouse_pad create_box half 3.5×6.5 cm
        "mouse": 0.040,
        "fan": 0.050,
        "pot": 0.090,
    }
    if key in exact:
        return float(exact[key])
    table = (
        ("coaster", 0.052),
        ("skillet", 0.095),
        ("basket", 0.080),
        ("plate", 0.090),
        ("scale", 0.075),
        ("displaystand", 0.055),
        ("phonestand", 0.045),
        ("stand", 0.045),
        ("pad", 0.055),
        ("phone", 0.035),
        ("container", 0.045),
        ("bread", 0.048),
        ("stapler", 0.050),
        ("pillbottle", 0.018),
        ("can", 0.032),
        ("cup", 0.037),
    )
    for token, rad in table:
        if token in key:
            return float(rad)
    if default is None:
        return None
    return float(default)


class PlanarFootprint:
    """World-XY AABB of a mesh relative to the actor origin."""

    def __init__(self, center_xy: np.ndarray, half_xy: np.ndarray):
        self.center_xy = np.asarray(center_xy, dtype=np.float64).reshape(2)
        self.half_xy = np.abs(np.asarray(half_xy, dtype=np.float64).reshape(2))

    def support(self, direction_xy: np.ndarray) -> float:
        """Distance from the origin to the AABB face in ``direction_xy``."""
        u = _unit_xy(direction_xy)
        return float(np.dot(self.center_xy, u) + np.dot(self.half_xy, np.abs(u)))


def pusher_object_radius(wrapper, name: str | None, entity=None) -> float:
    """Safe isotropic standoff: the larger of the named disc and the live AABB.

    ``min`` pulled the bumper inside whichever estimate was smaller. The car
    has no collision response, so the nose has to stay outside both.
    A degenerate model_data box (≤2 cm) is ignored in favour of the named disc.
    """
    named = object_xy_radius(name, default=None)
    fb = float(named) if named is not None else DEFAULT_OBJECT_RADIUS
    aabb = measure_actor_xy_radius(wrapper, fb, entity=entity)
    if named is None:
        return float(aabb)
    if float(aabb) <= 0.021:
        return float(named)
    return float(max(aabb, named))


def measure_actor_xy_radius(wrapper, fallback: float | None = None, entity=None) -> float:
    """World-XY footprint of the mesh AABB relative to the actor pose origin.

    RoboTwin cups/coasters are discs: the visual rim is the AABB *face*
    (half-width), not the box-corner hypot (×√2). Local ``center`` is usually
    along the asset up-axis; after the spawn quaternion that offset is Z and
    must not inflate the planar radius. Prefer :func:`pusher_object_radius`
    for car placement — JSON AABB overstates the visible disc.
    """
    fb = DEFAULT_OBJECT_RADIUS if fallback is None else float(fallback)
    data = getattr(wrapper, "config", None)
    if not isinstance(data, dict):
        data = getattr(wrapper, "config", None)
    if not isinstance(data, dict):
        return fb
    try:
        extents = np.asarray(data.get("extents", []), dtype=np.float64).reshape(-1)
        scale = np.asarray(data.get("scale", [1.0]), dtype=np.float64).reshape(-1)
        if extents.size < 3:
            return fb
        if scale.size == 1:
            scale3 = np.full(3, float(scale[0]))
        else:
            scale3 = np.abs(np.asarray(scale[:3], dtype=np.float64).reshape(-1))
            if scale3.size < 3:
                scale3 = np.pad(scale3, (0, 3 - int(scale3.size)), constant_values=1.0)
        half = np.abs(extents[:3]) * scale3 * 0.5
        center = np.asarray(data.get("center", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        center_s = center * scale3
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append(center_s + half * np.array([sx, sy, sz]))
        corners = np.stack(corners, axis=0)
        ent = entity or getattr(wrapper, "actor", None)
        if ent is not None and hasattr(ent, "get_pose"):
            pose = ent.get_pose()
            try:
                mat = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)
                world = corners @ mat[:3, :3].T
            except Exception:
                # sapien Pose.q is wxyz
                q = np.asarray(pose.q, dtype=np.float64).reshape(4)
                w, x, y, z = q
                R = np.array(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ],
                    dtype=np.float64,
                )
                world = corners @ R.T
            # Footprint = world-XY AABB half-size, plus planar offset of the
            # AABB center from the pose origin. Using max||xy|| of the 8
            # corners is the box *diagonal* (×√2) and pushed the car too far.
            xy = world[:, :2]
            half_xy = 0.5 * (xy.max(axis=0) - xy.min(axis=0))
            center_xy = 0.5 * (xy.max(axis=0) + xy.min(axis=0))
            radius = float(np.max(half_xy) + np.linalg.norm(center_xy))
        else:
            radius = float(np.max(half))
        return max(radius * float(RADIUS_MARGIN), 0.020)
    except Exception:
        return fb


def measure_planar_footprint(wrapper, entity=None) -> Optional[PlanarFootprint]:
    """Axis-aligned footprint used to clear corners, not only the long face."""
    data = getattr(wrapper, "config", None)
    if not isinstance(data, dict):
        return None
    try:
        extents = np.asarray(data.get("extents", []), dtype=np.float64).reshape(-1)
        scale = np.asarray(data.get("scale", [1.0]), dtype=np.float64).reshape(-1)
        if extents.size < 3:
            return None
        if scale.size == 1:
            scale3 = np.full(3, float(scale[0]))
        else:
            scale3 = np.abs(np.asarray(scale[:3], dtype=np.float64).reshape(-1))
            if scale3.size < 3:
                scale3 = np.pad(scale3, (0, 3 - int(scale3.size)), constant_values=1.0)
        half = np.abs(extents[:3]) * scale3 * 0.5
        center = np.asarray(data.get("center", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        center_s = center * scale3
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append(center_s + half * np.array([sx, sy, sz]))
        corners = np.stack(corners, axis=0)
        ent = entity or getattr(wrapper, "actor", None)
        if ent is not None and hasattr(ent, "get_pose"):
            pose = ent.get_pose()
            try:
                mat = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)
                world = corners @ mat[:3, :3].T
            except Exception:
                q = np.asarray(pose.q, dtype=np.float64).reshape(4)
                w, x, y, z = q
                rot = np.array(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ],
                    dtype=np.float64,
                )
                world = corners @ rot.T
            xy = world[:, :2]
        else:
            xy = corners[:, :2]
        half_xy = 0.5 * (xy.max(axis=0) - xy.min(axis=0))
        center_xy = 0.5 * (xy.max(axis=0) + xy.min(axis=0))
        if float(np.max(half_xy)) <= 0.021:
            return None
        return PlanarFootprint(center_xy, half_xy)
    except Exception:
        return None


def wheel_bottom_local_z() -> float:
    """Tire contact in the chassis frame. Origin is the chassis center."""
    return -float(HALF_HEIGHT) - float(WHEEL_RADIUS)


def car_origin_z(table_z: float) -> float:
    """Chassis-center height that puts the tire bottom on ``table_z``."""
    return float(table_z) - wheel_bottom_local_z()


def _rgba_mat(rgba: tuple[float, float, float, float]):
    import sapien

    return sapien.render.RenderMaterial(base_color=[float(c) for c in rgba])


def _attach_box(render, half, local_xyz, rgba, quat=None) -> None:
    import sapien

    shape = sapien.render.RenderShapeBox(
        [float(half[0]), float(half[1]), float(half[2])],
        _rgba_mat(rgba),
    )
    q = [1.0, 0.0, 0.0, 0.0] if quat is None else [float(v) for v in quat]
    shape.set_local_pose(
        sapien.Pose([float(local_xyz[0]), float(local_xyz[1]), float(local_xyz[2])], q)
    )
    render.attach(shape)


def _attach_cylinder(render, radius, half_length, local_xyz, rgba, quat) -> None:
    """SAPIEN cylinders run along local +X. ``quat`` points that axis at the axle."""
    import sapien

    shape = sapien.render.RenderShapeCylinder(
        float(radius),
        float(half_length),
        _rgba_mat(rgba),
    )
    shape.set_local_pose(
        sapien.Pose(
            [float(local_xyz[0]), float(local_xyz[1]), float(local_xyz[2])],
            [float(v) for v in quat],
        )
    )
    render.attach(shape)


def spawn_bumper_car(task_env, *, name: str, palette: str = "red"):
    """Kinematic bumper car. Tires sit on the table; the nose stops at FRONT_EXTENT.

    The body frame origin is the chassis center, matching Dynamic-LIBERO.
    Wheel centers are at local z=-HALF_HEIGHT, so with
    ``car_origin_z(table) = table + HALF_HEIGHT + WHEEL_RADIUS`` the tire
    bottom is exactly the table plane and the chassis bottom clears it by
    one wheel radius. An older build drew the wheels *above* that origin,
    which left the car about 3 cm in the air and drove the bumper through
    upright meshes.

    Collision is visual-sized and then disabled. PhysX will not separate a
    kinematic car from the object; clearance is entirely in ``sync``.
    """
    import sapien
    from envs.utils.create_actor import preprocess

    pal = PALETTES.get(palette) or PALETTES["red"]
    L, W, H, wr = HALF_LENGTH, HALF_WIDTH, HALF_HEIGHT, WHEEL_RADIUS
    scene, _ = preprocess(task_env, sapien.Pose([0.0, 0.0, 1.0]))

    entity = sapien.Entity()
    entity.set_name(str(name))

    rigid = sapien.physx.PhysxRigidDynamicComponent()
    # Nose at +FRONT_EXTENT, tail at -L, tire bottom at -(H+wr). Centered on
    # the origin this box reaches the visual nose and the tire contact, and
    # no farther forward.
    rigid.attach(
        sapien.physx.PhysxCollisionShapeBox(
            half_size=[float(FRONT_EXTENT), W * 1.02, H + wr],
            material=scene.default_physical_material,
        )
    )
    try:
        rigid.set_kinematic(True)
    except Exception:
        pass

    render = sapien.render.RenderBodyComponent()
    _attach_box(render, (L, W * 0.92, H), (0.0, 0.0, 0.0), pal["chassis"])
    cab_l, cab_w, cab_h = L * 0.36, W * 0.70, H * 1.05
    _attach_box(render, (cab_l, cab_w, cab_h), (-L * 0.18, 0.0, H + cab_h), pal["cabin"])
    _attach_box(
        render,
        (cab_l * 0.72, cab_w * 0.78, cab_h * 0.42),
        (-L * 0.18, 0.0, H + cab_h + cab_h * 0.35),
        (0.75, 0.88, 0.95, 0.55),
    )
    hood_l = L * 0.28
    _attach_box(
        render,
        (hood_l, W * 0.82, H * 0.36),
        (L * 0.48, 0.0, H + H * 0.36),
        pal["hood"],
    )
    bump_l = float(BUMPER_HALF)
    _attach_box(
        render,
        (bump_l, W * 1.02, H * 0.55),
        (L, 0.0, -H * 0.20),
        pal["bumper"],
    )
    # +90° about Z takes the cylinder axis from +X to +Y (the axle).
    axle = (0.707106781, 0.0, 0.0, 0.707106781)
    hubs = ((-L * 0.58, -1.0), (-L * 0.58, 1.0), (L * 0.52, -1.0), (L * 0.52, 1.0))
    for sx, side in hubs:
        sy = side * W * 0.72
        center = (sx, sy, -H)
        _attach_cylinder(render, wr, 0.007, center, pal["wheel"], axle)
        _attach_cylinder(render, wr * 0.45, 0.008, center, pal["hood"], axle)
        _attach_box(
            render,
            (wr * 0.95, 0.006, wr * 0.28),
            (sx, side * W * 0.78, -H + wr * 0.55),
            pal["chassis"],
        )
    for side in (-1.0, 1.0):
        _attach_box(
            render,
            (0.004, 0.007, 0.004),
            (FRONT_EXTENT - 0.006, side * W * 0.62, -H * 0.05),
            (0.95, 0.95, 0.85, 1.0),
        )

    entity.add_component(rigid)
    entity.add_component(render)
    scene.add_entity(entity)

    try:
        shapes = rigid.get_collision_shapes() or []
        for shape in shapes:
            groups = list(shape.get_collision_groups())
            shape.set_collision_groups([0, 0, groups[2], groups[3]])
    except Exception:
        pass
    return entity


class RoboTwinPusherCar:
    """Kinematic bumper car bound to one rails target + trajectory."""

    def __init__(self, entity, *, palette: str = "red", target_name: str = "cup"):
        self.entity = entity
        self.palette = palette
        self.target_name = target_name
        self.traj: Any = None
        self.obj_radius = object_xy_radius(target_name)
        self.footprint: Optional[PlanarFootprint] = None
        self.table_z = TABLE_Z
        self.hz = 20.0

    @property
    def standoff(self) -> float:
        return float(FRONT_EXTENT) + float(self.obj_radius) + float(CONTACT_GAP)

    def object_reach(self, forward_xy: np.ndarray) -> float:
        """How far the mesh extends from the rails origin toward the car.

        Takes the larger of the stored radius, the named disc, and the AABB
        support in the current heading so a corner approach cannot clip.
        """
        reach = float(self.obj_radius)
        named = object_xy_radius(self.target_name, default=None)
        if named is not None:
            reach = max(reach, float(named))
        if self.footprint is not None:
            reach = max(reach, self.footprint.support(-np.asarray(forward_xy, dtype=np.float64)))
        return float(reach)

    def clearance(self, forward_xy: np.ndarray) -> float:
        return float(FRONT_EXTENT) + self.object_reach(forward_xy) + float(CONTACT_GAP)

    def bind(
        self,
        traj: Any,
        *,
        table_z: float,
        hz: float = 20.0,
        obj_radius: Optional[float] = None,
        footprint: Optional[PlanarFootprint] = None,
    ):
        self.traj = traj
        self.table_z = float(table_z)
        self.hz = float(hz)
        if obj_radius is not None:
            self.obj_radius = float(obj_radius)
        if footprint is not None:
            self.footprint = footprint
        self.sync()

    def _car_z(self) -> float:
        return car_origin_z(self.table_z)

    def sync(self) -> None:
        if self.traj is None or self.entity is None:
            return
        import sapien

        xyz = np.asarray(self.traj.xyz_at(self.traj.t), dtype=np.float64).reshape(3)
        fwd = forward_xy_from_traj(self.traj, hz=self.hz)
        cxy = xyz[:2] - fwd * self.clearance(fwd)
        quat = _yaw_quat_wxyz(fwd)
        pose = sapien.Pose(
            [float(cxy[0]), float(cxy[1]), self._car_z()],
            quat.tolist(),
        )
        try:
            self.entity.set_pose(pose)
        except Exception:
            pass
        try:
            import sapien as _s

            comp = self.entity.find_component_by_type(_s.physx.PhysxRigidDynamicComponent)
            if comp is not None and hasattr(comp, "set_kinematic_target"):
                comp.set_kinematic_target(pose)
        except Exception:
            pass


def orient_curve_inward(traj: Any) -> None:
    """Flip a planar curve 180° in XY when it would drift away from the origin.

    ``s_wave`` is authored +X; RoboTwin cups spawn on the ±X rim, so an unflipped
    wave can walk off the table. Linear rails already use ``toward_center``.
    """
    if str(getattr(traj, "name", "")) != "curve":
        return
    pts = getattr(traj, "_pts", None)
    base = getattr(traj, "base_pos", None)
    if pts is None or base is None:
        return
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return
    end = pts[-1, :2]
    toward = -np.asarray(base, dtype=np.float64).reshape(3)[:2]
    if float(np.linalg.norm(end)) < 1e-9 or float(np.linalg.norm(toward)) < 1e-9:
        return
    if float(np.dot(end, toward)) >= 0.0:
        return
    pts = pts.copy()
    pts[:, 0] *= -1.0
    pts[:, 1] *= -1.0
    traj._pts = pts
    traj.rel_waypoints = [p.copy() for p in pts]


def clone_trajectory(traj: Any):
    """Independent copy; place rails re-resolve ``toward_center`` at their own reset."""
    from benchmarks.dynamic_libero.trajectories.base import clone_trajectory as _clone

    return _clone(traj)
