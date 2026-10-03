"""Native motion for the five Dynamic-RoboTwin extra tasks.

These are not linear rails. Each task advances on the policy clock with the
same ODE the demo recorders use: ballistic shuttlecock, hopping bunny,
rolling orange, steered toy car, or a PhysX cut shot. ``motion_scale`` is
the intensity axis shared by every accelerator panel. Scale ``1`` is the
catalog motion. Scale ``0`` holds the object in place (static sanity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..viz.diecast_drive import FRONT, HUBS, INTERCEPT_T, PATH, TOTAL, body_matrix, wheel_matrix

TABLE_Z = 0.741
G = 9.81

# Eval lob, not the 0.36 s cine probe. The arm has to intercept a bird that
# is already flying at t=0.
SHUTTLE_P0 = np.array([0.20, 0.00, 0.84], dtype=np.float64)
SHUTTLE_V0 = np.array([-0.55, 0.00, 3.05], dtype=np.float64)
SHUTTLE_CATCH_T = 0.55
SHUTTLE_K_M = 0.30

BUNNY_START = np.array([0.20, 0.10], dtype=np.float64)
BUNNY_END = np.array([-0.16, -0.03], dtype=np.float64)
BUNNY_HOP_HEIGHT = 0.055
BUNNY_HOP_PERIOD = 0.36
BUNNY_DURATION = 2.4
BUNNY_CATCH_T = 1.98

ORANGE_START = np.array([0.22, -0.20], dtype=np.float64)
ORANGE_END = np.array([-0.16, -0.02], dtype=np.float64)
ORANGE_RADIUS = 0.038
ORANGE_DURATION = 2.4

POOL_CUE_SPEED = 0.70
POOL_RADIUS = 0.030
POOL_HORIZON_S = 3.6


@dataclass(frozen=True)
class ExtraMotionProfile:
    task_name: str
    kind: str
    catch_t: float
    horizon_s: float
    motion_scale: float


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _yaw_pitch_quat(yaw: float, pitch: float) -> np.ndarray:
    hy, hp = 0.5 * float(yaw), 0.5 * float(pitch)
    q_yaw = np.array([np.cos(hy), 0.0, 0.0, np.sin(hy)], dtype=np.float64)
    q_pitch = np.array([np.cos(hp), 0.0, np.sin(hp), 0.0], dtype=np.float64)
    return _qmul(q_yaw, q_pitch)


def _axangle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = np.sin(0.5 * float(angle))
    return np.array(
        [np.cos(0.5 * float(angle)), axis[0] * s, axis[1] * s, axis[2] * s],
        dtype=np.float64,
    )


def _yaw_quat(yaw: float) -> np.ndarray:
    h = 0.5 * float(yaw)
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)], dtype=np.float64)


def _rotmat_to_quat(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def step_shuttlecock(
    position: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    *,
    k_m: float = SHUTTLE_K_M,
    g: float = G,
) -> tuple[np.ndarray, np.ndarray]:
    """Gravity plus quadratic drag. Same update as ``catch_shuttlecock.step_flight``."""
    v = np.asarray(velocity, dtype=np.float64).reshape(3).copy()
    p = np.asarray(position, dtype=np.float64).reshape(3).copy()
    speed = float(np.linalg.norm(v)) + 1e-9
    acc = np.array([0.0, 0.0, -float(g)], dtype=np.float64) - float(k_m) * speed * v
    v = v + acc * float(dt)
    p = p + v * float(dt)
    return p, v


def shuttlecock_initial(seed: int, motion_scale: float) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(int(seed) + 17)
    p0 = SHUTTLE_P0 + rng.uniform(-1.0, 1.0, size=3) * np.array([0.04, 0.04, 0.02])
    scale = float(rng.uniform(0.92, 1.08))
    yaw = float(rng.uniform(-0.10, 0.10))
    c, s = np.cos(yaw), np.sin(yaw)
    v = SHUTTLE_V0 * scale
    v0 = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], dtype=np.float64)
    catch_t = SHUTTLE_CATCH_T * float(rng.uniform(0.88, 1.12))
    rate = float(motion_scale)
    if rate > 0.0:
        v0 = v0 * rate
        catch_t = catch_t / rate
    return p0, v0, float(catch_t)


def bunny_pose(t: float) -> tuple[np.ndarray, np.ndarray]:
    u = _smoothstep(float(t) / BUNNY_DURATION)
    xy = (1.0 - u) * BUNNY_START + u * BUNNY_END
    if t >= BUNNY_DURATION:
        xy = BUNNY_END.copy()
    heading = BUNNY_END - BUNNY_START
    yaw = float(np.arctan2(heading[1], heading[0]))
    z = TABLE_Z + 0.002
    pitch = 0.0
    if 0.0 < t < BUNNY_DURATION:
        phase = (t % BUNNY_HOP_PERIOD) / BUNNY_HOP_PERIOD
        z += 4.0 * BUNNY_HOP_HEIGHT * phase * (1.0 - phase)
        pitch = 0.34 * float(np.sin(phase * np.pi)) * (1.0 if phase < 0.55 else 0.4)
    return np.array([xy[0], xy[1], z], dtype=np.float64), _yaw_pitch_quat(yaw, pitch)


def _roll_arc(t: float, dist: float, duration: float) -> float:
    duration = max(float(duration), 1e-6)
    t = float(np.clip(t, 0.0, duration))
    dist = float(dist)
    v_start = 0.12
    tau = min(0.70, 0.35 * duration)
    rest = duration - tau
    denom = tau * (0.5 * tau + rest)
    acc = (dist - v_start * duration) / max(denom, 1e-9)
    if t <= tau:
        s = v_start * t + 0.5 * acc * t * t
    else:
        v_c = v_start + acc * tau
        s = v_start * tau + 0.5 * acc * tau * tau + v_c * (t - tau)
    return float(np.clip(s, 0.0, dist))


def orange_pose(t: float) -> tuple[np.ndarray, np.ndarray]:
    direction = ORANGE_END - ORANGE_START
    dist = float(np.linalg.norm(direction))
    direction = direction / max(dist, 1e-9)
    s = _roll_arc(t, dist, ORANGE_DURATION)
    xy = ORANGE_START + direction * s
    axis = np.cross(np.array([0.0, 0.0, 1.0]), np.array([direction[0], direction[1], 0.0]))
    quat = _axangle_quat(axis, s / ORANGE_RADIUS)
    return np.array([xy[0], xy[1], TABLE_Z + ORANGE_RADIUS], dtype=np.float64), quat


def car_sample(t: float) -> dict:
    return PATH.sample(float(t))


def pool_strike_velocity(motion_scale: float) -> tuple[np.ndarray, np.ndarray]:
    speed = POOL_CUE_SPEED * max(float(motion_scale), 0.0)
    linear = np.array([0.0, -speed, 0.0], dtype=np.float64)
    angular = np.cross(np.array([0.0, 0.0, 1.0]), linear) / max(POOL_RADIUS, 1e-6)
    return linear, angular


def _wall_time(nominal_s: float, motion_scale: float) -> float:
    scale = float(motion_scale)
    if scale <= 0.0:
        return 8.0
    return float(nominal_s) / scale


def _bind_clock(env, pose_fn, *, duration: float, motion_scale: float) -> None:
    scale = float(motion_scale)

    def reset_flight() -> None:
        env.flight_t = 0.0

    def step_flight(dt: float) -> None:
        if scale <= 0.0:
            return
        env.flight_t = float(getattr(env, "flight_t", 0.0)) + float(dt) * scale

    def flight_pose():
        import sapien

        xyz, quat = pose_fn(float(getattr(env, "flight_t", 0.0)))
        return sapien.Pose(xyz.tolist(), quat.tolist())

    env.reset_flight = reset_flight
    env.step_flight = step_flight
    env.flight_pose = flight_pose
    env.extra_motion_kind = "kinematic"
    env.flight_t = 0.0
    env.CATCH_T = float(duration if scale <= 0.0 else duration / max(scale, 1e-6))


def _set_actor_pose(actor, pose) -> None:
    ent = getattr(actor, "actor", actor)
    try:
        ent.set_pose(pose)
    except Exception:
        return


def _set_velocity(actor, linear: np.ndarray, angular: np.ndarray) -> None:
    ent = getattr(actor, "actor", actor)
    try:
        import sapien

        comp = ent.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    except Exception:
        return
    if comp is None:
        return
    lin = np.asarray(linear, dtype=np.float64).reshape(3)
    ang = np.asarray(angular, dtype=np.float64).reshape(3)
    try:
        comp.set_linear_velocity(lin)
        comp.set_angular_velocity(ang)
    except Exception:
        try:
            comp.linear_velocity = lin
            comp.angular_velocity = ang
        except Exception:
            return


def install_extra_motion(env, task_name: str, *, seed: int, motion_scale: float) -> ExtraMotionProfile:
    """Attach the task ODE and return the wall-clock intercept window."""
    name = str(task_name)
    scale = float(motion_scale)
    env.extra_motion_scale = scale

    if name == "catch_shuttlecock":
        p0, v0, catch_t = shuttlecock_initial(seed, scale if scale > 0.0 else 1.0)
        env.P0 = p0
        env.V0 = v0 if scale > 0.0 else np.zeros(3, dtype=np.float64)
        env.K_M = float(SHUTTLE_K_M)
        env.CATCH_T = float(catch_t if scale > 0.0 else 8.0)
        if scale <= 0.0:
            orig = env.step_flight

            def step_flight(dt: float, _orig=orig) -> None:
                del dt
                return None

            env.step_flight = step_flight
        env.extra_motion_kind = "kinematic"
        return ExtraMotionProfile(name, "intercept", float(env.CATCH_T), float(env.CATCH_T) + 1.5, scale)

    if name == "catch_bunny_toy":
        _bind_clock(env, bunny_pose, duration=BUNNY_DURATION, motion_scale=scale)
        catch_t = _wall_time(BUNNY_CATCH_T, scale)
        return ExtraMotionProfile(name, "intercept", catch_t, _wall_time(BUNNY_DURATION, scale) + 1.0, scale)

    if name == "stop_rolling_orange":
        _bind_clock(env, orange_pose, duration=ORANGE_DURATION, motion_scale=scale)
        horizon = _wall_time(ORANGE_DURATION, scale) + 1.0
        return ExtraMotionProfile(name, "intercept", horizon - 1.0, horizon, scale)

    if name == "pursue_toycar":
        def pose_fn(t: float) -> tuple[np.ndarray, np.ndarray]:
            state = car_sample(t)
            xyz = np.array([state["xy"][0], state["xy"][1], TABLE_Z], dtype=np.float64)
            return xyz, _yaw_quat(float(state["yaw"]))

        _bind_clock(env, pose_fn, duration=TOTAL, motion_scale=scale)

        def apply_extra_visuals() -> None:
            import sapien

            t = float(getattr(env, "flight_t", 0.0))
            state = car_sample(t)
            body = body_matrix(state, TABLE_Z)
            _set_actor_pose(
                env.object,
                sapien.Pose(body[:3, 3].tolist(), _rotmat_to_quat(body[:3, :3]).tolist()),
            )
            wheels = list(getattr(env, "wheels", []) or [])
            for wheel, hub, front in zip(wheels, HUBS, FRONT):
                mat = wheel_matrix(state, hub, TABLE_Z, front=front)
                _set_actor_pose(
                    wheel,
                    sapien.Pose(mat[:3, 3].tolist(), _rotmat_to_quat(mat[:3, :3]).tolist()),
                )

        env.apply_extra_visuals = apply_extra_visuals
        catch_t = _wall_time(float(INTERCEPT_T), scale)
        return ExtraMotionProfile(name, "chase", catch_t, _wall_time(TOTAL, scale) + 1.0, scale)

    if name == "collide_pool_balls":
        env.extra_motion_kind = "physx"
        env._extra_struck = False

        def reset_flight() -> None:
            env._extra_struck = False
            cue = getattr(env, "cue", None)
            obj = getattr(env, "object", None)
            if cue is not None:
                _set_velocity(cue, np.zeros(3), np.zeros(3))
            if obj is not None:
                _set_velocity(obj, np.zeros(3), np.zeros(3))

        def step_flight(dt: float) -> None:
            del dt
            if scale <= 0.0 or getattr(env, "_extra_struck", False):
                pin_extra_physics()
                return
            cue = getattr(env, "cue", None)
            if cue is None:
                return
            linear, angular = pool_strike_velocity(scale)
            _set_velocity(cue, linear, angular)
            env._extra_struck = True
            pin_extra_physics()

        def pin_extra_physics() -> None:
            dt = None
            try:
                dt = float(env.scene.get_timestep())
            except Exception:
                dt = 1.0 / 250.0
            for actor in (getattr(env, "cue", None), getattr(env, "object", None)):
                if actor is None or not hasattr(env, "pin_to_slate"):
                    continue
                env.pin_to_slate(actor, dt)

        env.reset_flight = reset_flight
        env.step_flight = step_flight
        env.pin_extra_physics = pin_extra_physics
        if hasattr(env, "flight_pose"):
            delattr(env, "flight_pose")
        return ExtraMotionProfile(name, "physics_demo", POOL_HORIZON_S, POOL_HORIZON_S, scale)

    raise KeyError(f"No native extra motion for {task_name!r}")
