#!/usr/bin/env python3
"""Cinematic demo: vinyl tangerine rolls toward the table edge, then is stopped.

Uses FastWAM's bundled RoboTwin. Does not touch the live official-eval tree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import sapien
import transforms3d as t3d
from PIL import Image, ImageDraw, ImageFont

for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

ROOT = setup_sys_path()

from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge

# Left-arm top-down / side-pinch quats (wxyz), same table as RoboTwin GRASP_DIRECTION_DIC.
_HOVER_QUATS = (
    [-0.61239, 0.353523, -0.61239, -0.353524],  # top_down_little_right
    [-0.5, 0.5, -0.5, -0.5],  # top_down
    [-0.853532, 0.146484, -0.353542, -0.3536],  # left_arm_perf
)

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TABLE_Z = 0.741
RADIUS = 0.038
_ALOHA_GRIPPER_SCALE = (-0.01, 0.045)


def _gripper_from_width(width_m: float, squeeze_m: float = 0.001) -> float:
    """Map object width to Aloha ``set_gripper`` in [0, 1]. 0 is a 2 cm overlap."""
    inner = max(float(width_m) - float(squeeze_m), 0.0)
    joint = 0.5 * inner
    lo, hi = _ALOHA_GRIPPER_SCALE
    return float(np.clip((joint - lo) / (hi - lo), 0.0, 1.0))


def _topdown_along(xy) -> list[float]:
    """TCP X down, TCP Y along world-XY ``xy`` (finger opening)."""
    y = np.array([float(xy[0]), float(xy[1]), 0.0], dtype=np.float64)
    y = y / (np.linalg.norm(y) + 1e-9)
    x = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-9)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return t3d.quaternions.mat2quat(R).tolist()


# Finger mesh is ~8.6 cm tall. Pads sit ~3.7 cm above TCP. Keep fingertips
# on the table and the pads around the equator; a deeper drop hid the pads.
GRASP_GRIP = _gripper_from_width(2.0 * RADIUS, squeeze_m=0.016)
_PAD_ALONG_X = -0.002
_PAD_WORLD_X = 0.011
_PAD_WORLD_Y = 0.000
# Seat on the pad midline. 8 mm cine-left bias opened the opposite gap.
_FRUIT_TOWARD_CINE_LEFT = 0.002
START_XY = np.array([0.22, -0.20], dtype=np.float64)
END_XY = np.array([-0.16, -0.02], dtype=np.float64)
ROLL_DURATION = 2.4
# Motion-only: keep rolling toward the camera-near edge and drop off.
FALL_END_XY = np.array([0.10, 0.42], dtype=np.float64)
FALL_DURATION = 2.45
G_CINE = 3.4
TABLE_HALF = np.array([0.60, 0.35], dtype=np.float64)
_DEFAULT_GROUPS = [1, 1, 0, 0]
CAPTION = "Stop the orange from rolling off"
CAPTION_MOTION = "Orange rolling off the table"


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype(_FONT, max(22, W // 36))
    except OSError:
        fnt = ImageFont.load_default()
    bar = int(fnt.size * (len(lines) + 0.8))
    draw.rectangle([0, 0, W, bar], fill=(18, 10, 6))
    for i, ln in enumerate(lines):
        draw.text((14, 8 + i * fnt.size), ln, fill=(255, 232, 196), font=fnt)
    return np.asarray(im)


def _travel() -> tuple[np.ndarray, float]:
    disp = END_XY - START_XY
    dist = float(np.linalg.norm(disp))
    direction = disp / max(dist, 1e-9)
    return direction, dist


def _roll_arc(t: float, dist: float, duration: float) -> float:
    """Arc length of a shoved orange: accelerate, then coast.

    Flat-table rolling is not a conveyor. A push gives a low initial speed,
    then it picks up over ~0.7 s and holds a coasting speed into the catch.
    """
    duration = max(float(duration), 1e-6)
    t = float(np.clip(t, 0.0, duration))
    dist = float(dist)
    v_start = 0.12
    tau = min(0.70, 0.35 * duration)
    rest = duration - tau
    # s(T) = v_start * T + a * tau * (0.5 * tau + rest)
    denom = tau * (0.5 * tau + rest)
    a = (dist - v_start * duration) / max(denom, 1e-9)
    if t <= tau:
        s = v_start * t + 0.5 * a * t * t
    else:
        v_c = v_start + a * tau
        s = v_start * tau + 0.5 * a * tau * tau + v_c * (t - tau)
    return float(np.clip(s, 0.0, dist))


def _leaf_up_pose(p) -> sapien.Pose:
    """Identity quat: stem/leaf along world +Z, sphere origin at ``p``."""
    p = np.asarray(p, dtype=np.float64).reshape(3)
    return sapien.Pose([float(p[0]), float(p[1]), float(p[2])], [1.0, 0.0, 0.0, 0.0])


def _finger_inner_x(env) -> tuple[float, float] | tuple[None, None]:
    """Inner faces of the two left-arm fingers along world X (cine left/right)."""
    a7 = a8 = None
    for ln in env.robot.left_entity.get_links():
        name = ln.get_name()
        if name == "fl_link7":
            a7 = np.asarray(ln.compute_global_aabb_tight(), dtype=np.float64)
        elif name == "fl_link8":
            a8 = np.asarray(ln.compute_global_aabb_tight(), dtype=np.float64)
    if a7 is None or a8 is None:
        return None, None
    # fl_link7 is the +X / cine-left finger; inner face is AABB min X.
    return float(a7[0, 0]), float(a8[1, 0])


def _seat_in_jaw(env, y: float, z: float) -> sapien.Pose:
    """Put the sphere between the closed pads, biased toward the cine-left gap."""
    left_inner, right_inner = _finger_inner_x(env)
    if left_inner is None:
        return _leaf_up_pose([END_XY[0], y, z])
    mid = 0.5 * (left_inner + right_inner)
    x = mid + float(_FRUIT_TOWARD_CINE_LEFT)
    return _leaf_up_pose([x, y, z])


def _qslerp(q0, q1, u: float) -> list[float]:
    a = np.asarray(q0, dtype=np.float64).reshape(4)
    b = np.asarray(q1, dtype=np.float64).reshape(4)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    if float(np.dot(a, b)) < 0.0:
        b = -b
    u = float(np.clip(u, 0.0, 1.0))
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if d > 0.9995:
        q = (1.0 - u) * a + u * b
        return (q / (np.linalg.norm(q) + 1e-12)).tolist()
    omega = float(np.arccos(d))
    so = float(np.sin(omega))
    q = (np.sin((1.0 - u) * omega) / so) * a + (np.sin(u * omega) / so) * b
    return (q / (np.linalg.norm(q) + 1e-12)).tolist()


def orange_pose(t: float, *, ease: bool = True) -> sapien.Pose:
    direction, dist = _travel()
    if ease:
        u = float(np.clip(t / ROLL_DURATION, 0.0, 1.0))
        u = u * u * (3.0 - 2.0 * u)
        s = u * dist
    else:
        s = _roll_arc(t, dist, ROLL_DURATION)
    xy = START_XY + direction * s
    axis = np.cross(np.array([0.0, 0.0, 1.0]), np.array([direction[0], direction[1], 0.0]))
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    angle = s / RADIUS
    q = t3d.quaternions.axangle2quat(axis, angle)
    return sapien.Pose(
        [float(xy[0]), float(xy[1]), TABLE_Z + RADIUS],
        q.tolist(),
    )


def orange_pose_fall(t: float) -> sapien.Pose:
    """Linear roll toward the +Y table edge, then ballistic drop in-frame."""
    disp = FALL_END_XY - START_XY
    dist = float(np.linalg.norm(disp))
    direction = disp / max(dist, 1e-9)
    speed = dist / FALL_DURATION
    s = speed * max(t, 0.0)
    xy = START_XY + direction * s
    on_table = bool(abs(xy[0]) < TABLE_HALF[0] and abs(xy[1]) < TABLE_HALF[1])
    z = TABLE_Z + RADIUS
    if not on_table:
        # Keep the fruit just past the rim so the drop stays in the cine frame.
        xy[1] = min(float(xy[1]), float(TABLE_HALF[1] + 0.04))
        if abs(direction[1]) > 1e-6:
            s_leave = (TABLE_HALF[1] - START_XY[1]) / direction[1]
        else:
            s_leave = dist
        t_air = max(0.0, s / speed - s_leave / speed)
        z -= 0.5 * G_CINE * t_air * t_air
        z = max(z, 0.12)
    axis = np.cross(np.array([0.0, 0.0, 1.0]), np.array([direction[0], direction[1], 0.0]))
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    q = t3d.quaternions.axangle2quat(axis, s / RADIUS)
    return sapien.Pose([float(xy[0]), float(xy[1]), float(z)], q.tolist())


def _rigid(actor):
    ent = getattr(actor, "actor", actor)
    try:
        return ent, ent.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    except Exception:
        return ent, None


def _set_kinematic(actor, on: bool) -> None:
    _ent, comp = _rigid(actor)
    if comp is None:
        return
    try:
        comp.set_kinematic(bool(on))
    except Exception:
        pass


def _set_pose(actor, pose: sapien.Pose) -> None:
    ent, _comp = _rigid(actor)
    try:
        ent.set_pose(pose)
    except Exception:
        pass


def _iter_shapes(actor):
    _ent, comp = _rigid(actor)
    if comp is None:
        return
    shapes = []
    try:
        shapes = list(comp.collision_shapes)
    except Exception:
        pass
    if not shapes:
        try:
            shapes = list(comp.get_collision_shapes())
        except Exception:
            shapes = []
    yield from shapes


def _set_groups(actor, groups: list[int]) -> None:
    for shape in _iter_shapes(actor):
        try:
            shape.set_collision_groups(list(groups))
        except Exception:
            pass


def _ghost_physics(actor, *, hard: bool = False) -> None:
    _set_kinematic(actor, True)
    _set_groups(actor, [0, 0, 0, 0])
    _ent, comp = _rigid(actor)
    if comp is None:
        return
    try:
        comp.set_disable_gravity(True)
    except Exception:
        pass
    if hard:
        try:
            comp.disable()
        except Exception:
            pass


def _restore_collision(actor) -> None:
    _set_kinematic(actor, True)
    _set_groups(actor, _DEFAULT_GROUPS)


def _ee_pose(env, arm: str = "left") -> sapien.Pose:
    p = np.asarray(env.get_arm_pose(arm), dtype=np.float64)
    return sapien.Pose(p[:3].tolist(), p[3:].tolist())


def _tcp(env, arm: str = "left") -> np.ndarray:
    if arm == "left":
        return np.asarray(env.robot.get_left_tcp_pose(), dtype=np.float64)[:3]
    return np.asarray(env.robot.get_right_tcp_pose(), dtype=np.float64)[:3]


def _weld(actor, env, rel: sapien.Pose, arm: str = "left") -> None:
    _set_pose(actor, _ee_pose(env, arm) * rel)


def _arm_q(env, arm: str = "left") -> np.ndarray:
    joints = env.robot.left_arm_joints if arm == "left" else env.robot.right_arm_joints
    return np.array([float(j.get_drive_target()[0]) for j in joints], dtype=np.float64)


def _arm_qpos(env, arm: str = "left") -> np.ndarray:
    if arm == "left":
        q = np.asarray(env.robot.get_left_arm_real_jointState(), dtype=np.float64)
    else:
        q = np.asarray(env.robot.get_right_arm_real_jointState(), dtype=np.float64)
    return q[:-1]


def _drive_arm(env, q: np.ndarray, arm: str = "left") -> None:
    env.robot.set_arm_joints(q, np.zeros_like(q), arm)


def _snap_arm(env, q: np.ndarray, arm: str = "left") -> None:
    """Drive target plus qpos so the cine arm actually reaches in real time."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    _drive_arm(env, q, arm)
    entity = env.robot.left_entity if arm == "left" else env.robot.right_entity
    joints = env.robot.left_arm_joints if arm == "left" else env.robot.right_arm_joints
    active = list(entity.get_active_joints())
    qpos = np.asarray(entity.get_qpos(), dtype=np.float64).copy()
    for i, joint in enumerate(joints):
        qpos[active.index(joint)] = float(q[i])
    entity.set_qpos(qpos)


def _joint_path(env, xyz: np.ndarray, quat: list[float]) -> np.ndarray | None:
    pose7 = _pose7(xyz, quat)
    planner = env.robot.left_plan_path
    try:
        res = planner(pose7)
    except Exception as exc:
        print(f"[stop-orange] plan failed: {exc}", flush=True)
        return None
    if not isinstance(res, dict) or res.get("status") != "Success":
        return None
    pos = res.get("position")
    if pos is None:
        return None
    path = np.asarray(pos, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2:
        return None
    return path


def _ik_q(env, pose7, arm: str = "left") -> np.ndarray | None:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(-1).tolist()
    planner = env.robot.left_plan_path if arm == "left" else env.robot.right_plan_path
    entity = env.robot.left_entity if arm == "left" else env.robot.right_entity
    try:
        res = planner(pose7, last_qpos=np.asarray(entity.get_qpos()))
    except Exception as exc:
        print(f"[stop-orange] IK failed: {exc}", flush=True)
        return None
    if not isinstance(res, dict) or res.get("status") != "Success":
        return None
    return np.asarray(res["position"][-1], dtype=np.float64)


def _pose7(xyz: np.ndarray, quat: list[float]) -> list[float]:
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), *[float(q) for q in quat]]


def _first_ik(env, xyz: np.ndarray, quats: tuple = _HOVER_QUATS) -> tuple[np.ndarray, list[float]] | tuple[None, None]:
    for quat in quats:
        q = _ik_q(env, _pose7(xyz, quat))
        if q is not None:
            return q, list(quat)
    return None, None


def _install_cine_camera(env, *, wide: bool = False) -> Any:
    fovy = 56.0 if wide else 48.0
    cam = env.scene.add_camera(
        name="cine_camera", width=1280, height=720, fovy=np.deg2rad(fovy), near=0.05, far=10.0
    )
    if wide:
        # Full roll in frame: launch (screen-left, +X) → intercept at the left arm.
        pos = np.array([0.02, 0.62, 1.14], dtype=np.float64)
        look = np.array([0.02, -0.08, 0.80], dtype=np.float64)
    else:
        # Sit just past the +Y table edge, centered on the intercept X so neither
        # pad sits between the camera and the fruit (that reads as left-finger clip).
        pos = np.array([-0.16, 0.54, 1.10], dtype=np.float64)
        look = np.array([-0.16, -0.04, 0.78], dtype=np.float64)
    forward = look - pos
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    left = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = np.cross(forward, left)
    up = up / (np.linalg.norm(up) + 1e-9)
    left = np.cross(up, forward)
    mat = np.eye(4)
    mat[:3, :3] = np.stack([forward, left, up], axis=1)
    mat[:3, 3] = pos
    cam.entity.set_pose(sapien.Pose(mat))
    return cam


def _grab_rgb(env, cine) -> dict[str, np.ndarray]:
    env._update_render()
    out: dict[str, np.ndarray] = {}
    try:
        obs = env.get_obs()
        out["head"] = np.ascontiguousarray(obs["observation"]["head_camera"]["rgb"])
    except Exception:
        pass
    try:
        cine.take_picture()
        rgba = cine.get_picture("Color")
        out["cine"] = (np.asarray(rgba)[..., :3] * 255).clip(0, 255).astype(np.uint8)
    except Exception:
        pass
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--robotwin-root",
        default=os.environ.get("ROBOTWIN_ROOT", "/DATA/YuanZhen/FastWAM/third_party/RoboTwin"),
    )
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out-dir", default="evaluate_results/demo_videos/stop_rolling_orange")
    p.add_argument("--no-grasp", action="store_true", help="Roll off the table; arms stay home.")
    p.add_argument("--tag", default="", help="Filename infix, e.g. teacher → stop_rolling_orange_teacher_cine.mp4")
    args = p.parse_args()

    root = Path(args.robotwin_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.pop("VK_ICD_FILENAMES", None)
    os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)

    tw_args = bridge.load_task_args(root, task_config="demo_clean", task_name="place_object_basket")
    tw_args["_robotwin_root"] = str(root)
    tw_args["task_name"] = "stop_rolling_orange"
    tw_args["eval_video_log"] = False
    tw_args["render_freq"] = 0
    tw_args.setdefault("data_type", {})
    tw_args["data_type"]["rgb"] = True
    tw_args["data_type"]["third_view"] = True

    cwd = os.getcwd()
    os.chdir(root)
    try:
        from envs.stop_rolling_orange import stop_rolling_orange
        from envs.utils.action import Action

        env = stop_rolling_orange()
        env.setup_demo(now_ep_num=0, seed=int(args.seed), is_test=True, **tw_args)
        env.set_instruction(instruction="Stop the orange from rolling off.")
    finally:
        os.chdir(cwd)

    orange = env.object
    _set_kinematic(orange, True)
    _ghost_physics(orange)
    _set_pose(orange, orange_pose(0.0, ease=False))
    cine = _install_cine_camera(env, wide=not bool(args.no_grasp))

    frames_cine: list[np.ndarray] = []
    frames_head: list[np.ndarray] = []
    dt = 1.0 / float(args.fps)
    caption = CAPTION_MOTION if args.no_grasp else CAPTION

    def snap(label: str, t: float) -> None:
        rgbs = _grab_rgb(env, cine)
        cine_f = rgbs["cine"] if "cine" in rgbs else rgbs.get("head")
        if cine_f is not None:
            frames_cine.append(_annotate(cine_f, [caption, f"{label}   t={t:.2f}s"]))
        if "head" in rgbs:
            frames_head.append(rgbs["head"])

    def hold(n: int, label: str, t0: float, freeze: sapien.Pose | None = None) -> float:
        t = t0
        for _ in range(n):
            if freeze is not None:
                _set_pose(orange, freeze)
            else:
                _set_pose(orange, orange_pose(0.0))
            for _s in range(8):
                env.scene.step()
            snap(label, t)
            t += dt
        return t

    if args.no_grasp:
        t = hold(int(0.40 * args.fps), "ready", 0.0, freeze=orange_pose_fall(0.0))
        roll_t = 0.0
        last = orange_pose_fall(0.0)
        while roll_t < FALL_DURATION:
            last = orange_pose_fall(roll_t)
            _set_pose(orange, last)
            for _ in range(8):
                env.scene.step()
            off = abs(last.p[1]) >= TABLE_HALF[1] or abs(last.p[0]) >= TABLE_HALF[0]
            snap("falling" if off else "rolling", roll_t)
            roll_t += dt
            t += dt
            # Cine camera sits past the +Y rim; going further exits the frame.
            if float(last.p[1]) >= TABLE_HALF[1] - 0.04:
                break
        t = hold(int(0.12 * args.fps), "off the table", t, freeze=last)
        cine_path = out_dir / "stop_rolling_orange_motion_cine.mp4"
        head_path = out_dir / "stop_rolling_orange_motion_head.mp4"
        if frames_cine:
            imageio.mimsave(cine_path.as_posix(), frames_cine, fps=args.fps, quality=8)
            frame_dir = out_dir / "frames_motion"
            frame_dir.mkdir(parents=True, exist_ok=True)
            picks = {
                "m_01": 0,
                "m_02": max(len(frames_cine) // 4, 0),
                "m_03": max(len(frames_cine) // 2, 0),
                "m_04": max((3 * len(frames_cine)) // 4, 0),
                "m_05": len(frames_cine) - 1,
            }
            for name, idx in picks.items():
                Image.fromarray(frames_cine[idx]).save(frame_dir / f"{name}.png")
        if frames_head:
            imageio.mimsave(head_path.as_posix(), frames_head, fps=args.fps, quality=8)
        meta = {
            "n_cine": len(frames_cine),
            "n_head": len(frames_head),
            "cine": str(cine_path),
            "head": str(head_path),
            "task": "stop_rolling_orange",
            "mode": "roll_off",
            "seed": args.seed,
            "start_xy": START_XY.tolist(),
            "end_xy": FALL_END_XY.tolist(),
        }
        (out_dir / "recording_meta_motion.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2), flush=True)
        try:
            env.close_env(clear_cache=True)
        except Exception:
            pass
        return 0 if frames_cine else 1

    q_home = _arm_qpos(env)
    intercept = orange_pose(ROLL_DURATION, ease=False)
    center = np.asarray(intercept.p, dtype=np.float64)
    # Cine looks from +Y; open along world X so pads flank the fruit, not sit in front.
    aligned = (_topdown_along([1.0, 0.0]), _topdown_along([-1.0, 0.0]))
    # Sphere origin, not the stem/leaf. Raise TCP so the visible pads hit the equator.
    pinch_xyz = center + np.array([_PAD_WORLD_X, _PAD_WORLD_Y, _PAD_ALONG_X], dtype=np.float64)
    hover_xyz = pinch_xyz + np.array([0.0, 0.0, 0.20], dtype=np.float64)
    pinch_quats = aligned + _HOVER_QUATS
    q_pinch, pinch_quat = _first_ik(env, pinch_xyz, pinch_quats)
    grasp_ok = q_pinch is not None
    print(
        f"[stop-orange] pinch={grasp_ok} grasp_grip={GRASP_GRIP:.3f} "
        f"center={center.round(3).tolist()} pinch={pinch_xyz.round(3).tolist()}",
        flush=True,
    )

    def _calibrate(xyz: np.ndarray, quat: list[float], q0: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray]:
        plan = np.asarray(xyz, dtype=np.float64).copy()
        if q0 is None:
            return None, plan
        _snap_arm(env, q0)
        for _s in range(4):
            env.scene.step()
            _snap_arm(env, q0)
        delta = _tcp(env) - xyz
        cmd = xyz - delta
        q_cal = _ik_q(env, _pose7(cmd, quat))
        print(
            f"[stop-orange] tcp-cal {np.asarray(xyz).round(3).tolist()} "
            f"delta={delta.round(3).tolist()} ok={q_cal is not None}",
            flush=True,
        )
        if q_cal is None:
            return q0, plan
        _snap_arm(env, q_cal)
        for _s in range(4):
            env.scene.step()
            _snap_arm(env, q_cal)
        print(
            f"[stop-orange] tcp-cal after={float(np.linalg.norm(_tcp(env) - xyz)):.3f} "
            f"tcp={_tcp(env).round(3).tolist()}",
            flush=True,
        )
        return q_cal, cmd

    plan_xyz = np.asarray(pinch_xyz, dtype=np.float64).copy()
    q_hover = None
    plan_hover = hover_xyz.copy()
    if q_pinch is not None:
        q_pinch, plan_xyz = _calibrate(pinch_xyz, pinch_quat, q_pinch)
        # Hover in planner frame so actual TCP clears the fruit; 20 cm
        # desired often exceeds IK, so step down until it succeeds.
        for dz in (0.18, 0.15, 0.12, 0.09):
            hcmd = np.asarray(plan_xyz, dtype=np.float64).copy()
            hcmd[1] += 0.07
            hcmd[2] += float(dz)
            q_try = _ik_q(env, _pose7(hcmd, pinch_quat))
            if q_try is None:
                continue
            q_hover = q_try
            plan_hover = hcmd
            _snap_arm(env, q_hover)
            for _s in range(4):
                env.scene.step()
                _snap_arm(env, q_hover)
            print(
                f"[stop-orange] hover dz={dz:.2f} tcp={_tcp(env).round(3).tolist()}",
                flush=True,
            )
            break
        _snap_arm(env, q_home)
        _set_pose(orange, orange_pose(0.0, ease=False))

    hover_target = plan_hover if q_hover is not None else plan_xyz
    hover_q = q_hover if q_hover is not None else q_pinch
    path = _joint_path(env, hover_target, pinch_quat) if hover_q is not None else None
    if path is not None:
        d = np.linalg.norm(path - q_home.reshape(1, -1), axis=1)
        k0 = int(next((i for i, v in enumerate(d) if v > 0.04), 0))
        path = path[max(k0 - 1, 0) :]
    if q_pinch is None:
        print("[stop-orange] no pinch IK", flush=True)
    _snap_arm(env, q_home)
    env.robot.set_gripper(1.0, "left", gripper_eps=0.0)
    _set_pose(orange, orange_pose(0.0, ease=False))
    for _s in range(8):
        env.scene.step()
        _snap_arm(env, q_home)
        _set_pose(orange, orange_pose(0.0, ease=False))

    n_reach = max(int(round(ROLL_DURATION * args.fps)), 20)
    n_drop = max(int(0.18 * args.fps), 5)
    n_close_fly = max(int(0.16 * args.fps), 5)
    pd_steps = 4

    def _arm_at(i: int) -> np.ndarray:
        n_pre = max(n_reach - n_drop, 1)
        if i < n_pre:
            a = float(i + 1) / float(n_pre)
            if path is not None:
                idx = int(round(min(1.0, a) * float(len(path) - 1)))
                idx = int(np.clip(idx, 0, len(path) - 1))
                return path[idx]
            if hover_q is None:
                return q_home
            return (1.0 - a) * q_home + a * hover_q
        b = float(i + 1 - n_pre) / float(n_drop)
        b = float(np.clip(b, 0.0, 1.0))
        # Stay high until the last beat, then drop onto the fruit.
        b = b * b
        q_from = hover_q if hover_q is not None else q_home
        q_to = q_pinch if q_pinch is not None else q_from
        return (1.0 - b) * q_from + b * q_to

    t = 0.0
    for i in range(n_reach):
        t_roll = float(i + 1) * dt
        frac = float(i + 1) / float(n_reach)
        q_now = _arm_at(i)
        pose = orange_pose(t_roll, ease=False)
        grip = 1.0
        n_pre = max(n_reach - n_drop, 1)
        n_upright = max(n_pre - int(0.30 * args.fps), 0)
        phase = "rolling"
        if i >= n_upright:
            uu = float(i + 1 - n_upright) / float(max(n_pre - n_upright, 1))
            uu = float(np.clip(uu, 0.0, 1.0))
            uu = uu * uu * (3.0 - 2.0 * uu)
            up = _leaf_up_pose(pose.p)
            pose = sapien.Pose(list(pose.p), _qslerp(pose.q, up.q, uu))
        if i >= n_pre:
            phase = "intercept"
        if i >= n_reach - n_close_fly:
            a = float(i + 1 - (n_reach - n_close_fly)) / float(n_close_fly)
            a = float(np.clip(a, 0.0, 1.0))
            grip = (1.0 - a) * 1.0 + a * GRASP_GRIP
            phase = "catch"
            p = np.asarray(pose.p, dtype=np.float64).copy()
            p[0] += a * float(_FRUIT_TOWARD_CINE_LEFT)
            pose = sapien.Pose(p.tolist(), list(pose.q))
        _snap_arm(env, q_now)
        env.robot.set_gripper(float(grip), "left", gripper_eps=0.0)
        _set_pose(orange, pose)
        for _s in range(pd_steps):
            env.scene.step()
            _snap_arm(env, q_now)
            _set_pose(orange, pose)
        if i < 6 or i in (15, 21, 30, n_pre - 1, n_reach - 1):
            _, dist = _travel()
            s1 = _roll_arc(t_roll, dist, ROLL_DURATION)
            s0 = _roll_arc(max(t_roll - dt, 0.0), dist, ROLL_DURATION)
            print(
                f"[stop-orange] frame{i} t={t_roll:.3f} v={float((s1 - s0) / dt):.3f} "
                f"orange={np.asarray(pose.p).round(3).tolist()} "
                f"tcp={_tcp(env).round(3).tolist()} "
                f"arm_move={float(np.linalg.norm(q_now - q_home)):.3f}",
                flush=True,
            )
        snap(phase, t_roll)
        t = t_roll

    intercept = _leaf_up_pose(orange_pose(ROLL_DURATION, ease=False).p)
    _set_pose(orange, intercept)
    if q_pinch is not None:
        _snap_arm(env, q_pinch)
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        for _s in range(pd_steps):
            env.scene.step()
            _snap_arm(env, q_pinch)
            _set_pose(orange, intercept)
        intercept = _seat_in_jaw(env, float(intercept.p[1]), float(intercept.p[2]))
        _set_pose(orange, intercept)
        for _s in range(pd_steps):
            env.scene.step()
            _snap_arm(env, q_pinch)
            _set_pose(orange, intercept)
    tcp = _tcp(env)
    pinch = float(np.linalg.norm(tcp - pinch_xyz))
    left_inner, right_inner = _finger_inner_x(env)
    print(
        f"[stop-orange] pinch tcp-target={pinch:.3f} grip={GRASP_GRIP:.3f} "
        f"tcp={tcp.round(3).tolist()} center={center.round(3).tolist()} "
        f"held={np.asarray(intercept.p).round(3).tolist()} "
        f"innerX={[None if left_inner is None else round(left_inner, 3), None if right_inner is None else round(right_inner, 3)]}",
        flush=True,
    )

    n_close = max(int(0.10 * args.fps), 3)
    for _ in range(n_close):
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        if q_pinch is not None:
            _snap_arm(env, q_pinch)
        _set_pose(orange, intercept)
        for _s in range(8):
            env.scene.step()
            if q_pinch is not None:
                _snap_arm(env, q_pinch)
            _set_pose(orange, intercept)
        snap("catch", t)
        t += dt

    _ghost_physics(orange, hard=True)
    # Weld the sphere between the pads. Leaf stays world-up, out of the jaw.
    rel = _ee_pose(env).inv() * intercept
    _weld(orange, env, rel)

    def _step_visual(label: str, t_now: float, rel_pose: sapien.Pose, q: np.ndarray | None = None) -> None:
        if q is not None:
            _snap_arm(env, q)
        _weld(orange, env, rel_pose)
        for _s in range(8):
            env.scene.step()
            if q is not None:
                _snap_arm(env, q)
            _weld(orange, env, rel_pose)
        snap(label, t_now)

    # Pull back toward table center — the "saved from the edge" beat.
    direction, _dist = _travel()
    pull = -0.12 * direction
    ee0 = np.asarray(env.get_arm_pose("left"), dtype=np.float64)
    ee_pull = ee0.copy()
    ee_pull[0] += float(pull[0])
    ee_pull[1] += float(pull[1])
    q_pull = _ik_q(env, ee_pull)
    q_now = _arm_qpos(env)
    if q_pull is None:
        print("[stop-orange] pull-back IK failed", flush=True)
        q_pull = q_now
    n_pull = int(0.7 * args.fps)
    for i in range(max(n_pull, 1)):
        a = float(i + 1) / float(max(n_pull, 1))
        q_i = (1.0 - a) * q_now + a * q_pull
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        _step_visual("pull back", t, rel, q_i)
        t += dt

    ee0 = np.asarray(env.get_arm_pose("left"), dtype=np.float64)
    ee1 = ee0.copy()
    ee1[2] += 0.08
    q_lift = _ik_q(env, ee1)
    q_now = _arm_qpos(env)
    if q_lift is None:
        print("[stop-orange] lift IK failed, holding grasp pose", flush=True)
        q_lift = q_now
    n_lift = int(0.70 * args.fps)
    for i in range(max(n_lift, 1)):
        a = float(i + 1) / float(max(n_lift, 1))
        q_i = (1.0 - a) * q_now + a * q_lift
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        _step_visual("saved", t, rel, q_i)
        t += dt

    q_hold = _arm_qpos(env)
    for _ in range(int(0.45 * args.fps)):
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        _step_visual("saved", t, rel, q_hold)
        t += dt

    tag = f"_{args.tag}" if str(args.tag).strip() else ""
    cine_path = out_dir / f"stop_rolling_orange{tag}_cine.mp4"
    head_path = out_dir / f"stop_rolling_orange{tag}_head.mp4"
    if frames_cine:
        imageio.mimsave(cine_path.as_posix(), frames_cine, fps=args.fps, quality=8)
        frame_dir = out_dir / f"frames{tag or ''}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        picks = {
            "c_01": 0,
            "c_02": max(len(frames_cine) // 4, 0),
            "c_03": max(len(frames_cine) // 2, 0),
            "c_04": max((3 * len(frames_cine)) // 4, 0),
            "c_05": len(frames_cine) - 1,
            "catch": min(max(int(round(ROLL_DURATION * args.fps)) - 1, 0), len(frames_cine) - 1),
            "pull_283": min(max(int(round(2.83 * args.fps)) - 1, 0), len(frames_cine) - 1),
        }
        for name, idx in picks.items():
            Image.fromarray(frames_cine[idx]).save(frame_dir / f"{name}.png")
    if frames_head:
        imageio.mimsave(head_path.as_posix(), frames_head, fps=args.fps, quality=8)
    meta = {
        "n_cine": len(frames_cine),
        "n_head": len(frames_head),
        "cine": str(cine_path),
        "head": str(head_path),
        "task": "stop_rolling_orange",
        "mode": "teacher_roll_and_stop" if str(args.tag).strip() == "teacher" else "roll_and_stop",
        "seed": args.seed,
        "grasp_ok": grasp_ok,
        "pinch_tcp_target_m": pinch,
        "grasp_grip": GRASP_GRIP,
        "start_xy": START_XY.tolist(),
        "end_xy": END_XY.tolist(),
    }
    (out_dir / "recording_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    try:
        env.close_env(clear_cache=True)
    except Exception:
        pass
    return 0 if frames_cine else 1


if __name__ == "__main__":
    raise SystemExit(main())
