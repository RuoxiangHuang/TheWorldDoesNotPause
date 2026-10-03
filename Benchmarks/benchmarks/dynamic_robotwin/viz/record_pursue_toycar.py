#!/usr/bin/env python3
"""Cinematic extra: chase a steered die-cast coupe (not base-subset rails).

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
from benchmarks.dynamic_robotwin.viz.diecast_drive import (
    FRONT,
    HUBS,
    IDLE,
    PATH,
    TOTAL,
    body_matrix,
    cabin_world,
    wheel_matrix,
)

_HOVER_QUATS = (
    [-0.61239, 0.353523, -0.61239, -0.353524],
    [-0.5, 0.5, -0.5, -0.5],
    [-0.853532, 0.146484, -0.353542, -0.3536],
)

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TABLE_Z = 0.741
# Hull Y radius 2.3 cm; pads touch at joint=0 so inner gap ≈ 2 * joint.
CAR_PINCH_WIDTH = 0.046
# TCP at hull mid-height so fingertips sit on the cabin flanks, not the wheels.
PINCH_Z = TABLE_Z + 0.056
HOVER_ABOVE = 0.14
# Ease onto the car (lag shrinks), lock briefly, then drop on a low-curvature beat.
LOCK_END = 1.85
FOLLOW_END = 2.42
DROP_DUR = 0.24
CATCH_T = FOLLOW_END + DROP_DUR
PARK_XY = np.array([0.04, -0.12], dtype=np.float64)
_ALOHA_GRIPPER_SCALE = (-0.01, 0.045)
_DEFAULT_GROUPS = [1, 1, 0, 0]
CAPTION = "Chase the toy car"
CAPTION_MOTION = "Toy car slalom"


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


GRASP_GRIP = _gripper_from_width(CAR_PINCH_WIDTH, squeeze_m=-0.008)


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype(_FONT, max(22, W // 36))
    except OSError:
        fnt = ImageFont.load_default()
    bar = int(fnt.size * (len(lines) + 0.8))
    draw.rectangle([0, 0, W, bar], fill=(12, 8, 10))
    for i, ln in enumerate(lines):
        draw.text((14, 8 + i * fnt.size), ln, fill=(255, 220, 210), font=fnt)
    return np.asarray(im)


def _pose_T(T: np.ndarray) -> sapien.Pose:
    T = np.asarray(T, dtype=np.float64)
    q = t3d.quaternions.mat2quat(T[:3, :3])
    return sapien.Pose(T[:3, 3].tolist(), q.tolist())


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
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    _drive_arm(env, q, arm)
    entity = env.robot.left_entity if arm == "left" else env.robot.right_entity
    joints = env.robot.left_arm_joints if arm == "left" else env.robot.right_arm_joints
    active = list(entity.get_active_joints())
    qpos = np.asarray(entity.get_qpos(), dtype=np.float64).copy()
    for i, joint in enumerate(joints):
        qpos[active.index(joint)] = float(q[i])
    entity.set_qpos(qpos)


def _ik_q(env, pose7, arm: str = "left") -> np.ndarray | None:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(-1).tolist()
    planner = env.robot.left_plan_path if arm == "left" else env.robot.right_plan_path
    entity = env.robot.left_entity if arm == "left" else env.robot.right_entity
    try:
        res = planner(pose7, last_qpos=np.asarray(entity.get_qpos()))
    except Exception as exc:
        print(f"[pursue-toycar] IK failed: {exc}", flush=True)
        return None
    if not isinstance(res, dict) or res.get("status") != "Success":
        return None
    return np.asarray(res["position"][-1], dtype=np.float64)


def _pose7(xyz: np.ndarray, quat: list[float]) -> list[float]:
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), *[float(q) for q in quat]]


def _first_ik(env, xyz: np.ndarray, quats: tuple = _HOVER_QUATS):
    for quat in quats:
        q = _ik_q(env, _pose7(xyz, quat))
        if q is not None:
            return q, list(quat)
    return None, None


def _width_dir(yaw: float) -> np.ndarray:
    return np.array([-np.sin(float(yaw)), np.cos(float(yaw))], dtype=np.float64)


def _car_hover(state: dict) -> np.ndarray:
    cabin = cabin_world(state, TABLE_Z)
    hover = cabin.copy()
    hover[2] = PINCH_Z + HOVER_ABOVE
    return hover


def _car_pinch(state: dict) -> np.ndarray:
    cabin = cabin_world(state, TABLE_Z)
    pinch = cabin.copy()
    pinch[2] = PINCH_Z
    yaw = float(state["yaw"])
    fwd = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    w = _width_dir(yaw)
    # Rearward onto the cabin, tiny +width so the cine-near pad stays outside.
    pinch[0] += -0.012 * float(fwd[0]) + 0.004 * float(w[0])
    pinch[1] += -0.012 * float(fwd[1]) + 0.004 * float(w[1])
    return pinch


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _chase_aim(t: float) -> tuple[np.ndarray, dict, float]:
    """Hover that starts at a park pose and eases onto the (lagged) car."""
    t = float(t)
    u = _smoothstep((t - IDLE) / max(LOCK_END - IDLE, 1e-6))
    if t >= LOCK_END:
        u = 1.0
    lag = (1.0 - u) * 0.36
    st = PATH.sample(max(IDLE, t - lag))
    hover = _car_hover(st)
    park = np.array([PARK_XY[0], PARK_XY[1], PINCH_Z + HOVER_ABOVE + 0.07], dtype=np.float64)
    xyz = (1.0 - u) * park + u * hover
    return xyz, st, u


def _chase_quats(state: dict, u: float) -> tuple:
    if u < 0.45:
        return (_topdown_along([1.0, 0.0]), _topdown_along([-1.0, 0.0])) + _HOVER_QUATS
    return _car_quats(state)


def _car_quat(state: dict) -> list[float]:
    return _topdown_along(_width_dir(state["yaw"]))


def _car_quats(state: dict) -> tuple:
    w = _width_dir(state["yaw"])
    return (_topdown_along(w), _topdown_along(-w)) + _HOVER_QUATS


def _blend_xyz(state: dict, drop_a: float) -> np.ndarray:
    a = float(np.clip(drop_a, 0.0, 1.0))
    a = a * a
    hover = _car_hover(state)
    pinch = _car_pinch(state)
    return (1.0 - a) * hover + a * pinch


def _try_ik_cal(env, xyz, quats, q_hint=None, max_jump: float | None = 2.2):
    """IK in planner frame so actual TCP lands on ``xyz``.

    Prefer the joint solution closest to ``q_hint`` so chase interpolation
    does not swing through a different IK branch.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    best = None
    for quat in quats:
        if q_hint is not None:
            _snap_arm(env, q_hint)
            for _ in range(2):
                env.scene.step()
                _snap_arm(env, q_hint)
        q = _ik_q(env, _pose7(xyz, list(quat)))
        if q is None:
            continue
        _snap_arm(env, q)
        for _ in range(4):
            env.scene.step()
            _snap_arm(env, q)
        delta = _tcp(env) - xyz
        cmd = xyz - delta
        if q_hint is not None:
            _snap_arm(env, q_hint)
            for _ in range(2):
                env.scene.step()
                _snap_arm(env, q_hint)
        q_cal = _ik_q(env, _pose7(cmd, list(quat)))
        if q_cal is None:
            continue
        _snap_arm(env, q_cal)
        for _ in range(3):
            env.scene.step()
            _snap_arm(env, q_cal)
        err = float(np.linalg.norm(_tcp(env) - xyz))
        jump = 0.0 if q_hint is None else float(np.linalg.norm(q_cal - q_hint))
        cand = (jump, q_cal, list(quat), np.asarray(cmd, dtype=np.float64), err)
        if best is None or jump < best[0]:
            best = cand
        if q_hint is None or jump <= 0.85:
            break
    if best is None:
        return None, None, xyz, None
    jump, q_cal, quat, cmd, err = best
    if max_jump is not None and q_hint is not None and jump > max_jump:
        print(f"[pursue-toycar] skip jump={jump:.2f} xyz={xyz.round(3).tolist()}", flush=True)
        return None, None, xyz, None
    _snap_arm(env, q_cal)
    return q_cal, quat, cmd, err


def _install_cine_camera(env) -> Any:
    cam = env.scene.add_camera(
        name="cine_camera", width=1280, height=720, fovy=np.deg2rad(56.0), near=0.05, far=10.0
    )
    pos = np.array([0.04, 0.64, 1.16], dtype=np.float64)
    look = np.array([-0.02, -0.04, 0.82], dtype=np.float64)
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
    p.add_argument("--out-dir", default="evaluate_results/demo_videos/pursue_toycar")
    p.add_argument("--no-grasp", action="store_true", help="Drive only; arms stay home.")
    p.add_argument("--tag", default="", help="Filename infix, e.g. teacher → pursue_toycar_teacher_cine.mp4")
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
    tw_args["task_name"] = "pursue_toycar"
    tw_args["eval_video_log"] = False
    tw_args["render_freq"] = 0
    tw_args.setdefault("data_type", {})
    tw_args["data_type"]["rgb"] = True
    tw_args["data_type"]["third_view"] = True

    cwd = os.getcwd()
    os.chdir(root)
    try:
        from envs.pursue_toycar import pursue_toycar

        env = pursue_toycar()
        env.setup_demo(now_ep_num=0, seed=int(args.seed), is_test=True, **tw_args)
        env.set_instruction(instruction="Catch the toy car.")
    finally:
        os.chdir(cwd)

    car = env.object
    wheels = list(getattr(env, "wheels", []) or [])
    _set_kinematic(car, True)
    _ghost_physics(car)
    for w in wheels:
        _ghost_physics(w)

    def seat(state: dict) -> sapien.Pose:
        bp = _pose_T(body_matrix(state, TABLE_Z))
        _set_pose(car, bp)
        for i, w in enumerate(wheels):
            if i >= len(HUBS):
                break
            _set_pose(w, _pose_T(wheel_matrix(state, HUBS[i], TABLE_Z, front=FRONT[i])))
        return bp

    st0 = PATH.sample(0.0)
    seat(st0)
    cine = _install_cine_camera(env)

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

    def hold_motion(n: int, label: str, video_t: float, motion_t: float) -> float:
        st = PATH.sample(motion_t)
        for _ in range(n):
            seat(st)
            for _s in range(8):
                env.scene.step()
            snap(label, video_t)
            video_t += dt
        return video_t

    def drive_motion(m_end: float, label: str, video_t: float, motion_t: float) -> tuple[float, float]:
        while motion_t < m_end - 1e-9:
            seat(PATH.sample(motion_t))
            for _s in range(8):
                env.scene.step()
            snap(label, video_t)
            video_t += dt
            motion_t += dt
        return video_t, motion_t

    st_catch = PATH.sample(CATCH_T)
    cabin_catch = cabin_world(st_catch, TABLE_Z)
    print(
        f"[pursue-toycar] catch_t={CATCH_T:.2f}s lock_end={LOCK_END:.2f}s follow_end={FOLLOW_END:.2f}s "
        f"xy={np.asarray(st_catch['xy']).round(3).tolist()} yaw={float(st_catch['yaw']):.2f} "
        f"cabin={cabin_catch.round(3).tolist()} path_len={PATH.length:.3f}m "
        f"heading_span={float(PATH.heading[-1]-PATH.heading[0]):.2f}rad idle={IDLE:.2f}s",
        flush=True,
    )

    tag = f"_{str(args.tag).strip()}" if str(args.tag).strip() else ""

    if args.no_grasp:
        t, _mt = drive_motion(TOTAL, "driving", 0.0, IDLE)
        t = hold_motion(int(0.35 * args.fps), "escaped", t, TOTAL)
        cine_path = out_dir / f"pursue_toycar_motion{tag}_cine.mp4"
        head_path = out_dir / f"pursue_toycar_motion{tag}_head.mp4"
        meta_name = "recording_meta_motion.json"
        frame_dir = out_dir / "frames_motion"
        prefix = "m"
        mode = "drive_only"
        hover_ok = False
        grasp_ok = False
        pinch = None
    else:
        q_home = _arm_qpos(env)
        env.robot.set_gripper(1.0, "left", gripper_eps=0.0)

        blend_ts = np.linspace(IDLE + 0.12, LOCK_END, 10)
        lock_ts = np.linspace(LOCK_END, FOLLOW_END, 6)
        drop_ts = np.linspace(FOLLOW_END, CATCH_T, 8)
        key_t: list[float] = [IDLE]
        key_q: list[np.ndarray] = [q_home]
        q_hint = q_home
        n_hover_ok = 0
        for kt in blend_ts:
            xyz, st, u = _chase_aim(float(kt))
            seat(PATH.sample(float(kt)))
            q, quat, _cmd, err = _try_ik_cal(
                env, xyz, _chase_quats(st, u), q_hint, max_jump=None if n_hover_ok == 0 else 2.2
            )
            if q is None:
                print(f"[pursue-toycar] blend IK fail t={float(kt):.2f} u={u:.2f}", flush=True)
                continue
            key_t.append(float(kt))
            key_q.append(q)
            jump = float(np.linalg.norm(q - q_hint))
            q_hint = q
            n_hover_ok += 1
            print(
                f"[pursue-toycar] blend t={float(kt):.2f} u={u:.2f} aim={xyz.round(3).tolist()} "
                f"car={st['xy'].round(3).tolist()} tcp-err={err:.3f} jump={jump:.2f} "
                f"tcp={_tcp(env).round(3).tolist()}",
                flush=True,
            )

        for kt in lock_ts:
            st = PATH.sample(float(kt))
            seat(st)
            q, quat, _cmd, err = _try_ik_cal(
                env, _car_hover(st), _car_quats(st), q_hint, max_jump=2.2
            )
            if q is None:
                print(f"[pursue-toycar] lock IK fail t={float(kt):.2f}", flush=True)
                continue
            key_t.append(float(kt))
            key_q.append(q)
            jump = float(np.linalg.norm(q - q_hint))
            q_hint = q
            n_hover_ok += 1
            print(
                f"[pursue-toycar] lock t={float(kt):.2f} xy={st['xy'].round(3).tolist()} "
                f"tcp-err={err:.3f} jump={jump:.2f} tcp={_tcp(env).round(3).tolist()}",
                flush=True,
            )

        q_pinch = None
        pinch_xyz = _car_pinch(st_catch)
        n_pinch_ok = 0
        for kt in drop_ts:
            st = PATH.sample(float(kt))
            drop_a = 0.0 if CATCH_T <= FOLLOW_END else (float(kt) - FOLLOW_END) / DROP_DUR
            xyz = _blend_xyz(st, drop_a)
            seat(st)
            quats = _car_quats(st)
            q, quat, _cmd, err = _try_ik_cal(env, xyz, quats, q_hint, max_jump=2.2)
            if q is None:
                print(f"[pursue-toycar] drop IK fail t={float(kt):.2f} a={drop_a:.2f}", flush=True)
                continue
            key_t.append(float(kt))
            key_q.append(q)
            jump = float(np.linalg.norm(q - q_hint))
            q_hint = q
            n_pinch_ok += 1
            if drop_a >= 0.99:
                q_pinch = q
                pinch_xyz = _car_pinch(st)
            print(
                f"[pursue-toycar] drop t={float(kt):.2f} a={drop_a:.2f} "
                f"tcp-err={err:.3f} jump={jump:.2f} tcp={_tcp(env).round(3).tolist()}",
                flush=True,
            )

        hover_ok = n_hover_ok > 0
        grasp_ok = q_pinch is not None
        print(
            f"[pursue-toycar] hover_keys={n_hover_ok} drop_keys={n_pinch_ok} pinch={grasp_ok} "
            f"grasp_grip={GRASP_GRIP:.3f}",
            flush=True,
        )

        _snap_arm(env, q_home)
        env.robot.set_gripper(1.0, "left", gripper_eps=0.0)
        seat(PATH.sample(IDLE))
        for _s in range(8):
            env.scene.step()
            _snap_arm(env, q_home)
            seat(PATH.sample(IDLE))

        kt = np.asarray(key_t, dtype=np.float64) if key_t else np.array([IDLE], dtype=np.float64)
        kq = np.stack(key_q, axis=0) if key_q else q_home.reshape(1, -1)

        def _q_at(motion_t: float) -> np.ndarray:
            if len(kt) == 1:
                return kq[0]
            tq = float(np.clip(motion_t, float(kt[0]), float(kt[-1])))
            i = int(np.searchsorted(kt, tq, side="right") - 1)
            i = max(0, min(i, len(kt) - 2))
            span = max(float(kt[i + 1] - kt[i]), 1e-6)
            a = float(np.clip((tq - kt[i]) / span, 0.0, 1.0))
            return (1.0 - a) * kq[i] + a * kq[i + 1]

        n_chase = max(int(round((CATCH_T - IDLE) * args.fps)), 8)
        n_close_fly = 4
        pd_steps = 4
        t = 0.0
        last_q = q_home
        last_st = PATH.sample(IDLE)
        for i in range(n_chase):
            motion_t = IDLE + float(i + 1) * dt
            motion_t = min(motion_t, CATCH_T)
            st = PATH.sample(motion_t)
            q_now = _q_at(motion_t)
            last_q = q_now
            last_st = st
            grip = 1.0
            phase = "reach"
            if motion_t >= 0.55 * LOCK_END + 0.45 * IDLE:
                phase = "chase"
            if motion_t >= LOCK_END:
                phase = "chase"
            if motion_t >= FOLLOW_END:
                phase = "intercept"
            if motion_t >= CATCH_T - n_close_fly * dt:
                a = float(i + 1 - (n_chase - n_close_fly)) / float(max(n_close_fly, 1))
                a = float(np.clip(a, 0.0, 1.0))
                grip = (1.0 - a) * 1.0 + a * GRASP_GRIP
                phase = "catch"
            _snap_arm(env, q_now)
            env.robot.set_gripper(float(grip), "left", gripper_eps=0.0)
            seat(st)
            for _s in range(pd_steps):
                env.scene.step()
                _snap_arm(env, q_now)
                seat(st)
            if i < 4 or i in (10, 20, 30, 40, n_chase - 1):
                print(
                    f"[pursue-toycar] frame{i} t={motion_t:.3f} xy={st['xy'].round(3).tolist()} "
                    f"tcp={_tcp(env).round(3).tolist()} arm_move={float(np.linalg.norm(q_now - q_home)):.3f}",
                    flush=True,
                )
            snap(phase, motion_t)
            t = motion_t

        if q_pinch is not None:
            _snap_arm(env, q_pinch)
            last_q = q_pinch
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        catch_pose = _pose_T(body_matrix(last_st, TABLE_Z))
        seat(last_st)
        for _s in range(pd_steps):
            env.scene.step()
            _snap_arm(env, last_q)
            seat(last_st)
        tcp = _tcp(env)
        pinch = float(np.linalg.norm(tcp - pinch_xyz))
        print(
            f"[pursue-toycar] pinch tcp-target={pinch:.3f} grip={GRASP_GRIP:.3f} "
            f"tcp={tcp.round(3).tolist()} cabin={cabin_world(last_st, TABLE_Z).round(3).tolist()}",
            flush=True,
        )

        n_close = max(int(0.08 * args.fps), 2)
        for _ in range(n_close):
            env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
            _snap_arm(env, last_q)
            seat(last_st)
            for _s in range(8):
                env.scene.step()
                _snap_arm(env, last_q)
                seat(last_st)
            snap("catch", t)
            t += dt

        _ghost_physics(car, hard=True)
        for w in wheels:
            _ghost_physics(w, hard=True)
        rel = _ee_pose(env).inv() * catch_pose
        wheel_local = []
        for w in wheels:
            ent, _ = _rigid(w)
            wheel_local.append(catch_pose.inv() * ent.get_pose())

        def _step_visual(label: str, t_now: float, q: np.ndarray) -> None:
            _snap_arm(env, q)
            _weld(car, env, rel)
            bp = _ee_pose(env) * rel
            for w, wrel in zip(wheels, wheel_local):
                _set_pose(w, bp * wrel)
            for _s in range(8):
                env.scene.step()
                _snap_arm(env, q)
                _weld(car, env, rel)
                bp = _ee_pose(env) * rel
                for w, wrel in zip(wheels, wheel_local):
                    _set_pose(w, bp * wrel)
            snap(label, t_now)

        ee0 = np.asarray(env.get_arm_pose("left"), dtype=np.float64)
        ee1 = ee0.copy()
        ee1[2] += 0.10
        q_lift = _ik_q(env, ee1)
        q_now = _arm_qpos(env)
        if q_lift is None:
            print("[pursue-toycar] lift IK failed", flush=True)
            q_lift = q_now
        n_lift = int(0.75 * args.fps)
        for i in range(max(n_lift, 1)):
            a = float(i + 1) / float(max(n_lift, 1))
            q_i = (1.0 - a) * q_now + a * q_lift
            env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
            _step_visual("lift", t, q_i)
            t += dt
        q_hold = _arm_qpos(env)
        for _ in range(int(0.45 * args.fps)):
            env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
            _step_visual("caught", t, q_hold)
            t += dt
        cine_path = out_dir / f"pursue_toycar{tag}_cine.mp4"
        head_path = out_dir / f"pursue_toycar{tag}_head.mp4"
        meta_name = "recording_meta.json"
        frame_dir = out_dir / f"frames{tag or ''}"
        prefix = "c"
        mode = "teacher_chase_and_grasp" if str(args.tag).strip() == "teacher" else "pursue_and_grasp"

    if frames_cine:
        imageio.mimsave(cine_path.as_posix(), frames_cine, fps=args.fps, quality=8)
        frame_dir.mkdir(parents=True, exist_ok=True)
        picks = {
            f"{prefix}_01": 0,
            f"{prefix}_02": max(len(frames_cine) // 4, 0),
            f"{prefix}_03": max(len(frames_cine) // 2, 0),
            f"{prefix}_04": max((3 * len(frames_cine)) // 4, 0),
            f"{prefix}_05": len(frames_cine) - 1,
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
        "task": "pursue_toycar",
        "mode": mode,
        "seed": args.seed,
        "path_length_m": PATH.length,
        "heading_span_rad": float(PATH.heading[-1] - PATH.heading[0]),
        "catch_t": CATCH_T,
        "follow_end": FOLLOW_END,
        "lock_end": LOCK_END,
        "n_wheels": len(wheels),
    }
    if not args.no_grasp:
        meta["hover_ok"] = hover_ok
        meta["grasp_ok"] = grasp_ok
        meta["pinch_tcp_target_m"] = pinch
        meta["grasp_grip"] = GRASP_GRIP
        meta["car_pinch_width_m"] = CAR_PINCH_WIDTH
    (out_dir / meta_name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    try:
        env.close_env(clear_cache=True)
    except Exception:
        pass
    return 0 if frames_cine else 1


if __name__ == "__main__":
    raise SystemExit(main())
