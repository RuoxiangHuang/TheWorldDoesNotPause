#!/usr/bin/env python3
"""Cinematic extra: catch a hopping vinyl bunny mid-air by the torso.

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

_HOVER_QUATS = (
    [-0.61239, 0.353523, -0.61239, -0.353524],
    [-0.5, 0.5, -0.5, -0.5],
    [-0.853532, 0.146484, -0.353542, -0.3536],
)

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TABLE_Z = 0.741
HOP_HEIGHT = 0.055
HOP_PERIOD = 0.36
START_XY = np.array([0.20, 0.10], dtype=np.float64)
END_XY = np.array([-0.16, -0.03], dtype=np.float64)
HOP_DURATION = 2.4
# Visual AABB is 10.8 × 7.2 × 15.4 cm; pinch the mid-torso, not the 4.2 cm
# contact-flank pair (that width drives the pads through the vinyl).
TORSO_LOCAL = np.array([0.000, 0.0, 0.052], dtype=np.float64)
TORSO_PINCH_WIDTH = 0.072
# Top-down IK dies above ~0.94 m; stay just above the rump, not 14 cm up.
HOVER_ABOVE = 0.048
DROP_DUR = 0.24
# Apex of hop 6: still in the air, already in the left-arm workspace.
CATCH_T = 1.98
LOCK_END = 1.50
FOLLOW_END = CATCH_T - DROP_DUR
# Wrist turns in joint space while translating; one grasp quat the whole way.
APPROACH_END = 0.55
TURN_START = 0.55
TURN_END = 0.55
PARK_XY = np.array([-0.10, -0.05], dtype=np.float64)
PARK_Z = TABLE_Z + 0.16
_ALOHA_GRIPPER_SCALE = (-0.01, 0.045)
_DEFAULT_GROUPS = [1, 1, 0, 0]
CAPTION = "Catch the hopping rabbit toy"
CAPTION_MOTION = "Hopping rabbit toy"


def _gripper_from_width(width_m: float, squeeze_m: float = 0.001) -> float:
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


# Fully open is 9 cm — that never pinches the 7.2 cm torso. Close onto the flanks.
GRASP_GRIP = _gripper_from_width(TORSO_PINCH_WIDTH, squeeze_m=0.008)


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype(_FONT, max(22, W // 36))
    except OSError:
        fnt = ImageFont.load_default()
    bar = int(fnt.size * (len(lines) + 0.8))
    draw.rectangle([0, 0, W, bar], fill=(8, 8, 12))
    for i, ln in enumerate(lines):
        draw.text((14, 8 + i * fnt.size), ln, fill=(255, 246, 230), font=fnt)
    return np.asarray(im)


def _heading() -> np.ndarray:
    disp = END_XY - START_XY
    return disp / max(float(np.linalg.norm(disp)), 1e-9)


def _yaw() -> float:
    h = _heading()
    return float(np.arctan2(h[1], h[0]))


def _width_dir() -> np.ndarray:
    yaw = _yaw()
    return np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)


def _q_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    q_yaw = t3d.euler.euler2quat(0.0, 0.0, yaw, axes="sxyz")
    q_pitch = t3d.euler.euler2quat(0.0, pitch, 0.0, axes="sxyz")
    return t3d.quaternions.qmult(q_yaw, q_pitch)


def bunny_pose(t: float, *, hopping: bool) -> sapien.Pose:
    u = float(np.clip(t / HOP_DURATION, 0.0, 1.0))
    u = u * u * (3.0 - 2.0 * u)
    xy = (1.0 - u) * START_XY + u * END_XY
    if t >= HOP_DURATION:
        xy = END_XY.copy()
    yaw = _yaw()
    z = TABLE_Z + 0.002
    pitch = 0.0
    if hopping and 0.0 < t < HOP_DURATION:
        phase = (t % HOP_PERIOD) / HOP_PERIOD
        z += 4.0 * HOP_HEIGHT * phase * (1.0 - phase)
        pitch = 0.34 * float(np.sin(phase * np.pi)) * (1.0 if phase < 0.55 else 0.4)
    return sapien.Pose([float(xy[0]), float(xy[1]), float(z)], _q_yaw_pitch(yaw, pitch).tolist())


def _torso_world(pose: sapien.Pose) -> np.ndarray:
    t = pose.to_transformation_matrix()
    return t[:3, :3] @ TORSO_LOCAL + t[:3, 3]


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


def _ee_pose(env, arm: str = "left") -> sapien.Pose:
    p = np.asarray(env.get_arm_pose(arm), dtype=np.float64)
    return sapien.Pose(p[:3].tolist(), p[3:].tolist())


def _tcp(env, arm: str = "left") -> np.ndarray:
    if arm == "left":
        return np.asarray(env.robot.get_left_tcp_pose(), dtype=np.float64)[:3]
    return np.asarray(env.robot.get_right_tcp_pose(), dtype=np.float64)[:3]


def _weld(actor, env, rel: sapien.Pose, arm: str = "left") -> None:
    _set_pose(actor, _ee_pose(env, arm) * rel)


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
        print(f"[bunny] IK failed: {exc}", flush=True)
        return None
    if not isinstance(res, dict) or res.get("status") != "Success":
        return None
    return np.asarray(res["position"][-1], dtype=np.float64)


def _pose7(xyz: np.ndarray, quat: list[float]) -> list[float]:
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), *[float(q) for q in quat]]


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _qslerp(q0, q1, u: float) -> list[float]:
    """Short-arc slerp of wxyz quaternions."""
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


def _pinch_xyz(pose: sapien.Pose) -> np.ndarray:
    """TCP on the mid-torso so the pads squeeze the flanks, not the ears or air."""
    return _torso_world(pose)


def _hover_xyz(pose: sapien.Pose) -> np.ndarray:
    hover = _pinch_xyz(pose)
    hover[1] -= 0.10  # wait beside the hop so the bunny does not jump through the pads
    hover[2] += HOVER_ABOVE
    return hover


def _side_pinch(open_xy, tip_sign: float = 1.0) -> list[float]:
    """Horizontal pinch: tips in XY, opening in XY, TCP Z world-up."""
    y = np.array([float(open_xy[0]), float(open_xy[1]), 0.0], dtype=np.float64)
    y = y / (np.linalg.norm(y) + 1e-9)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x = np.cross(y, z)
    n = float(np.linalg.norm(x))
    x = (x / (n + 1e-9)) * (1.0 if float(tip_sign) >= 0.0 else -1.0)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-9)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-9)
    R = np.stack([x, y, z], axis=1)
    return t3d.quaternions.mat2quat(R).tolist()


def _tcp_z_toward_tail(quat) -> bool:
    """True when TCP Z (finger thickness) points toward the tail, not the ears."""
    z = np.asarray(t3d.quaternions.quat2mat(quat)[:, 2], dtype=np.float64)
    h = _heading()
    return float(np.dot(z[:2], h)) <= 0.0


def _side_quats() -> tuple:
    """Horizontal flank pinch from behind: tips toward the head, wrist on the tail."""
    h = _heading()
    opens = (_width_dir(), -_width_dir(), np.array([0.0, 1.0]), np.array([0.0, -1.0]))
    sides = []
    for open_xy in opens:
        for tip in (1.0, -1.0):
            quat = _side_pinch(open_xy, tip_sign=tip)
            x = np.asarray(t3d.quaternions.quat2mat(quat)[:, 0], dtype=np.float64)
            if float(np.dot(x[:2], h)) > 0.25:
                sides.append(quat)
    return tuple(sides)


def _top_quats() -> tuple:
    cands = (_topdown_along([0.0, 1.0]), _topdown_along([0.0, -1.0]))
    good = tuple(q for q in cands if _tcp_z_toward_tail(q))
    return good if good else cands


def _grasp_quat() -> list[float]:
    for quat in _side_quats() or _top_quats():
        return list(quat)
    return _topdown_along([0.0, 1.0])


def _pinch_quats() -> tuple:
    return _side_quats() + _top_quats()


def _chase_quats() -> tuple:
    return _pinch_quats() + (
        _topdown_along([0.0, 1.0]),
        _topdown_along([0.0, -1.0]),
    ) + _HOVER_QUATS


def _try_ik_cal(env, xyz, quats, q_hint=None, max_jump: float | None = 2.2):
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
        print(f"[bunny] skip jump={jump:.2f} xyz={xyz.round(3).tolist()}", flush=True)
        return None, None, xyz, None
    _snap_arm(env, q_cal)
    return q_cal, quat, cmd, err


def _install_cine_camera(env, *, teacher: bool = False) -> Any:
    fovy = 50.0 if teacher else 46.0
    cam = env.scene.add_camera(
        name="cine_camera", width=1280, height=720, fovy=np.deg2rad(fovy), near=0.05, far=10.0
    )
    if teacher:
        # Frontal cine: sit past the +Y table edge, looking -Y at the intercept.
        pos = np.array([-0.06, 0.58, 1.20], dtype=np.float64)
        look = np.array([-0.08, -0.04, 0.82], dtype=np.float64)
    else:
        pos = np.array([-0.04, 0.50, 1.18], dtype=np.float64)
        look = np.array([-0.10, -0.08, 0.82], dtype=np.float64)
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
    p.add_argument("--out-dir", default="evaluate_results/demo_videos/catch_bunny_toy")
    p.add_argument("--tag", default="", help="Filename infix, e.g. teacher → catch_bunny_toy_teacher_cine.mp4")
    p.add_argument("--no-grasp", action="store_true", help="Hop only; arms stay home.")
    args = p.parse_args()

    root = Path(args.robotwin_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag.strip()}" if str(args.tag).strip() else ""
    teacher = str(args.tag).strip() == "teacher" or not args.no_grasp

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.pop("VK_ICD_FILENAMES", None)
    os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)

    tw_args = bridge.load_task_args(root, task_config="demo_clean", task_name="place_object_basket")
    tw_args["_robotwin_root"] = str(root)
    tw_args["task_name"] = "catch_bunny_toy"
    tw_args["eval_video_log"] = False
    tw_args["render_freq"] = 0
    tw_args.setdefault("data_type", {})
    tw_args["data_type"]["rgb"] = True
    tw_args["data_type"]["third_view"] = True

    cwd = os.getcwd()
    os.chdir(root)
    try:
        from envs.catch_bunny_toy import catch_bunny_toy

        env = catch_bunny_toy()
        env.setup_demo(now_ep_num=0, seed=int(args.seed), is_test=True, **tw_args)
        env.set_instruction(instruction="Catch the hopping bunny.")
    finally:
        os.chdir(cwd)

    bunny = env.object
    _ghost_physics(bunny)
    _set_pose(bunny, bunny_pose(0.0, hopping=True))
    cine = _install_cine_camera(env, teacher=teacher)

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

    def seat(t_now: float) -> sapien.Pose:
        pose = bunny_pose(float(t_now), hopping=True)
        _set_pose(bunny, pose)
        return pose

    if args.no_grasp:
        t = 0.0
        n = int(round((HOP_DURATION + 0.6) * args.fps))
        for i in range(n):
            motion_t = float(i + 1) * dt
            pose = seat(min(motion_t, HOP_DURATION + 0.5))
            for _s in range(8):
                env.scene.step()
                _set_pose(bunny, pose)
            label = "hopping" if motion_t < HOP_DURATION else "landed"
            snap(label, motion_t)
            t = motion_t
        cine_path = out_dir / f"catch_bunny_toy{tag or '_motion'}_cine.mp4"
        head_path = out_dir / f"catch_bunny_toy{tag or '_motion'}_head.mp4"
        if frames_cine:
            imageio.mimsave(cine_path.as_posix(), frames_cine, fps=args.fps, quality=8)
        if frames_head:
            imageio.mimsave(head_path.as_posix(), frames_head, fps=args.fps, quality=8)
        meta = {
            "n_cine": len(frames_cine),
            "n_head": len(frames_head),
            "cine": str(cine_path),
            "head": str(head_path),
            "task": "catch_bunny_toy",
            "mode": "hop_only",
            "seed": args.seed,
        }
        (out_dir / "recording_meta_motion.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2), flush=True)
        try:
            env.close_env(clear_cache=True)
        except Exception:
            pass
        return 0 if frames_cine else 1

    catch_pose = bunny_pose(CATCH_T, hopping=True)
    catch_torso = _torso_world(catch_pose)
    gq = _grasp_quat()
    R = t3d.quaternions.quat2mat(gq)
    h = _heading()
    pinch_dbg = _pinch_xyz(catch_pose)
    print(
        f"[bunny] catch_t={CATCH_T:.2f}s lock_end={LOCK_END:.2f}s follow_end={FOLLOW_END:.2f}s "
        f"xy={np.asarray(catch_pose.p[:2]).round(3).tolist()} "
        f"z={float(catch_pose.p[2]):.3f} torso={catch_torso.round(3).tolist()} "
        f"pinch={pinch_dbg.round(3).tolist()} "
        f"z·head={float(np.dot(R[:2, 2], h)):.2f} grip={GRASP_GRIP:.3f}",
        flush=True,
    )

    q_home = _arm_qpos(env)
    env.robot.set_gripper(1.0, "left", gripper_eps=0.0)
    _snap_arm(env, q_home)
    seat(CATCH_T)
    q_pin, pinch_quat, _cmd_p, err_p = _try_ik_cal(
        env, _pinch_xyz(catch_pose), _side_quats(), q_home, max_jump=None
    )
    family = "side"
    if pinch_quat is None:
        family = "top"
        q_pin, pinch_quat, _cmd_p, err_p = _try_ik_cal(
            env, _pinch_xyz(catch_pose), _top_quats(), q_home, max_jump=None
        )
    q_hov, hover_quat, _cmd_h, err_h = _try_ik_cal(
        env,
        _hover_xyz(catch_pose),
        (pinch_quat,) if pinch_quat is not None else _pinch_quats(),
        q_pin if q_pin is not None else q_home,
        max_jump=None,
    )
    if pinch_quat is None:
        pinch_quat = hover_quat
    if hover_quat is None:
        hover_quat = pinch_quat
    z_dot = None
    if pinch_quat is not None:
        z_ax = np.asarray(t3d.quaternions.quat2mat(pinch_quat)[:, 2], dtype=np.float64)
        z_dot = float(np.dot(z_ax[:2], _heading()))
    print(
        f"[bunny] orient family={family} hover={hover_quat is not None} pinch={pinch_quat is not None} "
        f"hover-err={err_h} pinch-err={err_p} same_family={hover_quat==pinch_quat} "
        f"pinch_z·head={z_dot}",
        flush=True,
    )
    _snap_arm(env, q_home)
    seat(0.0)
    home_tcp_quat = np.asarray(env.robot.get_left_tcp_pose(), dtype=np.float64)[3:].tolist()

    def _orient(t: float) -> list[float]:
        grasp = list(pinch_quat) if pinch_quat is not None else list(_HOVER_QUATS[0])
        return grasp

    home_tcp = np.array([-0.22, -0.16, TABLE_Z + 0.22], dtype=np.float64)
    park = _hover_xyz(catch_pose).copy()
    park[0] = float(np.clip(park[0], -0.20, -0.05))
    park[1] = float(np.clip(park[1], -0.16, -0.04))
    park[2] = float(np.clip(park[2], 0.86, 0.93))

    def _clamp_ws(xyz: np.ndarray) -> np.ndarray:
        p = np.asarray(xyz, dtype=np.float64).copy()
        p[0] = float(np.clip(p[0], -0.22, -0.05))
        p[1] = float(np.clip(p[1], -0.16, -0.02))
        p[2] = float(np.clip(p[2], 0.86, 0.93))
        return p

    def _aim(t: float) -> tuple[np.ndarray, float]:
        t = float(t)
        settle = 0.35
        if t <= settle:
            u = 0.45 * _smoothstep(t / max(settle, 1e-6))
            return (1.0 - u) * home_tcp + u * park, u
        u = _smoothstep((t - settle) / max(LOCK_END - settle, 1e-6))
        lag = (1.0 - u) * 0.28
        pose = bunny_pose(max(0.0, t - lag), hopping=True)
        hover = _clamp_ws(_hover_xyz(pose))
        return (1.0 - u) * park + u * hover, u

    key_t: list[float] = [0.0]
    key_q: list[np.ndarray] = [q_home]
    q_hint = q_home
    n_hover_ok = 0
    blend_ts = np.unique(
        np.concatenate(
            [
                np.linspace(0.12, LOCK_END, 12),
                np.linspace(TURN_START, min(TURN_END, LOCK_END), 8),
            ]
        )
    )
    for kt in blend_ts:
        xyz, u = _aim(float(kt))
        seat(float(kt))
        ou = _smoothstep((float(kt) - TURN_START) / max(TURN_END - TURN_START, 1e-6))
        q, quat, _cmd, err = _try_ik_cal(
            env, xyz, (_orient(float(kt)),), q_hint, max_jump=None
        )
        if q is None:
            print(f"[bunny] blend IK fail t={float(kt):.2f} u={u:.2f}", flush=True)
            continue
        key_t.append(float(kt))
        key_q.append(q)
        jump = float(np.linalg.norm(q - q_hint))
        q_hint = q
        n_hover_ok += 1
        print(
            f"[bunny] blend t={float(kt):.2f} u={u:.2f} turn={ou:.2f} aim={xyz.round(3).tolist()} "
            f"tcp-err={err:.3f} jump={jump:.2f} tcp={_tcp(env).round(3).tolist()}",
            flush=True,
        )

    lock_ts = np.linspace(LOCK_END, FOLLOW_END, 10)
    for kt in lock_ts:
        pose = seat(float(kt))
        xyz = _clamp_ws(_hover_xyz(pose))
        ou = _smoothstep((float(kt) - TURN_START) / max(TURN_END - TURN_START, 1e-6))
        q, quat, _cmd, err = _try_ik_cal(env, xyz, (_orient(float(kt)),), q_hint, max_jump=None)
        if q is None:
            print(f"[bunny] lock IK fail t={float(kt):.2f}", flush=True)
            continue
        key_t.append(float(kt))
        key_q.append(q)
        jump = float(np.linalg.norm(q - q_hint))
        q_hint = q
        n_hover_ok += 1
        print(
            f"[bunny] lock t={float(kt):.2f} turn={ou:.2f} tcp-err={err:.3f} jump={jump:.2f} "
            f"tcp={_tcp(env).round(3).tolist()}",
            flush=True,
        )

    q_pinch = None
    pinch_xyz = _pinch_xyz(catch_pose)
    n_pinch_ok = 0
    drop_ts = np.linspace(FOLLOW_END, CATCH_T, 12)
    for kt in drop_ts:
        pose = seat(float(kt))
        drop_a = 0.0 if CATCH_T <= FOLLOW_END else (float(kt) - FOLLOW_END) / DROP_DUR
        a = drop_a * drop_a
        xyz = (1.0 - a) * _clamp_ws(_hover_xyz(pose)) + a * _pinch_xyz(pose)
        ou = _smoothstep((float(kt) - TURN_START) / max(TURN_END - TURN_START, 1e-6))
        q, quat, _cmd, err = _try_ik_cal(env, xyz, (_orient(float(kt)),), q_hint, max_jump=None)
        if q is None:
            print(f"[bunny] drop IK fail t={float(kt):.2f} a={drop_a:.2f}", flush=True)
            continue
        key_t.append(float(kt))
        key_q.append(q)
        jump = float(np.linalg.norm(q - q_hint))
        q_hint = q
        n_pinch_ok += 1
        if drop_a >= 0.99:
            q_pinch = q
            pinch_xyz = _pinch_xyz(pose)
        print(
            f"[bunny] drop t={float(kt):.2f} a={drop_a:.2f} turn={ou:.2f} tcp-err={err:.3f} "
            f"jump={jump:.2f} tcp={_tcp(env).round(3).tolist()}",
            flush=True,
        )

    grasp_ok = q_pinch is not None
    print(
        f"[bunny] hover_keys={n_hover_ok} drop_keys={n_pinch_ok} pinch={grasp_ok} grip={GRASP_GRIP:.3f}",
        flush=True,
    )

    _snap_arm(env, q_home)
    env.robot.set_gripper(1.0, "left", gripper_eps=0.0)
    seat(0.0)
    for _s in range(8):
        env.scene.step()
        _snap_arm(env, q_home)
        seat(0.0)

    kt = np.asarray(key_t, dtype=np.float64)
    kq = np.stack(key_q, axis=0)

    def _q_at(motion_t: float) -> np.ndarray:
        tq = float(np.clip(motion_t, float(kt[0]), float(kt[-1])))
        i = int(np.searchsorted(kt, tq, side="right") - 1)
        i = max(0, min(i, len(kt) - 2))
        span = max(float(kt[i + 1] - kt[i]), 1e-6)
        a = float(np.clip((tq - kt[i]) / span, 0.0, 1.0))
        return (1.0 - a) * kq[i] + a * kq[i + 1]

    n_chase = max(int(round(CATCH_T * args.fps)), 8)
    n_close_fly = max(int(round(DROP_DUR * args.fps)), 4)
    pd_steps = 4
    t = 0.0
    last_q = q_home
    last_pose = bunny_pose(0.0, hopping=True)
    for i in range(n_chase):
        motion_t = float(i + 1) * dt
        motion_t = min(motion_t, CATCH_T)
        pose = seat(motion_t)
        last_pose = pose
        q_now = _q_at(motion_t)
        last_q = q_now
        grip = 1.0
        phase = "hop"
        if motion_t >= 0.40:
            phase = "chase"
        if motion_t >= LOCK_END:
            phase = "intercept"
        if motion_t >= FOLLOW_END:
            a = 0.0 if CATCH_T <= FOLLOW_END else (motion_t - FOLLOW_END) / DROP_DUR
            a = float(np.clip(a, 0.0, 1.0))
            a = a * a
            grip = (1.0 - a) * 1.0 + a * GRASP_GRIP
            phase = "catch"
        _snap_arm(env, q_now)
        env.robot.set_gripper(float(grip), "left", gripper_eps=0.0)
        seat(motion_t)
        for _s in range(pd_steps):
            env.scene.step()
            _snap_arm(env, q_now)
            seat(motion_t)
        if i < 4 or i in (int(0.54 * args.fps), int(LOCK_END * args.fps), n_chase - 1):
            print(
                f"[bunny] frame{i} t={motion_t:.3f} bunny={np.asarray(pose.p).round(3).tolist()} "
                f"tcp={_tcp(env).round(3).tolist()} arm_move={float(np.linalg.norm(q_now - q_home)):.3f}",
                flush=True,
            )
        snap(phase, motion_t)
        t = motion_t

    if q_pinch is not None:
        _snap_arm(env, q_pinch)
        last_q = q_pinch
    env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
    _set_pose(bunny, last_pose)
    for _s in range(pd_steps):
        env.scene.step()
        _snap_arm(env, last_q)
        _set_pose(bunny, last_pose)
    tcp = _tcp(env)
    pinch = float(np.linalg.norm(tcp - pinch_xyz))
    dxy = tcp[:2] - pinch_xyz[:2]
    dz = float(tcp[2] - pinch_xyz[2])
    print(
        f"[bunny] pinch tcp-target={pinch:.3f} grip={GRASP_GRIP:.3f} "
        f"dxy={dxy.round(3).tolist()} dz={dz:.3f} "
        f"tcp={tcp.round(3).tolist()} torso={_torso_world(last_pose).round(3).tolist()}",
        flush=True,
    )

    n_close = max(int(0.12 * args.fps), 3)
    held = last_pose
    for _ in range(n_close):
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        _snap_arm(env, last_q)
        _set_pose(bunny, held)
        for _s in range(8):
            env.scene.step()
            _snap_arm(env, last_q)
            _set_pose(bunny, held)
        snap("catch", t)
        t += dt

    _ghost_physics(bunny, hard=True)
    rel = _ee_pose(env).inv() * held
    _weld(bunny, env, rel)

    def _step_visual(label: str, t_now: float, q: np.ndarray) -> None:
        _snap_arm(env, q)
        _weld(bunny, env, rel)
        for _s in range(8):
            env.scene.step()
            _snap_arm(env, q)
            _weld(bunny, env, rel)
        snap(label, t_now)

    ee0 = np.asarray(env.get_arm_pose("left"), dtype=np.float64)
    q_now = _arm_qpos(env)
    q_lift = None
    for dz in (0.08, 0.06, 0.10, 0.04, 0.12):
        ee1 = ee0.copy()
        ee1[2] += float(dz)
        q_lift = _ik_q(env, ee1)
        if q_lift is not None:
            break
    if q_lift is None:
        print("[bunny] lift IK failed", flush=True)
        q_lift = q_now
    n_lift = int(0.70 * args.fps)
    for i in range(max(n_lift, 1)):
        a = float(i + 1) / float(max(n_lift, 1))
        q_i = (1.0 - a) * q_now + a * q_lift
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        _step_visual("lift", t, q_i)
        t += dt
    q_hold = _arm_qpos(env)
    for _ in range(int(0.40 * args.fps)):
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        _step_visual("caught", t, q_hold)
        t += dt

    cine_path = out_dir / f"catch_bunny_toy{tag}_cine.mp4"
    head_path = out_dir / f"catch_bunny_toy{tag}_head.mp4"
    if frames_cine:
        imageio.mimsave(cine_path.as_posix(), frames_cine, fps=args.fps, quality=8)
        frame_dir = out_dir / f"frames{tag or ''}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        catch_idx = min(max(int(round(CATCH_T * args.fps)) - 1, 0), len(frames_cine) - 1)
        picks = {
            "c_01": 0,
            "c_02": max(len(frames_cine) // 4, 0),
            "c_03": max(len(frames_cine) // 2, 0),
            "c_04": max((3 * len(frames_cine)) // 4, 0),
            "c_05": len(frames_cine) - 1,
            "catch_197": catch_idx,
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
        "task": "catch_bunny_toy",
        "mode": "teacher_catch_hopping_torso",
        "seed": args.seed,
        "catch_t": CATCH_T,
        "lock_end": LOCK_END,
        "follow_end": FOLLOW_END,
        "grasp_ok": grasp_ok,
        "pinch_tcp_target_m": pinch,
        "grasp_grip": GRASP_GRIP,
        "torso_catch_xyz": _torso_world(last_pose).tolist(),
        "bunny_catch_xyz": np.asarray(last_pose.p, dtype=np.float64).tolist(),
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
