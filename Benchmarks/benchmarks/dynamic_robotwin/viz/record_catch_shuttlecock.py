#!/usr/bin/env python3
"""Cinematic extra: a shuttlecock lobs over the table and is caught mid-air.

Gravity + quadratic drag, cork aligned with velocity. Native SAPIEN visuals.
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
_ALOHA_GRIPPER_SCALE = (-0.01, 0.045)
CORK_WIDTH = 0.028
_DEFAULT_GROUPS = [1, 1, 0, 0]
CAPTION = "Catch the flying shuttlecock"
CAPTION_MOTION = "Shuttlecock over the table"


def _gripper_from_width(width_m: float, squeeze_m: float = 0.001) -> float:
    inner = max(float(width_m) - float(squeeze_m), 0.0)
    joint = 0.5 * inner
    lo, hi = _ALOHA_GRIPPER_SCALE
    return float(np.clip((joint - lo) / (hi - lo), 0.0, 1.0))


# Cork is 28 mm; extra 6 mm so pads kiss the cork instead of sinking into it.
GRASP_GRIP = _gripper_from_width(CORK_WIDTH, squeeze_m=-0.010)


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


def _fingers_up(open_xy) -> list[float]:
    """TCP X world-up: wrist below the cork, skirt continues past the fingertips."""
    x = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y = np.array([float(open_xy[0]), float(open_xy[1]), 0.0], dtype=np.float64)
    y = y / (np.linalg.norm(y) + 1e-9)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-9)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return t3d.quaternions.mat2quat(R).tolist()


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


def _grasp_quats() -> tuple:
    sides = []
    for open_xy in ((1.0, 0.0), (-1.0, 0.0)):
        for tip in (1.0, -1.0):
            sides.append(_side_pinch(open_xy, tip_sign=tip))
    return tuple(sides) + (
        _topdown_along([1.0, 0.0]),
        _topdown_along([-1.0, 0.0]),
    )


# Cork sits above pad center so the skirt is above the paddles, not through them.
_CORK_ABOVE_TCP = 0.016
# Reported TCP sits nearer the palm than the visible pads.
_PAD_ALONG_X = 0.028


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype(_FONT, max(22, W // 36))
    except OSError:
        fnt = ImageFont.load_default()
    bar = int(fnt.size * (len(lines) + 0.8))
    draw.rectangle([0, 0, W, bar], fill=(10, 14, 22))
    for i, ln in enumerate(lines):
        draw.text((14, 8 + i * fnt.size), ln, fill=(230, 244, 255), font=fnt)
    return np.asarray(im)


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
        print(f"[shuttlecock] plan failed: {exc}", flush=True)
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
    try:
        res = planner(pose7)
    except Exception as exc:
        print(f"[shuttlecock] IK failed: {exc}", flush=True)
        return None
    if not isinstance(res, dict) or res.get("status") != "Success":
        return None
    return np.asarray(res["position"][-1], dtype=np.float64)


def _pose7(xyz: np.ndarray, quat: list[float]) -> list[float]:
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), *[float(q) for q in quat]]


def _first_ik(env, xyz: np.ndarray, quats) -> tuple[np.ndarray, list[float]] | tuple[None, None]:
    for quat in quats:
        q = _ik_q(env, _pose7(xyz, quat))
        if q is not None:
            return q, list(quat)
    return None, None


def _install_cine_camera(env, *, fly_only: bool = False) -> Any:
    fovy = 48.0 if fly_only else 64.0
    cam = env.scene.add_camera(
        name="cine_camera", width=1280, height=720, fovy=np.deg2rad(fovy), near=0.05, far=10.0
    )
    # From +Y looking -Y: screen-left is +X (launch), screen-right is -X (left arm).
    if fly_only:
        pos = np.array([0.02, 0.70, 1.10], dtype=np.float64)
        look = np.array([0.02, 0.00, 0.92], dtype=np.float64)
    else:
        # Table in the foreground, full lob (launch → apex → catch) in frame.
        pos = np.array([0.08, 0.68, 1.16], dtype=np.float64)
        look = np.array([0.06, 0.00, 1.00], dtype=np.float64)
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


def _save_picks(frames: list[np.ndarray], frame_dir: Path, prefix: str) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    picks = {
        f"{prefix}01": 0,
        f"{prefix}02": max(n // 4, 0),
        f"{prefix}03": max(n // 2, 0),
        f"{prefix}04": max((3 * n) // 4, 0),
        f"{prefix}05": n - 1,
    }
    for name, idx in picks.items():
        Image.fromarray(frames[idx]).save(frame_dir / f"{name}.png")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--robotwin-root",
        default=os.environ.get("ROBOTWIN_ROOT", "/DATA/YuanZhen/FastWAM/third_party/RoboTwin"),
    )
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out-dir", default="evaluate_results/demo_videos/catch_shuttlecock")
    p.add_argument("--no-grasp", action="store_true", help="Fly only; arms stay home.")
    p.add_argument("--tag", default="", help="Filename infix, e.g. teacher → catch_shuttlecock_teacher_cine.mp4")
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
    tw_args["task_name"] = "catch_shuttlecock"
    tw_args["eval_video_log"] = False
    tw_args["render_freq"] = 0
    tw_args.setdefault("data_type", {})
    tw_args["data_type"]["rgb"] = True
    tw_args["data_type"]["third_view"] = True

    cwd = os.getcwd()
    os.chdir(root)
    try:
        from envs.catch_shuttlecock import catch_shuttlecock
        from envs.utils.action import Action

        env = catch_shuttlecock()
        env.setup_demo(now_ep_num=0, seed=int(args.seed), is_test=True, **tw_args)
        env.set_instruction(instruction="Catch the flying shuttlecock.")
    finally:
        os.chdir(cwd)

    bird = env.object
    _set_kinematic(bird, True)
    _ghost_physics(bird)
    env.reset_flight()
    _set_pose(bird, env.flight_pose())
    if args.no_grasp:
        # Wider, higher lob so the bird clearly crosses the table. Catch
        # height does not matter here because the arms stay home.
        env.P0 = np.array([0.30, 0.00, 0.90], dtype=np.float64)
        env.V0 = np.array([-1.15, 0.00, 2.45], dtype=np.float64)
        env.K_M = 0.28
        env.reset_flight()
        _set_pose(bird, env.flight_pose())
    else:
        # Same lob as extra/collect_oracle.py (the teacher that actually catches).
        env.P0 = np.array([0.20, 0.00, 0.84], dtype=np.float64)
        env.V0 = np.array([-0.55, 0.00, 3.05], dtype=np.float64)
        env.K_M = 0.30
        env.CATCH_T = 0.55
        env.reset_flight()
        _set_pose(bird, env.flight_pose())
        env.catch_pose = env.pose_at(env.CATCH_T)
    cine = _install_cine_camera(env, fly_only=bool(args.no_grasp))

    frames_cine: list[np.ndarray] = []
    frames_head: list[np.ndarray] = []
    dt = 1.0 / float(args.fps)
    try:
        phys_dt = float(env.scene.get_timestep())
    except Exception:
        phys_dt = 1.0 / 250.0
    substeps = max(int(round(dt / max(phys_dt, 1e-6))), 1)
    caption = CAPTION_MOTION if args.no_grasp else CAPTION

    def snap(label: str, t: float) -> None:
        rgbs = _grab_rgb(env, cine)
        cine_f = rgbs["cine"] if "cine" in rgbs else rgbs.get("head")
        if cine_f is not None:
            frames_cine.append(_annotate(cine_f, [caption, f"{label}   t={t:.2f}s"]))
        if "head" in rgbs:
            frames_head.append(rgbs["head"])

    def hold_start(n: int, label: str, t0: float) -> float:
        t = t0
        env.reset_flight()
        pose0 = env.flight_pose()
        for _ in range(n):
            _set_pose(bird, pose0)
            for _s in range(substeps):
                env.scene.step()
                _set_pose(bird, pose0)
            snap(label, t)
            t += dt
        return t

    def fly_frame(t_cap: float | None = None) -> sapien.Pose:
        for _s in range(substeps):
            step = phys_dt
            if t_cap is not None:
                remain = float(t_cap) - float(env.flight_t)
                if remain <= 1e-9:
                    break
                step = min(phys_dt, remain)
            env.step_flight(step)
            _set_pose(bird, env.flight_pose())
            env.scene.step()
            _set_pose(bird, env.flight_pose())
        return env.flight_pose()

    catch_pose = env.catch_pose
    catch_xyz = np.asarray(catch_pose.p, dtype=np.float64)
    print(
        f"[shuttlecock] catch={catch_xyz.round(3).tolist()} t={env.CATCH_T:.2f} "
        f"p0={env.P0.round(3).tolist()} substeps={substeps}",
        flush=True,
    )

    if args.no_grasp:
        t = hold_start(int(0.35 * args.fps), "ready", 0.0)
        env.reset_flight()
        last = env.flight_pose()
        while env.flight_p[2] > TABLE_Z + env.CORK_R and env.flight_t < 1.4:
            last = fly_frame()
            snap("flying", t)
            t += dt
        land_p = np.asarray(last.p, dtype=np.float64).copy()
        land_p[2] = TABLE_Z + env.CORK_R
        last = sapien.Pose(land_p.tolist(), list(last.q))
        for _ in range(int(0.35 * args.fps)):
            _set_pose(bird, last)
            for _s in range(substeps):
                env.scene.step()
                _set_pose(bird, last)
            snap("landed", t)
            t += dt
        cine_path = out_dir / "catch_shuttlecock_motion_cine.mp4"
        head_path = out_dir / "catch_shuttlecock_motion_head.mp4"
        if frames_cine:
            imageio.mimsave(cine_path.as_posix(), frames_cine, fps=args.fps, quality=8)
            _save_picks(frames_cine, out_dir / "frames_motion", "m_")
        if frames_head:
            imageio.mimsave(head_path.as_posix(), frames_head, fps=args.fps, quality=8)
        meta = {
            "n_cine": len(frames_cine),
            "n_head": len(frames_head),
            "cine": str(cine_path),
            "head": str(head_path),
            "task": "catch_shuttlecock",
            "mode": "fly_only",
            "seed": args.seed,
            "catch_xyz": catch_xyz.tolist(),
        }
        (out_dir / "recording_meta_motion.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2), flush=True)
        try:
            env.close_env(clear_cache=True)
        except Exception:
            pass
        return 0 if frames_cine else 1

    q_home = _arm_qpos(env)
    pinch_quats = _grasp_quats()
    candidates: list[tuple[np.ndarray, list[float], np.ndarray]] = []
    for quat in pinch_quats:
        x_hat = t3d.quaternions.quat2mat(np.asarray(quat, dtype=np.float64))[:, 0]
        cand = (
            catch_xyz
            + np.array([0.0, 0.0, -_CORK_ABOVE_TCP], dtype=np.float64)
            - _PAD_ALONG_X * x_hat
        )
        q_try = _ik_q(env, _pose7(cand, quat))
        if q_try is not None:
            candidates.append((q_try, list(quat), cand))
    q_pinch, pinch_quat, pinch_xyz = (candidates[0] if candidates else (None, None, catch_xyz.copy()))
    # Stay 3 cm behind the arc so the waiting gripper is not a wall.
    wait_xyz = pinch_xyz + np.array([0.00, -0.03, 0.00], dtype=np.float64)
    wait_quats = ((pinch_quat,) if pinch_quat is not None else ()) + pinch_quats
    q_wait, _ = _first_ik(env, wait_xyz, wait_quats)
    hover_ok = q_wait is not None
    grasp_ok = q_pinch is not None
    print(
        f"[shuttlecock] wait={hover_ok} pinch={grasp_ok} grip={GRASP_GRIP:.3f} "
        f"wait={wait_xyz.round(3).tolist()} pinch={pinch_xyz.round(3).tolist()} "
        f"cork={catch_xyz.round(3).tolist()}",
        flush=True,
    )

    def interp_arm(q_goal: np.ndarray, n: int, label: str, t0: float, flying: bool, grip: float) -> float:
        q0 = _arm_q(env)
        t_now = t0
        steps = max(int(n), 1)
        pose0 = env.flight_pose()
        for i in range(steps):
            a = float(i + 1) / float(steps)
            _drive_arm(env, (1.0 - a) * q0 + a * q_goal)
            env.robot.set_gripper(float(grip), "left", gripper_eps=0.0)
            if flying:
                fly_frame()
            else:
                _set_pose(bird, pose0)
                for _s in range(substeps):
                    env.scene.step()
                    _set_pose(bird, pose0)
            snap(label, t_now)
            t_now += dt
        return t_now

    plan_xyz = np.asarray(pinch_xyz, dtype=np.float64).copy()
    if q_pinch is not None:
        # Planner pose and get_left_tcp_pose() disagree; measure it before
        # recording so the cine path already aims the real pads at the cork.
        _snap_arm(env, q_pinch)
        for _s in range(4):
            env.scene.step()
            _snap_arm(env, q_pinch)
        delta = _tcp(env) - pinch_xyz
        cmd = pinch_xyz - delta
        q_cal = _ik_q(env, _pose7(cmd, pinch_quat))
        print(
            f"[shuttlecock] tcp-cal delta={delta.round(3).tolist()} "
            f"cmd={cmd.round(3).tolist()} ok={q_cal is not None}",
            flush=True,
        )
        if q_cal is not None:
            q_pinch = q_cal
            plan_xyz = cmd
            _snap_arm(env, q_pinch)
            for _s in range(4):
                env.scene.step()
                _snap_arm(env, q_pinch)
            print(
                f"[shuttlecock] tcp-cal after={float(np.linalg.norm(_tcp(env) - pinch_xyz)):.3f} "
                f"tcp={_tcp(env).round(3).tolist()}",
                flush=True,
            )
        _snap_arm(env, q_home)
        env.reset_flight()
        _set_pose(bird, env.flight_pose())

    # Real-time lob and reach on the same clock. Do not freeze the bird at
    # P0, and do not give the arm a head start — first cine frame already
    # has both moving.
    env.reset_flight()
    t = 0.0
    catch_t = float(env.CATCH_T)
    path = _joint_path(env, plan_xyz, pinch_quat) if q_pinch is not None else None
    if q_pinch is None:
        print("[shuttlecock] no pinch IK", flush=True)
    _snap_arm(env, q_home)
    env.robot.set_gripper(1.0, "left", gripper_eps=0.0)
    env.reset_flight()
    _set_pose(bird, env.flight_pose())
    for _s in range(max(substeps, 8)):
        env.scene.step()
        _snap_arm(env, q_home)
        _set_pose(bird, env.flight_pose())

    n_reach = max(int(round(catch_t * args.fps)), 12)
    n_close_fly = 2
    pd_steps = max(substeps, 4)

    def _set_flight_t(t_bird: float) -> None:
        t_bird = float(np.clip(t_bird, 0.0, catch_t))
        pose = env.pose_at(t_bird)
        env.flight_t = t_bird
        env.flight_p = np.asarray(pose.p, dtype=np.float64).copy()
        env.flight_q = np.asarray(pose.q, dtype=np.float64).copy()
        _set_pose(bird, pose)

    def _arm_at(frac: float) -> np.ndarray:
        a = float(np.clip(frac, 0.0, 1.0))
        if path is not None:
            idx = int(round(a * float(len(path) - 1)))
            idx = int(np.clip(idx, 0, len(path) - 1))
            return path[idx]
        if q_pinch is None:
            return q_home
        return (1.0 - a) * q_home + a * q_pinch

    for i in range(n_reach):
        # i=0 → dt, not 0, so frame 0 is already in motion.
        t_bird = float(i + 1) * dt
        frac = float(i + 1) / float(n_reach)
        q_now = _arm_at(frac)
        _snap_arm(env, q_now)
        grip = 1.0
        phase = "reach" if frac < 0.92 else "intercept"
        if i >= n_reach - n_close_fly:
            a = float(i + 1 - (n_reach - n_close_fly)) / float(n_close_fly)
            grip = (1.0 - a) * 1.0 + a * GRASP_GRIP
            phase = "catch"
        env.robot.set_gripper(float(grip), "left", gripper_eps=0.0)
        _set_flight_t(t_bird)
        for _s in range(pd_steps):
            env.scene.step()
            _snap_arm(env, q_now)
            _set_pose(bird, env.flight_pose())
        if i < 6:
            print(
                f"[shuttlecock] frame{i} t={t_bird:.3f} bird={np.asarray(env.flight_p).round(3).tolist()} "
                f"tcp={_tcp(env).round(3).tolist()} arm_move={float(np.linalg.norm(q_now - q_home)):.3f}",
                flush=True,
            )
        snap(phase, t_bird)
        t = t_bird

    _set_flight_t(catch_t)
    if q_pinch is not None:
        _snap_arm(env, q_pinch)
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        for _s in range(pd_steps):
            env.scene.step()
            _snap_arm(env, q_pinch)
            _set_pose(bird, env.flight_pose())
    print(
        f"[shuttlecock] after-reach tcp-pinch={float(np.linalg.norm(pinch_xyz - _tcp(env))):.3f} "
        f"tcp={_tcp(env).round(3).tolist()}",
        flush=True,
    )

    n_close = max(int(0.06 * args.fps), 2)
    pose_catch = env.flight_pose()
    for _ in range(n_close):
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        if q_pinch is not None:
            _snap_arm(env, q_pinch)
        _set_pose(bird, pose_catch)
        for _s in range(substeps):
            env.scene.step()
            if q_pinch is not None:
                _snap_arm(env, q_pinch)
            _set_pose(bird, pose_catch)
        snap("catch", t)
        t += dt
    env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
    q_hold_arm = _arm_qpos(env)
    _snap_arm(env, q_hold_arm)

    catch_now = env.flight_pose()
    tcp = _tcp(env)
    pinch = float(np.linalg.norm(tcp - np.asarray(catch_now.p, dtype=np.float64)))
    print(
        f"[shuttlecock] at-catch tcp-cork={pinch:.3f} grip={GRASP_GRIP:.3f} "
        f"tcp={tcp.round(3).tolist()} bird={np.asarray(catch_now.p).round(3).tolist()} "
        f"flight_t={float(env.flight_t):.3f} v={np.asarray(env.flight_v).round(3).tolist()}",
        flush=True,
    )

    _ghost_physics(bird, hard=True)
    # Cork stays on the parabola; feathers-up so the skirt sits above the pads.
    cork_p = np.asarray(catch_now.p, dtype=np.float64)
    q_up = env.cork_along_quat(np.array([0.0, 0.0, -1.0], dtype=np.float64), fallback=np.asarray(catch_now.q))
    held = sapien.Pose(cork_p.tolist(), np.asarray(q_up, dtype=np.float64).tolist())
    rel = _ee_pose(env).inv() * held
    _weld(bird, env, rel)

    def _step_visual(label: str, t_now: float, rel_pose: sapien.Pose) -> None:
        _weld(bird, env, rel_pose)
        for _s in range(substeps):
            env.scene.step()
            _weld(bird, env, rel_pose)
        snap(label, t_now)

    ee0 = np.asarray(env.get_arm_pose("left"), dtype=np.float64)
    q_now = _arm_qpos(env)
    q_lift = None
    for dz in (0.06, 0.04, 0.08, 0.03):
        ee1 = ee0.copy()
        ee1[2] += float(dz)
        q_lift = _ik_q(env, ee1)
        if q_lift is not None:
            break
    if q_lift is None:
        print("[shuttlecock] lift IK failed, holding grasp pose", flush=True)
        q_lift = q_now
    n_lift = int(0.70 * args.fps)
    for i in range(max(n_lift, 1)):
        a = float(i + 1) / float(max(n_lift, 1))
        _drive_arm(env, (1.0 - a) * q_now + a * q_lift)
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        _step_visual("lift", t, rel)
        t += dt

    q_hold = _arm_q(env)
    for _ in range(int(0.55 * args.fps)):
        _drive_arm(env, q_hold)
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        _step_visual("lift", t, rel)
        t += dt

    tag = f"_{args.tag}" if str(args.tag).strip() else ""
    cine_path = out_dir / f"catch_shuttlecock{tag}_cine.mp4"
    head_path = out_dir / f"catch_shuttlecock{tag}_head.mp4"
    if frames_cine:
        imageio.mimsave(cine_path.as_posix(), frames_cine, fps=args.fps, quality=8)
        _save_picks(frames_cine, out_dir / f"frames{tag or ''}", "c_")
    if frames_head:
        imageio.mimsave(head_path.as_posix(), frames_head, fps=args.fps, quality=8)
    meta = {
        "n_cine": len(frames_cine),
        "n_head": len(frames_head),
        "cine": str(cine_path),
        "head": str(head_path),
        "task": "catch_shuttlecock",
        "mode": "teacher_fly_and_catch" if str(args.tag).strip() == "teacher" else "fly_and_catch",
        "seed": args.seed,
        "hover_ok": hover_ok,
        "grasp_ok": grasp_ok,
        "pinch_tcp_cork_m": pinch,
        "grasp_grip": GRASP_GRIP,
        "catch_xyz": catch_xyz.tolist(),
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
