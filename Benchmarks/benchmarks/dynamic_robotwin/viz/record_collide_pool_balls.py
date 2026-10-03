#!/usr/bin/env python3
"""Cinematic extra: two phenolic pool balls collide under real PhysX.

Sphere-sphere contact, ivory restitution, felt table. Not kinematic rails.
Teacher cine: catch the red object ball while it is still rolling after the cut.
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
RADIUS = 0.030
CUE_SPEED = 0.70
# Teacher: object sits -X of the cue line so the red ball cuts into the left arm.
TEACHER_OBJ_XY = np.array([-0.028, -0.05], dtype=np.float64)
CENTER_Z = TABLE_Z + RADIUS
# TCP is the pad center. Sit just above the equator so pads wrap the 6 cm ball
# instead of brushing the top (the old +2 cm looked like grasping air).
PINCH_Z = CENTER_Z + 0.008
HOVER_ABOVE = 0.14
DROP_DUR = 0.22
_ALOHA_GRIPPER_SCALE = (-0.01, 0.045)
_DEFAULT_GROUPS = [1, 1, 0, 0]
# Match collide_pool_balls._add_cushions: table 1.2×0.7, wall half 0.020×0.038.
# Inner face minus radius, plus 8 mm so the full ball stays on the white cloth.
RAIL_INNER_X = 0.60 - 0.020 - 0.020 - RADIUS - 0.008  # 0.522
RAIL_INNER_Y = 0.35 - 0.020 - 0.020 - RADIUS - 0.008  # 0.272
RAIL_RESTITUTION = 0.72
CAPTION = "Pool balls collide"
CAPTION_TEACHER = "Catch the red ball after the break"
PARK_XY = np.array([-0.10, -0.08], dtype=np.float64)


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


# Diameter 6 cm; slight squeeze so both pads sit on the sphere, not around a gap.
GRASP_GRIP = _gripper_from_width(2.0 * RADIUS, squeeze_m=0.010)
PINCH_QUATS = (_topdown_along([1.0, 0.0]), _topdown_along([-1.0, 0.0])) + _HOVER_QUATS


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    im = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype(_FONT, max(22, W // 36))
    except OSError:
        fnt = ImageFont.load_default()
    bar = int(fnt.size * (len(lines) + 0.8))
    draw.rectangle([0, 0, W, bar], fill=(12, 18, 14))
    for i, ln in enumerate(lines):
        draw.text((14, 8 + i * fnt.size), ln, fill=(210, 255, 220), font=fnt)
    return np.asarray(im)


def _rigid(actor):
    ent = getattr(actor, "actor", actor)
    try:
        return ent, ent.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    except Exception:
        return ent, None


def _xy(actor) -> np.ndarray:
    ent, _ = _rigid(actor)
    return np.asarray(ent.get_pose().p, dtype=np.float64)[:2]


def _xyz(actor) -> np.ndarray:
    ent, _ = _rigid(actor)
    return np.asarray(ent.get_pose().p, dtype=np.float64)


def _copy_pose(actor) -> sapien.Pose:
    ent, _ = _rigid(actor)
    p = ent.get_pose()
    return sapien.Pose(list(p.p), list(p.q))


def _set_pose(actor, pose: sapien.Pose) -> None:
    ent, _ = _rigid(actor)
    try:
        ent.set_pose(pose)
    except Exception:
        pass


def _set_kinematic(actor, on: bool) -> None:
    _ent, comp = _rigid(actor)
    if comp is None:
        return
    try:
        comp.set_kinematic(bool(on))
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


def _bounce_in_rails(actor) -> None:
    """Keep a live ball on the cloth; ghost groups otherwise ignore the cushions."""
    ent, comp = _rigid(actor)
    if ent is None or comp is None:
        return
    pose = ent.get_pose()
    p = np.asarray(pose.p, dtype=np.float64).copy()
    v = _get_velocity(actor)
    w = _get_angular(actor)
    hit = False
    if p[0] > RAIL_INNER_X:
        p[0] = RAIL_INNER_X
        v[0] = -RAIL_RESTITUTION * abs(float(v[0]))
        hit = True
    elif p[0] < -RAIL_INNER_X:
        p[0] = -RAIL_INNER_X
        v[0] = RAIL_RESTITUTION * abs(float(v[0]))
        hit = True
    if p[1] > RAIL_INNER_Y:
        p[1] = RAIL_INNER_Y
        v[1] = -RAIL_RESTITUTION * abs(float(v[1]))
        hit = True
    elif p[1] < -RAIL_INNER_Y:
        p[1] = -RAIL_INNER_Y
        v[1] = RAIL_RESTITUTION * abs(float(v[1]))
        hit = True
    if not hit:
        return
    ent.set_pose(sapien.Pose(p.tolist(), list(pose.q)))
    _set_velocity(actor, v, w)


def _set_velocity(actor, linear: np.ndarray, angular: np.ndarray | None = None) -> None:
    _ent, comp = _rigid(actor)
    if comp is None:
        return
    lin = np.asarray(linear, dtype=np.float64).reshape(3)
    try:
        comp.set_linear_velocity(lin)
    except Exception:
        try:
            comp.linear_velocity = lin
        except Exception:
            pass
    if angular is None:
        return
    ang = np.asarray(angular, dtype=np.float64).reshape(3)
    try:
        comp.set_angular_velocity(ang)
    except Exception:
        try:
            comp.angular_velocity = ang
        except Exception:
            pass


def _get_angular(actor) -> np.ndarray:
    _ent, comp = _rigid(actor)
    if comp is None:
        return np.zeros(3)
    try:
        return np.asarray(comp.angular_velocity, dtype=np.float64).reshape(3)
    except Exception:
        try:
            return np.asarray(comp.get_angular_velocity(), dtype=np.float64).reshape(3)
        except Exception:
            return np.zeros(3)


def _get_velocity(actor) -> np.ndarray:
    _ent, comp = _rigid(actor)
    if comp is None:
        return np.zeros(3)
    try:
        return np.asarray(comp.linear_velocity, dtype=np.float64).reshape(3)
    except Exception:
        try:
            return np.asarray(comp.get_linear_velocity(), dtype=np.float64).reshape(3)
        except Exception:
            return np.zeros(3)


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
        print(f"[pool] IK failed: {exc}", flush=True)
        return None
    if not isinstance(res, dict) or res.get("status") != "Success":
        return None
    return np.asarray(res["position"][-1], dtype=np.float64)


def _pose7(xyz: np.ndarray, quat: list[float]) -> list[float]:
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), *[float(q) for q in quat]]


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _pinch_xyz(center: np.ndarray) -> np.ndarray:
    pinch = np.asarray(center, dtype=np.float64).copy()
    pinch[2] = PINCH_Z
    # +X: Aloha pad midpoint is slightly -X of TCP (cine-right finger ran through the ball).
    # +Y: visible rubber sits distal of the reported TCP (toward the cine camera).
    pinch[0] += 0.006
    pinch[1] += 0.008
    return pinch


def _hover_xyz(center: np.ndarray) -> np.ndarray:
    hover = _pinch_xyz(center)
    hover[2] = PINCH_Z + HOVER_ABOVE
    return hover


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
        print(f"[pool] skip jump={jump:.2f} xyz={xyz.round(3).tolist()}", flush=True)
        return None, None, xyz, None
    _snap_arm(env, q_cal)
    return q_cal, quat, cmd, err


def _install_cine_camera(env, *, teacher: bool = False) -> Any:
    fovy = 52.0 if teacher else 54.0
    cam = env.scene.add_camera(
        name="cine_camera", width=1280, height=720, fovy=np.deg2rad(fovy), near=0.05, far=10.0
    )
    if teacher:
        pos = np.array([-0.06, 0.60, 1.22], dtype=np.float64)
        look = np.array([-0.08, -0.06, 0.80], dtype=np.float64)
    else:
        pos = np.array([0.00, 0.62, 1.38], dtype=np.float64)
        look = np.array([0.00, -0.04, 0.772], dtype=np.float64)
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


def _strike(cue, target_xy: np.ndarray, speed: float) -> np.ndarray:
    """Launch the cue along -Y with no-slip roll; ``target_xy`` is unused."""
    del target_xy
    v = np.array([0.0, -float(speed), 0.0], dtype=np.float64)
    omega = np.cross(np.array([0.0, 0.0, 1.0]), v) / max(RADIUS, 1e-6)
    _set_velocity(cue, v, omega)
    return v


def _seat_start(env, cue, obj, obj_xy: np.ndarray) -> None:
    z = CENTER_Z
    cue_xy = np.asarray(env.CUE_XY, dtype=np.float64)
    _set_pose(cue, sapien.Pose([float(cue_xy[0]), float(cue_xy[1]), z], [1, 0, 0, 0]))
    _set_pose(obj, sapien.Pose([float(obj_xy[0]), float(obj_xy[1]), z], [1, 0, 0, 0]))
    _set_velocity(cue, np.zeros(3), np.zeros(3))
    _set_velocity(obj, np.zeros(3), np.zeros(3))


def _dry_run(env, cue, obj, *, fps: int, duration: float = 3.6) -> dict:
    dt = 1.0 / float(fps)
    substeps = max(int(round(dt / max(float(env.scene.get_timestep()), 1e-6))), 1)
    _seat_start(env, cue, obj, TEACHER_OBJ_XY)
    for _ in range(8):
        env.scene.step()
        env.pin_to_slate(cue)
        env.pin_to_slate(obj)
        _bounce_in_rails(cue)
        _bounce_in_rails(obj)
    _strike(cue, TEACHER_OBJ_XY, CUE_SPEED)
    n = max(int(round(duration * fps)), 8)
    cue_xyz = np.zeros((n, 3), dtype=np.float64)
    obj_xyz = np.zeros((n, 3), dtype=np.float64)
    cue_q = np.zeros((n, 4), dtype=np.float64)
    obj_q = np.zeros((n, 4), dtype=np.float64)
    obj_v = np.zeros((n, 3), dtype=np.float64)
    cue_v = np.zeros((n, 3), dtype=np.float64)
    cue_w = np.zeros((n, 3), dtype=np.float64)
    impact_i = None
    min_gap = 1e9
    for i in range(n):
        for _ in range(substeps):
            env.scene.step()
            env.pin_to_slate(cue)
            env.pin_to_slate(obj)
            _bounce_in_rails(cue)
            _bounce_in_rails(obj)
        cp = _copy_pose(cue)
        op = _copy_pose(obj)
        cue_xyz[i] = np.asarray(cp.p, dtype=np.float64)
        obj_xyz[i] = np.asarray(op.p, dtype=np.float64)
        cue_q[i] = np.asarray(cp.q, dtype=np.float64)
        obj_q[i] = np.asarray(op.q, dtype=np.float64)
        obj_v[i] = _get_velocity(obj)
        cue_v[i] = _get_velocity(cue)
        cue_w[i] = _get_angular(cue)
        gap = float(np.linalg.norm(cue_xyz[i, :2] - obj_xyz[i, :2]))
        if gap < min_gap:
            min_gap = gap
        if impact_i is None and float(np.linalg.norm(obj_v[i, :2])) > 0.08:
            impact_i = i
    if impact_i is None:
        impact_i = int(0.6 * fps)
    catch_i = None
    best_i = None
    best_score = -1e9
    for i in range(impact_i + int(0.35 * fps), n):
        xy = obj_xyz[i, :2]
        v = float(np.linalg.norm(obj_v[i, :2]))
        if v < 0.07:
            break
        in_ws = (-0.22 <= xy[0] <= 0.06) and (-0.22 <= xy[1] <= 0.06)
        sep = float(np.linalg.norm(xy - cue_xyz[i, :2]))
        score = (0.25 if in_ws else -0.4) + min(sep, 0.28) + 0.15 * min(v, 0.4)
        if score > best_score:
            best_score = score
            best_i = i
        if in_ws and sep > 0.16 and i >= impact_i + int(0.65 * fps):
            catch_i = i
            break
    if catch_i is None:
        catch_i = int(best_i if best_i is not None else min(impact_i + int(0.70 * fps), n - 1))
    return {
        "dt": dt,
        "n": n,
        "cue_xyz": cue_xyz,
        "obj_xyz": obj_xyz,
        "cue_q": cue_q,
        "obj_q": obj_q,
        "obj_v": obj_v,
        "cue_v": cue_v,
        "cue_w": cue_w,
        "impact_i": int(impact_i),
        "catch_i": int(catch_i),
        "min_gap": min_gap,
    }


def _pose_at(xyz: np.ndarray, quat: np.ndarray) -> sapien.Pose:
    return sapien.Pose(np.asarray(xyz, dtype=np.float64).tolist(), np.asarray(quat, dtype=np.float64).tolist())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--robotwin-root",
        default=os.environ.get("ROBOTWIN_ROOT", "/DATA/YuanZhen/FastWAM/third_party/RoboTwin"),
    )
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out-dir", default="evaluate_results/demo_videos/collide_pool_balls")
    p.add_argument("--tag", default="", help="Filename infix, e.g. teacher → collide_pool_balls_teacher_cine.mp4")
    args = p.parse_args()
    teacher = str(args.tag).strip() == "teacher"

    root = Path(args.robotwin_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.pop("VK_ICD_FILENAMES", None)
    os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
    try:
        sapien.render.clear_cache()
    except Exception:
        pass

    tw_args = bridge.load_task_args(root, task_config="demo_clean", task_name="place_object_basket")
    tw_args["_robotwin_root"] = str(root)
    tw_args["task_name"] = "collide_pool_balls"
    tw_args["eval_video_log"] = False
    tw_args["render_freq"] = 0
    tw_args.setdefault("data_type", {})
    tw_args["data_type"]["rgb"] = True
    tw_args["data_type"]["third_view"] = True

    cwd = os.getcwd()
    os.chdir(root)
    try:
        from envs.collide_pool_balls import collide_pool_balls

        env = collide_pool_balls()
        env.setup_demo(now_ep_num=0, seed=int(args.seed), is_test=True, **tw_args)
        env.set_instruction(
            instruction="Catch the red ball after the collision." if teacher else "Watch the pool balls collide."
        )
    finally:
        os.chdir(cwd)

    cue = env.cue
    obj = env.object
    cine = _install_cine_camera(env, teacher=teacher)
    dt = 1.0 / float(args.fps)
    substeps = max(int(round(dt / max(float(env.scene.get_timestep()), 1e-6))), 1)
    caption = CAPTION_TEACHER if teacher else CAPTION

    frames_cine: list[np.ndarray] = []
    frames_head: list[np.ndarray] = []

    def snap(label: str, t: float) -> None:
        rgbs = _grab_rgb(env, cine)
        cine_f = rgbs["cine"] if "cine" in rgbs else rgbs.get("head")
        if cine_f is not None:
            frames_cine.append(_annotate(cine_f, [caption, f"{label}   t={t:.2f}s"]))
        if "head" in rgbs:
            frames_head.append(rgbs["head"])

    def step_physics() -> None:
        for _ in range(substeps):
            env.scene.step()
            env.pin_to_slate(cue)
            env.pin_to_slate(obj)

    tag = f"_{str(args.tag).strip()}" if str(args.tag).strip() else ""

    if not teacher:
        t = 0.0
        impact_t = None
        min_gap = 1e9
        for _ in range(int(0.45 * args.fps)):
            step_physics()
            snap("ready", t)
            t += dt
        v_cue = _strike(cue, env.OBJ_XY, CUE_SPEED)
        print(
            f"[pool] cue={_xyz(cue).round(3).tolist()} obj={_xyz(obj).round(3).tolist()} "
            f"v={v_cue.round(3).tolist()} substeps={substeps}",
            flush=True,
        )
        done_t = t + 3.8
        label = "rolling"
        while t < done_t:
            step_physics()
            gap = float(np.linalg.norm(_xy(cue) - _xy(obj)))
            if gap < min_gap:
                min_gap = gap
            if impact_t is None and gap < 2.0 * RADIUS + 0.004:
                impact_t = t
                label = "impact"
            elif impact_t is not None and t > impact_t + 0.12:
                label = "scatter"
            vc = float(np.linalg.norm(_get_velocity(cue)[:2]))
            vo = float(np.linalg.norm(_get_velocity(obj)[:2]))
            wc = float(np.linalg.norm(_get_angular(cue)))
            wo = float(np.linalg.norm(_get_angular(obj)))
            if impact_t is not None and t > impact_t + 0.8 and vc < 0.04 and vo < 0.04 and wc < 1.2 and wo < 1.2:
                label = "rest"
            snap(label, t)
            t += dt
            if label == "rest" and t > impact_t + 1.15:
                for _ in range(int(0.35 * args.fps)):
                    step_physics()
                    snap("rest", t)
                    t += dt
                break
        cine_path = out_dir / f"collide_pool_balls{tag}_cine.mp4"
        head_path = out_dir / f"collide_pool_balls{tag}_head.mp4"
        cue_end = _xyz(cue)
        obj_end = _xyz(obj)
        meta = {
            "n_cine": len(frames_cine),
            "n_head": len(frames_head),
            "cine": str(cine_path),
            "head": str(head_path),
            "task": "collide_pool_balls",
            "mode": "physx_sphere_collision",
            "seed": args.seed,
            "radius_m": RADIUS,
            "cue_speed_mps": CUE_SPEED,
            "impact_t": impact_t,
            "min_center_gap_m": min_gap,
            "final_separation_m": float(np.linalg.norm(cue_end[:2] - obj_end[:2])),
            "cue_end": cue_end.tolist(),
            "obj_end": obj_end.tolist(),
        }
        if frames_cine:
            imageio.mimsave(cine_path.as_posix(), frames_cine, fps=args.fps, quality=8)
            frame_dir = out_dir / "frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            picks = {
                "c_01": 0,
                "c_02": max(len(frames_cine) // 4, 0),
                "c_03": max(len(frames_cine) // 2, 0),
                "c_04": max((3 * len(frames_cine)) // 4, 0),
                "c_05": len(frames_cine) - 1,
            }
            for name, idx in picks.items():
                Image.fromarray(frames_cine[idx]).save(frame_dir / f"{name}.png")
        if frames_head:
            imageio.mimsave(head_path.as_posix(), frames_head, fps=args.fps, quality=8)
        (out_dir / "recording_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2), flush=True)
        try:
            env.close_env(clear_cache=True)
        except Exception:
            pass
        return 0 if frames_cine else 1

    traj = _dry_run(env, cue, obj, fps=args.fps)
    impact_i = traj["impact_i"]
    catch_i = traj["catch_i"]
    obj_xyz = traj["obj_xyz"]
    cue_xyz = traj["cue_xyz"]
    impact_t = (impact_i + 1) * dt
    catch_t = (catch_i + 1) * dt
    lock_end = max(impact_t + 0.35, catch_t - DROP_DUR)
    print(
        f"[pool] teacher impact_t={impact_t:.2f}s catch_t={catch_t:.2f}s lock_end={lock_end:.2f}s "
        f"red={obj_xyz[catch_i].round(3).tolist()} v={float(np.linalg.norm(traj['obj_v'][catch_i, :2])):.3f} "
        f"min_gap={traj['min_gap']:.4f} cue_ymax={float(np.max(np.abs(cue_xyz[:, 1]))):.3f} "
        f"cue_xmax={float(np.max(np.abs(cue_xyz[:, 0]))):.3f}",
        flush=True,
    )
    for i in (0, impact_i, (impact_i + catch_i) // 2, catch_i):
        i = int(np.clip(i, 0, traj["n"] - 1))
        print(
            f"[pool] traj i={i} t={i * dt:.2f} cue={cue_xyz[i].round(3).tolist()} "
            f"red={obj_xyz[i].round(3).tolist()} v={float(np.linalg.norm(traj['obj_v'][i, :2])):.3f}",
            flush=True,
        )

    _ghost_physics(cue)
    _ghost_physics(obj)
    q_home = _arm_qpos(env)
    env.robot.set_gripper(1.0, "left", gripper_eps=0.0)
    _snap_arm(env, q_home)
    _set_pose(cue, _pose_at(cue_xyz[0], traj["cue_q"][0]))
    _set_pose(obj, _pose_at(obj_xyz[0], traj["obj_q"][0]))

    def _red_at(t: float) -> np.ndarray:
        idx = int(np.clip(round(float(t) / dt) - 1, 0, traj["n"] - 1))
        return obj_xyz[idx].copy()

    def _aim(t: float) -> tuple[np.ndarray, float]:
        t = float(t)
        park = np.array([PARK_XY[0], PARK_XY[1], PINCH_Z + HOVER_ABOVE + 0.06], dtype=np.float64)
        home_tcp = np.array([-0.22, -0.16, PINCH_Z + HOVER_ABOVE + 0.08], dtype=np.float64)
        settle = impact_t + 0.12
        if t <= settle:
            u = 0.55 * _smoothstep(t / max(settle, 1e-6))
            return (1.0 - u) * home_tcp + u * park, u
        u = _smoothstep((t - settle) / max(lock_end - settle, 1e-6))
        lag = (1.0 - u) * 0.16
        red = _red_at(max(impact_t, t - lag))
        hover = _hover_xyz(red)
        return (1.0 - u) * park + u * hover, u

    key_t: list[float] = [0.0]
    key_q: list[np.ndarray] = [q_home]
    q_hint = q_home
    n_hover_ok = 0
    blend_ts = np.linspace(0.12, lock_end, 12)
    for kt in blend_ts:
        xyz, u = _aim(float(kt))
        _set_pose(obj, _pose_at(_red_at(float(kt)), traj["obj_q"][min(int(kt / dt), traj["n"] - 1)]))
        q, quat, _cmd, err = _try_ik_cal(
            env, xyz, PINCH_QUATS, q_hint, max_jump=None if n_hover_ok == 0 else 2.2
        )
        if q is None:
            print(f"[pool] blend IK fail t={float(kt):.2f} u={u:.2f}", flush=True)
            continue
        key_t.append(float(kt))
        key_q.append(q)
        jump = float(np.linalg.norm(q - q_hint))
        q_hint = q
        n_hover_ok += 1
        print(
            f"[pool] blend t={float(kt):.2f} u={u:.2f} aim={xyz.round(3).tolist()} "
            f"tcp-err={err:.3f} jump={jump:.2f} tcp={_tcp(env).round(3).tolist()}",
            flush=True,
        )

    q_pinch = None
    pinch_xyz = _pinch_xyz(obj_xyz[catch_i])
    n_pinch_ok = 0
    drop_ts = np.linspace(lock_end, catch_t, 8)
    for kt in drop_ts:
        red = _red_at(float(kt))
        drop_a = 0.0 if catch_t <= lock_end else (float(kt) - lock_end) / DROP_DUR
        a = drop_a * drop_a
        xyz = (1.0 - a) * _hover_xyz(red) + a * _pinch_xyz(red)
        q, quat, _cmd, err = _try_ik_cal(env, xyz, PINCH_QUATS, q_hint, max_jump=2.2)
        if q is None:
            print(f"[pool] drop IK fail t={float(kt):.2f} a={drop_a:.2f}", flush=True)
            continue
        key_t.append(float(kt))
        key_q.append(q)
        jump = float(np.linalg.norm(q - q_hint))
        q_hint = q
        n_pinch_ok += 1
        if drop_a >= 0.99:
            q_pinch = q
            pinch_xyz = _pinch_xyz(red)
        print(
            f"[pool] drop t={float(kt):.2f} a={drop_a:.2f} tcp-err={err:.3f} jump={jump:.2f} "
            f"tcp={_tcp(env).round(3).tolist()}",
            flush=True,
        )

    grasp_ok = q_pinch is not None
    print(
        f"[pool] hover_keys={n_hover_ok} drop_keys={n_pinch_ok} pinch={grasp_ok} grip={GRASP_GRIP:.3f}",
        flush=True,
    )

    _snap_arm(env, q_home)
    env.robot.set_gripper(1.0, "left", gripper_eps=0.0)
    _set_pose(cue, _pose_at(cue_xyz[0], traj["cue_q"][0]))
    _set_pose(obj, _pose_at(obj_xyz[0], traj["obj_q"][0]))
    for _s in range(8):
        env.scene.step()
        _snap_arm(env, q_home)

    kt = np.asarray(key_t, dtype=np.float64)
    kq = np.stack(key_q, axis=0)

    def _q_at(motion_t: float) -> np.ndarray:
        tq = float(np.clip(motion_t, float(kt[0]), float(kt[-1])))
        i = int(np.searchsorted(kt, tq, side="right") - 1)
        i = max(0, min(i, len(kt) - 2))
        span = max(float(kt[i + 1] - kt[i]), 1e-6)
        a = float(np.clip((tq - kt[i]) / span, 0.0, 1.0))
        return (1.0 - a) * kq[i] + a * kq[i + 1]

    def _seat_i(i: int) -> None:
        i = int(np.clip(i, 0, traj["n"] - 1))
        _set_pose(cue, _pose_at(cue_xyz[i], traj["cue_q"][i]))
        _set_pose(obj, _pose_at(obj_xyz[i], traj["obj_q"][i]))

    n_chase = catch_i + 1
    # Keep the jaw open until the pads already flank the ball, then pinch.
    n_close_fly = 2
    pd_steps = 4
    t = 0.0
    last_q = q_home
    last_i = 0
    for i in range(n_chase):
        motion_t = float(i + 1) * dt
        motion_t = min(motion_t, catch_t)
        q_now = _q_at(motion_t)
        last_q = q_now
        last_i = i
        grip = 1.0
        phase = "break"
        if motion_t >= impact_t:
            phase = "scatter"
        if motion_t >= lock_end:
            phase = "intercept"
        if motion_t >= catch_t - n_close_fly * dt:
            a = float(i + 1 - (n_chase - n_close_fly)) / float(max(n_close_fly, 1))
            a = float(np.clip(a, 0.0, 1.0))
            grip = (1.0 - a) * 1.0 + a * GRASP_GRIP
            phase = "catch"
        _snap_arm(env, q_now)
        env.robot.set_gripper(float(grip), "left", gripper_eps=0.0)
        _seat_i(i)
        for _s in range(pd_steps):
            env.scene.step()
            _snap_arm(env, q_now)
            _seat_i(i)
        if i < 4 or i in (impact_i, (impact_i + catch_i) // 2, n_chase - 1):
            print(
                f"[pool] frame{i} t={motion_t:.3f} red={obj_xyz[i].round(3).tolist()} "
                f"tcp={_tcp(env).round(3).tolist()} arm_move={float(np.linalg.norm(q_now - q_home)):.3f}",
                flush=True,
            )
        snap(phase, motion_t)
        t = motion_t

    if q_pinch is not None:
        _snap_arm(env, q_pinch)
        last_q = q_pinch
    env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
    catch_pose = _pose_at(obj_xyz[last_i], traj["obj_q"][last_i])
    _set_pose(obj, catch_pose)

    # White ball keeps rolling on felt and cushions; ignore the gripper/red ball.
    _set_kinematic(cue, False)
    _set_groups(cue, _DEFAULT_GROUPS)
    _ent_c, comp_c = _rigid(cue)
    if comp_c is not None:
        try:
            comp_c.set_disable_gravity(False)
        except Exception:
            pass
    _set_pose(cue, _pose_at(cue_xyz[last_i], traj["cue_q"][last_i]))
    _set_velocity(cue, traj["cue_v"][last_i], traj["cue_w"][last_i])
    print(
        f"[pool] cue live v={np.asarray(traj['cue_v'][last_i, :2]).round(3).tolist()} "
        f"xy={cue_xyz[last_i, :2].round(3).tolist()}",
        flush=True,
    )

    def _step_cue() -> None:
        env.scene.step()
        env.pin_to_slate(cue)
        _bounce_in_rails(cue)

    for _s in range(pd_steps):
        _step_cue()
        _snap_arm(env, last_q)
        _set_pose(obj, catch_pose)
    tcp = _tcp(env)
    pinch = float(np.linalg.norm(tcp - pinch_xyz))
    dxy = tcp[:2] - obj_xyz[last_i, :2]
    dz = float(tcp[2] - obj_xyz[last_i, 2])
    print(
        f"[pool] pinch tcp-target={pinch:.3f} grip={GRASP_GRIP:.3f} "
        f"dxy={dxy.round(3).tolist()} dz={dz:.3f} "
        f"tcp={tcp.round(3).tolist()} red={obj_xyz[last_i].round(3).tolist()}",
        flush=True,
    )

    n_close = max(int(0.12 * args.fps), 3)
    for _ in range(n_close):
        env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
        _snap_arm(env, last_q)
        _set_pose(obj, catch_pose)
        for _s in range(8):
            _step_cue()
            _snap_arm(env, last_q)
            _set_pose(obj, catch_pose)
        snap("catch", t)
        t += dt

    _ghost_physics(obj, hard=True)
    rel = _ee_pose(env).inv() * catch_pose

    def _step_visual(label: str, t_now: float, q: np.ndarray) -> None:
        _snap_arm(env, q)
        _weld(obj, env, rel)
        for _s in range(8):
            _step_cue()
            _snap_arm(env, q)
            _weld(obj, env, rel)
        snap(label, t_now)

    ee0 = np.asarray(env.get_arm_pose("left"), dtype=np.float64)
    ee1 = ee0.copy()
    ee1[2] += 0.10
    q_lift = _ik_q(env, ee1)
    q_now = _arm_qpos(env)
    if q_lift is None:
        print("[pool] lift IK failed", flush=True)
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

    cue_end = _xyz(cue)
    print(
        f"[pool] cue_end={cue_end.round(3).tolist()} on_cloth={abs(cue_end[0]) < RAIL_INNER_X and abs(cue_end[1]) < RAIL_INNER_Y}",
        flush=True,
    )
    cine_path = out_dir / f"collide_pool_balls{tag}_cine.mp4"
    head_path = out_dir / f"collide_pool_balls{tag}_head.mp4"
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
        "task": "collide_pool_balls",
        "mode": "teacher_catch_red_after_break",
        "seed": args.seed,
        "radius_m": RADIUS,
        "cue_speed_mps": CUE_SPEED,
        "impact_t": impact_t,
        "catch_t": catch_t,
        "lock_end": lock_end,
        "grasp_ok": grasp_ok,
        "pinch_tcp_target_m": pinch,
        "grasp_grip": GRASP_GRIP,
        "red_catch_xyz": obj_xyz[last_i].tolist(),
        "cue_end": cue_end.tolist(),
        "min_center_gap_m": traj["min_gap"],
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
