#!/usr/bin/env python3
"""Collect extra intercept oracle demos (8-GPU parallel).

Each rank writes episode folders under ``--out-dir/raw/rank{r}/``. Convert with
``python -m benchmarks.dynamic_robotwin.extra.to_lerobot``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import sapien
from PIL import Image

for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

ROOT = setup_sys_path()

from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge
from benchmarks.dynamic_robotwin.viz.record_catch_shuttlecock import (
    GRASP_GRIP,
    TABLE_Z,
    _arm_q,
    _drive_arm,
    _ee_pose,
    _first_ik,
    _ghost_physics,
    _ik_q,
    _pose7,
    _set_kinematic,
    _set_pose,
    _tcp,
    _grasp_quats,
    _weld,
)


INSTRUCTION = "Catch the flying shuttlecock."
FPS = 20
# Cine probe stays 0.36 s. Acquisition uses a slightly slower lob so the
# arm can intercept while the bird actually flies (eval starts flight at t=0).
BASE_P0 = np.array([0.20, 0.00, 0.84], dtype=np.float64)
BASE_V0 = np.array([-0.55, 0.00, 3.05], dtype=np.float64)
BASE_CATCH_T = 0.55
K_M = 0.30


def _cams(env) -> dict[str, np.ndarray]:
    env._update_render()
    obs = env.get_obs()
    o = obs["observation"]
    return {
        "head": np.ascontiguousarray(o["head_camera"]["rgb"]),
        "left": np.ascontiguousarray(o["left_camera"]["rgb"]),
        "right": np.ascontiguousarray(o["right_camera"]["rgb"]),
        "qpos": np.asarray(obs["joint_action"]["vector"], dtype=np.float32),
    }


def _resize(rgb: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if rgb.shape[0] == h and rgb.shape[1] == w:
        return np.ascontiguousarray(rgb)
    return np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR), dtype=np.uint8)


def _jitter(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    p0 = BASE_P0 + rng.uniform(-1.0, 1.0, size=3) * np.array([0.04, 0.04, 0.02])
    scale = float(rng.uniform(0.92, 1.08))
    yaw = float(rng.uniform(-0.10, 0.10))
    c, s = np.cos(yaw), np.sin(yaw)
    v = BASE_V0 * scale
    v0 = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], dtype=np.float64)
    catch_t = BASE_CATCH_T * float(rng.uniform(0.88, 1.12))
    return p0, v0, catch_t


def _interp_arm(env, q_goal: np.ndarray, n: int, grip: float, flying: bool, bird, phys_dt: float, sub: int):
    q0 = _arm_q(env)
    steps = max(int(n), 1)
    pose0 = env.flight_pose()
    for i in range(steps):
        a = float(i + 1) / float(steps)
        _drive_arm(env, (1.0 - a) * q0 + a * q_goal)
        env.robot.set_gripper(float(grip), "left", gripper_eps=0.0)
        if flying:
            for _ in range(sub):
                env.step_flight(phys_dt)
                _set_pose(bird, env.flight_pose())
                env.scene.step()
                _set_pose(bird, env.flight_pose())
        else:
            _set_pose(bird, pose0)
            for _ in range(sub):
                env.scene.step()
                _set_pose(bird, pose0)
        yield _cams(env)


def _hold(env, n: int, bird, pose, phys_dt: float, sub: int, grip: float | None = None):
    for _ in range(max(int(n), 1)):
        if grip is not None:
            env.robot.set_gripper(float(grip), "left", gripper_eps=0.0)
        _set_pose(bird, pose)
        for _ in range(sub):
            env.scene.step()
            _set_pose(bird, pose)
        yield _cams(env)


def collect_episode(env, tw_args, seed: int, phys_dt: float, sub: int) -> dict | None:
    cwd = os.getcwd()
    root = Path(tw_args["_robotwin_root"])
    os.chdir(root)
    try:
        env.setup_demo(now_ep_num=0, seed=int(seed), is_test=True, **tw_args)
        env.set_instruction(instruction=INSTRUCTION)
    finally:
        os.chdir(cwd)
    rng = np.random.default_rng(int(seed))
    p0, v0, catch_t = _jitter(rng)
    env.P0 = p0
    env.V0 = v0
    env.K_M = K_M
    env.CATCH_T = float(catch_t)
    env.reset_flight()
    bird = env.object
    _set_kinematic(bird, True)
    _ghost_physics(bird)
    _set_pose(bird, env.flight_pose())
    catch_pose = env.pose_at(env.CATCH_T)
    catch_xyz = np.asarray(catch_pose.p, dtype=np.float64)
    if catch_xyz[2] < TABLE_Z + 0.08:
        print(f"[collect] skip low catch z={catch_xyz[2]:.3f}", flush=True)
        return None
    pinch_quats = _grasp_quats()
    q_pinch, pinch_quat, pinch_xyz = None, None, catch_xyz.copy()
    for dz in (0.0, -0.02, 0.02, -0.04):
        cand = catch_xyz + np.array([0.0, 0.0, dz], dtype=np.float64)
        q_try, quat_try = _first_ik(env, cand, pinch_quats)
        if q_try is not None:
            q_pinch, pinch_quat, pinch_xyz = q_try, quat_try, cand
            break
    if q_pinch is None:
        print(f"[collect] skip no pinch IK catch={catch_xyz.round(3).tolist()}", flush=True)
        return None
    wait_xyz = pinch_xyz + np.array([0.00, -0.03, 0.00], dtype=np.float64)
    wait_quats = (pinch_quat,) + pinch_quats
    q_wait, _ = _first_ik(env, wait_xyz, wait_quats)
    if q_wait is None:
        q_wait = q_pinch

    frames: list[dict] = []
    env.reset_flight()
    q0 = _arm_q(env)
    q_goal = q_wait if q_wait is not None else q_pinch
    # Reach while the bird flies — matches ExtraNativeDriver (flight starts at t=0).
    catch_t = float(env.CATCH_T)
    max_reach = max(8, int(round(catch_t * FPS)) + 4)
    for i in range(max_reach):
        t_frac = min(1.0, (float(env.flight_t) + 1.0 / FPS) / max(catch_t, 1e-6))
        if t_frac < 0.70:
            a = t_frac / 0.70
            q = (1.0 - a) * q0 + a * q_goal
        else:
            b = (t_frac - 0.70) / 0.30
            q = (1.0 - b) * q_goal + b * q_pinch
        _drive_arm(env, q)
        env.robot.set_gripper(1.0, "left", gripper_eps=0.0)
        for _ in range(sub):
            env.step_flight(phys_dt)
            _set_pose(bird, env.flight_pose())
            env.scene.step()
            _set_pose(bird, env.flight_pose())
        frames.append(_cams(env))
        if float(env.flight_t) >= catch_t:
            break
    catch_now = env.pose_at(min(float(env.flight_t), float(env.CATCH_T)))
    env.flight_p = np.asarray(catch_now.p, dtype=np.float64).copy()
    env.flight_q = np.asarray(catch_now.q, dtype=np.float64).copy()
    env.flight_t = float(env.CATCH_T)
    _set_pose(bird, catch_now)
    tcp = _tcp(env)
    if float(np.linalg.norm(np.asarray(catch_now.p) - tcp)) > 0.14:
        print(
            f"[collect] skip far pinch dist={float(np.linalg.norm(np.asarray(catch_now.p)-tcp)):.3f}",
            flush=True,
        )
        return None
    n_close = 6
    cork0 = np.asarray(catch_now.p, dtype=np.float64)
    for i in range(n_close):
        a = float(i + 1) / float(n_close)
        env.robot.set_gripper((1.0 - a) * 1.0 + a * GRASP_GRIP, "left", gripper_eps=0.0)
        p = (1.0 - a) * cork0 + a * _tcp(env)
        pose_i = sapien.Pose(p.tolist(), list(catch_now.q))
        _set_pose(bird, pose_i)
        for _ in range(sub):
            env.scene.step()
            _set_pose(bird, pose_i)
        frames.append(_cams(env))
    rel = _ee_pose(env).inv() * sapien.Pose(_tcp(env).tolist(), list(catch_now.q))
    lift_xyz = pinch_xyz + np.array([0.0, 0.0, 0.08], dtype=np.float64)
    q_lift, _ = _first_ik(env, lift_xyz, (pinch_quat,) + pinch_quats)
    n_lift = 16
    if q_lift is not None:
        q0 = _arm_q(env)
        for i in range(n_lift):
            a = float(i + 1) / float(n_lift)
            _drive_arm(env, (1.0 - a) * q0 + a * q_lift)
            env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
            _weld(bird, env, rel, "left")
            for _s in range(sub):
                env.scene.step()
                _weld(bird, env, rel, "left")
            frames.append(_cams(env))
    else:
        for _ in range(n_lift):
            _weld(bird, env, rel, "left")
            env.robot.set_gripper(GRASP_GRIP, "left", gripper_eps=0.0)
            for _s in range(sub):
                env.scene.step()
                _weld(bird, env, rel, "left")
            frames.append(_cams(env))

    success = False
    try:
        success = bool(env.check_success())
    except Exception:
        tcp = _tcp(env)
        p = np.asarray(bird.get_pose().p, dtype=np.float64)
        success = float(np.linalg.norm(p - tcp)) < 0.16 and float(p[2] - TABLE_Z) > 0.10
    if not success or len(frames) < 22:
        tcp = _tcp(env)
        p = np.asarray(bird.get_pose().p, dtype=np.float64)
        print(
            f"[collect] skip success={success} n={len(frames)} "
            f"dist={float(np.linalg.norm(p-tcp)):.3f} z={float(p[2]-TABLE_Z):.3f}",
            flush=True,
        )
        return None
    head = np.stack([_resize(f["head"], (480, 640)) for f in frames], axis=0)
    left = np.stack([_resize(f["left"], (480, 640)) for f in frames], axis=0)
    right = np.stack([_resize(f["right"], (480, 640)) for f in frames], axis=0)
    qpos = np.stack([f["qpos"] for f in frames], axis=0).astype(np.float32)
    action = np.concatenate([qpos[1:], qpos[-1:]], axis=0)
    return {
        "head": head,
        "left": left,
        "right": right,
        "qpos": qpos,
        "action": action,
        "instruction": INSTRUCTION,
        "success": True,
        "seed": int(seed),
        "catch_t": float(env.CATCH_T),
        "p0": p0.tolist(),
        "v0": v0.tolist(),
        "n": int(len(frames)),
    }


def _make_env(robotwin_root: Path, seed: int):
    tw_args = bridge.load_task_args(robotwin_root, task_config="demo_clean", task_name="place_object_basket")
    tw_args["_robotwin_root"] = str(robotwin_root)
    tw_args["task_name"] = "catch_shuttlecock"
    tw_args["eval_video_log"] = False
    tw_args["render_freq"] = 0
    tw_args.setdefault("data_type", {})
    tw_args["data_type"]["rgb"] = True
    tw_args["data_type"]["third_view"] = True
    cwd = os.getcwd()
    os.chdir(robotwin_root)
    try:
        from envs.catch_shuttlecock import catch_shuttlecock

        env = catch_shuttlecock()
        env.setup_demo(now_ep_num=0, seed=int(seed), is_test=True, **tw_args)
        env.set_instruction(instruction=INSTRUCTION)
    finally:
        os.chdir(cwd)
    return env, tw_args


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--episodes-per-rank", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument(
        "--out-dir",
        default="evaluate_results/extra_oracle/catch_shuttlecock",
    )
    p.add_argument(
        "--robotwin-root",
        default=os.environ.get("ROBOTWIN_ROOT", "/DATA/YuanZhen/FastWAM/third_party/RoboTwin"),
    )
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.pop("VK_ICD_FILENAMES", None)
    os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)

    root = Path(args.robotwin_root).resolve()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    rank_dir = out / "raw" / f"rank{int(args.rank)}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    env, _tw = _make_env(root, seed=int(args.seed_base) + int(args.rank))
    try:
        phys_dt = float(env.scene.get_timestep())
    except Exception:
        phys_dt = 1.0 / 250.0
    sub = max(int(round((1.0 / FPS) / max(phys_dt, 1e-6))), 1)

    saved = 0
    attempts = 0
    max_attempts = int(args.episodes_per_rank) * 8
    seed = int(args.seed_base) + int(args.rank) * 10_000
    print(
        f"[collect] rank={args.rank}/{args.world_size} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"target={args.episodes_per_rank} sub={sub}",
        flush=True,
    )
    while saved < int(args.episodes_per_rank) and attempts < max_attempts:
        attempts += 1
        seed += int(args.world_size)
        try:
            ep = collect_episode(env, _tw, seed, phys_dt, sub)
        except Exception as exc:
            print(f"[collect] rank={args.rank} seed={seed} err={exc}", flush=True)
            ep = None
        if ep is None:
            continue
        ep_dir = rank_dir / f"ep{saved:05d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.save(ep_dir / "head.npy", ep["head"])
        np.save(ep_dir / "left.npy", ep["left"])
        np.save(ep_dir / "right.npy", ep["right"])
        np.save(ep_dir / "qpos.npy", ep["qpos"])
        np.save(ep_dir / "action.npy", ep["action"])
        meta = {k: ep[k] for k in ("instruction", "success", "seed", "catch_t", "p0", "v0", "n")}
        (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        saved += 1
        print(
            f"[collect] rank={args.rank} saved={saved}/{args.episodes_per_rank} "
            f"n={ep['n']} catch_t={ep['catch_t']:.2f} seed={ep['seed']}",
            flush=True,
        )
        # Recreate env every few successes — SAPIEN RT caches get unhappy.
        if saved % 5 == 0 and saved < int(args.episodes_per_rank):
            try:
                env.close_env(clear_cache=True)
            except Exception:
                pass
            env, _tw = _make_env(root, seed=seed + 1)
    try:
        env.close_env(clear_cache=True)
    except Exception:
        pass
    summary = {"rank": args.rank, "saved": saved, "attempts": attempts}
    (rank_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
