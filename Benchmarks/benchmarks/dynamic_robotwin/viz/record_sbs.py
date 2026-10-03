#!/usr/bin/env python3
"""Wall-clock Dynamic-RoboTwin side-by-side recorder."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import imageio
import numpy as np
import torch

import sys
from pathlib import Path
for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

ROOT = setup_sys_path()

from benchmarks.dynamic_libero.trajectories import build_trajectory
from benchmarks.dynamic_libero.viz.annotate import annotate
from benchmarks.common.grasp import add_grasp_cli_args, parse_grasp_mode
from benchmarks.dynamic_robotwin.env.driver import RealtimeRoboTwinDriver, wrap_env_with_driver
from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge
from benchmarks.dynamic_robotwin.env.task_registry import (
    default_instruction,
    default_release_radius,
    resolve_traj_kwargs,
)
from benchmarks.dynamic_robotwin.eval.measure_latency import (
    freeze_segments,
    measure_latency_varying,
    n_freeze_from_latency,
)


def _grasp_overlay(driver: RealtimeRoboTwinDriver) -> list[str]:
    phase = getattr(driver.rails_phase, "value", driver.rails_phase)
    lines = [f"grasp={driver.grasp_mode} phase={phase}"]
    dist, _ = driver._tcp_dist_and_closing()
    if dist is not None:
        lines.append(f"dist={dist * 100:.1f}cm released={driver.released}")
    else:
        lines.append(f"released={driver.released}")
    return lines


def _head_rgb(env) -> np.ndarray:
    obs = env.get_obs()
    rgb = obs["observation"]["head_camera"]["rgb"]
    return np.ascontiguousarray(rgb)


def rollout_sbs(
    name: str,
    task_name: str,
    seed: int,
    ctx: bridge.EpisodeContext,
    *,
    trajectory_kind: str,
    speed: float,
    trajectory_kwargs: dict[str, Any],
    n_freeze: float,
    release_radius: Optional[float],
    physics_per_tick: int,
    robotwin_root,
    task_config: str,
    fps: int,
    latency_ms: float,
    tail_hold: int = 15,
    grasp_mode: str = "ghost",
    engage_alpha: float = 1.75,
    scene=None,
):
    from benchmarks.common.scene import (
        resolve_scene_condition,
        trajectory_speed_for_condition,
    )

    if scene is None:
        scene = resolve_scene_condition(speed=float(speed), catalog_job=False)
    traj_kw = resolve_traj_kwargs(task_name, dict(trajectory_kwargs or {}))
    release = (
        float(release_radius)
        if release_radius is not None
        else default_release_radius(task_name)
    )
    traj = build_trajectory(
        trajectory_kind,
        speed=trajectory_speed_for_condition(scene, float(speed)),
        **traj_kw,
    )
    args = bridge.load_task_args(robotwin_root, task_config=task_config, task_name=task_name)
    args["_robotwin_root"] = str(robotwin_root)
    env = bridge.create_task_env(task_name, robotwin_root)
    instr = default_instruction(task_name)
    driver = RealtimeRoboTwinDriver(
        env,
        traj,
        task_name=task_name,
        release_radius=release,
        physics_per_tick=physics_per_tick,
        grasp_mode=grasp_mode,
        engage_alpha=engage_alpha,
        scene=scene,
    )
    wrap_env_with_driver(env, driver)

    freeze_ticks = float(n_freeze)
    frames: list = []
    pending: list = []
    escaped = False
    try:
        bridge.setup_episode(env, args, seed=seed, instruction=instr, ep_num=0)
        driver.on_episode_start()
        driver.begin_policy()
        obs = env.get_obs()

        while env.take_action_cnt < env.step_lim and not env.eval_success:
            if len(pending) == 0:
                chunk = bridge.predict_action_chunk(obs, instr, ctx)
                segs = freeze_segments(freeze_ticks)
                n_seg = max(len(segs), 1)
                elapsed = 0.0
                for f, dt in enumerate(segs):
                    driver.tick(dt)
                    elapsed += dt
                    think_ms = elapsed / max(float(fps), 1.0) * 1000.0
                    frames.append(
                        annotate(
                            _head_rgb(env),
                            [
                                f"accel={name}",
                                f"THINKING... {think_ms:.0f} ms "
                                f"(freeze {elapsed:.2f}/{freeze_ticks:.2f} ticks, "
                                f"seg {f+1}/{n_seg}, lat~{latency_ms:.0f}ms)",
                                f"robot HOLD, object @ {speed*100:.1f} cm/tick "
                                f"drift {driver.drift_m*100:.1f} cm",
                            ]
                            + _grasp_overlay(driver),
                        )
                    )
                    if driver.escaped:
                        escaped = True
                        break
                if escaped:
                    break
                pending = [
                    np.asarray(chunk[i], dtype=np.float32)
                    for i in range(min(ctx.replan_steps, len(chunk)))
                ]

            if not pending:
                break
            frames.append(
                annotate(
                    np.ascontiguousarray(obs["observation"]["head_camera"]["rgb"]),
                    [
                        f"accel={name}",
                        f"executing (object speed {speed*100:.1f} cm/tick, SAME both panels)",
                        f"drift {driver.drift_m*100:.1f} cm "
                        f"[{'chasing' if driver.rails_active else 'released'}]",
                    ]
                    + _grasp_overlay(driver),
                )
            )
            env.take_action(pending.pop(0), action_type="qpos")
            if driver.escaped:
                escaped = True
                break
            if env.eval_success:
                break
            if len(pending) == 0:
                obs = env.get_obs()

        success = bool(env.eval_success) and not escaped
        final = _head_rgb(env)
        for _ in range(tail_hold):
            frames.append(
                annotate(final, [f"accel={name}", f"task: {instr}", "outcome:"], ok=success)
            )
    finally:
        bridge.close_task_env(env)
    return frames, success


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default=bridge.DEFAULT_CKPT)
    p.add_argument("--stats", type=str, default=bridge.DEFAULT_STATS)
    p.add_argument("--task", type=str, default=bridge.DEFAULT_TASK)
    p.add_argument("--task-name", type=str, default="place_empty_cup")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--accels", type=str, default="off,pace")
    p.add_argument("--trajectory", type=str, default="linear")
    p.add_argument("--speed", type=float, default=0.003)
    p.add_argument("--axis", type=str, default=None)
    p.add_argument("--direction", type=float, default=None)
    p.add_argument("--release-radius", type=float, default=None)
    p.add_argument("--physics-per-tick", type=int, default=12)
    p.add_argument("--policy-hz", type=float, default=20.0)
    p.add_argument("--replan-steps", type=int, default=8)
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--latency-k", type=int, default=8)
    p.add_argument("--latency-warmup", type=int, default=3)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--robotwin-root", type=str, default=str(bridge.DEFAULT_ROBOTWIN_ROOT))
    p.add_argument("--task-config", type=str, default="demo_clean")
    p.add_argument("--out-dir", type=str, default="evaluate_results/dynamic_robotwin/viz")
    p.add_argument(
        "--grasp-modes",
        type=str,
        default="",
        help="Comma-separated ghost,hybrid_a,hybrid_b for grasp SBS (uses first --accels entry)",
    )
    p.add_argument(
        "--dynamic-modes",
        type=str,
        default="none",
        help="Dynamic stress modes; use none for grasp demos.",
    )
    add_grasp_cli_args(p)
    from benchmarks.common.scene import add_scene_cli_args

    add_scene_cli_args(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    accels = [x.strip() for x in args.accels.split(",") if x.strip()]
    grasp_modes = [parse_grasp_mode(x.strip()) for x in args.grasp_modes.split(",") if x.strip()]
    if grasp_modes:
        panel_names = grasp_modes
        accels = [accels[0] if accels else "full"]
    else:
        panel_names = accels
    from benchmarks.common.scene import scene_kwargs_from_args
    from benchmarks.dynamic_robotwin.env.dynamic_tasks import is_catalog_eval

    scene_cond = scene_kwargs_from_args(
        args,
        speed=float(args.speed),
        catalog_job=is_catalog_eval(str(args.task_name)),
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj_kw: dict[str, Any] = {}
    if args.axis is not None:
        traj_kw["axis"] = args.axis
    if args.direction is not None:
        traj_kw["direction"] = args.direction
        traj_kw["toward_center"] = False

    clips: dict[str, list] = {}
    meta: dict[str, Any] = {
        "protocol": "real_time_v2",
        "task_name": args.task_name,
        "seed": args.seed,
        "trajectory": args.trajectory,
        "speed": args.speed,
        "fps": args.fps,
        "grasp_modes": grasp_modes or [getattr(args, "grasp_mode", "ghost")],
        "configs": {},
    }

    for panel in panel_names:
        grasp_mode = panel if grasp_modes else getattr(args, "grasp_mode", "ghost")
        name = panel
        print(f"\n=== panel={name} grasp={grasp_mode} ===", flush=True)
        ctx = bridge.load_policy(
            checkpoint=args.checkpoint,
            stats=args.stats,
            task=args.task,
            device=args.device,
            num_inference_steps=args.num_inference_steps,
            replan_steps=args.replan_steps,
            seed=args.seed,
            accel=accels[0],
            policy_hz=args.policy_hz,
        )
        lat, raw = measure_latency_varying(
            args.task_name,
            ctx,
            robotwin_root=args.robotwin_root,
            task_config=args.task_config,
            seed=args.seed,
            k=args.latency_k,
            warmup=args.latency_warmup,
        )
        n_freeze = scene_cond.apply_n_freeze(n_freeze_from_latency(lat, args.policy_hz))
        latency_ms = lat * 1000.0
        print(f"latency={latency_ms:.1f}ms n_freeze={n_freeze:.3f}", flush=True)
        t0 = time.time()
        frames, success = rollout_sbs(
            name,
            args.task_name,
            args.seed,
            ctx,
            trajectory_kind=args.trajectory,
            speed=args.speed,
            trajectory_kwargs=traj_kw,
            n_freeze=n_freeze,
            release_radius=args.release_radius,
            physics_per_tick=args.physics_per_tick,
            robotwin_root=args.robotwin_root,
            task_config=args.task_config,
            fps=args.fps,
            latency_ms=latency_ms,
            grasp_mode=grasp_mode,
            engage_alpha=float(getattr(args, "engage_alpha", 1.75)),
            scene=scene_cond,
        )
        suffix = f"grasp-{grasp_mode}" if grasp_modes else f"accel-{name}"
        path = out / (
            f"rtl_tw_{args.task_name}_s{args.seed}_{args.trajectory}_v{args.speed}_{suffix}.mp4"
        )
        imageio.mimsave(path.as_posix(), frames, fps=args.fps)
        clips[name] = frames
        meta["configs"][name] = {
            "grasp_mode": grasp_mode,
            "engage_alpha": float(getattr(args, "engage_alpha", 1.75)),
            "latency_ms": round(latency_ms, 1),
            "n_freeze": n_freeze,
            "success": bool(success),
            "frames": len(frames),
            "path": path.as_posix(),
            "record_wall_s": round(time.time() - t0, 1),
        }
        print(f"{'SUCCESS' if success else 'FAIL'} -> {path}", flush=True)
        del ctx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(clips) >= 2:
        names = panel_names[: min(len(panel_names), 2)] if grasp_modes else accels[:2]
        a, b = clips[names[0]], clips[names[1]]
        n = max(len(a), len(b))
        a = a + [a[-1]] * (n - len(a))
        b = b + [b[-1]] * (n - len(b))
        sbs = [np.concatenate([fa, fb], axis=1) for fa, fb in zip(a, b)]
        tag = "grasp-SBS" if grasp_modes else "SIDEBYSIDE"
        sbs_path = out / (
            f"rtl_tw_{args.task_name}_s{args.seed}_{args.trajectory}_v{args.speed}_{tag}.mp4"
        )
        imageio.mimsave(sbs_path.as_posix(), sbs, fps=args.fps)
        meta["side_by_side"] = {
            "left": names[0],
            "right": names[1],
            "path": sbs_path.as_posix(),
            "grasp_comparison": bool(grasp_modes),
        }
        print(f"SBS -> {sbs_path}", flush=True)

    meta_path = out / (
        f"rtl_tw_{args.task_name}_s{args.seed}_{args.trajectory}_v{args.speed}_meta.json"
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"meta -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
