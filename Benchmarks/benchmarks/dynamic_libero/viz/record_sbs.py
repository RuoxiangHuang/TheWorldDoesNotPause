#!/usr/bin/env python3
"""Wall-clock Dynamic-LIBERO side-by-side recorder.

Uses ``RealtimeDriver`` (this package) — does **not** import FastWAM
``dynamic_libero_sr``. Object speed is identical across panels; latency appears
as THINKING freeze frames.
"""

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

from benchmarks.dynamic_libero.env import libero_bridge as bridge
from benchmarks.dynamic_libero.env.driver import RealtimeDriver, wrap_env_with_driver
from benchmarks.common.grasp import add_grasp_cli_args, parse_grasp_mode
from benchmarks.common.language import apply_language_mode, parse_language_mode
from benchmarks.dynamic_libero.env.movers import (
    MoverSpec,
    mover_assets,
    parse_mover_language,
    parse_movers_flag,
    rewrite_language,
)
from benchmarks.dynamic_libero.eval.measure_latency import (
    freeze_segments,
    measure_latency_varying,
    n_freeze_from_latency,
)
from benchmarks.dynamic_libero.trajectories import build_trajectory
from benchmarks.dynamic_libero.viz.annotate import annotate, render_agent


def _grasp_overlay(driver: RealtimeDriver, obs: dict | None = None) -> list[str]:
    phase = getattr(driver.rails_phase, "value", driver.rails_phase)
    lines = [f"grasp={driver.grasp_mode} phase={phase}"]
    dist_m = None
    if obs is not None and driver.target is not None:
        rel = obs.get(f"{driver.target}_to_robot0_eef_pos")
        if rel is not None:
            dist_m = float(np.linalg.norm(np.asarray(rel, dtype=np.float64)))
    if dist_m is not None:
        lines.append(f"dist={dist_m * 100:.1f}cm released={driver.released}")
    else:
        lines.append(f"released={driver.released}")
    return lines


@torch.no_grad()
def rollout_sbs(
    name: str,
    task,
    init_state,
    ctx: bridge.EpisodeContext,
    *,
    trajectory_kind: str,
    speed: float,
    trajectory_kwargs: dict[str, Any],
    n_freeze: float,
    release_radius: float,
    max_steps: int,
    render_res: int,
    fps: int,
    latency_ms: float,
    tail_hold: int = 20,
    mover_spec: Optional[MoverSpec] = None,
    grasp_mode: str = "ghost",
    engage_alpha: float = 1.75,
    task_desc: str | None = None,
    scene=None,
):
    from benchmarks.common.scene import (
        resolve_scene_condition,
        trajectory_speed_for_condition,
    )

    if scene is None:
        scene = resolve_scene_condition(speed=float(speed), catalog_job=False)
    traj = build_trajectory(
        trajectory_kind,
        speed=trajectory_speed_for_condition(scene, float(speed)),
        **trajectory_kwargs,
    )
    env, desc = bridge.get_libero_env(task, render_res, ctx.cfg.get("seed"))
    if task_desc is not None:
        desc = task_desc
    driver = RealtimeDriver(
        env,
        traj,
        release_radius=release_radius,
        mover=mover_spec,
        grasp_mode=grasp_mode,
        engage_alpha=engage_alpha,
        scene=scene,
    )
    wrap_env_with_driver(env, driver)

    replan_steps = int(ctx.cfg.EVALUATION.get("replan_steps", 10))
    num_steps_wait = int(ctx.cfg.EVALUATION.get("num_steps_wait", 30))
    freeze_ticks = float(n_freeze)

    env.reset()
    obs = env.set_init_state(init_state)
    frames: list = []
    pending: list = []
    done = False
    escaped = False
    t = 0

    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(bridge.dummy_action())
                t += 1
                continue

            if not driver.motion_enabled:
                driver.begin_policy()

            if len(pending) == 0:
                chunk, _, _ = bridge.predict_action_chunk(obs, desc, ctx)
                segs = freeze_segments(freeze_ticks)
                n_seg = max(len(segs), 1)
                elapsed = 0.0
                for f, dt in enumerate(segs):
                    driver.tick(dt)
                    elapsed += dt
                    think_ms = elapsed / max(float(fps), 1.0) * 1000.0
                    drift_cm = driver.drift_m * 100.0
                    frames.append(
                        annotate(
                            render_agent(env),
                            [
                                f"accel={name}",
                                f"THINKING...  {think_ms:.0f} ms  "
                                f"(freeze {elapsed:.2f}/{freeze_ticks:.2f} ticks, "
                                f"seg {f + 1}/{n_seg}, lat~{latency_ms:.0f}ms)",
                                f"robot FROZEN, object @ {speed * 100:.1f} cm/tick "
                                f"-> drift {drift_cm:.1f} cm",
                            ]
                            + _grasp_overlay(driver, obs),
                        )
                    )
                pending = chunk[:replan_steps].tolist()

            agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            rails = "chasing" if driver.rails_active else "caught/static"
            frames.append(
                annotate(
                    agent,
                    [
                        f"accel={name}",
                        f"executing  (object speed {speed * 100:.1f} cm/tick, SAME both panels)",
                        f"object drift {driver.drift_m * 100:.1f} cm  [{rails}]",
                    ]
                    + _grasp_overlay(driver, obs),
                )
            )
            obs, _, done, info = env.step(pending.pop(0))
            t += 1
            if isinstance(info, dict) and info.get("escaped"):
                escaped = True
            if driver.escaped:
                escaped = True
            if done or escaped:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    success = bool(done) and not escaped
    final = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    for _ in range(tail_hold):
        frames.append(
            annotate(final, [f"accel={name}", f"task: {desc}", "outcome:"], ok=success)
        )
    return frames, success


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default=bridge.DEFAULT_CKPT)
    p.add_argument("--stats", type=str, default=bridge.DEFAULT_STATS)
    p.add_argument("--task", type=str, default=bridge.DEFAULT_TASK)
    p.add_argument("--task-suite", type=str, default="libero_object")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-idx", type=int, default=0)
    p.add_argument("--accels", type=str, default="off,pace")
    p.add_argument("--trajectory", type=str, default="linear")
    p.add_argument("--speed", type=float, default=0.003)
    p.add_argument("--axis", type=str, default="x")
    p.add_argument("--direction", type=float, default=1.0)
    p.add_argument("--release-radius", type=float, default=0.06)
    p.add_argument("--render-res", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=280)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--latency-k", type=int, default=8)
    p.add_argument("--latency-warmup", type=int, default=3)
    p.add_argument(
        "--out-dir",
        type=str,
        default="evaluate_results/dynamic_libero/viz",
    )
    p.add_argument("--movers", type=str, default="on", choices=["on", "off"])
    p.add_argument("--mover-language", type=str, default="rewrite", choices=["rewrite", "keep"])
    p.add_argument(
        "--language-mode",
        type=str,
        default="keep",
        choices=["keep", "motion", "scene"],
        help="keep=primary instruction; motion=B-track suffix.",
    )
    p.add_argument(
        "--grasp-modes",
        type=str,
        default="",
        help="Comma-separated ghost,hybrid_a,hybrid_b for grasp SBS (uses first --accels entry)",
    )
    add_grasp_cli_args(p)
    from benchmarks.common.scene import add_scene_cli_args

    add_scene_cli_args(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    accels = [x.strip() for x in args.accels.split(",") if x.strip()]
    grasp_modes = [parse_grasp_mode(x.strip()) for x in args.grasp_modes.split(",") if x.strip()]
    if grasp_modes:
        panel_names = grasp_modes
        accel_for_grasp = accels[0] if accels else "full"
        accels = [accel_for_grasp]
    else:
        panel_names = accels
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj_kw = {"axis": args.axis, "direction": args.direction}
    movers_on = parse_movers_flag(args.movers)
    lang_mode = parse_mover_language(args.mover_language)
    from benchmarks.common.scene import scene_kwargs_from_args

    scene_cond = scene_kwargs_from_args(
        args, speed=float(args.speed), catalog_job=False
    )

    clips: dict[str, list] = {}
    meta: dict[str, Any] = {
        "protocol": "real_time_v1",
        "mode": "wallclock",
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "init_idx": args.init_idx,
        "trajectory": args.trajectory,
        "speed": args.speed,
        "fps": args.fps,
        "movers": movers_on,
        "mover_language": lang_mode,
        "language_mode": args.language_mode,
        "grasp_modes": grasp_modes or [getattr(args, "grasp_mode", "ghost")],
        "configs": {},
    }

    with mover_assets(args.task_suite, args.task_id, enabled=movers_on) as mover_spec:
        for panel in panel_names:
            grasp_mode = panel if grasp_modes else getattr(args, "grasp_mode", "ghost")
            name = panel if grasp_modes else panel
            print(f"\n=== panel={name} grasp={grasp_mode} ===", flush=True)
            ctx = bridge.load_policy(
                checkpoint=args.checkpoint,
                stats=args.stats,
                task=args.task,
                task_suite_name=args.task_suite,
                task_id=args.task_id,
                seed=args.seed,
                device=args.device,
                num_inference_steps=args.num_inference_steps,
                accel=accels[0],
            )
            task = ctx.task_suite.get_task(args.task_id)
            desc = task.language
            if movers_on and lang_mode == "rewrite" and mover_spec is not None:
                desc = rewrite_language(desc, mover_spec)
            noun = mover_spec.category if mover_spec is not None else None
            effective_lang = parse_language_mode(args.language_mode) if scene_cond.advances_rails() else "keep"
            desc = apply_language_mode(desc, effective_lang, noun=noun)
            lat, raw = measure_latency_varying(
                task, ctx, k=args.latency_k, warmup=args.latency_warmup
            )
            ctrl = float(ctx.cfg.EVALUATION.get("control_freq", args.fps))
            n_freeze = scene_cond.apply_n_freeze(n_freeze_from_latency(lat, ctrl))
            latency_ms = lat * 1000.0
            print(f"latency={latency_ms:.1f}ms n_freeze={n_freeze:.3f}", flush=True)

            inits = list(ctx.task_suite.get_task_init_states(args.task_id))
            t0 = time.time()
            frames, success = rollout_sbs(
                name,
                task,
                inits[args.init_idx],
                ctx,
                trajectory_kind=args.trajectory,
                speed=args.speed,
                trajectory_kwargs=traj_kw,
                n_freeze=n_freeze,
                release_radius=args.release_radius,
                max_steps=args.max_steps,
                render_res=args.render_res,
                fps=args.fps,
                latency_ms=latency_ms,
                mover_spec=mover_spec,
                grasp_mode=grasp_mode,
                engage_alpha=float(getattr(args, "engage_alpha", 1.75)),
                task_desc=desc,
                scene=scene_cond,
            )
            suffix = f"grasp-{grasp_mode}" if grasp_modes else f"accel-{name}"
            path = out / (
                f"rtl_{args.task_suite}_t{args.task_id}_i{args.init_idx}"
                f"_{args.trajectory}_v{args.speed}_{suffix}.mp4"
            )
            imageio.mimsave(path.as_posix(), frames, fps=args.fps)
            clips[name] = frames
            meta["configs"][name] = {
                "grasp_mode": grasp_mode,
                "engage_alpha": float(getattr(args, "engage_alpha", 1.75)),
                "latency_ms": round(latency_ms, 1),
                "latency_raw_ms": [round(x * 1000, 1) for x in raw],
                "n_freeze": n_freeze,
                "pursuit_lag_m": args.speed * n_freeze,
                "success": bool(success),
                "frames": len(frames),
                "path": path.as_posix(),
                "record_wall_s": round(time.time() - t0, 1),
                "task_description": desc,
            }
            print(
                f"{'SUCCESS' if success else 'FAIL'} -> {path} ({time.time() - t0:.0f}s)",
                flush=True,
            )
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
            f"rtl_{args.task_suite}_t{args.task_id}_i{args.init_idx}"
            f"_{args.trajectory}_v{args.speed}_{tag}.mp4"
        )
        imageio.mimsave(sbs_path.as_posix(), sbs, fps=args.fps)
        meta["side_by_side"] = {
            "left": names[0],
            "right": names[1],
            "path": sbs_path.as_posix(),
            "object_speed_aligned": True,
            "grasp_comparison": bool(grasp_modes),
        }
        print(f"SBS -> {sbs_path}", flush=True)

    meta_path = out / (
        f"rtl_{args.task_suite}_t{args.task_id}_i{args.init_idx}"
        f"_{args.trajectory}_v{args.speed}_meta.json"
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"meta -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
