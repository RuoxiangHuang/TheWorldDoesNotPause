"""Paired off vs full (or any accel pair) Dynamic-LIBERO evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
from benchmarks.dynamic_libero.env.suite_registry import parse_init_ids, parse_task_ids
from benchmarks.dynamic_libero.eval.cli_common import (
    add_dynamic_task_args,
    add_motive_arg,
    add_smooth_turn_args,
    dynamic_eval_units,
    grasp_kwargs_from_args,
    language_mode_from_args,
    motive_from_args,
    mover_language_from_args,
    movers_enabled_from_args,
    protocol_tag,
    rails_variant_from_args,
    resolve_trajectory_from_args,
    target_registry_from_args,
)
from benchmarks.dynamic_libero.trajectories.complexity import COMPLEXITY_CHOICES
from benchmarks.common.latency import LatencyMode, LatencyTrace, parse_latency_mode
from benchmarks.common.policy.factory import make_policy_client
from benchmarks.common.protocol import (
    DEFAULT_BACKEND,
    DEFAULT_DEADLINE_MS,
    RealtimeBackend,
    blind_window_ratio,
    effective_replan_steps,
)
from benchmarks.common.workspace import parse_aabb
from benchmarks.common.grasp import add_grasp_cli_args
from benchmarks.dynamic_libero.eval.measure_latency import (
    measure_latency_varying,
    n_freeze_from_latency,
)
from benchmarks.dynamic_libero.eval.rollout import run_episode


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired Dynamic-LIBERO eval")
    p.add_argument("--checkpoint", type=str, default=bridge.DEFAULT_CKPT)
    p.add_argument("--task-suite", type=str, default="libero_object")
    p.add_argument("--task-ids", type=str, default="0", help="Comma-separated task ids or 'all'")
    p.add_argument("--init-ids", type=str, default="0,1,2,3,4", help="Comma-separated init ids")
    p.add_argument("--accels", type=str, default="off,pace", help="Comma-separated accel presets")
    p.add_argument("--trajectory", type=str, default="linear")
    p.add_argument(
        "--traj-complexity",
        type=str,
        default="none",
        choices=list(COMPLEXITY_CHOICES),
        help="easy|medium|hard|chaotic preset (overrides --trajectory when not none)",
    )
    p.add_argument("--speed", type=float, default=0.003)
    p.add_argument("--axis", type=str, default="x")
    p.add_argument("--direction", type=float, default=1.0)
    p.add_argument(
        "--toward-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inward rails (default on). Does not change BDDL or language.",
    )
    p.add_argument("--radius", type=float, default=0.04)
    p.add_argument("--omega", type=float, default=0.03)
    p.add_argument("--amplitude", type=float, default=0.03)
    p.add_argument("--wavelength", type=float, default=0.20, help="sine wavelength in metres")
    p.add_argument("--n-waypoints", type=int, default=5)
    p.add_argument("--extent", type=float, default=0.12)
    p.add_argument("--release-radius", type=float, default=0.06)
    p.add_argument(
        "--workspace-aabb",
        type=str,
        default=None,
        help="Override escape AABB as xmin,ymin,zmin,xmax,ymax,zmax (metres)",
    )
    p.add_argument(
        "--no-escape-check",
        action="store_true",
        help="Disable workspace escape detection (default: on with suite defaults)",
    )
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--latency-k", type=int, default=8)
    p.add_argument("--latency-warmup", type=int, default=3)
    p.add_argument(
        "--fixed-latency",
        type=float,
        default=None,
        help="Force constant latency (seconds); implies --latency-mode constant",
    )
    p.add_argument(
        "--latency-mode",
        type=str,
        default="measured",
        choices=["measured", "replay", "constant"],
        help="Per-replan coupling: measured (default) / replay / constant (legacy)",
    )
    p.add_argument(
        "--latency-trace-dir",
        type=str,
        default=None,
        help="Write/read per-episode latency JSON traces under this directory",
    )
    p.add_argument(
        "--deadline-ms",
        type=float,
        default=DEFAULT_DEADLINE_MS,
        help="Deadline D for miss rate Pr[L>D] (default 50 ms)",
    )
    p.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND.value,
        choices=["async", "freeze"],
        help="Protocol default is async (prior chunk else ZOH); freeze is ablation",
    )
    p.add_argument(
        "--policy-url",
        type=str,
        default=None,
        help="HTTP policy server URL (e.g. http://127.0.0.1:8765). Omit for in-process.",
    )
    p.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help=f"Locked default {10}; pass --allow-replan-override to change",
    )
    p.add_argument("--allow-replan-override", action="store_true")
    p.add_argument(
        "--movers",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Optional toy-car asset swap (secondary track). Official primary: off.",
    )
    p.add_argument(
        "--mover-language",
        type=str,
        default="keep",
        choices=["rewrite", "keep"],
        help="Official primary track: keep.",
    )
    p.add_argument(
        "--language-mode",
        type=str,
        default="keep",
        choices=["keep", "motion", "scene"],
        help="keep=primary; motion=B-track motion suffix.",
    )
    p.add_argument(
        "--no-target-registry",
        action="store_true",
        help="Disable MoverSpec rails targeting (legacy BDDL inference).",
    )
    add_grasp_cli_args(p)
    add_motive_arg(p)
    add_dynamic_task_args(p)
    add_smooth_turn_args(p)
    from benchmarks.common.scene import add_scene_cli_args

    add_scene_cli_args(p)
    p.add_argument("--out-dir", type=str, default="evaluate_results/dynamic_libero/paired")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    units = dynamic_eval_units(args)
    if units:
        args.task_suite = units[0].suite
        task_ids = [u.task_id for u in units]
    else:
        task_ids = parse_task_ids(args.task_ids, args.task_suite)
        units = None
    init_ids = parse_init_ids(args.init_ids)
    accels = [x.strip() for x in args.accels.split(",") if x.strip()]
    _, traj_kw = resolve_trajectory_from_args(args)
    aabb_override = parse_aabb(args.workspace_aabb) if args.workspace_aabb else None
    escape_check = not args.no_escape_check
    movers_on = movers_enabled_from_args(args)
    mover_lang = mover_language_from_args(args)
    language_mode = language_mode_from_args(args)
    target_registry = target_registry_from_args(args)
    motive = motive_from_args(args)
    grasp_kw = grasp_kwargs_from_args(args)
    from benchmarks.common.scene import scene_kwargs_from_args

    scene_cond = scene_kwargs_from_args(
        args, speed=float(args.speed), catalog_job=units is not None
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{args.trajectory}_v{args.speed}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "protocol": protocol_tag(),
        "trajectory": args.trajectory,
        "speed": args.speed,
        "task_suite": args.task_suite,
        "task_ids": task_ids,
        "init_ids": init_ids,
        "accels": {},
        "paired_delta": {},
        "backend": args.backend,
        "policy_url": args.policy_url,
        "movers": movers_on,
        "mover_language": mover_lang,
        "language_mode": language_mode,
        "target_registry": target_registry,
        "motive": motive,
    }

    per_accel_success: dict[str, list[bool]] = {}

    for accel in accels:
        print(f"\n=== accel={accel} ===", flush=True)
        backend = RealtimeBackend(args.backend)
        if args.policy_url:
            policy = make_policy_client("libero", policy_url=args.policy_url)
            meta_ctx = bridge.load_env_context(
                task_suite_name=args.task_suite,
                task_id=task_ids[0],
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
            )
            task0 = meta_ctx.task_suite.get_task(task_ids[0])
            if args.fixed_latency is not None:
                lat = float(args.fixed_latency)
                raw = [lat]
            else:
                from benchmarks.common.policy.measure_http import measure_http_latency_libero

                lat, raw = measure_http_latency_libero(
                    args.policy_url,
                    task0,
                    suite=meta_ctx.task_suite,
                    task_id=task_ids[0],
                    seed=args.seed,
                    k=args.latency_k,
                    warmup=args.latency_warmup,
                )
            rsteps = effective_replan_steps(
                "libero", args.replan_steps or 10, allow_override=args.allow_replan_override
            )
            ctrl = float(meta_ctx.control_freq)
            n_freeze = n_freeze_from_latency(lat, ctrl)
            bwr = blind_window_ratio(n_freeze, rsteps)
            ctx = None
        else:
            ctx = bridge.load_policy(
                checkpoint=args.checkpoint,
                task_suite_name=args.task_suite,
                task_id=task_ids[0],
                seed=args.seed,
                device=args.device,
                num_inference_steps=args.num_inference_steps,
                accel=accel,
            )
            task = ctx.task_suite.get_task(task_ids[0])
            if args.fixed_latency is not None:
                lat = float(args.fixed_latency)
                raw = [lat]
            else:
                lat, raw = measure_latency_varying(
                    task, ctx, k=args.latency_k, warmup=args.latency_warmup
                )
            ctrl = float(ctx.cfg.EVALUATION.get("control_freq", 20))
            n_freeze = n_freeze_from_latency(lat, ctrl)
            rsteps = effective_replan_steps(
                "libero", args.replan_steps or ctx.cfg.EVALUATION.get("replan_steps"),
                allow_override=args.allow_replan_override,
            )
            bwr = blind_window_ratio(n_freeze, rsteps)
            policy = make_policy_client("libero", ctx=ctx)
        print(
            f"latency={lat*1000:.1f}ms n_freeze={n_freeze:.3f} "
            f"blind_window={bwr:.3f} replan_steps={rsteps} backend={backend.value}",
            flush=True,
        )

        rows: list[dict[str, Any]] = []
        jobs = units if units is not None else [None]
        for job in jobs:
            tids = [job.task_id] if job is not None else task_ids
            for tid in tids if job is None else [job.task_id]:
                if ctx is not None:
                    task = ctx.task_suite.get_task(tid)
                    inits = list(ctx.task_suite.get_task_init_states(tid))
                    episode_ctx = ctx
                else:
                    episode_ctx = bridge.load_env_context(
                        task_suite_name=args.task_suite,
                        task_id=tid,
                        seed=args.seed,
                        num_inference_steps=args.num_inference_steps,
                    )
                    task = episode_ctx.task_suite.get_task(tid)
                    inits = list(episode_ctx.task_suite.get_task_init_states(tid))
                dyn_kw = {}
                if job is not None:
                    dyn_kw = {
                        "rails_variant": job.variant,
                        "dynamic_task_id": job.id,
                        "place_target_name": job.place_target,
                    }
                else:
                    dyn_kw = {"rails_variant": rails_variant_from_args(args)}
                for iid in init_ids:
                    if iid >= len(inits):
                        continue
                    print(f"  task={tid} init={iid} ...", flush=True)
                    episode_ctx.cfg.EVALUATION.task_id = tid
                    result = run_episode(
                        task,
                        inits[iid],
                        episode_ctx,
                        trajectory_kind=args.trajectory,
                        speed=args.speed,
                        trajectory_kwargs=traj_kw,
                        n_freeze=scene_cond.apply_n_freeze(n_freeze),
                        release_radius=args.release_radius,
                        max_steps=args.max_steps,
                        backend=backend,
                        policy=policy,
                        replan_steps=args.replan_steps,
                        allow_replan_override=args.allow_replan_override,
                        task_suite_name=args.task_suite,
                        workspace_aabb=aabb_override,
                        escape_check=escape_check,
                        task_id=tid,
                        movers_enabled=movers_on,
                        mover_language=mover_lang,
                        language_mode=language_mode,
                        target_registry=target_registry,
                        motive=motive,
                        scene=scene_cond,
                        **grasp_kw,
                        **dyn_kw,
                        init_id=iid,
                    )
                    result.update(
                        {
                            "task_id": tid,
                            "init_id": iid,
                            "accel": accel,
                            "latency_s": lat,
                            "dynamic_task_id": None if job is None else job.id,
                            "rails_variant": dyn_kw.get("rails_variant"),
                        }
                    )
                    rows.append(result)
                    print(
                        f"    success={result['success']} escaped={result['escaped']} steps={result['steps']}",
                        flush=True,
                    )

        succ = [bool(r["success"]) for r in rows]
        per_accel_success[accel] = succ
        sr = float(np.mean(succ)) if succ else 0.0
        summary["accels"][accel] = {
            "latency_s": lat,
            "latency_ms": lat * 1000.0,
            "latency_raw": raw,
            "n_freeze": n_freeze,
            "pursuit_lag_m": float(args.speed) * n_freeze,
            "blind_window_ratio": bwr,
            "backend": backend.value,
            "replan_steps": rsteps,
            "sr": sr,
            "n": len(succ),
            "episodes": rows,
        }
        with open(out_dir / f"episodes_{accel}.json", "w") as f:
            json.dump(summary["accels"][accel], f, indent=2)

        # Free GPU before next accel
        if ctx is not None:
            del ctx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Paired ΔSR for first two accels
    if len(accels) >= 2:
        a0, a1 = accels[0], accels[1]
        s0, s1 = per_accel_success[a0], per_accel_success[a1]
        n = min(len(s0), len(s1))
        delta = float(np.mean([float(s1[i]) - float(s0[i]) for i in range(n)])) if n else 0.0
        summary["paired_delta"] = {
            f"SR({a1})-SR({a0})": delta,
            "SR_" + a0: float(np.mean(s0)) if s0 else 0.0,
            "SR_" + a1: float(np.mean(s1)) if s1 else 0.0,
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("accels", "paired_delta") if k in summary}, indent=2))
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
