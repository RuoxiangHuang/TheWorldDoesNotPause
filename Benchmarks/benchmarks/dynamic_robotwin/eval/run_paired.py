"""Paired off vs full Dynamic-RoboTwin evaluation."""

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

from benchmarks.common.language import add_language_mode_arg, parse_language_mode
from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge
from benchmarks.dynamic_robotwin.env.dynamic_tasks import resolve_eval_spec
from benchmarks.dynamic_robotwin.env.eval_config import add_dynamics_cli_args
from benchmarks.dynamic_robotwin.env.task_registry import parse_dynamic_modes
from benchmarks.common.policy.factory import make_policy_client
from benchmarks.common.protocol import (
    DEFAULT_BACKEND,
    RealtimeBackend,
    blind_window_ratio,
    effective_replan_steps,
)
from benchmarks.common.workspace import parse_aabb
from benchmarks.dynamic_robotwin.eval.cli_common import grasp_kwargs_from_args, protocol_tag
from benchmarks.common.grasp import add_grasp_cli_args
from benchmarks.dynamic_robotwin.eval.measure_latency import (
    measure_latency_varying,
    n_freeze_from_latency,
)
from benchmarks.dynamic_robotwin.eval.rollout import run_episode


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired Dynamic-RoboTwin eval")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--stats", type=str, default=bridge.DEFAULT_STATS)
    p.add_argument("--task", type=str, default=bridge.DEFAULT_TASK)
    p.add_argument("--task-name", type=str, default="place_empty_cup")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--accels", type=str, default="off,pace")
    p.add_argument("--trajectory", type=str, default="linear")
    p.add_argument("--speed", type=float, default=0.001)
    p.add_argument("--axis", type=str, default=None)
    p.add_argument("--direction", type=float, default=None)
    p.add_argument("--release-radius", type=float, default=None)
    p.add_argument(
        "--physics-per-tick",
        type=int,
        default=None,
        help="PhysX scene.step count per policy tick; default auto-calibrate per episode",
    )
    p.add_argument(
        "--no-calibrate-physics",
        action="store_true",
        help="Skip auto-calibration; use --physics-per-tick or default 12",
    )
    p.add_argument(
        "--workspace-aabb",
        type=str,
        default=None,
        help="Override escape AABB as xmin,ymin,zmin,xmax,ymax,zmax (metres)",
    )
    p.add_argument(
        "--no-escape-check",
        action="store_true",
        help="Disable workspace escape detection (default: on with task defaults)",
    )
    p.add_argument("--policy-hz", type=float, default=20.0)
    p.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="Locked default 8; pass --allow-replan-override to change",
    )
    p.add_argument("--allow-replan-override", action="store_true")
    p.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND.value,
        choices=["freeze", "async"],
        help="Protocol default is async (prior chunk else ZOH); freeze is ablation",
    )
    p.add_argument("--policy-url", type=str, default=None)
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--latency-k", type=int, default=8)
    p.add_argument("--latency-warmup", type=int, default=3)
    p.add_argument("--fixed-latency", type=float, default=None, help="Override measured latency (seconds)")
    p.add_argument("--robotwin-root", type=str, default=str(bridge.DEFAULT_ROBOTWIN_ROOT))
    p.add_argument("--task-config", type=str, default="demo_clean")
    p.add_argument("--out-dir", type=str, default="evaluate_results/dynamic_robotwin/paired")
    add_dynamics_cli_args(p)
    add_language_mode_arg(p)
    add_grasp_cli_args(p)
    from benchmarks.common.scene import add_scene_cli_args

    add_scene_cli_args(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))

    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    accels = [x.strip() for x in args.accels.split(",") if x.strip()]
    traj_kw: dict[str, Any] = {}
    if args.axis is not None:
        traj_kw["axis"] = args.axis
    if args.direction is not None:
        traj_kw["direction"] = args.direction
        traj_kw["toward_center"] = False
    aabb_override = parse_aabb(args.workspace_aabb) if args.workspace_aabb else None
    escape_check = not args.no_escape_check
    auto_calibrate_physics = not args.no_calibrate_physics
    grasp_kw = grasp_kwargs_from_args(args)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{args.task_name}_{args.trajectory}_v{args.speed}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "protocol": protocol_tag(),
        "task_name": args.task_name,
        "trajectory": args.trajectory,
        "speed": args.speed,
        "seeds": seeds,
        "accels": {},
        "paired_delta": {},
        "backend": args.backend,
        "policy_url": args.policy_url,
        "language_mode": parse_language_mode(args.language_mode),
    }
    per_accel_success: dict[str, list[bool]] = {}

    for accel in accels:
        print(f"\n=== accel={accel} ===", flush=True)
        rsteps = effective_replan_steps(
            "robotwin", args.replan_steps or 8, allow_override=args.allow_replan_override
        )
        backend = RealtimeBackend(args.backend)
        spec = resolve_eval_spec(args.task_name)
        instr = spec.language
        if args.policy_url:
            from dataclasses import dataclass

            @dataclass
            class _StubCtx:
                replan_steps: int
                policy_hz: float

            ctx = _StubCtx(replan_steps=rsteps, policy_hz=float(args.policy_hz))
            if args.fixed_latency is not None:
                lat = float(args.fixed_latency)
                raw = [lat]
            else:
                from benchmarks.common.policy.measure_http import measure_http_latency_robotwin

                lat, raw = measure_http_latency_robotwin(
                    args.policy_url,
                    args.task_name,
                    instr,
                    robotwin_root=args.robotwin_root,
                    task_config=args.task_config,
                    seed=seeds[0],
                    k=args.latency_k,
                    warmup=args.latency_warmup,
                )
            policy = make_policy_client("robotwin", policy_url=args.policy_url)
        else:
            ctx = bridge.load_policy(
                checkpoint=args.checkpoint or bridge.DEFAULT_CKPT,
                stats=args.stats,
                task=args.task,
                device=args.device,
                num_inference_steps=args.num_inference_steps,
                replan_steps=rsteps,
                seed=args.seed,
                accel=accel,
                policy_hz=args.policy_hz,
            )
            if args.fixed_latency is not None:
                lat = float(args.fixed_latency)
                raw = [lat]
            else:
                lat, raw = measure_latency_varying(
                    args.task_name,
                    ctx,
                    robotwin_root=args.robotwin_root,
                    task_config=args.task_config,
                    seed=seeds[0],
                    k=args.latency_k,
                    warmup=args.latency_warmup,
                )
            policy = make_policy_client("robotwin", ctx=ctx)
        from benchmarks.common.scene import scene_kwargs_from_args
        from benchmarks.dynamic_robotwin.env.dynamic_tasks import is_catalog_eval

        scene_cond = scene_kwargs_from_args(
            args,
            speed=float(args.speed),
            catalog_job=is_catalog_eval(str(args.task_name)),
        )
        n_freeze = scene_cond.apply_n_freeze(n_freeze_from_latency(lat, args.policy_hz))
        bwr = blind_window_ratio(n_freeze, rsteps)
        print(
            f"latency={lat*1000:.1f}ms n_freeze={n_freeze:.3f} "
            f"blind_window={bwr:.3f} replan_steps={rsteps} backend={backend.value}",
            flush=True,
        )

        rows = []
        for sd in seeds:
            print(f"  seed={sd} ...", flush=True)
            try:
                result = run_episode(
                    args.task_name,
                    sd,
                    ctx,
                    trajectory_kind=args.trajectory,
                    speed=args.speed,
                    trajectory_kwargs=traj_kw,
                    n_freeze=n_freeze,
                    release_radius=args.release_radius,
                    physics_per_tick=args.physics_per_tick,
                    auto_calibrate_physics=auto_calibrate_physics,
                    policy_hz=args.policy_hz,
                    robotwin_root=args.robotwin_root,
                    task_config=args.task_config,
                    backend=backend,
                    policy=policy,
                    replan_steps=rsteps,
                    allow_replan_override=args.allow_replan_override,
                    workspace_aabb=aabb_override,
                    escape_check=escape_check,
                    dynamic_modes=parse_dynamic_modes(args.dynamic_modes),
                    contact_switch=args.contact_trigger,
                    occlusion=(args.occlusion == "on"),
                    occlusion_duty=args.occlusion_duty,
                    contact_chain=(args.contact_chain == "on"),
                    secondary_rails=args.secondary_rails,
                    language_mode=args.language_mode,
                    scene=scene_cond,
                    **grasp_kw,
                )
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                result = {
                    "success": False,
                    "escaped": False,
                    "steps": 0,
                    "n_replans": 0,
                    "n_freeze": n_freeze,
                    "drift_m": 0.0,
                    "released": False,
                    "trajectory": args.trajectory,
                    "speed": args.speed,
                    "task_name": args.task_name,
                    "seed": sd,
                    "error": str(e),
                }
            result.update({"accel": accel, "latency_s": lat})
            rows.append(result)
            print(
                f"    success={result['success']} escaped={result.get('escaped')} steps={result['steps']}",
                flush=True,
            )

        succ = [bool(r["success"]) for r in rows]
        per_accel_success[accel] = succ
        summary["accels"][accel] = {
            "latency_s": lat,
            "latency_ms": lat * 1000.0,
            "latency_raw": raw,
            "n_freeze": n_freeze,
            "pursuit_lag_m": float(args.speed) * n_freeze,
            "blind_window_ratio": bwr,
            "backend": backend.value,
            "replan_steps": rsteps,
            "sr": float(np.mean(succ)) if succ else 0.0,
            "n": len(succ),
            "episodes": rows,
        }
        with open(out_dir / f"episodes_{accel}.json", "w") as f:
            json.dump(summary["accels"][accel], f, indent=2)
        del ctx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    print(json.dumps(summary.get("paired_delta", {}), indent=2))
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
