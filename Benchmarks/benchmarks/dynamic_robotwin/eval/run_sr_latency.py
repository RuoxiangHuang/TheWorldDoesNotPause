#!/usr/bin/env python3
"""Hardware-decoupled SR(L) characteristic curve for Dynamic-RoboTwin.

Injects synthetic latencies ``L`` (seconds). Anchors:
  - L=0     → oracle (n_freeze=0)
  - L=inf   → open-loop (thinking never finishes; robot never acts)
"""

from __future__ import annotations

import argparse
import json
import math
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
from benchmarks.common.protocol import DEFAULT_BACKEND
from benchmarks.dynamic_robotwin.eval.cli_common import protocol_tag
from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge
from benchmarks.dynamic_robotwin.env.eval_config import add_dynamics_cli_args
from benchmarks.dynamic_robotwin.env.task_registry import parse_dynamic_modes
from benchmarks.dynamic_robotwin.eval.measure_latency import (
    OPEN_LOOP_LATENCY_S,
    n_freeze_from_latency,
)
from benchmarks.dynamic_robotwin.eval.rollout import run_episode
from benchmarks.dynamic_robotwin.metrics.aggregate import write_curve_csv

DEFAULT_LATENCIES = [0.0, 0.025, 0.05, 0.1, 0.2, 0.4]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SR(L) characteristic curve (RoboTwin)")
    p.add_argument("--checkpoint", type=str, default=bridge.DEFAULT_CKPT)
    p.add_argument("--stats", type=str, default=bridge.DEFAULT_STATS)
    p.add_argument("--task", type=str, default=bridge.DEFAULT_TASK)
    p.add_argument("--task-name", type=str, default="place_empty_cup")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--accel", type=str, default="off")
    p.add_argument("--trajectory", type=str, default="linear")
    p.add_argument("--speed", type=float, default=0.001)
    p.add_argument("--axis", type=str, default=None)
    p.add_argument("--direction", type=float, default=None)
    p.add_argument(
        "--latencies",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LATENCIES),
    )
    p.add_argument("--include-open-loop", action="store_true")
    p.add_argument("--release-radius", type=float, default=None)
    p.add_argument("--physics-per-tick", type=int, default=12)
    p.add_argument("--policy-hz", type=float, default=20.0)
    p.add_argument("--replan-steps", type=int, default=8)
    p.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND.value,
        choices=["freeze", "async"],
    )
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--robotwin-root", type=str, default=str(bridge.DEFAULT_ROBOTWIN_ROOT))
    p.add_argument("--task-config", type=str, default="demo_clean")
    p.add_argument("--out-dir", type=str, default="evaluate_results/dynamic_robotwin/sr_latency")
    add_dynamics_cli_args(p)
    add_language_mode_arg(p)
    from benchmarks.common.scene import add_scene_cli_args

    add_scene_cli_args(p)
    return p.parse_args()


def _parse_latencies(s: str, include_open_loop: bool) -> list[float]:
    out: list[float] = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in ("inf", "+inf", "infinity", "open", "open_loop", "open-loop"):
            out.append(OPEN_LOOP_LATENCY_S)
        else:
            out.append(float(tok))
    if include_open_loop and not any(not math.isfinite(x) for x in out):
        out.append(OPEN_LOOP_LATENCY_S)
    return out


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))

    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    latencies = _parse_latencies(args.latencies, args.include_open_loop)
    traj_kw: dict[str, Any] = {}
    if args.axis is not None:
        traj_kw["axis"] = args.axis
    if args.direction is not None:
        traj_kw["direction"] = args.direction
        traj_kw["toward_center"] = False

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{args.task_name}_{args.trajectory}_v{args.speed}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from benchmarks.common.scene import scene_kwargs_from_args
    from benchmarks.dynamic_robotwin.env.dynamic_tasks import is_catalog_eval

    scene_cond = scene_kwargs_from_args(
        args,
        speed=float(args.speed),
        catalog_job=is_catalog_eval(str(args.task_name)),
    )

    print(f"=== Load accel={args.accel} (synthetic L injection) ===", flush=True)
    ctx = bridge.load_policy(
        checkpoint=args.checkpoint,
        stats=args.stats,
        task=args.task,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        replan_steps=args.replan_steps,
        seed=args.seed,
        accel=args.accel,
        policy_hz=args.policy_hz,
    )

    cells: list[dict[str, Any]] = []
    try:
        for lat in latencies:
            n_freeze = scene_cond.apply_n_freeze(n_freeze_from_latency(lat, args.policy_hz))
            lat_label = "inf" if not math.isfinite(lat) else f"{lat:.6g}"
            nf_label = "inf" if not math.isfinite(n_freeze) else f"{n_freeze:.3f}"
            lat_print = lat_label if lat_label == "inf" else f"{lat_label}s"
            print(f"--- L={lat_print} n_freeze={nf_label} ---", flush=True)
            rows = []
            for sd in seeds:
                try:
                    r = run_episode(
                        args.task_name,
                        sd,
                        ctx,
                        trajectory_kind=args.trajectory,
                        speed=args.speed,
                        trajectory_kwargs=traj_kw,
                        n_freeze=n_freeze,
                        release_radius=args.release_radius,
                        physics_per_tick=args.physics_per_tick,
                        robotwin_root=args.robotwin_root,
                        task_config=args.task_config,
                        backend=args.backend,
                        replan_steps=args.replan_steps,
                        dynamic_modes=parse_dynamic_modes(args.dynamic_modes),
                        contact_switch=args.contact_trigger,
                        occlusion=(args.occlusion == "on"),
                        occlusion_duty=args.occlusion_duty,
                        contact_chain=(args.contact_chain == "on"),
                        secondary_rails=args.secondary_rails,
                        language_mode=args.language_mode,
                        scene=scene_cond,
                    )
                except Exception as e:
                    print(f"  seed={sd}: ERROR {e}", flush=True)
                    r = {
                        "success": False,
                        "escaped": False,
                        "steps": 0,
                        "error": str(e),
                        "task_name": args.task_name,
                        "seed": sd,
                        "trajectory": args.trajectory,
                        "speed": args.speed,
                        "drift_m": 0.0,
                    }
                r.update(
                    {
                        "accel": "injected_L",
                        "latency_s": lat if math.isfinite(lat) else None,
                        "latency_anchor": lat_label,
                    }
                )
                rows.append(r)
                print(
                    f"  seed={sd}: success={r.get('success')} escaped={r.get('escaped')} "
                    f"drift={r.get('drift_m', 0):.4f}",
                    flush=True,
                )
            sr = float(np.mean([bool(x.get("success")) for x in rows])) if rows else 0.0
            cell = {
                "accel": "injected_L",
                "policy_accel": args.accel,
                "trajectory": args.trajectory,
                "speed": args.speed,
                "latency_s": lat if math.isfinite(lat) else None,
                "latency_ms": (lat * 1000.0) if math.isfinite(lat) else None,
                "latency_anchor": lat_label,
                "n_freeze": n_freeze if math.isfinite(n_freeze) else None,
                "n_freeze_anchor": nf_label,
                "pursuit_lag_m": (float(args.speed) * n_freeze) if math.isfinite(n_freeze) else None,
                "sr": sr,
                "n": len(rows),
                "episodes": rows,
            }
            cells.append(cell)
            with open(out_dir / f"cell_L{lat_label}.json", "w") as f:
                json.dump(cell, f, indent=2)
    finally:
        del ctx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    curve = [
        {
            "latency_anchor": c["latency_anchor"],
            "latency_ms": c["latency_ms"],
            "n_freeze": c["n_freeze"],
            "sr": c["sr"],
            "n": c["n"],
            "pursuit_lag_m": c["pursuit_lag_m"],
        }
        for c in cells
    ]
    payload = {
        "protocol": protocol_tag(),
        "curve_type": "SR(L)",
        "task_name": args.task_name,
        "seeds": seeds,
        "trajectory": args.trajectory,
        "speed": args.speed,
        "policy_accel": args.accel,
        "policy_hz": args.policy_hz,
        "backend": args.backend,
        "language_mode": parse_language_mode(args.language_mode),
        "anchors": {"oracle": "L=0", "open_loop": "L=inf"},
        "curve": curve,
        "cells": cells,
    }
    with open(out_dir / "sr_latency.json", "w") as f:
        json.dump(payload, f, indent=2)
    csv_cells = []
    for c in cells:
        csv_cells.append(
            {
                "accel": c["latency_anchor"],
                "trajectory": args.trajectory,
                "speed": args.speed,
                "sr": c["sr"],
                "n": c["n"],
                "latency_ms": c["latency_ms"] if c["latency_ms"] is not None else -1.0,
                "n_freeze": c["n_freeze"] if c["n_freeze"] is not None else -1.0,
                "pursuit_lag_m": c["pursuit_lag_m"] if c["pursuit_lag_m"] is not None else -1.0,
            }
        )
    write_curve_csv(csv_cells, out_dir / "sr_latency.csv")

    lines = [
        "# SR(L) characteristic curve — Dynamic-RoboTwin",
        "",
        f"- task: `{args.task_name}`",
        f"- trajectory: `{args.trajectory}` @ speed `{args.speed}`",
        f"- policy_accel (weights only): `{args.accel}`",
        "",
        "| L | latency_ms | n_freeze | pursuit_lag_m | SR | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in curve:
        lat_ms = "inf" if c["latency_ms"] is None else f"{c['latency_ms']:.1f}"
        nf = "inf" if c["n_freeze"] is None else f"{c['n_freeze']:.3f}"
        lag = "n/a" if c["pursuit_lag_m"] is None else f"{c['pursuit_lag_m']:.4f}"
        lines.append(
            f"| {c['latency_anchor']} | {lat_ms} | {nf} | {lag} | {c['sr']:.3f} | {c['n']} |"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"curve": curve}, indent=2))
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
