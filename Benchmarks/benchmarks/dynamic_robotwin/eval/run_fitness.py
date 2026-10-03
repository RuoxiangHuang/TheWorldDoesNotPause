#!/usr/bin/env python3
"""Diagnostic latency-sensitivity scan for Dynamic-RoboTwin.

This does **not** decide main-track eligibility. Main vs extension is
semantic (language still true, scene legal, check_success correct).

For each task, estimate:
  - SR_original @ official spawn, no apparatus
  - SR_slow / SR_fast @ apparatus_drive with synthetic L
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

ROOT = setup_sys_path()

from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge
from benchmarks.dynamic_robotwin.env.official import (
    OFFICIAL,
    contract_banner,
    parse_task_names,
)
from benchmarks.dynamic_robotwin.eval.measure_latency import n_freeze_from_latency
from benchmarks.dynamic_robotwin.eval.rollout import run_episode
from benchmarks.dynamic_robotwin.metrics.aggregate import bootstrap_ci


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic-RoboTwin task fitness scan")
    p.add_argument("--checkpoint", type=str, default=bridge.DEFAULT_CKPT)
    p.add_argument("--stats", type=str, default=bridge.DEFAULT_STATS)
    p.add_argument("--task-names", type=str, default="all")
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in OFFICIAL.seeds))
    p.add_argument("--accel", type=str, default="full")
    p.add_argument("--trajectory", type=str, default="linear")
    p.add_argument("--probe-speed", type=float, default=0.001)
    p.add_argument("--latency-slow-ms", type=float, default=200.0)
    p.add_argument("--latency-fast-ms", type=float, default=0.0)
    p.add_argument("--min-static-sr", type=float, default=0.2)
    p.add_argument("--min-delta-sr", type=float, default=0.1)
    p.add_argument("--policy-hz", type=float, default=OFFICIAL.policy_hz)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--robotwin-root", type=str, default=str(bridge.DEFAULT_ROBOTWIN_ROOT))
    p.add_argument(
        "--dynamics-track",
        type=str,
        default="rails_only",
        choices=["primary", "rails_only"],
        help="rails_only isolates latency coupling; primary includes §9 stresses",
    )
    p.add_argument("--out-dir", type=str, default="evaluate_results/dynamic_robotwin/fitness")
    return p.parse_args()


def _dyn_kwargs(track: str) -> dict[str, Any]:
    if track == "rails_only":
        return {
            "dynamic_modes": [],
            "contact_switch": "off",
            "occlusion": False,
            "contact_chain": False,
            "secondary_rails": "off",
        }
    return {
        "dynamic_modes": list(OFFICIAL.dynamic_modes),
        "contact_switch": OFFICIAL.contact_switch,
        "occlusion": OFFICIAL.occlusion,
        "contact_chain": OFFICIAL.contact_chain,
        "secondary_rails": OFFICIAL.secondary_rails,
    }


def _sr_for(
    ctx,
    *,
    task_name: str,
    seeds: list[int],
    speed: float,
    n_freeze: float,
    trajectory: str,
    robotwin_root: str,
    dyn: dict[str, Any],
    scene=None,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        r = run_episode(
            task_name,
            seed,
            ctx,
            trajectory_kind=trajectory,
            speed=speed,
            n_freeze=n_freeze,
            robotwin_root=robotwin_root,
            backend=OFFICIAL.backend,
            replan_steps=OFFICIAL.replan_steps,
            policy_hz=OFFICIAL.policy_hz,
            scene=scene,
            **dyn,
        )
        r.update({"task_name": task_name, "seed": seed})
        rows.append(r)
    succ = [1.0 if bool(x.get("success")) else 0.0 for x in rows]
    ci = bootstrap_ci(succ, n_boot=1000, seed=0)
    return {
        "sr": float(ci["mean"]) if succ else 0.0,
        "ci95": [ci["lo"], ci["hi"]],
        "n": len(rows),
        "episodes": rows,
    }


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))

    print(contract_banner(), flush=True)
    tasks = parse_task_names(args.task_names)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    nf_slow = n_freeze_from_latency(args.latency_slow_ms / 1000.0, args.policy_hz)
    nf_fast = n_freeze_from_latency(args.latency_fast_ms / 1000.0, args.policy_hz)
    dyn = _dyn_kwargs(args.dynamics_track)
    from benchmarks.common.scene import scene_from_preset

    scene_original = scene_from_preset("original", speed=0.0)
    scene_drive = scene_from_preset("apparatus_drive", speed=float(args.probe_speed))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, Any]] = []
    for task_name in tasks:
        print(f"\n######## fitness task={task_name} ########", flush=True)
        ctx = bridge.load_policy(
            checkpoint=args.checkpoint,
            stats=args.stats,
            task=bridge.DEFAULT_TASK,
            seed=args.seed,
            device=args.device,
            num_inference_steps=4,
            accel=args.accel,
        )
        try:
            print(f"  static...", flush=True)
            static = _sr_for(
                ctx,
                task_name=task_name,
                seeds=seeds,
                speed=0.0,
                n_freeze=0.0,
                trajectory=args.trajectory,
                robotwin_root=args.robotwin_root,
                dyn={**dyn, "dynamic_modes": [], "contact_switch": "off"},
                scene=scene_original,
            )
            print(f"  slow L={args.latency_slow_ms}ms...", flush=True)
            slow = _sr_for(
                ctx,
                task_name=task_name,
                seeds=seeds,
                speed=args.probe_speed,
                n_freeze=nf_slow,
                trajectory=args.trajectory,
                robotwin_root=args.robotwin_root,
                dyn=dyn,
                scene=scene_drive,
            )
            print(f"  fast L={args.latency_fast_ms}ms...", flush=True)
            fast = _sr_for(
                ctx,
                task_name=task_name,
                seeds=seeds,
                speed=args.probe_speed,
                n_freeze=nf_fast,
                trajectory=args.trajectory,
                robotwin_root=args.robotwin_root,
                dyn=dyn,
                scene=scene_drive,
            )
            delta = float(fast["sr"]) - float(slow["sr"])
            eligible = float(static["sr"]) >= float(args.min_static_sr)
            fitness = delta if eligible else 0.0
            passed = eligible and delta >= float(args.min_delta_sr)
            row = {
                "task_name": task_name,
                "sr_static": static["sr"],
                "sr_slow": slow["sr"],
                "sr_fast": fast["sr"],
                "delta_sr": delta,
                "fitness": fitness,
                "eligible": eligible,
                "passed": passed,
                "n": static["n"],
                "probe_speed": args.probe_speed,
                "latency_slow_ms": args.latency_slow_ms,
                "latency_fast_ms": args.latency_fast_ms,
                "dynamics_track": args.dynamics_track,
            }
            rows_out.append(row)
            print(
                f"    static={static['sr']:.2f} slow={slow['sr']:.2f} "
                f"fast={fast['sr']:.2f} Δ={delta:+.2f} pass={passed}",
                flush=True,
            )
            with open(out_dir / f"task_{task_name}.json", "w") as f:
                json.dump({**row, "static": static, "slow": slow, "fast": fast}, f, indent=2)
        finally:
            del ctx

    passed = [r for r in rows_out if r["passed"]]
    summary = {
        "protocol": OFFICIAL.protocol,
        "banner": contract_banner(),
        "checkpoint": args.checkpoint,
        "tasks": tasks,
        "criteria": {
            "min_static_sr": args.min_static_sr,
            "min_delta_sr": args.min_delta_sr,
            "probe_speed": args.probe_speed,
            "latency_slow_ms": args.latency_slow_ms,
            "latency_fast_ms": args.latency_fast_ms,
            "dynamics_track": args.dynamics_track,
        },
        "n_tasks": len(rows_out),
        "n_passed": len(passed),
        "passed_tasks": [r["task_name"] for r in passed],
        "rows": rows_out,
        "macro_mean_delta": float(np.mean([r["delta_sr"] for r in rows_out])) if rows_out else 0.0,
    }
    with open(out_dir / "fitness.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Dynamic-RoboTwin task fitness",
        "",
        f"Passed {len(passed)}/{len(rows_out)} "
        f"(static≥{args.min_static_sr}, ΔSR≥{args.min_delta_sr})",
        "",
        "| task | SR_static | SR_slow | SR_fast | ΔSR | pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for r in rows_out:
        lines.append(
            f"| {r['task_name']} | {r['sr_static']:.2f} | {r['sr_slow']:.2f} | "
            f"{r['sr_fast']:.2f} | {r['delta_sr']:+.2f} | "
            f"{'Y' if r['passed'] else ''} |"
        )
    (out_dir / "fitness.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"n_passed": len(passed), "out": str(out_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
