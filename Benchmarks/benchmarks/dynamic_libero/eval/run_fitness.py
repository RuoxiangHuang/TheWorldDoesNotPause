#!/usr/bin/env python3
"""Diagnostic latency-sensitivity scan for Dynamic-LIBERO.

This does **not** decide main-track eligibility. Main vs extension is
semantic (language still true, scene legal, BDDL/check_success correct).

For each task, estimate:
  - SR_original @ official spawn, no apparatus
  - SR_slow     @ apparatus_drive with a high synthetic latency
  - SR_fast     @ same modified scene with low / zero freeze
  - ΔSR         = SR_fast − SR_slow (diagnostic only)
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

from benchmarks.dynamic_libero.env import libero_bridge as bridge
from benchmarks.dynamic_libero.env.official import OFFICIAL, contract_banner
from benchmarks.dynamic_libero.env.suite_registry import parse_init_ids, resolve_suites
from benchmarks.dynamic_libero.eval.cli_common import (
    add_dynamic_task_args,
    add_motive_arg,
    dynamic_eval_units,
    motive_from_args,
)
from benchmarks.dynamic_libero.eval.measure_latency import n_freeze_from_latency
from benchmarks.dynamic_libero.eval.rollout import run_episode
from benchmarks.dynamic_libero.metrics.aggregate import bootstrap_ci


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic-LIBERO task fitness scan")
    p.add_argument("--checkpoint", type=str, default=bridge.DEFAULT_CKPT)
    p.add_argument("--suites", type=str, default="trained")
    p.add_argument("--task-ids", type=str, default="all")
    p.add_argument("--init-ids", type=str, default=",".join(str(i) for i in OFFICIAL.init_ids))
    p.add_argument("--accel", type=str, default="full", help="Weights stack (L is synthetic)")
    p.add_argument("--trajectory", type=str, default="linear")
    p.add_argument("--probe-speed", type=float, default=0.0005)
    p.add_argument(
        "--latency-slow-ms",
        type=float,
        default=400.0,
        help="Slow-stack proxy. Object-0 cliff is ~230–400 ms; 0 vs 200 sits on the plateau.",
    )
    p.add_argument(
        "--latency-fast-ms",
        type=float,
        default=0.0,
        help="Fast-stack proxy (0 or ~80 ms). Do not use 0 vs 200 as the official gate.",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="freeze",
        choices=["freeze", "async"],
        help="Fitness default is freeze (robot holds while the object moves). "
        "async is an ablation — stale actions can wash out ΔSR.",
    )
    p.add_argument(
        "--toward-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inward rails only; does not change BDDL or language.",
    )
    p.add_argument("--min-static-sr", type=float, default=0.2)
    p.add_argument("--min-delta-sr", type=float, default=0.1, help="Pass threshold for fitness")
    p.add_argument("--control-freq", type=float, default=OFFICIAL.control_hz)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=OFFICIAL.max_steps)
    p.add_argument("--out-dir", type=str, default="evaluate_results/dynamic_libero/fitness")
    p.add_argument(
        "--reuse-static-from",
        type=str,
        default=None,
        help="Reuse SR_static from a prior fitness.json tree (skip static rollouts).",
    )
    add_motive_arg(p)
    add_dynamic_task_args(p)
    return p.parse_args()


def _static_cache(root: str | None) -> dict[tuple[str, int], dict[str, Any]]:
    """Latest SR_static per (suite, task_id) under ``root``."""
    if not root:
        return {}
    cache: dict[tuple[str, int], tuple[float, dict[str, Any]]] = {}
    for path in sorted(Path(root).rglob("fitness.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        mtime = path.stat().st_mtime
        for row in data.get("rows") or []:
            suite = row.get("suite")
            tid = row.get("task_id")
            if suite is None or tid is None:
                continue
            key = (str(suite), int(tid))
            prev = cache.get(key)
            if prev is None or mtime >= prev[0]:
                cache[key] = (
                    mtime,
                    {
                        "sr": float(row.get("sr_static") or 0.0),
                        "ci95": row.get("ci95_static") or [0.0, 0.0],
                        "n": int(row.get("n") or 0),
                        "episodes": [],
                        "reused": True,
                    },
                )
    return {k: v[1] for k, v in cache.items()}


def _sr_for(
    ctx,
    *,
    suite: str,
    task_id: int,
    init_ids: list[int],
    speed: float,
    n_freeze: float,
    trajectory: str,
    max_steps: int,
    backend: str,
    toward_center: bool,
    motive: str = "auto",
    rails_variant: str = "pick",
    dynamic_task_id: str | None = None,
    place_target_name: str | None = None,
    scene=None,
) -> dict[str, Any]:
    task = ctx.task_suite.get_task(task_id)
    inits = list(ctx.task_suite.get_task_init_states(task_id))
    traj_kw: dict[str, Any] = {"axis": "x", "toward_center": bool(toward_center)}
    if not toward_center:
        traj_kw["direction"] = 1.0
    rows = []
    for iid in init_ids:
        if iid >= len(inits):
            continue
        ctx.cfg.EVALUATION.task_id = task_id
        r = run_episode(
            task,
            inits[iid],
            ctx,
            trajectory_kind=trajectory,
            speed=speed,
            trajectory_kwargs=traj_kw,
            n_freeze=n_freeze,
            max_steps=max_steps,
            task_suite_name=suite,
            task_id=task_id,
            movers_enabled=False,
            mover_language="keep",
            language_mode="keep",
            target_registry=True,
            backend=backend,
            grasp_mode="ghost",
            latency_mode="constant",
            motive=motive,
            rails_variant=rails_variant,
            dynamic_task_id=dynamic_task_id,
            place_target_name=place_target_name,
            init_id=iid,
            scene=scene,
        )
        r.update({"task_id": task_id, "init_id": iid, "dynamic_task_id": dynamic_task_id})
        rows.append(r)
    succ = [1.0 if bool(x["success"]) else 0.0 for x in rows]
    ci = bootstrap_ci(succ, n_boot=1000, seed=0)
    return {
        "sr": float(ci["mean"]) if succ else 0.0,
        "ci95": [ci["lo"], ci["hi"]],
        "n": len(rows),
        "episodes": rows,
    }


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))

    print(contract_banner(), flush=True)
    suites = resolve_suites(args.suites)
    init_ids = parse_init_ids(args.init_ids)
    nf_slow = n_freeze_from_latency(args.latency_slow_ms / 1000.0, args.control_freq)
    nf_fast = n_freeze_from_latency(args.latency_fast_ms / 1000.0, args.control_freq)
    motive = motive_from_args(args)
    units = dynamic_eval_units(args)
    from benchmarks.common.scene import scene_from_preset

    scene_original = scene_from_preset("original", speed=0.0)
    scene_drive = scene_from_preset("apparatus_drive", speed=float(args.probe_speed))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    static_cache = _static_cache(args.reuse_static_from)
    rows_out: list[dict[str, Any]] = []
    if units:
        grouped: dict[str, list] = {}
        for u in units:
            grouped.setdefault(u.suite, []).append(u)
        suite_jobs: list[tuple[str, list | None]] = list(grouped.items())
        suites = [s for s, _ in suite_jobs]
    else:
        suite_jobs = [(s, None) for s in suites]

    for suite, suite_units in suite_jobs:
        from benchmarks.dynamic_libero.env.suite_registry import parse_task_ids

        if suite_units is None:
            jobs = [
                {
                    "task_id": tid,
                    "variant": "pick",
                    "id": None,
                    "place": None,
                }
                for tid in parse_task_ids(args.task_ids, suite)
            ]
        else:
            jobs = [
                {
                    "task_id": u.task_id,
                    "variant": u.variant,
                    "id": u.id,
                    "place": u.place_target,
                }
                for u in suite_units
            ]
        print(
            f"\n######## fitness suite={suite} jobs={[j['id'] or j['task_id'] for j in jobs]} ########",
            flush=True,
        )
        ctx = bridge.load_policy(
            checkpoint=args.checkpoint,
            task_suite_name=suite,
            task_id=jobs[0]["task_id"],
            seed=args.seed,
            device=args.device,
            num_inference_steps=4,
            accel=args.accel,
        )
        try:
            static_by_tid: dict[int, dict[str, Any]] = {}
            for job in jobs:
                tid = int(job["task_id"])
                variant = str(job["variant"])
                dyn_id = job["id"]
                label = dyn_id or f"t{tid}"
                reused = None
                if tid in static_by_tid:
                    static = static_by_tid[tid]
                    reused = {"reused": True}
                else:
                    reused = static_cache.get((suite, tid))
                    if reused is not None:
                        print(
                            f"  {label} static (reused {reused['sr']:.2f})...",
                            flush=True,
                        )
                        static = reused
                    else:
                        print(f"  {label} static...", flush=True)
                        static = _sr_for(
                            ctx,
                            suite=suite,
                            task_id=tid,
                            init_ids=init_ids,
                            speed=0.0,
                            n_freeze=0.0,
                            trajectory=args.trajectory,
                            max_steps=args.max_steps,
                            backend=args.backend,
                            toward_center=bool(args.toward_center),
                            motive=motive,
                            rails_variant=variant,
                            dynamic_task_id=dyn_id,
                            place_target_name=job["place"],
                            scene=scene_original,
                        )
                    static_by_tid[tid] = static
                print(f"  {label} slow L={args.latency_slow_ms}ms...", flush=True)
                slow = _sr_for(
                    ctx,
                    suite=suite,
                    task_id=tid,
                    init_ids=init_ids,
                    speed=args.probe_speed,
                    n_freeze=nf_slow,
                    trajectory=args.trajectory,
                    max_steps=args.max_steps,
                    backend=args.backend,
                    toward_center=bool(args.toward_center),
                    motive=motive,
                    rails_variant=variant,
                    dynamic_task_id=dyn_id,
                    place_target_name=job["place"],
                    scene=scene_drive,
                )
                print(f"  {label} fast L={args.latency_fast_ms}ms...", flush=True)
                fast = _sr_for(
                    ctx,
                    suite=suite,
                    task_id=tid,
                    init_ids=init_ids,
                    speed=args.probe_speed,
                    n_freeze=nf_fast,
                    trajectory=args.trajectory,
                    max_steps=args.max_steps,
                    backend=args.backend,
                    toward_center=bool(args.toward_center),
                    motive=motive,
                    rails_variant=variant,
                    dynamic_task_id=dyn_id,
                    place_target_name=job["place"],
                    scene=scene_drive,
                )
                delta = float(fast["sr"]) - float(slow["sr"])
                eligible = float(static["sr"]) >= float(args.min_static_sr)
                fitness = delta if eligible else 0.0
                passed = eligible and delta >= float(args.min_delta_sr)
                row = {
                    "suite": suite,
                    "task_id": tid,
                    "dynamic_task_id": dyn_id,
                    "rails_variant": variant,
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
                    "n_freeze_slow": nf_slow,
                    "n_freeze_fast": nf_fast,
                    "backend": args.backend,
                    "toward_center": bool(args.toward_center),
                    "latency_mode": "constant",
                    "motive": motive,
                    "static_reused": bool(reused is not None),
                }
                rows_out.append(row)
                print(
                    f"    static={static['sr']:.2f} slow={slow['sr']:.2f} "
                    f"fast={fast['sr']:.2f} Δ={delta:+.2f} pass={passed}",
                    flush=True,
                )
                stem = dyn_id.replace(".", "_") if dyn_id else f"task_{suite}_{tid}"
                with open(out_dir / f"{stem}.json", "w") as f:
                    json.dump({**row, "static": static, "slow": slow, "fast": fast}, f, indent=2)
        finally:
            del ctx

    passed = [r for r in rows_out if r["passed"]]
    summary = {
        "protocol": OFFICIAL.protocol,
        "banner": contract_banner(),
        "checkpoint": args.checkpoint,
        "suites": suites,
        "criteria": {
            "min_static_sr": args.min_static_sr,
            "min_delta_sr": args.min_delta_sr,
            "probe_speed": args.probe_speed,
            "latency_slow_ms": args.latency_slow_ms,
            "latency_fast_ms": args.latency_fast_ms,
            "backend": args.backend,
            "toward_center": bool(args.toward_center),
            "latency_mode": "constant",
            "language_mode": "keep",
            "movers": "off",
            "motive": motive,
        },
        "n_tasks": len(rows_out),
        "n_passed": len(passed),
        "passed_tasks": [
            {
                "suite": r["suite"],
                "task_id": r["task_id"],
                "dynamic_task_id": r.get("dynamic_task_id"),
                "rails_variant": r.get("rails_variant"),
            }
            for r in passed
        ],
        "rows": rows_out,
        "macro_mean_delta": float(np.mean([r["delta_sr"] for r in rows_out])) if rows_out else 0.0,
    }
    with open(out_dir / "fitness.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown table
    lines = [
        "# Dynamic-LIBERO task fitness",
        "",
        f"Passed {len(passed)}/{len(rows_out)} "
        f"(static≥{args.min_static_sr}, ΔSR≥{args.min_delta_sr})",
        "",
        "| id | suite | task | variant | SR_static | SR_slow | SR_fast | ΔSR | pass |",
        "|---|---|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for r in rows_out:
        did = r.get("dynamic_task_id") or f"{r['suite']}.t{r['task_id']:02d}"
        lines.append(
            f"| {did} | {r['suite']} | {r['task_id']} | {r.get('rails_variant', 'pick')} | "
            f"{r['sr_static']:.2f} | {r['sr_slow']:.2f} | {r['sr_fast']:.2f} | "
            f"{r['delta_sr']:+.2f} | {'Y' if r['passed'] else ''} |"
        )
    (out_dir / "fitness.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"n_passed": len(passed), "out": str(out_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
