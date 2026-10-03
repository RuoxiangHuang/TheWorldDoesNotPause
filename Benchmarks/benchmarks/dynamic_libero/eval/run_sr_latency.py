#!/usr/bin/env python3
"""Hardware-decoupled SR(L) characteristic curve for Dynamic-LIBERO.

Injects synthetic latencies ``L`` (seconds) rather than measuring wall-clock
inference. Anchors:
  - L=0     → oracle (n_freeze=0)
  - L=inf   → open-loop (thinking never finishes; robot never acts)

Any method can later map its measured L onto this curve to read expected SR
without re-running the full dynamic suite on every machine.
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
    target_registry_from_args,
    traj_kwargs_from_args,
)
from benchmarks.common.protocol import DEFAULT_BACKEND
from benchmarks.common.grasp import add_grasp_cli_args
from benchmarks.dynamic_libero.eval.measure_latency import (
    OPEN_LOOP_LATENCY_S,
    n_freeze_from_latency,
)
from benchmarks.dynamic_libero.eval.rollout import run_episode
from benchmarks.dynamic_libero.metrics.aggregate import write_curve_csv

DEFAULT_LATENCIES = [0.0, 0.025, 0.05, 0.1, 0.2, 0.4]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SR(L) characteristic curve (LIBERO)")
    p.add_argument("--checkpoint", type=str, default=bridge.DEFAULT_CKPT)
    p.add_argument("--task-suite", type=str, default="libero_object")
    p.add_argument("--task-ids", type=str, default="0", help="Comma-separated task ids or 'all'")
    p.add_argument("--init-ids", type=str, default="0,1,2,3,4")
    p.add_argument(
        "--accel",
        type=str,
        default="off",
        help="Policy stack used while injecting synthetic L (weights only; L is forced)",
    )
    p.add_argument("--trajectory", type=str, default="linear")
    p.add_argument("--speed", type=float, default=0.003)
    p.add_argument("--axis", type=str, default="x")
    p.add_argument("--direction", type=float, default=1.0)
    p.add_argument(
        "--toward-center",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--backend",
        type=str,
        default="freeze",
        choices=["freeze", "async"],
        help="SR(L) paper default is freeze so the curve is not washed out by stale actions.",
    )
    p.add_argument(
        "--latencies",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LATENCIES),
        help="Comma-separated synthetic latencies in seconds",
    )
    p.add_argument(
        "--include-open-loop",
        action="store_true",
        help="Append L=inf open-loop anchor",
    )
    p.add_argument("--release-radius", type=float, default=0.06)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--control-freq", type=float, default=20.0)
    p.add_argument(
        "--movers",
        type=str,
        default="off",
        choices=["on", "off"],
    )
    p.add_argument(
        "--mover-language",
        type=str,
        default="keep",
        choices=["rewrite", "keep"],
    )
    p.add_argument(
        "--language-mode",
        type=str,
        default="keep",
        choices=["keep", "motion", "scene"],
        help="keep=primary; motion=B-track motion suffix.",
    )
    p.add_argument("--no-target-registry", action="store_true")
    add_grasp_cli_args(p)
    add_motive_arg(p)
    add_dynamic_task_args(p)
    add_smooth_turn_args(p)
    from benchmarks.common.scene import add_scene_cli_args

    add_scene_cli_args(p)
    p.add_argument("--out-dir", type=str, default="evaluate_results/dynamic_libero/sr_latency")
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
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    task_ids = parse_task_ids(args.task_ids, args.task_suite)
    init_ids = parse_init_ids(args.init_ids)
    latencies = _parse_latencies(args.latencies, args.include_open_loop)
    traj_kw = traj_kwargs_from_args(args)
    movers_on = movers_enabled_from_args(args)
    mover_lang = mover_language_from_args(args)
    language_mode = language_mode_from_args(args)
    target_registry = target_registry_from_args(args)
    grasp_kw = grasp_kwargs_from_args(args)
    motive = motive_from_args(args)
    from benchmarks.common.scene import scene_kwargs_from_args

    units = dynamic_eval_units(args)
    scene_cond = scene_kwargs_from_args(
        args, speed=float(args.speed), catalog_job=bool(units)
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{args.trajectory}_v{args.speed}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Load accel={args.accel} (synthetic L injection) ===", flush=True)
    ctx = bridge.load_policy(
        checkpoint=args.checkpoint,
        task_suite_name=args.task_suite,
        task_id=task_ids[0],
        seed=args.seed,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        accel=args.accel,
    )

    cells: list[dict[str, Any]] = []
    try:
        for lat in latencies:
            n_freeze = scene_cond.apply_n_freeze(n_freeze_from_latency(lat, args.control_freq))
            lat_label = "inf" if not math.isfinite(lat) else f"{lat:.6g}"
            nf_label = "inf" if not math.isfinite(n_freeze) else f"{n_freeze:.3f}"
            lat_print = lat_label if lat_label == "inf" else f"{lat_label}s"
            print(f"--- L={lat_print} n_freeze={nf_label} ---", flush=True)
            rows = []
            for tid in task_ids:
                task = ctx.task_suite.get_task(tid)
                inits = list(ctx.task_suite.get_task_init_states(tid))
                for iid in init_ids:
                    if iid >= len(inits):
                        continue
                    ctx.cfg.EVALUATION.task_id = tid
                    r = run_episode(
                        task,
                        inits[iid],
                        ctx,
                        trajectory_kind=args.trajectory,
                        speed=args.speed,
                        trajectory_kwargs=traj_kw,
                        n_freeze=n_freeze,
                        release_radius=args.release_radius,
                        max_steps=args.max_steps,
                        task_suite_name=args.task_suite,
                        task_id=tid,
                        movers_enabled=movers_on,
                        mover_language=mover_lang,
                        language_mode=language_mode,
                        target_registry=target_registry,
                        backend=args.backend,
                        latency_mode="constant",
                        motive=motive,
                        rails_variant=rails_variant_from_args(args),
                        init_id=iid,
                        scene=scene_cond,
                        **grasp_kw,
                    )
                    r.update(
                        {
                            "task_id": tid,
                            "init_id": iid,
                            "accel": f"injected_L",
                            "latency_s": lat if math.isfinite(lat) else None,
                            "latency_anchor": lat_label,
                        }
                    )
                    rows.append(r)
                    print(
                        f"  t{tid}i{iid}: success={r['success']} escaped={r['escaped']} "
                        f"drift={r['drift_m']:.4f}",
                        flush=True,
                    )
            sr = float(np.mean([bool(x["success"]) for x in rows])) if rows else 0.0
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
        "task_suite": args.task_suite,
        "task_ids": task_ids,
        "init_ids": init_ids,
        "trajectory": args.trajectory,
        "speed": args.speed,
        "policy_accel": args.accel,
        "control_freq": args.control_freq,
        "movers": movers_on,
        "mover_language": mover_lang,
        "language_mode": language_mode,
        "target_registry": target_registry,
        "motive": motive,
        "backend": args.backend,
        "toward_center": bool(args.toward_center),
        "anchors": {"oracle": "L=0", "open_loop": "L=inf"},
        "curve": curve,
        "cells": cells,
    }
    with open(out_dir / "sr_latency.json", "w") as f:
        json.dump(payload, f, indent=2)
    # Flatten for CSV reuse
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
        "# SR(L) characteristic curve — Dynamic-LIBERO",
        "",
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
