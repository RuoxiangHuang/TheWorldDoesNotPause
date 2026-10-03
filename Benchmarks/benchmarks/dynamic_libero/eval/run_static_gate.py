#!/usr/bin/env python3
"""Static SR gate: compare movers on vs off at speed=0 across trained suites.

Reports absolute SR delta from visual/language swap alone (no dynamics).
Requires GPU + checkpoint for full policy rollouts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

setup_sys_path()

from benchmarks.dynamic_libero.env import libero_bridge as bridge
from benchmarks.dynamic_libero.env.suite_registry import TRAINED_SUITES, parse_init_ids
from benchmarks.dynamic_libero.eval.rollout import run_episode


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static SR gate (movers on vs off, speed=0)")
    p.add_argument("--checkpoint", type=str, default=bridge.DEFAULT_CKPT)
    p.add_argument("--init-ids", type=str, default="0")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-dir", type=str, default="evaluate_results/dynamic_libero/static_gate")
    return p.parse_args()


def _run_condition(ctx, suite: str, task_ids: list[int], init_ids: list[int], movers_on: bool) -> dict:
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
                trajectory_kind="linear",
                speed=0.0,
                trajectory_kwargs={"axis": "x", "direction": 1.0},
                n_freeze=0.0,
                task_suite_name=suite,
                task_id=tid,
                movers_enabled=movers_on,
                mover_language="rewrite" if movers_on else "keep",
                target_registry=True,
                max_steps=400,
                init_id=iid,
            )
            r.update({"task_id": tid, "init_id": iid, "movers": movers_on})
            rows.append(r)
    sr = float(np.mean([bool(x["success"]) for x in rows])) if rows else 0.0
    return {"sr": sr, "n": len(rows), "episodes": rows}


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    init_ids = parse_init_ids(args.init_ids)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"suites": {}, "macro": {}}
    all_on: list[bool] = []
    all_off: list[bool] = []

    for suite in TRAINED_SUITES:
        print(f"\n=== {suite} ===", flush=True)
        ctx = bridge.load_policy(
            checkpoint=args.checkpoint,
            task_suite_name=suite,
            task_id=0,
            seed=args.seed,
            device=args.device,
            accel="off",
        )
        task_ids = list(range(10))
        off = _run_condition(ctx, suite, task_ids, init_ids, movers_on=False)
        on = _run_condition(ctx, suite, task_ids, init_ids, movers_on=True)
        delta = on["sr"] - off["sr"]
        summary["suites"][suite] = {
            "sr_movers_off": off["sr"],
            "sr_movers_on": on["sr"],
            "delta_sr": delta,
            "n": on["n"],
        }
        all_off.extend([bool(e["success"]) for e in off["episodes"]])
        all_on.extend([bool(e["success"]) for e in on["episodes"]])
        print(
            f"  off={off['sr']:.3f} on={on['sr']:.3f} delta={delta:+.3f}",
            flush=True,
        )
        del ctx

    summary["macro"] = {
        "sr_movers_off": float(np.mean(all_off)) if all_off else 0.0,
        "sr_movers_on": float(np.mean(all_on)) if all_on else 0.0,
        "delta_sr": (
            float(np.mean(all_on)) - float(np.mean(all_off)) if all_on and all_off else 0.0
        ),
        "n_episodes": len(all_on),
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["macro"], indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
