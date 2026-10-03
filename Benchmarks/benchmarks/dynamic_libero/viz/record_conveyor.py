#!/usr/bin/env python3
"""Record Dynamic-LIBERO conveyor-motive demos (libero_object by default)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

setup_sys_path()

from benchmarks.common.viz.record_object_motion import record_libero
from benchmarks.dynamic_libero.env.suite_registry import suite_n_tasks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="evaluate_results/demo_videos")
    p.add_argument("--task-suite", default="libero_object")
    p.add_argument("--task-ids", default="all", help="Comma ids or 'all'")
    p.add_argument("--trajectory", default="linear")
    p.add_argument("--speed", type=float, default=0.006)
    p.add_argument("--axis", default="x")
    p.add_argument("--direction", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--render-res", type=int, default=512)
    p.add_argument("--init-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--motive", default="conveyor")
    args = p.parse_args()

    n = suite_n_tasks(args.task_suite)
    if args.task_ids.strip().lower() in {"all", "*"}:
        task_ids = list(range(n))
    else:
        task_ids = [int(x) for x in args.task_ids.split(",") if x.strip() != ""]

    out_dir = Path(args.out_dir)
    video_dir = out_dir / "dynamic_libero" / "conveyor"
    video_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    for i, tid in enumerate(task_ids):
        out_name = f"conveyor/{args.task_suite}_t{tid:02d}_{args.trajectory}.mp4"
        print(f"\n[{i+1}/{len(task_ids)}] {out_name}", flush=True)
        ns = Namespace(
            task_suite=args.task_suite,
            task_id=tid,
            init_idx=args.init_idx,
            seed=args.seed,
            trajectory=args.trajectory,
            traj_complexity="none",
            speed=args.speed,
            axis=args.axis,
            direction=args.direction,
            steps=args.steps,
            fps=args.fps,
            render_res=args.render_res,
            release_radius=0.06,
            out_dir=str(out_dir),
            out_name=out_name,
            caption=False,
            movers="",
            mover_language="keep",
            motive=args.motive,
            demo_toycar=None,
        )
        meta = record_libero(ns)
        results.append(meta)
        print(
            f"  motive={meta.get('motive')} conveyor={meta.get('conveyor')} "
            f"frames={meta['frames']} escaped={meta['escaped']}",
            flush=True,
        )

    meta_path = video_dir / "recording_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "suite": args.task_suite,
                "motive": args.motive,
                "n_videos": len(results),
                "elapsed_s": round(time.time() - t0, 1),
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {meta_path} ({len(results)} videos)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
