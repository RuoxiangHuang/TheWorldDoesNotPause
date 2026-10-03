#!/usr/bin/env python3
"""Record object-motion demos for all 40 Dynamic-LIBERO trained-suite tasks."""

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
from benchmarks.dynamic_libero.env.suite_registry import TRAINED_SUITES, suite_n_tasks


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="evaluate_results/demo_videos")
    p.add_argument("--trajectory", default="linear")
    p.add_argument("--traj-complexity", default="none")
    p.add_argument("--speed", type=float, default=0.006)
    p.add_argument("--axis", default="x")
    p.add_argument("--direction", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--render-res", type=int, default=512)
    p.add_argument("--release-radius", type=float, default=0.06)
    p.add_argument("--init-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--movers", default="on", choices=["on", "off"])
    p.add_argument("--mover-language", default="rewrite", choices=["rewrite", "keep"])
    p.add_argument(
        "--suites",
        default="trained",
        help="Comma-separated suite names or 'trained' (default: all 4 train suites)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks whose output mp4 already exists (for resume)",
    )
    return p.parse_args()


def _resolve_suites(spec: str) -> list[str]:
    if spec.strip().lower() == "trained":
        return list(TRAINED_SUITES)
    return [s.strip() for s in spec.split(",") if s.strip()]


def main() -> int:
    args = _parse_args()
    suites = _resolve_suites(args.suites)
    out_dir = Path(args.out_dir)
    video_dir = out_dir / "dynamic_libero" / "all_tasks"
    video_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    t0 = time.time()
    total = sum(suite_n_tasks(s) for s in suites)
    done = 0

    for suite in suites:
        n = suite_n_tasks(suite)
        for task_id in range(n):
            done += 1
            out_name = f"all_tasks/{suite}_t{task_id:02d}.mp4"
            dst = out_dir / "dynamic_libero" / out_name
            if args.skip_existing and dst.is_file() and dst.stat().st_size > 0:
                print(f"\n[{done}/{total}] skip {out_name} (exists)", flush=True)
                continue
            print(f"\n[{done}/{total}] {suite} task_id={task_id} -> {out_name}", flush=True)
            ns = Namespace(
                task_suite=suite,
                task_id=task_id,
                init_idx=args.init_idx,
                seed=args.seed,
                trajectory=args.trajectory,
                traj_complexity=args.traj_complexity,
                speed=args.speed,
                axis=args.axis,
                direction=args.direction,
                steps=args.steps,
                fps=args.fps,
                render_res=args.render_res,
                release_radius=args.release_radius,
                out_dir=str(out_dir),
                out_name=out_name,
                caption=False,
                movers=args.movers,
                mover_language=args.mover_language,
                demo_toycar=args.movers == "on",
                motive="auto",
            )
            meta = record_libero(ns)
            results.append(meta)
            print(
                f"  frames={meta['frames']} escaped={meta['escaped']} "
                f"duration={meta['duration_s']}s",
                flush=True,
            )

    meta_path = video_dir / "recording_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_tasks": len(results),
                "suites": suites,
                "out_dir": str(video_dir.resolve()),
                "results": results,
            },
            indent=2,
        )
    )
    elapsed = time.time() - t0
    print(f"\nRecorded {len(results)} videos in {elapsed:.1f}s -> {video_dir}", flush=True)
    print(f"Meta -> {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
