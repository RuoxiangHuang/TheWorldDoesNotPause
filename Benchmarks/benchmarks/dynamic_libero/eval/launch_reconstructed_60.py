#!/usr/bin/env python3
"""Shard the 60-task reconstructed Dynamic-LIBERO set across GPUs.

FastWAM = accel off; FasterWAM = accel full. Catalog trajectories are kept
(linear object / spatial; unique curves for irregular extras).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/DATA/YuanZhen/FasterWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt",
    )
    p.add_argument("--init-ids", type=str, default="0,1,2,3,4")
    p.add_argument("--accels", type=str, default="off,pace")
    p.add_argument("--speeds", type=str, default="0.0,0.001,0.003")
    p.add_argument("--trajectories", type=str, default="linear")
    p.add_argument("--backend", type=str, default="freeze")
    p.add_argument(
        "--language-mode",
        type=str,
        default="keep",
        choices=["keep", "motion", "scene"],
    )
    p.add_argument(
        "--task-set",
        type=str,
        default="reconstructed",
        choices=["reconstructed", "spatial", "irregular"],
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="evaluate_results/dynamic_libero/reconstructed_60_off_vs_full",
    )
    p.add_argument("--seed", type=int, default=0)
    args, extra = p.parse_known_args()

    root = setup_sys_path()
    os.environ.setdefault("PYTHONPATH", f"{root}/Benchmarks:{root}/FastWAM/src")
    from benchmarks.dynamic_libero.env.dynamic_tasks import (
        irregular_tasks,
        reconstructed_tasks,
        spatial_tasks,
    )

    if args.task_set == "spatial":
        tasks = spatial_tasks()
    elif args.task_set == "irregular":
        tasks = irregular_tasks()
    else:
        tasks = reconstructed_tasks()
        if len(tasks) != 60:
            raise SystemExit(f"expected 60 reconstructed tasks, got {len(tasks)}")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    shards: list[list[str]] = [[] for _ in gpus]
    for i, t in enumerate(tasks):
        shards[i % len(gpus)].append(t.id)

    py = sys.executable
    procs = []
    for gpu, ids in zip(gpus, shards):
        if not ids:
            continue
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        env.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(root / "checkpoints"))
        env["PYTHONPATH"] = f"{root}/Benchmarks:{root}/FastWAM/src:{env.get('PYTHONPATH', '')}"
        cmd = [
            py,
            "-m",
            "benchmarks.dynamic_libero.eval.run_sr",
            "--checkpoint",
            args.checkpoint,
            "--dynamic-tasks",
            ",".join(ids),
            "--init-ids",
            args.init_ids,
            "--accels",
            args.accels,
            "--trajectories",
            args.trajectories,
            "--speeds",
            args.speeds,
            "--backend",
            args.backend,
            "--movers",
            "off",
            "--language-mode",
            args.language_mode,
            "--seed",
            str(args.seed),
            "--device",
            "cuda:0",
            "--out-dir",
            str(Path(args.out_dir) / f"gpu{gpu}"),
            *extra,
        ]
        print(f"LAUNCH gpu={gpu} n={len(ids)}: {ids[0]} … {ids[-1]}", flush=True)
        log = Path(args.out_dir) / "logs"
        log.mkdir(parents=True, exist_ok=True)
        lf = open(log / f"gpu{gpu}.log", "w")
        procs.append(
            subprocess.Popen(cmd, cwd=str(root), env=env, stdout=lf, stderr=subprocess.STDOUT)
        )

    codes = [pr.wait() for pr in procs]
    if any(c != 0 for c in codes):
        raise SystemExit(max(codes) if max(codes) > 0 else 1)


if __name__ == "__main__":
    main()
