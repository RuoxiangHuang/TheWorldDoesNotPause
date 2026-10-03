"""Shard an SR sweep across multiple GPUs (one process per device).

Example:
  python -m benchmarks.dynamic_libero.eval.launch_multi_gpu \\
    --gpus 0,1,2,3 --checkpoint ... --trajectories linear,sine \\
    --speeds 0,0.003,0.006 --accels off,pace
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
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--task-suite", type=str, default="libero_object")
    p.add_argument("--task-ids", type=str, default="0")
    p.add_argument("--init-ids", type=str, default="0,1,2,3,4")
    p.add_argument("--accels", type=str, default="off,pace")
    p.add_argument("--trajectories", type=str, default="linear")
    p.add_argument("--speeds", type=str, default="0.0,0.003,0.006")
    p.add_argument("--out-dir", type=str, default="evaluate_results/dynamic_libero/sr_sweep")
    p.add_argument("--seed", type=int, default=0)
    args, extra = p.parse_known_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    trajs = [t.strip() for t in args.trajectories.split(",") if t.strip()]
    # Shard by trajectory (simple, reproducible).
    shards = [[] for _ in gpus]
    for i, t in enumerate(trajs):
        shards[i % len(gpus)].append(t)

    root = setup_sys_path()
    py = sys.executable
    procs = []
    for gpu, shard in zip(gpus, shards):
        if not shard:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        env.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(root / "checkpoints"))
        cmd = [
            py,
            "-m",
            "benchmarks.dynamic_libero.eval.run_sr",
            "--checkpoint",
            args.checkpoint,
            "--task-suite",
            args.task_suite,
            "--task-ids",
            args.task_ids,
            "--init-ids",
            args.init_ids,
            "--accels",
            args.accels,
            "--trajectories",
            ",".join(shard),
            "--speeds",
            args.speeds,
            "--seed",
            str(args.seed),
            "--device",
            "cuda:0",
            "--out-dir",
            str(Path(args.out_dir) / f"gpu{gpu}"),
            *extra,
        ]
        print("LAUNCH:", " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, cwd=str(root), env=env))

    codes = [pr.wait() for pr in procs]
    if any(c != 0 for c in codes):
        raise SystemExit(max(codes))


if __name__ == "__main__":
    main()