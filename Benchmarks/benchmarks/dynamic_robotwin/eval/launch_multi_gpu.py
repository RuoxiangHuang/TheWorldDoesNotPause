"""Shard an SR sweep across GPUs (one process per device, by task name)."""

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
    p.add_argument("--task-names", type=str, default="place_empty_cup,grab_roller")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--accels", type=str, default="off,pace")
    p.add_argument("--trajectories", type=str, default="linear")
    p.add_argument("--speeds", type=str, default="0.0,0.003,0.006")
    p.add_argument("--out-dir", type=str, default="evaluate_results/dynamic_robotwin/sr_sweep")
    args, extra = p.parse_known_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    tasks = [t.strip() for t in args.task_names.split(",") if t.strip()]
    shards = [[] for _ in gpus]
    for i, t in enumerate(tasks):
        shards[i % len(gpus)].append(t)

    root = setup_sys_path()
    py = sys.executable
    procs = []
    for gpu, shard in zip(gpus, shards):
        if not shard:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(root / "checkpoints"))
        cmd = [
            py,
            "-m",
            "benchmarks.dynamic_robotwin.eval.run_sr",
            "--checkpoint",
            args.checkpoint,
            "--task-names",
            ",".join(shard),
            "--seeds",
            args.seeds,
            "--accels",
            args.accels,
            "--trajectories",
            args.trajectories,
            "--speeds",
            args.speeds,
            "--device",
            "cuda:0",
            "--out-dir",
            str(Path(args.out_dir) / f"gpu{gpu}"),
            *extra,
        ]
        print("LAUNCH:", " ".join(cmd), flush=True)
        log_path = Path(args.out_dir) / f"gpu{gpu}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "a", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, cwd=str(root), env=env, stdout=log_fh, stderr=subprocess.STDOUT))

    codes = [pr.wait() for pr in procs]
    if any(c != 0 for c in codes):
        raise SystemExit(max(codes))


if __name__ == "__main__":
    main()