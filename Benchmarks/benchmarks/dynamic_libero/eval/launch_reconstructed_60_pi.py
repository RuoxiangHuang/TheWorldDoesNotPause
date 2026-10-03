#!/usr/bin/env python3
"""Shard reconstructed-60 Dynamic-LIBERO across GPUs for PI vs FasterPI.

Same grid as FastWAM vs FasterWAM:
  60 catalog tasks, N=5 inits, freeze, movers=off, language=keep,
  speeds {0, 2, 3, 5, 6, 8, 10, 30} e-4, catalog trajectories.

Single-process per GPU: openpi env loads π₀.₅ + FasterPI + LIBERO sim
in one interpreter so measured L is model wall-clock (no HTTP RTT).

PI = eager (no accel). FasterPI = compile+chunk_residual_cache+step_cache+compile_prefix.
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

FASTERPI = Path("/DATA/YuanZhen/FasterPI")
OPENPI_SRC = Path("/DATA/YuanZhen/PI/openpi/src")
OPENPI_PY = Path("/DATA/miniconda3_corl/envs_corl/openpi/bin/python")
OPENPI_DATA = Path("/DATA/huangsiqiao/.cache/openpi")

DEFAULT_SPEEDS = "0.0,0.0002,0.0003,0.0005,0.0006,0.0008,0.001,0.003"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--init-ids", type=str, default="0,1,2,3,4")
    p.add_argument("--speeds", type=str, default=DEFAULT_SPEEDS)
    p.add_argument("--trajectories", type=str, default="linear")
    p.add_argument("--backend", type=str, default="freeze")
    p.add_argument(
        "--language-mode",
        type=str,
        default="keep",
        choices=["keep", "motion", "scene"],
    )
    p.add_argument(
        "--place-freeze",
        type=str,
        default="off",
        help="Match published FastWAM SR–Speed grid (no receptacle park).",
    )
    p.add_argument(
        "--speedups",
        type=str,
        default="pi=eager,fasterpi=compile+chunk_residual_cache+step_cache+compile_prefix",
        help="Comma list of label=spec (PI eager vs FasterPI stack).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="evaluate_results/dynamic_libero/pi_vs_fasterpi",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--python", type=str, default=str(OPENPI_PY))
    p.add_argument("--fasterpi", type=str, default=str(FASTERPI))
    p.add_argument("--openpi-src", type=str, default=str(OPENPI_SRC))
    p.add_argument("--openpi-data", type=str, default=str(OPENPI_DATA))
    p.add_argument(
        "--libero-ckpt",
        type=str,
        default=str(FASTERPI / "ckpt/pi05_libero_pytorch"),
    )
    p.add_argument(
        "--libero-norm",
        type=str,
        default="/DATA/YuanZhen/cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",
    )
    p.add_argument(
        "--task-set",
        type=str,
        default="reconstructed",
        choices=["reconstructed", "spatial", "irregular"],
    )
    p.add_argument(
        "--dynamic-tasks",
        type=str,
        default="",
        help="Optional comma-separated catalog ids (overrides --task-set sharding).",
    )
    args = p.parse_args()

    root = setup_sys_path()
    from benchmarks.dynamic_libero.env.dynamic_tasks import (
        irregular_tasks,
        reconstructed_tasks,
        spatial_tasks,
    )

    if args.dynamic_tasks.strip():
        ids_all = [x.strip() for x in args.dynamic_tasks.split(",") if x.strip()]
    elif args.task_set == "spatial":
        ids_all = [t.id for t in spatial_tasks()]
    elif args.task_set == "irregular":
        ids_all = [t.id for t in irregular_tasks()]
    else:
        tasks = reconstructed_tasks()
        if len(tasks) != 60:
            raise SystemExit(f"expected 60 reconstructed tasks, got {len(tasks)}")
        ids_all = [t.id for t in tasks]

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    shards: list[list[str]] = [[] for _ in gpus]
    for i, tid in enumerate(ids_all):
        shards[i % len(gpus)].append(tid)

    out_root = Path(args.out_dir)
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    py = str(args.python)
    pythonpath = (
        f"{args.fasterpi}:{args.openpi_src}:{root}/Benchmarks:"
        f"{os.environ.get('PYTHONPATH', '')}"
    )

    procs = []
    for gpu, ids in zip(gpus, shards):
        if not ids:
            continue
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["MUJOCO_GL"] = "egl"
        env["PYOPENGL_PLATFORM"] = "egl"
        env["OPENPI_DATA_HOME"] = str(args.openpi_data)
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["PYTHONPATH"] = pythonpath
        cmd = [
            py,
            "-u",
            "-m",
            "benchmarks.dynamic_libero.eval.run_sr",
            "--pi-speedups",
            args.speedups,
            "--pi-ckpt",
            args.libero_ckpt,
            "--pi-norm",
            args.libero_norm,
            "--dynamic-tasks",
            ",".join(ids),
            "--init-ids",
            args.init_ids,
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
            "--place-freeze",
            args.place_freeze,
            "--seed",
            str(args.seed),
            "--device",
            "cuda:0",
            "--out-dir",
            str(out_root / f"gpu{gpu}"),
        ]
        print(f"LAUNCH gpu={gpu} n={len(ids)} in-process: {ids[0]} … {ids[-1]}", flush=True)
        lf = open(logs / f"gpu{gpu}.log", "w")
        procs.append(
            subprocess.Popen(cmd, cwd=str(root), env=env, stdout=lf, stderr=subprocess.STDOUT)
        )

    codes = [pr.wait() for pr in procs]
    if any(c != 0 for c in codes):
        raise SystemExit(max(codes) if max(codes) > 0 else 1)


if __name__ == "__main__":
    main()
