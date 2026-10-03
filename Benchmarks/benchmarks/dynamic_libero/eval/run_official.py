#!/usr/bin/env python3
"""Official Dynamic-LIBERO suite entrypoint (frozen contract).

Runs the primary track:
  movers=off, language=keep, target_registry=on, backend=async,
  N_init=5, trajectories={linear,sine}, official speed grid,
  task_ids=all across trained suites.

This wraps ``run_sr`` / ``run_suite`` knobs so comparable reports share one
contract banner and JSON ``official_contract`` block.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

ROOT = setup_sys_path()

from benchmarks.dynamic_libero.env import libero_bridge as bridge
from benchmarks.dynamic_libero.env.official import (
    OFFICIAL,
    apply_official_args,
    contract_banner,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official Dynamic-LIBERO evaluation")
    p.add_argument("--checkpoint", type=str, default=bridge.DEFAULT_CKPT)
    p.add_argument(
        "--mode",
        type=str,
        default="sr",
        choices=["sr", "paired", "suite"],
        help="sr=speed×traj grid; paired=one (traj,speed); suite=multi-suite paired",
    )
    p.add_argument(
        "--suites",
        type=str,
        default="trained",
        help="For mode=suite / sr suite override (comma or trained)",
    )
    p.add_argument("--task-suite", type=str, default="libero_object", help="For mode=sr|paired")
    p.add_argument("--task-ids", type=str, default="all")
    p.add_argument("--accels", type=str, default="off,pace")
    p.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help="Override single trajectory (paired/suite). Default: official grid.",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Override single speed (paired/suite). Default: official working point 0.001.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fixed-latency", type=float, default=None)
    p.add_argument(
        "--movers-track",
        type=str,
        default="primary",
        choices=["primary", "toycar"],
        help="primary=movers off+keep; toycar=secondary asset-swap track",
    )
    p.add_argument(
        "--language-track",
        type=str,
        default="primary",
        choices=["primary", "motion"],
        help="primary=keep instruction; motion=B-track motion suffix",
    )
    p.add_argument("--dry-run", action="store_true", help="Print argv and exit")
    p.add_argument(
        "--out-dir",
        type=str,
        default="evaluate_results/dynamic_libero/official",
    )
    return p.parse_args()


def _build_cmd(args: argparse.Namespace) -> list[str]:
    c = OFFICIAL
    py = sys.executable
    stamp_parent = Path(args.out_dir)
    movers = "on" if args.movers_track == "toycar" else "off"
    lang = "rewrite" if args.movers_track == "toycar" else "keep"
    language_mode = "motion" if args.language_track == "motion" else c.language_mode
    init_ids = ",".join(str(i) for i in c.init_ids)
    speeds = ",".join(str(s) for s in c.speeds)
    trajs = ",".join(c.trajectories)

    common = [
        "--checkpoint",
        args.checkpoint,
        "--init-ids",
        init_ids,
        "--accels",
        args.accels,
        "--backend",
        c.backend,
        "--movers",
        movers,
        "--mover-language",
        lang,
        "--language-mode",
        language_mode,
        "--release-radius",
        str(c.release_radius_m),
        "--max-steps",
        str(c.max_steps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.fixed_latency is not None:
        common += ["--fixed-latency", str(args.fixed_latency)]

    if args.mode == "sr":
        return [
            py,
            "-m",
            "benchmarks.dynamic_libero.eval.run_sr",
            *common,
            "--task-suite",
            args.task_suite,
            "--task-ids",
            args.task_ids,
            "--trajectories",
            trajs,
            "--speeds",
            speeds,
            "--out-dir",
            str(stamp_parent / "sr"),
        ]

    traj = args.trajectory or "linear"
    speed = 0.001 if args.speed is None else float(args.speed)
    if args.mode == "paired":
        return [
            py,
            "-m",
            "benchmarks.dynamic_libero.eval.run_paired",
            *common,
            "--task-suite",
            args.task_suite,
            "--task-ids",
            args.task_ids,
            "--trajectory",
            traj,
            "--speed",
            str(speed),
            "--out-dir",
            str(stamp_parent / "paired"),
        ]

    # suite
    return [
        py,
        "-m",
        "benchmarks.dynamic_libero.eval.run_suite",
        *common,
        "--suites",
        args.suites,
        "--task-ids",
        args.task_ids,
        "--trajectory",
        traj,
        "--speed",
        str(speed),
        "--out-dir",
        str(stamp_parent / "suite"),
    ]


def main() -> None:
    args = _parse_args()
    apply_official_args(args, force=True)
    print(contract_banner(), flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_root)

    meta: dict[str, Any] = {
        "banner": contract_banner(),
        "official_contract": OFFICIAL.to_dict(),
        "movers_track": args.movers_track,
        "language_track": args.language_track,
        "mode": args.mode,
        "checkpoint": args.checkpoint,
    }
    (out_root / "official_meta.json").write_text(json.dumps(meta, indent=2))

    cmd = _build_cmd(args)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)
    print(f"Official run finished → {out_root}", flush=True)


if __name__ == "__main__":
    main()
