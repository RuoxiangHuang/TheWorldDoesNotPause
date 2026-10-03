#!/usr/bin/env python3
"""Official Dynamic-RoboTwin suite entrypoint (frozen contract).

Primary track: rails_collide=False (ghost sliding), backend=async,
paper dynamics mix, N_seed=5, trajectories={linear,sine}, official speed grid.
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

from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge
from benchmarks.dynamic_robotwin.env.official import (
    OFFICIAL,
    apply_official_args,
    contract_banner,
    parse_task_names,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official Dynamic-RoboTwin evaluation")
    p.add_argument("--checkpoint", type=str, default=bridge.DEFAULT_CKPT)
    p.add_argument("--stats", type=str, default=bridge.DEFAULT_STATS)
    p.add_argument(
        "--mode",
        type=str,
        default="sr",
        choices=["sr", "paired"],
        help="sr=speed×traj grid; paired=one (traj,speed)",
    )
    p.add_argument(
        "--task-names",
        type=str,
        default="all",
        help="Whitelist names, 'all' (official ghost-rails), 'recon' reconstructions, 'extra' appendix, or dotted catalog ids.",
    )
    p.add_argument("--accels", type=str, default="off,pace")
    p.add_argument("--trajectory", type=str, default=None)
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fixed-latency", type=float, default=None)
    p.add_argument("--robotwin-root", type=str, default=str(bridge.DEFAULT_ROBOTWIN_ROOT))
    p.add_argument(
        "--dynamics-track",
        type=str,
        default="primary",
        choices=["primary", "rails_only"],
        help="primary=paper dynamics mix; rails_only=dynamic-modes none",
    )
    p.add_argument(
        "--language-track",
        type=str,
        default="primary",
        choices=["primary", "motion"],
        help="primary=keep instruction; motion=B-track motion suffix",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--out-dir",
        type=str,
        default="evaluate_results/dynamic_robotwin/official",
    )
    return p.parse_args()


def _build_cmd(args: argparse.Namespace) -> list[str]:
    c = OFFICIAL
    py = sys.executable
    tasks = parse_task_names(args.task_names)
    seeds = ",".join(str(s) for s in c.seeds)
    speeds = ",".join(str(s) for s in c.speeds)
    trajs = ",".join(c.trajectories)
    dyn = c.dynamics_cli()
    if args.dynamics_track == "rails_only":
        dyn = {
            **dyn,
            "dynamic_modes": "none",
            "contact_trigger": "off",
            "occlusion": "off",
            "contact_chain": "off",
        }

    common = [
        "--checkpoint",
        args.checkpoint,
        "--stats",
        args.stats,
        "--seeds",
        seeds,
        "--accels",
        args.accels,
        "--backend",
        c.backend,
        "--policy-hz",
        str(c.policy_hz),
        "--replan-steps",
        str(c.replan_steps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--robotwin-root",
        args.robotwin_root,
        "--dynamic-modes",
        dyn["dynamic_modes"],
        "--contact-trigger",
        dyn["contact_trigger"],
        "--occlusion",
        dyn["occlusion"],
        "--contact-chain",
        dyn["contact_chain"],
        "--secondary-rails",
        dyn["secondary_rails"],
        "--language-mode",
        "motion" if args.language_track == "motion" else c.language_mode,
    ]
    if args.fixed_latency is not None:
        common += ["--fixed-latency", str(args.fixed_latency)]

    if args.mode == "sr":
        return [
            py,
            "-m",
            "benchmarks.dynamic_robotwin.eval.run_sr",
            *common,
            "--task-names",
            ",".join(tasks),
            "--trajectories",
            trajs,
            "--speeds",
            speeds,
            "--out-dir",
            str(Path(args.out_dir) / "sr"),
        ]

    traj = args.trajectory or "linear"
    speed = 0.001 if args.speed is None else float(args.speed)
    # paired CLI takes a single --task-name
    return [
        py,
        "-m",
        "benchmarks.dynamic_robotwin.eval.run_paired",
        *common,
        "--task-name",
        tasks[0],
        "--trajectory",
        traj,
        "--speed",
        str(speed),
        "--out-dir",
        str(Path(args.out_dir) / "paired"),
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
        "dynamics_track": args.dynamics_track,
        "language_track": args.language_track,
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "task_names": parse_task_names(args.task_names),
    }
    (out_root / "official_meta.json").write_text(json.dumps(meta, indent=2))

    cmd = _build_cmd(args)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return

    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)
    print(f"Official run finished → {out_root}", flush=True)


if __name__ == "__main__":
    main()
