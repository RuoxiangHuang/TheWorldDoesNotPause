#!/usr/bin/env python3
"""Appendix extra eval under the Dynamic-RoboTwin real_time_v2 contract.

The five tasks stay outside the 95-task main table. ``off`` and ``pace``
share one checkpoint. Latency is measured per replan unless ``--latency-s``
selects a constant SR(L) point. ``--motion-scale 0`` holds the native motion
(static sanity). ``--world-clock pause`` zeroes thinking-time advance.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

ROOT = setup_sys_path()

from benchmarks.common.latency import n_delay_from_latency
from benchmarks.common.scene import resolve_scene_condition
from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge
from benchmarks.dynamic_robotwin.env.official import parse_task_names
from benchmarks.dynamic_robotwin.eval.rollout import run_episode


def _ctx(ckpt: str, accel: str, device: str):
    bridge.ensure_runtime_env()
    return bridge.load_policy(
        checkpoint=ckpt,
        accel=accel,
        device=device,
        task=bridge.DEFAULT_TASK,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-names", default="extra")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--checkpoint", default=str(bridge.DEFAULT_CKPT))
    p.add_argument("--accels", default="off,pace")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--latency-s",
        type=float,
        default=None,
        help="Constant latency in seconds. Omit to measure each replan.",
    )
    p.add_argument(
        "--l0",
        action="store_true",
        help="Constant L=0 fitness gate. Same checkpoint, listed accels.",
    )
    p.add_argument("--motion-scale", type=float, default=1.0)
    p.add_argument("--world-clock", default="realtime", choices=("realtime", "pause"))
    p.add_argument("--backend", default="async", choices=("async", "freeze"))
    p.add_argument("--out-dir", default="")
    args = p.parse_args()

    tasks = parse_task_names(args.task_names)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    accels = [x.strip() for x in args.accels.split(",") if x.strip()]
    latency_s = 0.0 if args.l0 else args.latency_s
    latency_mode = "constant" if latency_s is not None else "measured"
    scene = resolve_scene_condition(speed=0.0, catalog_job=False, world_clock=args.world_clock)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out_dir or f"evaluate_results/extra_eval/{stamp}")
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for accel in accels:
        print(
            f"=== extra eval accel={accel} ckpt={args.checkpoint} "
            f"latency={latency_mode} motion_scale={args.motion_scale} ===",
            flush=True,
        )
        ctx = _ctx(args.checkpoint, accel, args.device)
        n_freeze = 0.0 if latency_s is None else n_delay_from_latency(latency_s, ctx.policy_hz)
        for task in tasks:
            for seed in seeds:
                r = run_episode(
                    task,
                    seed,
                    ctx,
                    n_freeze=n_freeze,
                    backend=args.backend,
                    latency_mode=latency_mode,
                    motion_scale=args.motion_scale,
                    scene=scene,
                )
                r.update(
                    {
                        "accel": accel,
                        "ckpt": args.checkpoint,
                        "latency_s": latency_s,
                    }
                )
                rows.append(r)
                print(
                    f"  {task} seed={seed} accel={accel} success={int(r['success'])} "
                    f"caught={r.get('caught')} dist={r.get('min_tcp_dist')} "
                    f"steps={r['steps']} n_freeze={r.get('n_freeze')}",
                    flush=True,
                )
        try:
            del ctx
        except Exception:
            pass

    summary = {"n": len(rows), "protocol": "real_time_v2", "by_accel": {}}
    for accel in accels:
        sub = [r for r in rows if r["accel"] == accel]
        sr = float(sum(bool(r["success"]) for r in sub) / max(len(sub), 1))
        summary["by_accel"][accel] = {"n": len(sub), "sr": sr}
    (out / "episodes.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8"
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
