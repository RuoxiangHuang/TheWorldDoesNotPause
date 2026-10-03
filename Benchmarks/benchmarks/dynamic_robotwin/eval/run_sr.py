"""SR sweep over speed × trajectory × accel for Dynamic-RoboTwin."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys
from pathlib import Path
for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

ROOT = setup_sys_path()

from benchmarks.common.persist import append_jsonl, dump_partial_cell
from benchmarks.common.language import add_language_mode_arg, parse_language_mode
from benchmarks.common.protocol import DEFAULT_BACKEND
from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge
from benchmarks.dynamic_robotwin.env.eval_config import add_dynamics_cli_args
from benchmarks.dynamic_robotwin.env.task_registry import parse_dynamic_modes
from benchmarks.common.policy.factory import make_policy_client
from benchmarks.dynamic_robotwin.eval.cli_common import (
    grasp_kwargs_from_args,
    parse_pi_speedups,
    protocol_tag,
    stub_policy_ctx,
)
from benchmarks.dynamic_robotwin.env.dynamic_tasks import is_catalog_eval, resolve_eval_spec
from benchmarks.common.grasp import add_grasp_cli_args
from benchmarks.dynamic_robotwin.eval.measure_latency import (
    measure_latency_varying,
    n_freeze_from_latency,
)
from benchmarks.dynamic_robotwin.eval.rollout import run_episode
from benchmarks.dynamic_robotwin.metrics.aggregate import (
    aggregate_sr_table,
    enrich_table_with_curve_metrics,
    write_curve_csv,
)

DEFAULT_SPEEDS = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.006, 0.010]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic-RoboTwin SR sweep")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=bridge.DEFAULT_CKPT,
        help="FastWAM checkpoint. Not required with --policy-url or --pi-speedups.",
    )
    p.add_argument("--stats", type=str, default=bridge.DEFAULT_STATS)
    p.add_argument("--task", type=str, default=bridge.DEFAULT_TASK)
    p.add_argument(
        "--task-names",
        type=str,
        default="place_empty_cup",
        help="Whitelist names, dotted catalog ids, 'recon' reconstructions, 'extra' appendix, or 'all'.",
    )
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--accels", type=str, default="off,pace")
    p.add_argument(
        "--trajectories",
        type=str,
        default="linear,sine",
        help="Official required: linear,sine",
    )
    p.add_argument("--speeds", type=str, default=",".join(str(s) for s in DEFAULT_SPEEDS))
    p.add_argument("--axis", type=str, default=None, help="Override task default axis")
    p.add_argument("--direction", type=float, default=None, help="Override task default direction")
    p.add_argument(
        "--release-radius",
        type=float,
        default=None,
        help="Override task default TCP release radius (default ~0.12)",
    )
    p.add_argument("--physics-per-tick", type=int, default=12)
    p.add_argument("--policy-hz", type=float, default=20.0)
    p.add_argument("--replan-steps", type=int, default=8)
    p.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND.value,
        choices=["freeze", "async"],
    )
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--latency-k", type=int, default=8)
    p.add_argument("--latency-warmup", type=int, default=3)
    p.add_argument("--fixed-latency", type=float, default=None, help="Override measured latency (seconds)")
    p.add_argument(
        "--delivery-delay-s",
        type=float,
        default=0.0,
        help="Extra wait after compute, before the executor installs the chunk. "
        "World time and stale actions keep running. Not added to GPU compute time.",
    )
    p.add_argument(
        "--latency-mode",
        type=str,
        default=None,
        choices=["measured", "constant", "replay"],
        help="Episode coupling. Default measured; --fixed-latency implies constant.",
    )
    p.add_argument(
        "--cell-label",
        type=str,
        default="",
        help="Table label. π₀.₅ stacks must be off (eager) or cache.",
    )
    p.add_argument(
        "--policy-url",
        type=str,
        default=None,
        help="Remote HTTP policy (openpi env server). Prefer this for RoboTwin π "
        "because sapien lives in the FastWAM env.",
    )
    p.add_argument(
        "--pi-speedups",
        type=str,
        default=None,
        help="In-process OpenPI: comma list of label=spec, e.g. off=eager,cache=cache. "
        "Requires sapien in the same env as openpi.",
    )
    p.add_argument(
        "--pi-speedup",
        type=str,
        default=None,
        help="Single in-process FasterPI spec (use with one --accels label).",
    )
    p.add_argument(
        "--pi-ckpt",
        type=str,
        default="/DATA/YuanZhen/FasterPI/ckpt/pi05_robotwin2",
    )
    p.add_argument("--robotwin-root", type=str, default=str(bridge.DEFAULT_ROBOTWIN_ROOT))
    p.add_argument("--task-config", type=str, default="demo_clean")
    p.add_argument("--out-dir", type=str, default="evaluate_results/dynamic_robotwin/sr_sweep")
    add_dynamics_cli_args(p)
    add_language_mode_arg(p)
    add_grasp_cli_args(p)
    from benchmarks.common.scene import add_scene_cli_args

    add_scene_cli_args(p)
    return p.parse_args()


def _load_openpi_policy(ckpt: str, spec: str, device: str, replan_steps: int, num_steps: int = 10):
    from fasterpi.realtime.loader import load_robotwin_policy, warmup_policy

    rt = load_robotwin_policy(
        ckpt=ckpt,
        device=device,
        speedup_dirs=spec,
        replan_steps=int(replan_steps),
        num_steps=int(num_steps),
    )
    warmup_policy(rt, n=4)
    return rt


def _traj_kwargs(kind: str, args: argparse.Namespace) -> dict[str, Any]:
    """Build trajectory kwargs; omit axis/direction when unset so task defaults apply."""
    kw: dict[str, Any] = {}
    if args.axis is not None:
        kw["axis"] = args.axis
    if args.direction is not None:
        kw["direction"] = args.direction
        kw["toward_center"] = False
    if kind == "circle":
        kw.setdefault("radius", 0.04)
        kw.setdefault("omega", 0.03)
    elif kind == "sine":
        kw.setdefault("amplitude", 0.03)
        kw.setdefault("wavelength", 0.20)
    elif kind == "stop_and_go":
        kw.setdefault("duty", 0.6)
        kw.setdefault("period", 40)
    elif kind == "polyline":
        kw.setdefault("waypoints", [[0, 0, 0], [0.08, 0, 0], [0.08, 0.06, 0]])
    return kw


def main() -> None:
    args = _parse_args()
    skip_path = Path(args.out_dir) / "SKIP"
    if skip_path.is_file():
        note = skip_path.read_text(errors="replace").strip()
        print(f"[skip] {args.out_dir} ({note})", flush=True)
        return
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))

    from benchmarks.dynamic_robotwin.env.official import parse_task_names

    task_names = parse_task_names(args.task_names)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    accels = [x.strip() for x in args.accels.split(",") if x.strip()]
    pi_pairs: list[tuple[str, str]] | None = None
    if args.pi_speedups:
        pi_pairs = parse_pi_speedups(args.pi_speedups)
        if not pi_pairs:
            raise SystemExit("--pi-speedups parsed empty")
    elif args.pi_speedup:
        spec = str(args.pi_speedup).strip()
        if len(accels) == 1:
            pi_pairs = [(accels[0], spec)]
        else:
            label = "off" if spec in ("eager", "none", "") else "cache"
            pi_pairs = [(label, spec)]
    if args.policy_url and pi_pairs:
        raise SystemExit("use either --policy-url or --pi-speedups, not both")
    if args.policy_url and len(accels) != 1:
        raise SystemExit(
            "with --policy-url pass exactly one --accels label "
            "(restart the HTTP server to switch off vs cache)"
        )
    eval_jobs: list[tuple[str, str | None]]
    if pi_pairs is not None:
        eval_jobs = [(label, spec) for label, spec in pi_pairs]
    else:
        eval_jobs = [(a, None) for a in accels]
    trajectories = [x.strip() for x in args.trajectories.split(",") if x.strip()]
    speeds = [float(x) for x in args.speeds.split(",") if x.strip() != ""]
    grasp_kw = grasp_kwargs_from_args(args)
    spec0 = resolve_eval_spec(task_names[0])
    print(
        f"[protocol] backend={args.backend} speeds={speeds} "
        f"modes={args.dynamic_modes!r} replan={args.replan_steps} "
        "(v=0 is still async: object still, robot may run the stale tail)",
        flush=True,
    )
    if args.latency_mode:
        ep_latency_mode = str(args.latency_mode)
    elif args.fixed_latency is not None:
        ep_latency_mode = "constant"
    else:
        ep_latency_mode = "measured"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []

    for accel, pi_spec in eval_jobs:
        print(f"\n=== Load accel={accel} ===", flush=True)
        policy = None
        ctx: Any
        if pi_spec is not None:
            print(f"in-process OpenPI speedup={pi_spec!r}", flush=True)
            policy = _load_openpi_policy(
                args.pi_ckpt,
                pi_spec,
                args.device,
                int(args.replan_steps),
                int(args.num_inference_steps),
            )
            ctx = stub_policy_ctx(
                replan_steps=int(args.replan_steps), policy_hz=float(args.policy_hz)
            )
            if args.fixed_latency is not None:
                lat = float(args.fixed_latency)
                raw = [lat]
            else:
                from fasterpi.realtime.measure import measure_robotwin_latency

                lat, raw = measure_robotwin_latency(
                    policy,
                    spec0.task_name,
                    spec0.language or spec0.task_name,
                    robotwin_root=args.robotwin_root,
                    task_config=args.task_config,
                    seed=seeds[0],
                    k=args.latency_k,
                    warmup=args.latency_warmup,
                )
        elif args.policy_url:
            print(f"HTTP OpenPI {args.policy_url}", flush=True)
            policy = make_policy_client("robotwin", policy_url=args.policy_url)
            ctx = stub_policy_ctx(
                replan_steps=int(args.replan_steps), policy_hz=float(args.policy_hz)
            )
            if args.fixed_latency is not None:
                lat = float(args.fixed_latency)
                raw = [lat]
            else:
                from benchmarks.common.policy.measure_http import (
                    measure_http_latency_robotwin,
                )

                lat, raw = measure_http_latency_robotwin(
                    args.policy_url,
                    task_names[0],
                    spec0.language or spec0.task_name,
                    robotwin_root=args.robotwin_root,
                    task_config=args.task_config,
                    seed=seeds[0],
                    k=args.latency_k,
                    warmup=args.latency_warmup,
                )
        else:
            ctx = bridge.load_policy(
                checkpoint=args.checkpoint,
                stats=args.stats,
                task=args.task,
                device=args.device,
                num_inference_steps=args.num_inference_steps,
                replan_steps=args.replan_steps,
                seed=args.seed,
                accel=accel,
                policy_hz=args.policy_hz,
            )
            if args.fixed_latency is not None:
                lat = float(args.fixed_latency)
                raw = [lat]
            else:
                lat, raw = measure_latency_varying(
                    task_names[0],
                    ctx,
                    robotwin_root=args.robotwin_root,
                    task_config=args.task_config,
                    seed=seeds[0],
                    k=args.latency_k,
                    warmup=args.latency_warmup,
                )
        n_freeze = n_freeze_from_latency(lat, args.policy_hz)
        print(
            f"latency={lat*1000:.1f}ms n_freeze={n_freeze:.3f} "
            f"latency_mode={ep_latency_mode} cell_label={str(args.cell_label).strip() or accel}",
            flush=True,
        )

        for traj in trajectories:
            for speed in speeds:
                from benchmarks.common.scene import iter_scene_conditions

                catalog_job = any(is_catalog_eval(n) for n in task_names)
                for scene_label, scene_cond in iter_scene_conditions(
                    args, speed=speed, catalog_job=catalog_job
                ):
                    nf = scene_cond.apply_n_freeze(float(n_freeze))
                    print(
                        f"--- {accel} | {traj} @ {speed} scene={scene_label} "
                        f"rails={scene_cond.rails} (n_freeze={nf:.3f}) ---",
                        flush=True,
                    )
                    rows = []
                    cell_path = out_dir / f"cell_{accel}_{traj}_v{speed}_{scene_label}.json"
                    jsonl_path = cell_path.with_suffix(".jsonl")
                    cell_meta = {
                        "accel": accel,
                        "trajectory": traj,
                        "speed": speed,
                        "scene_preset": scene_label,
                        "scene": scene_cond.to_dict(),
                        "replan_steps": int(args.replan_steps),
                        "latency_s": lat,
                        "latency_ms": lat * 1000.0,
                        "latency_raw": raw,
                        "n_freeze": nf,
                        "backend": args.backend,
                        "latency_mode": ep_latency_mode,
                        "delivery_delay_s": float(args.delivery_delay_s),
                        "nfe": int(args.num_inference_steps),
                        "cell_label": (str(args.cell_label).strip() or accel),
                        "pursuit_lag_m": (
                            float(speed) * nf if scene_cond.advances_rails() else 0.0
                        ),
                    }
                    for task_name in task_names:
                        for sd in seeds:
                            try:
                                r = run_episode(
                                    task_name,
                                    sd,
                                    ctx,
                                    trajectory_kind=traj,
                                    speed=speed,
                                    trajectory_kwargs=_traj_kwargs(traj, args),
                                    n_freeze=nf,
                                    release_radius=args.release_radius,
                                    physics_per_tick=args.physics_per_tick,
                                    policy_hz=args.policy_hz,
                                    robotwin_root=args.robotwin_root,
                                    task_config=args.task_config,
                                    backend=args.backend,
                                    policy=policy,
                                    replan_steps=args.replan_steps,
                                    dynamic_modes=parse_dynamic_modes(args.dynamic_modes),
                                    contact_switch=args.contact_trigger,
                                    occlusion=(args.occlusion == "on"),
                                    occlusion_duty=args.occlusion_duty,
                                    contact_chain=(args.contact_chain == "on"),
                                    secondary_rails=args.secondary_rails,
                                    language_mode=args.language_mode,
                                    scene=scene_cond,
                                    latency_mode=ep_latency_mode,
                                    delivery_delay_s=float(args.delivery_delay_s),
                                    **grasp_kw,
                                )
                            except Exception as e:
                                print(f"  {task_name} s{sd}: ERROR {e}", flush=True)
                                r = {
                                    "success": False,
                                    "escaped": False,
                                    "steps": 0,
                                    "error": str(e),
                                    "task_name": task_name,
                                    "seed": sd,
                                    "trajectory": traj,
                                    "speed": speed,
                                    "grasp_hold": None,
                                }
                            r.update(
                                {
                                    "accel": accel,
                                    "latency_s": lat,
                                    "dynamic_task_id": task_name,
                                    "seed": sd,
                                    "scene_preset": scene_label,
                                }
                            )
                            rows.append(r)
                            append_jsonl(jsonl_path, r)
                            dump_partial_cell(cell_path, cell_meta, rows)
                            print(
                                f"  {task_name} s{sd}: success={r.get('success')} "
                                f"released={r.get('rails_released')} hold={r.get('grasp_hold')} "
                                f"escaped={r.get('escaped')}",
                                flush=True,
                            )
                    sr = float(np.mean([bool(x.get("success")) for x in rows])) if rows else 0.0
                    cell = dict(cell_meta)
                    cell.update({"sr": sr, "n": len(rows), "episodes": rows, "partial": False})
                    cells.append(cell)
                    dump_partial_cell(cell_path, cell_meta, rows, partial=False)

        if policy is not None and pi_spec is not None:
            close_fn = getattr(policy, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        del ctx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = enrich_table_with_curve_metrics(aggregate_sr_table(cells))
    payload = {
        "protocol": protocol_tag(),
        "task_names": task_names,
        "seeds": seeds,
        "backend": args.backend,
        "latency_mode": ep_latency_mode,
        "language_mode": parse_language_mode(args.language_mode),
        "policy_url": args.policy_url,
        "pi_speedups": args.pi_speedups or args.pi_speedup,
        "cells": cells,
        "table": table,
    }
    with open(out_dir / "sr_sweep.json", "w") as f:
        json.dump(payload, f, indent=2)
    write_curve_csv(cells, out_dir / "sr_curves.csv")
    print(json.dumps(table, indent=2))
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
