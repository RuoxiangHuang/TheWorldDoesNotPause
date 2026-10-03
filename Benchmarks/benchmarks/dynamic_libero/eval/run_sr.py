"""SR sweep over speed × trajectory × accel for Dynamic-LIBERO."""

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
from benchmarks.dynamic_libero.env import libero_bridge as bridge
from benchmarks.dynamic_libero.env.suite_registry import parse_init_ids, parse_task_ids

_SUITE_CACHE: dict[str, Any] = {}


def _libero_suite(name: str):
    from libero.libero import benchmark as libero_benchmark

    name = str(name).strip()
    if name not in _SUITE_CACHE:
        _SUITE_CACHE[name] = libero_benchmark.get_benchmark_dict()[name]()
    return _SUITE_CACHE[name]


def _bind_suite(ctx, suite_name: str, task_id: int):
    ctx.cfg.EVALUATION.task_suite_name = suite_name
    ctx.cfg.EVALUATION.task_id = int(task_id)
    ctx.task_suite = _libero_suite(suite_name)
    return ctx.task_suite
from benchmarks.common.protocol import DEFAULT_BACKEND
from benchmarks.dynamic_libero.eval.cli_common import (
    add_dynamic_task_args,
    add_motive_arg,
    add_smooth_turn_args,
    dynamic_eval_units,
    grasp_kwargs_from_args,
    language_mode_from_args,
    motive_from_args,
    mover_language_from_args,
    movers_enabled_from_args,
    protocol_tag,
    rails_variant_from_args,
    target_registry_from_args,
    traj_kwargs_from_args,
)
from benchmarks.common.grasp import add_grasp_cli_args
from benchmarks.common.policy.factory import make_policy_client
from benchmarks.dynamic_libero.env.official import parse_speed_list
from benchmarks.dynamic_libero.eval.measure_latency import measure_latency_varying, n_freeze_from_latency
from benchmarks.dynamic_libero.eval.rollout import run_episode
from benchmarks.dynamic_libero.metrics.aggregate import (
    aggregate_sr_table,
    enrich_table_with_curve_metrics,
    write_curve_csv,
)


DEFAULT_SPEEDS = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.006, 0.010]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic-LIBERO SR sweep")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="FastWAM checkpoint. Not required with --policy-url or --pi-speedups.",
    )
    p.add_argument("--task-suite", type=str, default="libero_object")
    p.add_argument("--task-ids", type=str, default="0", help="Comma-separated task ids or 'all'")
    p.add_argument("--init-ids", type=str, default="0,1,2,3,4")
    p.add_argument("--accels", type=str, default="off,pace")
    p.add_argument(
        "--trajectories",
        type=str,
        default="linear,sine",
        help="Comma-separated: linear,sine,circle,polyline,stop_and_go,random,smooth_turn. "
        "smooth_turn is a diagnostic slice (not the official linear/sine grid).",
    )
    p.add_argument(
        "--speeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SPEEDS),
        help="Comma-separated m/tick, or alias official|paper. "
        "paper = async SR–speed curve including v=0 and v=0.0006. "
        "Backend stays async at every speed; freeze is a separate --backend flag.",
    )
    p.add_argument("--axis", type=str, default="x")
    p.add_argument("--direction", type=float, default=1.0)
    p.add_argument(
        "--toward-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inward rails (default on). Does not change BDDL goals or language.",
    )
    p.add_argument("--n-waypoints", type=int, default=5, help="For trajectory=random")
    p.add_argument("--extent", type=float, default=0.12, help="For trajectory=random (metres)")
    p.add_argument("--release-radius", type=float, default=0.06)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-inference-steps", type=int, default=4)
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
        help="Episode coupling. Default measured; --fixed-latency implies constant "
        "so injected L actually advances the world (Cache+matched-delay cells).",
    )
    p.add_argument(
        "--cell-label",
        type=str,
        default="",
        help="Optional table label (e.g. cache_delay). Default: the accel name.",
    )
    p.add_argument(
        "--movers",
        type=str,
        default="off",
        choices=["on", "off"],
    )
    p.add_argument(
        "--mover-language",
        type=str,
        default="keep",
        choices=["rewrite", "keep"],
    )
    p.add_argument(
        "--language-mode",
        type=str,
        default="keep",
        choices=["keep", "motion", "scene"],
        help="keep=primary; motion=B-track suffix; scene=spatial carrier locative rewrite.",
    )
    p.add_argument("--no-target-registry", action="store_true")
    add_grasp_cli_args(p)
    add_motive_arg(p)
    add_dynamic_task_args(p)
    add_smooth_turn_args(p)
    from benchmarks.common.scene import add_scene_cli_args

    add_scene_cli_args(p)
    p.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND.value,
        choices=["freeze", "async"],
        help="Protocol default is async; freeze is thinking-hold ablation.",
    )
    p.add_argument(
        "--replan-steps",
        type=str,
        default="10",
        help="Comma-separated execution prefix in control ticks. The model still "
        "emits a full action chunk; only this many ticks run before the next "
        "observation. Locked protocol value is 10.",
    )
    p.add_argument(
        "--allow-replan-override",
        action="store_true",
        help="Required when any --replan-steps value is not the locked LIBERO 10.",
    )
    p.add_argument(
        "--policy-url",
        type=str,
        default=None,
        help="Remote HTTP policy (legacy). Prefer --pi-speedups for in-process OpenPI.",
    )
    p.add_argument(
        "--pi-speedups",
        type=str,
        default=None,
        help="In-process OpenPI: comma list of label=spec, e.g. "
        "pi=eager,fasterpi=compile+chunk_residual_cache+step_cache+compile_prefix. Loads policy in this process "
        "(measured L is model wall-clock, no HTTP RTT).",
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
        default="/DATA/YuanZhen/FasterPI/ckpt/pi05_libero_pytorch",
    )
    p.add_argument(
        "--pi-norm",
        type=str,
        default="/DATA/YuanZhen/cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",
    )
    p.add_argument("--out-dir", type=str, default="evaluate_results/dynamic_libero/sr_sweep")
    return p.parse_args()


def _parse_pi_speedups(spec: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            label, s = item.split("=", 1)
            pairs.append((label.strip(), s.strip()))
        else:
            label = "pi" if item in ("eager", "none", "") else "fasterpi"
            pairs.append((label, item))
    return pairs


def _parse_replan_steps(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in str(raw or "10").split(",") if x.strip()]
    if not vals:
        return [10]
    if any(v <= 0 for v in vals):
        raise SystemExit("--replan-steps must be positive")
    return vals


def _fmt_turn(r: dict[str, Any]) -> str:
    t_ev = r.get("turn_event_t")
    if t_ev is None:
        t_ev = r.get("event_time")
    if t_ev is None:
        return ""

    def f(x: Any) -> str:
        return "—" if x is None else f"{float(x):.3f}"

    missed = r.get("event_experienced") is False
    if missed:
        return f" event_t={float(t_ev):.2f} missed"
    return (
        f" event_t={float(t_ev):.2f}"
        f" obs_s={f(r.get('turn_event_to_first_obs_s'))}"
        f" take_s={f(r.get('turn_first_obs_to_takeover_s'))}"
        f" corr_s={f(r.get('turn_event_to_correction_s'))}"
    )


def _load_openpi_policy(ckpt: str, norm: str, spec: str, device: str, num_steps: int = 10):
    from fasterpi.realtime.loader import load_libero_policy, warmup_policy

    rt = load_libero_policy(
        ckpt=ckpt,
        norm_assets=norm,
        device=device,
        speedup_dirs=spec,
        num_steps=int(num_steps),
    )
    warmup_policy(rt, n=4)
    return rt


def _traj_kwargs(kind: str, args: argparse.Namespace) -> dict[str, Any]:
    ns = argparse.Namespace(**vars(args))
    ns.trajectory = kind
    return traj_kwargs_from_args(ns)


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    units = dynamic_eval_units(args)
    if units:
        args.task_suite = units[0].suite
        task_ids = [u.task_id for u in units]
    else:
        task_ids = parse_task_ids(args.task_ids, args.task_suite)
        units = None
    init_ids = parse_init_ids(args.init_ids)
    accels = [x.strip() for x in args.accels.split(",") if x.strip()]
    pi_pairs: list[tuple[str, str]] | None = None
    if args.pi_speedups:
        pi_pairs = _parse_pi_speedups(args.pi_speedups)
        if not pi_pairs:
            raise SystemExit("--pi-speedups parsed empty")
    elif args.pi_speedup:
        spec = str(args.pi_speedup).strip()
        if len(accels) == 1:
            pi_pairs = [(accels[0], spec)]
        else:
            label = "pi" if spec in ("eager", "none", "") else "fasterpi"
            pi_pairs = [(label, spec)]
    if args.policy_url and pi_pairs:
        raise SystemExit("use either --policy-url or --pi-speedups, not both")
    if args.policy_url:
        if len(accels) != 1:
            raise SystemExit(
                "with --policy-url pass exactly one --accels label "
                "(restart the HTTP server to switch PI vs FasterPI)"
            )
    elif pi_pairs is None and not args.checkpoint:
        raise SystemExit(
            "--checkpoint is required unless --policy-url or --pi-speedups is set"
        )
    eval_jobs: list[tuple[str, str | None]]
    if pi_pairs is not None:
        eval_jobs = [(label, spec) for label, spec in pi_pairs]
    else:
        eval_jobs = [(a, None) for a in accels]
    trajectories = [x.strip() for x in args.trajectories.split(",") if x.strip()]
    if units and all(getattr(u, "track", "main") == "react" for u in units):
        if trajectories != ["smooth_turn"]:
            print(
                "[react track] forcing --trajectories smooth_turn "
                "(aligned pick + one world-clock event; not the official grid)",
                flush=True,
            )
            trajectories = ["smooth_turn"]
    speeds = parse_speed_list(args.speeds, default=tuple(DEFAULT_SPEEDS))
    print(
        f"[protocol] backend={args.backend} speeds={speeds} "
        f"scene_preset={getattr(args, 'scene_preset', 'auto')} "
        "(world-clock is independent of speed; v=0 no longer zeroes n_freeze)",
        flush=True,
    )
    replan_list = _parse_replan_steps(args.replan_steps)
    if any(r != 10 for r in replan_list) and not args.allow_replan_override:
        raise SystemExit(
            "non-default --replan-steps requires --allow-replan-override "
            "(LIBERO protocol lock is 10; this is a frequency ablation, not the main table)"
        )
    movers_on = movers_enabled_from_args(args)
    mover_lang = mover_language_from_args(args)
    language_mode = language_mode_from_args(args)
    target_registry = target_registry_from_args(args)
    grasp_kw = grasp_kwargs_from_args(args)
    motive = motive_from_args(args)
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
        if pi_spec is not None:
            print(f"in-process OpenPI speedup={pi_spec!r}", flush=True)
            policy = _load_openpi_policy(
                args.pi_ckpt, args.pi_norm, pi_spec, args.device, args.num_inference_steps
            )
            ctx = bridge.load_native_env_context(
                task_suite_name=args.task_suite,
                task_id=task_ids[0],
                seed=args.seed,
                device=args.device,
            )
            task0 = ctx.task_suite.get_task(task_ids[0])
            if args.fixed_latency is not None:
                lat = float(args.fixed_latency)
                raw = [lat]
            else:
                from fasterpi.realtime.measure import measure_libero_latency

                lat, raw = measure_libero_latency(
                    policy,
                    task0,
                    suite=ctx.task_suite,
                    task_id=task_ids[0],
                    seed=args.seed,
                    k=args.latency_k,
                    warmup=args.latency_warmup,
                )
        elif args.policy_url:
            policy = make_policy_client("libero", policy_url=args.policy_url)
            ctx = bridge.load_env_context(
                task_suite_name=args.task_suite,
                task_id=task_ids[0],
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
            )
            task0 = ctx.task_suite.get_task(task_ids[0])
            if args.fixed_latency is not None:
                lat = float(args.fixed_latency)
                raw = [lat]
            else:
                from benchmarks.common.policy.measure_http import (
                    measure_http_latency_libero,
                )

                lat, raw = measure_http_latency_libero(
                    args.policy_url,
                    task0,
                    suite=ctx.task_suite,
                    task_id=task_ids[0],
                    seed=args.seed,
                    k=args.latency_k,
                    warmup=args.latency_warmup,
                )
        else:
            ctx = bridge.load_policy(
                checkpoint=args.checkpoint,
                task_suite_name=args.task_suite,
                task_id=task_ids[0],
                seed=args.seed,
                device=args.device,
                num_inference_steps=args.num_inference_steps,
                accel=accel,
            )
            task0 = ctx.task_suite.get_task(task_ids[0])
            if args.fixed_latency is not None:
                lat = float(args.fixed_latency)
                raw = [lat]
            else:
                lat, raw = measure_latency_varying(
                    task0, ctx, k=args.latency_k, warmup=args.latency_warmup
                )
        ctrl = float(ctx.cfg.EVALUATION.get("control_freq", 20))
        n_freeze = n_freeze_from_latency(lat, ctrl)
        print(
            f"latency={lat*1000:.1f}ms n_freeze={n_freeze:.3f} "
            f"latency_mode={ep_latency_mode} cell_label={str(args.cell_label).strip() or accel}",
            flush=True,
        )

        for traj in trajectories:
            for rsteps in replan_list:
                period_s = float(lat) + float(rsteps) / ctrl
                fb_hz = (1.0 / period_s) if period_s > 0 else 0.0
                for speed in speeds:
                    from benchmarks.common.scene import iter_scene_conditions

                    scene_jobs = iter_scene_conditions(
                        args, speed=speed, catalog_job=units is not None
                    )
                    for scene_label, scene_cond in scene_jobs:
                        nf = scene_cond.apply_n_freeze(float(n_freeze))
                        print(
                            f"--- {accel} | r={rsteps} | {traj} @ {speed} "
                            f"scene={scene_label} rails={scene_cond.rails} "
                            f"(n_freeze={nf:.3f} T_fb={period_s*1000:.0f}ms {fb_hz:.2f}Hz) ---",
                            flush=True,
                        )
                        rows = []
                        cell_name = (
                            f"cell_{accel}_{traj}_v{speed}_r{rsteps}_{scene_label}.json"
                        )
                        cell_path = out_dir / cell_name
                        jsonl_path = cell_path.with_suffix(".jsonl")
                        cell_meta = {
                            "accel": accel,
                            "trajectory": traj,
                            "speed": speed,
                            "scene_preset": scene_label,
                            "scene": scene_cond.to_dict(),
                            "replan_steps": int(rsteps),
                            "control_hz": ctrl,
                            "feedback_period_s": period_s,
                            "feedback_hz": fb_hz,
                            "latency_s": lat,
                            "latency_ms": lat * 1000.0,
                            "latency_raw": raw,
                            "n_freeze": nf,
                            "backend": args.backend,
                            "latency_mode": ep_latency_mode,
                            "delivery_delay_s": float(args.delivery_delay_s),
                            "nfe": int(args.num_inference_steps),
                            "cell_label": (str(args.cell_label).strip() or accel),
                            "fixed_latency_s": (
                                None if args.fixed_latency is None else float(args.fixed_latency)
                            ),
                            "pursuit_lag_m": (
                                float(speed) * nf if scene_cond.advances_rails() else 0.0
                            ),
                        }
                        jobs = units if units is not None else [None]
                        for job in jobs:
                            tid = job.task_id if job is not None else None
                            if job is None:
                                break
                            suite_name = job.suite
                            ts = _bind_suite(ctx, suite_name, tid)
                            dyn_kw = {
                                "rails_variant": job.variant,
                                "dynamic_task_id": job.id,
                                "place_target_name": job.place_target,
                            }
                            task = ts.get_task(tid)
                            inits = list(ts.get_task_init_states(tid))
                            for iid in init_ids:
                                if iid >= len(inits):
                                    continue
                                r = run_episode(
                                    task,
                                    inits[iid],
                                    ctx,
                                    trajectory_kind=traj,
                                    speed=speed,
                                    trajectory_kwargs=_traj_kwargs(traj, args),
                                    n_freeze=nf,
                                    release_radius=args.release_radius,
                                    max_steps=args.max_steps,
                                    task_suite_name=suite_name,
                                    task_id=tid,
                                    movers_enabled=movers_on,
                                    mover_language=mover_lang,
                                    language_mode=language_mode,
                                    target_registry=target_registry,
                                    backend=args.backend,
                                    motive=motive,
                                    policy=policy,
                                    replan_steps=rsteps,
                                    allow_replan_override=bool(args.allow_replan_override) or rsteps == 10,
                                    latency_mode=ep_latency_mode,
                                    delivery_delay_s=float(args.delivery_delay_s),
                                    init_id=iid,
                                    scene=scene_cond,
                                    **grasp_kw,
                                    **dyn_kw,
                                )
                                r.update(
                                    {
                                        "task_id": tid,
                                        "init_id": iid,
                                        "accel": accel,
                                        "latency_s": lat,
                                        "dynamic_task_id": job.id,
                                        "rails_variant": job.variant,
                                        "suite": suite_name,
                                        "scene_preset": scene_label,
                                    }
                                )
                                rows.append(r)
                                append_jsonl(jsonl_path, r)
                                dump_partial_cell(cell_path, cell_meta, rows)
                                print(
                                    f"  {job.id} i{iid}: success={r['success']} "
                                    f"released={r.get('rails_released')} "
                                    f"hold={r.get('grasp_hold')} escaped={r['escaped']}"
                                    f"{_fmt_turn(r)}",
                                    flush=True,
                                )
                        if units is None:
                            for tid in task_ids:
                                task = ctx.task_suite.get_task(tid)
                                inits = list(ctx.task_suite.get_task_init_states(tid))
                                for iid in init_ids:
                                    if iid >= len(inits):
                                        continue
                                    ctx.cfg.EVALUATION.task_id = tid
                                    r = run_episode(
                                        task,
                                        inits[iid],
                                        ctx,
                                        trajectory_kind=traj,
                                        speed=speed,
                                        trajectory_kwargs=_traj_kwargs(traj, args),
                                        n_freeze=nf,
                                        release_radius=args.release_radius,
                                        max_steps=args.max_steps,
                                        task_suite_name=args.task_suite,
                                        task_id=tid,
                                        movers_enabled=movers_on,
                                        mover_language=mover_lang,
                                        language_mode=language_mode,
                                        target_registry=target_registry,
                                        backend=args.backend,
                                        motive=motive,
                                        rails_variant=rails_variant_from_args(args),
                                        policy=policy,
                                        replan_steps=rsteps,
                                        allow_replan_override=bool(args.allow_replan_override) or rsteps == 10,
                                        latency_mode=ep_latency_mode,
                                        delivery_delay_s=float(args.delivery_delay_s),
                                        init_id=iid,
                                        scene=scene_cond,
                                        **grasp_kw,
                                    )
                                    r.update(
                                        {
                                            "task_id": tid,
                                            "init_id": iid,
                                            "accel": accel,
                                            "latency_s": lat,
                                            "scene_preset": scene_label,
                                        }
                                    )
                                    rows.append(r)
                                    append_jsonl(jsonl_path, r)
                                    dump_partial_cell(cell_path, cell_meta, rows)
                                    print(
                                        f"  t{tid}i{iid}: success={r['success']} "
                                        f"released={r.get('rails_released')} "
                                        f"hold={r.get('grasp_hold')} escaped={r['escaped']}"
                                        f"{_fmt_turn(r)}",
                                        flush=True,
                                    )
                        sr = float(np.mean([bool(x["success"]) for x in rows])) if rows else 0.0
                        cell = dict(cell_meta)
                        cell.update({"sr": sr, "n": len(rows), "episodes": rows, "partial": False})
                        cells.append(cell)
                        dump_partial_cell(cell_path, cell_meta, rows, partial=False)

        if policy is not None:
            try:
                policy.close()
            except Exception:
                pass
            del policy
        del ctx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = enrich_table_with_curve_metrics(aggregate_sr_table(cells))
    payload = {
        "protocol": protocol_tag(),
        "task_suite": args.task_suite,
        "task_ids": task_ids,
        "dynamic_tasks": None if units is None else [u.id for u in units],
        "init_ids": init_ids,
        "movers": movers_on,
        "mover_language": mover_lang,
        "language_mode": language_mode,
        "target_registry": target_registry,
        "motive": motive,
        "backend": args.backend,
        "replan_steps": replan_list,
        "policy_url": args.policy_url,
        "pi_speedups": args.pi_speedups or args.pi_speedup,
        "eval_mode": (
            "inprocess_openpi"
            if pi_pairs is not None
            else ("http" if args.policy_url else "fastwam")
        ),
        "smooth_turn_diagnostic": any(
            str(t).strip().lower() in ("smooth_turn", "turn") for t in trajectories
        ),
        "latency_mode": ep_latency_mode,
        "cell_label": str(args.cell_label).strip() or None,
        "fixed_latency_s": None if args.fixed_latency is None else float(args.fixed_latency),
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
