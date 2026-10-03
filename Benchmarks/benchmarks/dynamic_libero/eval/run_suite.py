"""Paired Real-Time eval across multiple LIBERO suites (full-suite extension)."""

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

from benchmarks.dynamic_libero.env import libero_bridge as bridge
from benchmarks.dynamic_libero.env.suite_registry import (
    ALL_SUITES,
    TRAINED_SUITES,
    is_trained_suite,
    resolve_suites,
)
from benchmarks.dynamic_libero.eval.cli_common import (
    add_paired_common_args,
    dynamic_eval_units,
    grasp_kwargs_from_args,
    language_mode_from_args,
    motive_from_args,
    mover_language_from_args,
    movers_enabled_from_args,
    protocol_tag,
    rails_variant_from_args,
    resolve_task_and_init_ids,
    resolve_trajectory_from_args,
    target_registry_from_args,
)
from benchmarks.dynamic_libero.eval.measure_latency import measure_latency_varying, n_freeze_from_latency
from benchmarks.dynamic_libero.eval.rollout import run_episode
from benchmarks.common.policy.factory import make_policy_client
from benchmarks.common.protocol import RealtimeBackend, blind_window_ratio, effective_replan_steps
from benchmarks.common.workspace import parse_aabb


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired Dynamic-LIBERO eval across multiple suites"
    )
    add_paired_common_args(p)
    p.add_argument(
        "--suites",
        type=str,
        default="trained",
        help="Comma-separated suite names, or 'trained' (4 train suites) / 'all' (+ libero_90 OOD)",
    )
    p.add_argument("--out-dir", type=str, default="evaluate_results/dynamic_libero/suite_paired")
    return p.parse_args()


def _paired_delta(accels: list[str], per_accel_success: dict[str, list[bool]]) -> dict[str, Any]:
    if len(accels) < 2:
        return {}
    a0, a1 = accels[0], accels[1]
    s0, s1 = per_accel_success.get(a0, []), per_accel_success.get(a1, [])
    n = min(len(s0), len(s1))
    delta = float(np.mean([float(s1[i]) - float(s0[i]) for i in range(n)])) if n else 0.0
    return {
        f"SR({a1})-SR({a0})": delta,
        "SR_" + a0: float(np.mean(s0)) if s0 else 0.0,
        "SR_" + a1: float(np.mean(s1)) if s1 else 0.0,
        "n_paired": n,
    }


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))

    suites = resolve_suites(args.suites)
    init_ids = resolve_task_and_init_ids("0", args.init_ids, suites[0])[1]
    accels = [x.strip() for x in args.accels.split(",") if x.strip()]
    _, traj_kw = resolve_trajectory_from_args(args)
    aabb_override = parse_aabb(args.workspace_aabb) if args.workspace_aabb else None
    escape_check = not args.no_escape_check
    movers_on = movers_enabled_from_args(args)
    mover_lang = mover_language_from_args(args)
    language_mode = language_mode_from_args(args)
    target_registry = target_registry_from_args(args)
    grasp_kw = grasp_kwargs_from_args(args)
    motive = motive_from_args(args)
    from benchmarks.common.scene import scene_kwargs_from_args

    units = dynamic_eval_units(args)
    scene_cond = scene_kwargs_from_args(
        args, speed=float(args.speed), catalog_job=bool(units)
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{'_'.join(suites)}_{args.trajectory}_v{args.speed}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "protocol": protocol_tag(),
        "checkpoint": args.checkpoint,
        "suites": suites,
        "trained_suites": list(TRAINED_SUITES),
        "all_suites": list(ALL_SUITES),
        "trajectory": args.trajectory,
        "speed": args.speed,
        "init_ids": init_ids,
        "task_ids_spec": args.task_ids,
        "backend": args.backend,
        "policy_url": args.policy_url,
        "movers": movers_on,
        "mover_language": mover_lang,
        "language_mode": language_mode,
        "target_registry": target_registry,
        "motive": motive,
        "suite_results": {},
        "macro": {},
    }

    macro_per_accel: dict[str, list[bool]] = {a: [] for a in accels}

    for suite_name in suites:
        task_ids, _ = resolve_task_and_init_ids(args.task_ids, args.init_ids, suite_name)
        print(f"\n######## suite={suite_name} tasks={task_ids} ########", flush=True)

        suite_summary: dict[str, Any] = {
            "task_suite": suite_name,
            "trained_in_release_ckpt": is_trained_suite(suite_name),
            "task_ids": task_ids,
            "init_ids": init_ids,
            "accels": {},
            "paired_delta": {},
        }
        per_accel_success: dict[str, list[bool]] = {}

        for accel in accels:
            print(f"\n=== suite={suite_name} accel={accel} ===", flush=True)
            backend = RealtimeBackend(args.backend)
            if args.policy_url:
                policy = make_policy_client("libero", policy_url=args.policy_url)
                if args.fixed_latency is None:
                    raise ValueError("--policy-url requires --fixed-latency")
                lat = float(args.fixed_latency)
                raw = [lat]
                rsteps = effective_replan_steps(
                    "libero", args.replan_steps or 10, allow_override=args.allow_replan_override
                )
                n_freeze = n_freeze_from_latency(lat, 20.0)
                bwr = blind_window_ratio(n_freeze, rsteps)
                ctx = None
            else:
                ctx = bridge.load_policy(
                    checkpoint=args.checkpoint,
                    task_suite_name=suite_name,
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
                rsteps = effective_replan_steps(
                    "libero",
                    args.replan_steps or ctx.cfg.EVALUATION.get("replan_steps"),
                    allow_override=args.allow_replan_override,
                )
                bwr = blind_window_ratio(n_freeze, rsteps)
                policy = make_policy_client("libero", ctx=ctx)

            print(
                f"latency={lat*1000:.1f}ms n_freeze={n_freeze:.3f} "
                f"blind_window={bwr:.3f} replan_steps={rsteps}",
                flush=True,
            )

            rows: list[dict[str, Any]] = []
            for tid in task_ids:
                if ctx is not None:
                    task = ctx.task_suite.get_task(tid)
                    inits = list(ctx.task_suite.get_task_init_states(tid))
                    episode_ctx = ctx
                else:
                    episode_ctx = bridge.load_env_context(
                        task_suite_name=suite_name,
                        task_id=tid,
                        seed=args.seed,
                        num_inference_steps=args.num_inference_steps,
                    )
                    task = episode_ctx.task_suite.get_task(tid)
                    inits = list(episode_ctx.task_suite.get_task_init_states(tid))
                for iid in init_ids:
                    if iid >= len(inits):
                        continue
                    print(f"  {suite_name} task={tid} init={iid} ...", flush=True)
                    episode_ctx.cfg.EVALUATION.task_id = tid
                    result = run_episode(
                        task,
                        inits[iid],
                        episode_ctx,
                        trajectory_kind=args.trajectory,
                        speed=args.speed,
                        trajectory_kwargs=traj_kw,
                        n_freeze=scene_cond.apply_n_freeze(n_freeze),
                        release_radius=args.release_radius,
                        max_steps=args.max_steps,
                        backend=backend,
                        policy=policy,
                        replan_steps=args.replan_steps,
                        allow_replan_override=args.allow_replan_override,
                        task_suite_name=suite_name,
                        workspace_aabb=aabb_override,
                        escape_check=escape_check,
                        task_id=tid,
                        movers_enabled=movers_on,
                        mover_language=mover_lang,
                        language_mode=language_mode,
                        target_registry=target_registry,
                        motive=motive,
                        rails_variant=rails_variant_from_args(args),
                        init_id=iid,
                        scene=scene_cond,
                        **grasp_kw,
                    )
                    result.update(
                        {
                            "task_suite": suite_name,
                            "task_id": tid,
                            "init_id": iid,
                            "accel": accel,
                            "latency_s": lat,
                        }
                    )
                    rows.append(result)
                    print(
                        f"    success={result['success']} escaped={result['escaped']}",
                        flush=True,
                    )

            succ = [bool(r["success"]) for r in rows]
            per_accel_success[accel] = succ
            macro_per_accel[accel].extend(succ)
            sr = float(np.mean(succ)) if succ else 0.0
            suite_summary["accels"][accel] = {
                "latency_s": lat,
                "latency_ms": lat * 1000.0,
                "latency_raw": raw,
                "n_freeze": scene_cond.apply_n_freeze(n_freeze),
                "pursuit_lag_m": (
                    float(args.speed) * scene_cond.apply_n_freeze(n_freeze)
                    if scene_cond.advances_rails()
                    else 0.0
                ),
                "blind_window_ratio": bwr,
                "backend": backend.value,
                "replan_steps": rsteps,
                "sr": sr,
                "n": len(succ),
                "episodes": rows,
            }
            with open(out_dir / f"{suite_name}_episodes_{accel}.json", "w") as f:
                json.dump(suite_summary["accels"][accel], f, indent=2)

            if ctx is not None:
                del ctx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        suite_summary["paired_delta"] = _paired_delta(accels, per_accel_success)
        summary["suite_results"][suite_name] = suite_summary

    macro: dict[str, Any] = {"accels": {}, "paired_delta": _paired_delta(accels, macro_per_accel)}
    for accel in accels:
        succ = macro_per_accel[accel]
        macro["accels"][accel] = {
            "sr": float(np.mean(succ)) if succ else 0.0,
            "n": len(succ),
        }
    summary["macro"] = macro

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== macro SR ===", flush=True)
    for accel in accels:
        print(f"  {accel}: SR={macro['accels'][accel]['sr']:.3f} n={macro['accels'][accel]['n']}")
    if macro.get("paired_delta"):
        print(json.dumps(macro["paired_delta"], indent=2))
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
