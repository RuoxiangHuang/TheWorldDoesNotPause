#!/usr/bin/env python3
"""CLIRA probe-2 runner: cost split, cache age, executor alignment.

Observation stream from an eager dump policy (same as exp-1). Shadows after
the episode. NFE is fixed at --nfe (default 10).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from pathlib import Path as _P

for _anc in _P(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

setup_sys_path()

from benchmarks.dynamic_libero.env import libero_bridge as bridge
from benchmarks.dynamic_libero.eval.cli_common import (
    grasp_kwargs_from_args,
    language_mode_from_args,
    motive_from_args,
    mover_language_from_args,
    movers_enabled_from_args,
    target_registry_from_args,
)
from benchmarks.dynamic_libero.eval.coupling_probe2_common import AGES
from benchmarks.dynamic_libero.eval.rollout import run_episode
from benchmarks.common.policy.factory import make_inprocess_policy
from benchmarks.dynamic_libero.eval.run_coupling import (
    DumpingPolicy,
    _bin_kind,
    _jobs,
    _suite,
)
from benchmarks.common.grasp import add_grasp_cli_args


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", type=str, required=True, choices=["fastwam", "pi"])
    p.add_argument("--dynamic-tasks", type=str, default="")
    p.add_argument("--speeds", type=str, default="0.0,0.0003,0.0006,0.001")
    p.add_argument("--init-ids", type=str, default="0,1")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--out-jsonl", type=str, required=True)
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/DATA/YuanZhen/FasterWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dump-nfe", type=int, default=4)
    p.add_argument("--nfe", type=int, default=10)
    p.add_argument("--ages", type=str, default=",".join(str(a) for a in AGES))
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", type=str, default="freeze")
    p.add_argument("--pi-ckpt", type=str, default="/DATA/YuanZhen/FasterPI/ckpt/pi05_libero_pytorch")
    p.add_argument(
        "--pi-norm",
        type=str,
        default="/DATA/YuanZhen/cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",
    )
    add_grasp_cli_args(p)
    p.add_argument("--movers", type=str, default="off", choices=["on", "off"])
    p.add_argument("--mover-language", type=str, default="keep")
    p.add_argument("--language-mode", type=str, default="keep", choices=["keep", "motion", "scene"])
    p.add_argument("--no-target-registry", action="store_true")
    p.add_argument("--motive", type=str, default="auto")
    p.add_argument("--release-radius", type=float, default=0.06)
    p.add_argument("--trajectories", type=str, default="linear")
    args = p.parse_args()
    if not str(args.dynamic_tasks).strip():
        from benchmarks.dynamic_libero.eval.run_coupling import COUPLING_TASKS

        args.dynamic_tasks = ",".join(COUPLING_TASKS)
    return args


def _json_default(o):
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def main() -> None:
    args = _parse()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(args)
    ages = tuple(int(x) for x in args.ages.split(",") if x.strip())
    print(
        f"[probe2] backbone={args.backbone} shard={args.shard_index}/{args.shard_count} "
        f"jobs={len(jobs)} nfe={args.nfe} ages={ages} out={out}",
        flush=True,
    )
    if not jobs:
        print("[probe2] empty shard, exit", flush=True)
        return
    suite_cache: dict = {}
    movers_on = movers_enabled_from_args(args)
    mover_lang = mover_language_from_args(args)
    language_mode = language_mode_from_args(args)
    target_registry = target_registry_from_args(args)
    grasp_kw = grasp_kwargs_from_args(args)
    motive = motive_from_args(args)

    if args.backbone == "fastwam":
        from benchmarks.dynamic_libero.eval.coupling_probe2_fastwam import FastWAMProbe2

        ctx = bridge.load_policy(
            checkpoint=args.checkpoint,
            task_suite_name=jobs[0][0].suite if jobs else "libero_object",
            task_id=jobs[0][0].task_id if jobs else 0,
            seed=args.seed,
            device=args.device,
            num_inference_steps=args.dump_nfe,
            accel="off",
        )
        inner = make_inprocess_policy("libero", ctx)
        probe = FastWAMProbe2(ctx, nfe=args.nfe, ages=ages)
        policy = DumpingPolicy(inner)
    else:
        from fasterpi.realtime.loader import load_libero_policy, warmup_policy
        from benchmarks.dynamic_libero.eval.coupling_probe2_pi import PiProbe2

        rt = load_libero_policy(
            ckpt=args.pi_ckpt,
            norm_assets=args.pi_norm,
            device=args.device,
            speedup_dirs="eager",
        )
        warmup_policy(rt, n=2)
        ctx = bridge.load_native_env_context(
            task_suite_name=jobs[0][0].suite if jobs else "libero_object",
            task_id=jobs[0][0].task_id if jobs else 0,
            seed=args.seed,
            device=args.device,
        )
        probe = PiProbe2(rt, nfe=args.nfe, ages=ages)
        policy = DumpingPolicy(rt)

    n_ok = 0
    n_fail = 0
    with out.open("a") as fh:
        for ji, (job, speed, iid) in enumerate(jobs):
            ts = _suite(job.suite, suite_cache)
            if args.backbone == "fastwam":
                ctx.cfg.EVALUATION.task_suite_name = job.suite
                ctx.cfg.EVALUATION.task_id = int(job.task_id)
            ctx.task_suite = ts
            task = ts.get_task(job.task_id)
            inits = list(ts.get_task_init_states(job.task_id))
            if iid >= len(inits):
                print(f"[skip] {job.id} init={iid} (only {len(inits)})", flush=True)
                continue
            print(f"[{ji+1}/{len(jobs)}] {job.id} v={speed:g} init={iid}", flush=True)
            policy.reset()
            probe.reset()
            try:
                r = run_episode(
                    task,
                    inits[iid],
                    ctx,
                    trajectory_kind="linear",
                    speed=speed,
                    trajectory_kwargs={"toward_center": True},
                    n_freeze=0.0,
                    release_radius=args.release_radius,
                    max_steps=args.max_steps,
                    task_suite_name=job.suite,
                    task_id=job.task_id,
                    movers_enabled=movers_on,
                    mover_language=mover_lang,
                    language_mode=language_mode,
                    target_registry=target_registry,
                    backend=args.backend,
                    motive=motive,
                    policy=policy,
                    rails_variant=job.variant,
                    dynamic_task_id=job.id,
                    place_target_name=job.place_target,
                    init_id=iid,
                    **grasp_kw,
                )
            except Exception as exc:
                n_fail += 1
                print(f"[episode-fail] {job.id} {exc}", flush=True)
                traceback.print_exc()
                continue
            kind = _bin_kind(job.variant, job.suite, speed)
            n_pairs = 0
            for ci, frame in enumerate(policy.frames):
                try:
                    m = probe.measure(frame["obs"], frame["instruction"])
                except Exception as exc:
                    print(f"[probe-fail] {job.id} c={ci} {exc}", flush=True)
                    traceback.print_exc()
                    break
                row = {
                    "probe": "probe2",
                    "backbone": args.backbone,
                    "task_id": job.id,
                    "suite": job.suite,
                    "variant": job.variant,
                    "speed": speed,
                    "init_id": iid,
                    "replan": ci,
                    "bin_kind": kind,
                    "success": bool(r.get("success", False)),
                    "n_replans": int(r.get("n_replans") or len(policy.frames)),
                    "phase": frame.get("phase"),
                    "object_speed": frame.get("object_speed"),
                    "eef_dist": frame.get("eef_dist"),
                    "released": frame.get("released"),
                    "t_utc": datetime.now(timezone.utc).isoformat(),
                    **m,
                }
                fh.write(json.dumps(row, default=_json_default) + "\n")
                if m.get("delta_cam_age1") is not None:
                    n_pairs += 1
            fh.flush()
            n_ok += 1
            print(
                f"  frames={len(policy.frames)} pairs={n_pairs} success={r.get('success')}",
                flush=True,
            )
    print(f"[probe2] done ok={n_ok} fail={n_fail} -> {out}", flush=True)


if __name__ == "__main__":
    main()
