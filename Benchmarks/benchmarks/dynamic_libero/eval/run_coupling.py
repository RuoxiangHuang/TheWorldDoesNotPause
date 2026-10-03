#!/usr/bin/env python3
"""CLIRA exp-1: δ_w vs reduced-budget action error on Dynamic-LIBERO.

Observation stream comes from a full (eager) policy with measured latency.
Shadow A/B forwards run AFTER each episode and do not enter n_freeze.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sys
from pathlib import Path as _P

for _anc in _P(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

ROOT = setup_sys_path()

from benchmarks.dynamic_libero.env import libero_bridge as bridge
from benchmarks.dynamic_libero.env.dynamic_tasks import parse_dynamic_tasks
from benchmarks.dynamic_libero.eval.cli_common import (
    grasp_kwargs_from_args,
    language_mode_from_args,
    motive_from_args,
    mover_language_from_args,
    movers_enabled_from_args,
    target_registry_from_args,
)
from benchmarks.common.grasp import add_grasp_cli_args
from benchmarks.dynamic_libero.eval.rollout import run_episode
from benchmarks.common.policy.factory import make_inprocess_policy

COUPLING_TASKS = [
    "libero_object.t00.pick",
    "libero_object.t00.place",
    "libero_object.t00.both",
    "libero_object.t04.pick",
    "libero_object.t04.place",
    "libero_object.t04.both",
    "libero_object.t07.pick",
    "libero_object.t07.place",
    "libero_object.t02.pick",
    "libero_object.t02.place",
    "libero_object.t03.pick.irregular",
    "libero_spatial.t00.pick.carrier",
    "libero_spatial.t02.pick.carrier",
    "libero_spatial.t05.pick.carrier",
    "libero_spatial.t08.pick.carrier",
    "libero_spatial.t02.place",
]
DEFAULT_SPEEDS = [0.0, 0.0003, 0.0006, 0.001]


def _suite(name: str, cache: dict):
    from libero.libero import benchmark as libero_benchmark

    if name not in cache:
        cache[name] = libero_benchmark.get_benchmark_dict()[name]()
    return cache[name]


def _copy_obs(obs: dict) -> dict:
    keys = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    out = {}
    for k in keys:
        if k not in obs:
            continue
        v = obs[k]
        out[k] = np.array(v, copy=True)
    return out


def _driver_meta(driver: Any, obs: Any) -> dict[str, Any]:
    if driver is None:
        return {}
    hz = float(getattr(driver, "control_hz", 20.0) or 20.0)
    traj = getattr(driver, "trajectory", None)
    speed = 0.0
    if traj is not None:
        t = float(getattr(traj, "t", 0.0))
        try:
            v = np.asarray(traj.velocity_at(t, hz=hz), dtype=np.float64).reshape(-1)
            speed = float(np.linalg.norm(v)) / max(hz, 1e-6)
        except Exception:
            speed = 0.0
    phase = getattr(driver, "rails_phase", "pursuit")
    phase_s = str(getattr(phase, "value", phase) or "pursuit").lower()
    dist = None
    fn = getattr(driver, "_eef_dist", None)
    if callable(fn) and obs is not None:
        try:
            dist = float(fn(obs))
        except Exception:
            dist = None
    return {
        "phase": phase_s,
        "object_speed": speed,
        "eef_dist": dist,
        "released": bool(getattr(driver, "released", False)),
    }


def _bin_kind(variant: str, suite: str, speed: float) -> str:
    if float(speed) <= 0:
        return "static"
    if "spatial" in suite:
        return "spatial_carrier"
    if variant == "place":
        return "fine_contact"
    if variant == "both":
        return "dual_motion"
    return "object_motion"


class DumpingPolicy:
    def __init__(self, inner: Any):
        self.inner = inner
        self.frames: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.frames = []
        fn = getattr(self.inner, "reset", None)
        if callable(fn):
            fn()

    def predict(self, obs: Any, instruction: str, **kwargs: Any) -> np.ndarray:
        rec = {
            "obs": _copy_obs(obs),
            "instruction": str(instruction),
            **_driver_meta(kwargs.get("driver"), obs),
        }
        self.frames.append(rec)
        return np.asarray(self.inner.predict(obs, instruction, **kwargs))

    def close(self) -> None:
        fn = getattr(self.inner, "close", None)
        if callable(fn):
            fn()


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", type=str, required=True, choices=["fastwam", "pi"])
    p.add_argument("--dynamic-tasks", type=str, default=",".join(COUPLING_TASKS))
    p.add_argument("--speeds", type=str, default=",".join(str(s) for s in DEFAULT_SPEEDS))
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
    p.add_argument("--dump-nfe", type=int, default=4, help="FastWAM NFE used to generate the stream")
    p.add_argument("--nfe-full", type=int, default=10)
    p.add_argument("--nfe-mid", type=int, default=8)
    p.add_argument("--nfe-cheap", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", type=str, default="freeze")
    p.add_argument("--trajectories", type=str, default="linear")
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
    return p.parse_args()


def _jobs(args: argparse.Namespace) -> list[tuple[Any, float, int]]:
    tasks = parse_dynamic_tasks(args.dynamic_tasks)
    if not tasks:
        tasks = parse_dynamic_tasks(",".join(COUPLING_TASKS))
    speeds = [float(x) for x in args.speeds.split(",") if x.strip() != ""]
    inits = [int(x) for x in args.init_ids.split(",") if x.strip() != ""]
    jobs = []
    for t in tasks:
        for sp in speeds:
            for iid in inits:
                jobs.append((t, sp, iid))
    shard = [j for i, j in enumerate(jobs) if i % args.shard_count == args.shard_index]
    return shard


def main() -> None:
    args = _parse()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(args)
    print(
        f"[coupling] backbone={args.backbone} shard={args.shard_index}/{args.shard_count} "
        f"jobs={len(jobs)} out={out}",
        flush=True,
    )
    if not jobs:
        print("[coupling] empty shard, exit", flush=True)
        return
    suite_cache: dict[str, Any] = {}
    movers_on = movers_enabled_from_args(args)
    mover_lang = mover_language_from_args(args)
    language_mode = language_mode_from_args(args)
    target_registry = target_registry_from_args(args)
    grasp_kw = grasp_kwargs_from_args(args)
    motive = motive_from_args(args)

    if args.backbone == "fastwam":
        from benchmarks.dynamic_libero.eval.coupling_fastwam import FastWAMCouplingProbe

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
        probe = FastWAMCouplingProbe(
            ctx, nfe_full=args.nfe_full, nfe_mid=args.nfe_mid, nfe_cheap=args.nfe_cheap
        )
        policy = DumpingPolicy(inner)
    else:
        from fasterpi.realtime.loader import load_libero_policy, warmup_policy
        from benchmarks.dynamic_libero.eval.coupling_pi import PiCouplingProbe

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
        probe = PiCouplingProbe(
            rt, nfe_full=args.nfe_full, nfe_mid=args.nfe_mid, nfe_cheap=args.nfe_cheap
        )
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
            print(
                f"[{ji+1}/{len(jobs)}] {job.id} v={speed:g} init={iid}",
                flush=True,
            )
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
                fh.write(json.dumps(row) + "\n")
                if m.get("delta_cos") is not None:
                    n_pairs += 1
            fh.flush()
            n_ok += 1
            print(
                f"  frames={len(policy.frames)} pairs={n_pairs} success={r.get('success')}",
                flush=True,
            )
    print(f"[coupling] done ok={n_ok} fail={n_fail} -> {out}", flush=True)


if __name__ == "__main__":
    main()
