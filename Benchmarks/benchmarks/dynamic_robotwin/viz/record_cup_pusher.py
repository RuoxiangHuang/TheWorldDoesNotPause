#!/usr/bin/env python3
"""Record Dynamic-RoboTwin bumper-car reconstructions (no policy).

Original meshes stay. A bumper car rides the rails behind the moving actor.
Instruction / check_success are unchanged.

Suites (one unified reconstruction batch; seed/wave2 are aliases):
  recon        15 skills × pick/place/both linear + pick/place irregular (75)
  recon-both   both-linear only (standoff iteration)
  recon-demo   both linear + pick/place irregular (45)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_DEFAULT_BASE = "evaluate_results/demo_videos/dynamic_robotwin"


def _annotate(frame_rgb: np.ndarray, lines: list[str], *, scale: int = 4) -> np.ndarray:
    im = Image.fromarray(np.ascontiguousarray(frame_rgb)).convert("RGB")
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.Resampling.LANCZOS)
    W, H = im.size
    draw = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype(_FONT, max(18, W // 42))
    except OSError:
        fnt = ImageFont.load_default()
    bar_h = int(fnt.size * (len(lines) + 0.7))
    draw.rectangle([0, 0, W, bar_h], fill=(0, 0, 0))
    for i, ln in enumerate(lines):
        draw.text((10, 6 + i * fnt.size), ln, fill=(255, 255, 255), font=fnt)
    return np.asarray(im)


for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break

from bootstrap_paths import setup_sys_path

ROOT = setup_sys_path()

from benchmarks.dynamic_libero.trajectories import build_trajectory
from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge
from benchmarks.dynamic_robotwin.env.dynamic_tasks import (
    get_dynamic_task,
    recon_both_tasks,
    recon_demo_tasks,
    recon_tasks,
)
from benchmarks.dynamic_robotwin.env.driver import RealtimeRoboTwinDriver, wrap_env_with_driver
from benchmarks.dynamic_robotwin.env.dynamics import build_dynamic_config
from benchmarks.dynamic_robotwin.env.physics import _hold_qpos_action
from benchmarks.dynamic_robotwin.env.task_registry import default_instruction, resolve_traj_kwargs


def _noun(attr: str | None) -> str:
    return str(attr or "target").replace("_", " ").upper()


def _who_line(task) -> str:
    pick = _noun(task.pick_attr)
    place = _noun(task.place_attr)
    irr = task.trajectory_kind in ("curve", "irregular")
    path = "irregular s_wave" if irr else "linear"
    if task.variant == "pick":
        return f"red car pushes {pick} | {path} ({place} still)"
    if task.variant == "place":
        return f"blue car pushes {place} | {path} ({pick} still)"
    return f"red car pushes {pick} + blue car pushes {place} | both {path}"


def _ensure_demo_attrs(env, task) -> None:
    from benchmarks.dynamic_robotwin.env.robotwin_bridge import ensure_play_once_eval_attrs

    ensure_play_once_eval_attrs(env)


def _traj_kwargs(task) -> tuple[str, dict]:
    kind, spec = task.traj_spec()
    if kind == "linear":
        return kind, {**resolve_traj_kwargs(task.task_name, {}), **spec}
    return kind, dict(spec)


def record_one(task, args: argparse.Namespace) -> dict[str, Any]:
    kind, traj_kw = _traj_kwargs(task)
    traj = build_trajectory(kind, speed=args.speed, **traj_kw)
    tw_args = bridge.load_task_args(
        args.robotwin_root, task_config=args.task_config, task_name=task.task_name
    )
    tw_args["_robotwin_root"] = str(args.robotwin_root)
    env = bridge.create_task_env(task.task_name, args.robotwin_root)
    instr = default_instruction(task.task_name)
    driver = RealtimeRoboTwinDriver(
        env,
        traj,
        task_name=task.task_name,
        release_radius=0.12,
        physics_per_tick=args.physics_per_tick,
        dynamic=build_dynamic_config("none"),
        rails_variant=task.variant,
        spawn_pushers=True,
        place_attr=task.place_attr,
    )
    wrap_env_with_driver(env, driver)
    frames: list[np.ndarray] = []
    pusher_meta: list[dict[str, Any]] = []
    try:
        bridge.setup_episode(env, tw_args, seed=args.seed, instruction=instr, ep_num=0)
        _ensure_demo_attrs(env, task)
        driver.on_episode_start()
        driver.begin_policy()
        pusher_meta = [
            {
                "target": str(c.target_name),
                "obj_radius_cm": round(float(c.obj_radius) * 100.0, 2),
                "standoff_cm": round(float(c.standoff) * 100.0, 2),
            }
            for c in (getattr(driver, "_pushers", []) or [])
        ]
        hold = _hold_qpos_action(env)
        for t in range(args.steps):
            env.take_action(hold, action_type="qpos")
            rgb = np.ascontiguousarray(env.get_obs()["observation"]["head_camera"]["rgb"])
            path = kind if kind != "curve" else f"curve:{task.curve_id}"
            frames.append(
                _annotate(
                    rgb,
                    [
                        f"Dynamic-RoboTwin  {task.id}",
                        _who_line(task),
                        f"{path}  {args.speed * 100:.1f} cm/tick  original meshes  keep language  {t + 1}/{args.steps}",
                    ],
                    scale=int(args.scale),
                )
            )
            if driver.escaped:
                break
    finally:
        bridge.close_task_env(env)

    out = Path(args.out_dir) / task.task_name / f"{task.id.replace('.', '_')}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out.as_posix(), frames, fps=args.fps, quality=8)
    return {
        "id": task.id,
        "path": str(out.resolve()),
        "frames": len(frames),
        "variant": task.variant,
        "trajectory": kind,
        "curve_id": task.curve_id,
        "escaped": bool(driver.escaped),
        "n_pushers": int(len(getattr(driver, "_pushers", []) or [])),
        "pick_rails": bool(driver.pick_rails_active),
        "place_rails": bool(driver.place_rails_active),
        "instruction": instr,
        "pushers": pusher_meta,
        "task_name": task.task_name,
        "pick_attr": task.pick_attr,
        "place_attr": task.place_attr,
    }


def _select_tasks(args: argparse.Namespace):
    if str(args.ids).strip():
        return [get_dynamic_task(x.strip()) for x in str(args.ids).split(",") if x.strip()]
    suite = str(args.suite).strip().lower()
    if suite in (
        "recon",
        "dynamic",
        "catalog",
        "seed",
        "wave2",
        "wave-2",
    ):
        return recon_tasks()
    if suite in ("recon-both", "wave2-both", "wave2_both"):
        return recon_both_tasks()
    if suite in ("recon-demo", "wave2-demo"):
        return recon_demo_tasks()
    raise ValueError(f"Unknown suite {args.suite!r}")


def _write_indexes(base: Path, rows: list[dict[str, Any]]) -> None:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(str(r["task_name"]), []).append(r)
    for name, group in by_task.items():
        folder = base / name
        folder.mkdir(parents=True, exist_ok=True)
        meta = folder / "recording_meta.json"
        merged: dict[str, dict[str, Any]] = {}
        if meta.is_file():
            try:
                old = json.loads(meta.read_text(encoding="utf-8"))
                for r in old.get("rows", []):
                    merged[str(r.get("id"))] = r
            except (OSError, json.JSONDecodeError):
                pass
        for r in group:
            merged[str(r.get("id"))] = r
        group = list(merged.values())
        meta.write_text(json.dumps({"protocol": "real_time_v2", "rows": group}, indent=2), encoding="utf-8")
        lines = [
            f"Dynamic-RoboTwin bumper-car reconstructions: {name}",
            f"Directory: {folder.resolve()}",
            "Original meshes. Car rides rails behind the moving actor.",
            "Instruction and check_success unchanged.",
            "",
            "file                                      who                     path",
            "--------------------------------------------------------------------------------",
        ]
        for r in group:
            who = f"{r.get('pick_attr')}/{r.get('place_attr')} [{r['variant']}]"
            path = r["trajectory"] if not r.get("curve_id") else r["curve_id"]
            lines.append(f"{Path(r['path']).name:42s}  {who:22s}  {path}")
        (folder / "INDEX.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robotwin-root", default=os.environ.get("ROBOTWIN_ROOT", "/DATA/YuanZhen/RoboTwin"))
    p.add_argument("--task-config", default="demo_clean")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=0.003)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--physics-per-tick", type=int, default=12)
    p.add_argument("--scale", type=int, default=4, help="Upscale head-camera RGB before encoding.")
    p.add_argument(
        "--out-dir",
        default="",
        help=f"Base directory (default: { _DEFAULT_BASE }; videos go in <out-dir>/<task_name>/).",
    )
    p.add_argument("--ids", default="", help="Comma-separated catalog ids (overrides --suite).")
    p.add_argument(
        "--suite",
        default="recon-demo",
        help="recon | recon-both | recon-demo (seed/wave2 aliases; ignored if --ids is set).",
    )
    args = p.parse_args()
    os.environ.pop("VK_ICD_FILENAMES", None)
    os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else ROOT / _DEFAULT_BASE
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    args.out_dir = str(out_dir)
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    tasks = _select_tasks(args)
    rows = []
    for task in tasks:
        print(f"=== {task.id} ===", flush=True)
        row = record_one(task, args)
        print(json.dumps(row, indent=2), flush=True)
        rows.append(row)
    _write_indexes(out_dir, rows)
    top = out_dir / "recording_meta.json"
    top.write_text(json.dumps({"protocol": "real_time_v2", "suite": args.suite, "rows": rows}, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} clips under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
