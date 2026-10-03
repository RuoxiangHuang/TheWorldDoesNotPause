"""Real RoboTwin + SAPIEN smoke: verify dynamics fields are non-empty.

No policy weights required — holds qpos and forces a contact by spoofing TCP.
Writes JSON under evaluate_results/dynamic_robotwin/smoke_dynamics/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

ROOT = setup_sys_path()

from benchmarks.dynamic_libero.trajectories import build_trajectory
from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge
from benchmarks.dynamic_robotwin.env.driver import RealtimeRoboTwinDriver, wrap_env_with_driver
from benchmarks.dynamic_robotwin.env.dynamics import build_dynamic_config, resolve_secondary_rails
from benchmarks.dynamic_robotwin.env.physics import _hold_qpos_action
from benchmarks.dynamic_robotwin.env.task_registry import (
    default_instruction,
    get_task_meta,
    resolve_traj_kwargs,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic-RoboTwin SAPIEN dynamics smoke")
    p.add_argument("--task-name", type=str, default="place_empty_cup")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--speed", type=float, default=0.003)
    p.add_argument("--trajectory", type=str, default="linear")
    p.add_argument("--robotwin-root", type=str, default=str(bridge.DEFAULT_ROBOTWIN_ROOT))
    p.add_argument("--task-config", type=str, default="demo_clean")
    p.add_argument("--physics-per-tick", type=int, default=12)
    p.add_argument(
        "--out-dir",
        type=str,
        default="evaluate_results/dynamic_robotwin/smoke_dynamics",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    os.environ.setdefault("MUJOCO_GL", "egl")

    meta = get_task_meta(args.task_name)
    modes = list(meta.get("dynamic_modes") or ["contact_trigger", "occlusion", "contact_chain"])
    if meta.get("stress") == "handoff" and "handoff" not in modes:
        modes.append("handoff")
    sec = resolve_secondary_rails(args.task_name, "auto", modes=modes)
    dyn = build_dynamic_config(
        modes,
        contact_switch="medium",
        occlusion=True,
        occlusion_duty=0.5,
        occlusion_period=20.0,
        contact_chain=True,
        contact_chain_n_aux=3,
        secondary_rails=sec,
    )

    traj_kw = resolve_traj_kwargs(args.task_name, {})
    traj = build_trajectory(args.trajectory, speed=args.speed, **traj_kw)
    tw_args = bridge.load_task_args(
        args.robotwin_root, task_config=args.task_config, task_name=args.task_name
    )
    tw_args["_robotwin_root"] = str(args.robotwin_root)
    env = bridge.create_task_env(args.task_name, args.robotwin_root)
    instr = default_instruction(args.task_name)
    driver = RealtimeRoboTwinDriver(
        env,
        traj,
        task_name=args.task_name,
        physics_per_tick=args.physics_per_tick,
        dynamic=dyn,
    )
    wrap_env_with_driver(env, driver)

    report: dict[str, Any] = {
        "task_name": args.task_name,
        "modes": list(dyn.modes),
        "secondary_rails_resolved": sec,
        "ok": False,
        "checks": {},
    }
    try:
        bridge.setup_episode(env, tw_args, seed=args.seed, instruction=instr, ep_num=0)
        driver.on_episode_start()
        driver.begin_policy()
        hold = _hold_qpos_action(env)

        # Advance long enough for occlusion duty cycle to register active ticks.
        for _ in range(max(8, args.steps // 2)):
            env.take_action(hold, action_type="qpos")
            if driver.escaped:
                break

        # Force a contact event: spoof left TCP onto the moving target.
        if driver.entity is not None:
            tgt = np.asarray(driver.entity.get_pose().p, dtype=np.float64)

            def _near_tcp():
                return np.array(
                    [tgt[0], tgt[1], tgt[2], 0.0, 0.0, 0.0, 1.0], dtype=np.float64
                )

            env.robot.get_left_tcp_pose = _near_tcp  # type: ignore[method-assign]
            driver._maybe_contact_trigger()

        # Propagate contact-chain hops via hop_delay.
        for _ in range(max(6, args.steps // 2)):
            env.take_action(hold, action_type="qpos")
            if driver.escaped:
                break

        stats = driver.episode_dynamics_stats()
        report["stats"] = {
            k: stats[k]
            for k in (
                "contact_events",
                "trajectory_switches",
                "occlusion_active_ticks",
                "contact_chain_impulses",
                "contact_chain_hops",
                "contact_chain_events",
                "secondary_rails",
                "dynamic_modes",
                "rails_drive",
                "ghost_sliding",
                "ghost_collision_filter",
            )
            if k in stats
        }
        checks = {
            "contact_events_nonempty": len(stats.get("contact_events") or []) > 0,
            "trajectory_switches_nonempty": len(stats.get("trajectory_switches") or []) > 0,
            "occlusion_active_ticks_positive": float(stats.get("occlusion_active_ticks") or 0) > 0,
            "contact_chain_multi_hop": int(stats.get("contact_chain_hops") or 0) >= 2,
        }
        if sec:
            checks["secondary_rails_active"] = bool(stats.get("secondary_rails"))
        report["checks"] = checks
        report["ok"] = all(checks.values())
    finally:
        bridge.close_task_env(env)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task_name}_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}", flush=True)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
