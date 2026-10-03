#!/usr/bin/env python3
"""Record short object-motion demos (no policy / GPU required).

Highlights dynamic target trajectories under Real-Time protocol rails.
Robot holds still; the pick target moves (ghost pin, conveyor, or a toy car
that rides the rails and pushes the original mesh).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import imageio
import numpy as np

for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

ROOT = setup_sys_path()

from benchmarks.dynamic_libero.trajectories import build_trajectory
from benchmarks.dynamic_libero.viz.annotate import annotate, render_agent


def _save_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path.as_posix(), frames, fps=fps)


def _annotate_frame(
    rgb: np.ndarray,
    *,
    enabled: bool,
    benchmark: str,
    task_label: str,
    trajectory: str,
    speed: float,
    drift_cm: float,
    step: int,
    total: int,
    extra: str = "",
) -> np.ndarray:
    """Optionally overlay captions. Default path returns the raw frame unchanged."""
    if not enabled:
        return rgb
    lines = [
        f"Dynamic-{benchmark} · object motion demo",
        f"task: {task_label}  traj={trajectory}  speed={speed*100:.1f} cm/tick",
        f"drift {drift_cm:.1f} cm  frame {step+1}/{total}  (robot hold, target rails)",
    ]
    if extra:
        lines.append(extra)
    return annotate(rgb, lines)


def record_libero(args: argparse.Namespace) -> dict[str, Any]:
    from dataclasses import replace

    from benchmarks.dynamic_libero.env import libero_bridge as bridge
    from benchmarks.dynamic_libero.env.conveyor import ConveyorBelt, layout_conveyor_for_driver
    from benchmarks.dynamic_libero.env.driver import RealtimeDriver, wrap_env_with_driver
    from benchmarks.dynamic_libero.env.dynamic_tasks import (
        default_place_target,
        get_dynamic_task,
        parse_rails_variant,
    )
    from benchmarks.dynamic_libero.env.motives import keeps_original_mesh, resolve_motive
    from benchmarks.dynamic_libero.env.pusher import PusherCar
    from benchmarks.dynamic_libero.env.movers import (
        get_mover_spec,
        mover_assets,
        parse_mover_language,
        parse_movers_flag,
        rewrite_language,
    )
    from benchmarks.dynamic_libero.eval.cli_common import resolve_trajectory_from_args

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    dyn = None
    dyn_id = getattr(args, "dynamic_task", None)
    if dyn_id:
        dyn = get_dynamic_task(dyn_id)
        args.task_suite = dyn.suite
        args.task_id = dyn.task_id
    variant = parse_rails_variant(
        dyn.variant if dyn is not None else getattr(args, "rails_variant", "auto"),
        default="pick",
    )
    place_name = None if dyn is None else dyn.place_target
    if place_name is None:
        place_name = default_place_target(args.task_suite, int(args.task_id))

    if dyn is not None and getattr(dyn, "motive", None):
        motive = resolve_motive(args.task_suite, dyn.motive)
    else:
        motive = resolve_motive(
            args.task_suite, getattr(args, "motive", None) or getattr(args, "motion_motive", None)
        )
    raw_variant = str(getattr(args, "rails_variant", "auto") or "auto").strip().lower()
    if motive == "carrier" and dyn is None and raw_variant in ("", "auto", "default"):
        variant = "both"
    # Motives own default appearance; explicit --movers still wins when set via CLI.
    movers_flag = getattr(args, "movers", None)
    if movers_flag is None or str(movers_flag).strip() == "":
        movers_on = motive == "toycar"
    else:
        movers_on = parse_movers_flag(movers_flag)
    if getattr(args, "demo_toycar", None) is False:
        movers_on = False
    if motive in ("conveyor", "pusher") or keeps_original_mesh(motive):
        if motive != "toycar":
            movers_on = False  # original pick mesh
    lang_mode = parse_mover_language(getattr(args, "mover_language", "rewrite"))
    if motive in ("conveyor", "pusher", "carrier", "ghost"):
        lang_mode = "keep"

    ctx = bridge.load_env_context(
        task_suite_name=args.task_suite,
        task_id=args.task_id,
        seed=args.seed,
    )
    task = ctx.task_suite.get_task(args.task_id)

    # Ensure optional attrs exist when invoked with a minimal Namespace.
    for k, v in (
        ("traj_complexity", "none"),
        ("n_waypoints", 5),
        ("extent", 0.12),
        ("amplitude", 0.03),
        ("wavelength", 0.20),
        ("radius", 0.04),
        ("omega", 0.03),
    ):
        if not hasattr(args, k):
            setattr(args, k, v)
    kind, traj_kw = resolve_trajectory_from_args(args)
    if dyn is not None:
        kind, spec_kw = dyn.traj_spec()
        if spec_kw or kind != "linear":
            traj_kw = spec_kw
            args.trajectory = kind
    traj = build_trajectory(kind, speed=args.speed, **traj_kw)

    rails_spec = get_mover_spec(args.task_suite, args.task_id)
    if motive == "conveyor":
        rails_spec = replace(rails_spec, motion="conveyor", height_offset=0.0, noun=rails_spec.category.replace("_", " "))

    belt: ConveyorBelt | None = None
    pusher: PusherCar | None = None
    place_pusher: PusherCar | None = None
    with mover_assets(args.task_suite, args.task_id, enabled=movers_on) as swapped:
        env, desc = bridge.get_libero_env(task, args.render_res, ctx.cfg.get("seed"))
        if motive == "conveyor":
            belt = ConveyorBelt()
            belt.install(env)
            mover_spec = rails_spec
        elif motive in ("pusher", "carrier"):
            from benchmarks.dynamic_libero.env.pusher import install_cars, make_pusher_scene

            pusher, place_pusher = make_pusher_scene(motive, variant)
            cars = [c for c in (pusher, place_pusher) if c is not None]
            if cars:
                install_cars(env, cars)
            mover_spec = rails_spec
        else:
            mover_spec = swapped if swapped is not None else rails_spec

    if movers_on and lang_mode == "rewrite" and mover_spec is not None and motive == "toycar":
        desc = rewrite_language(desc, mover_spec)

    driver = RealtimeDriver(
        env,
        traj,
        release_radius=float(args.release_radius),
        mover=mover_spec if (movers_on or motive in ("conveyor", "pusher", "carrier")) else rails_spec,
        pusher=pusher,
        place_pusher=place_pusher,
        rails_variant=variant,
        place_target_name=place_name,
        clear_path=bool(dyn is None or getattr(dyn, "clear_path", True)),
    )
    wrap_env_with_driver(env, driver)

    num_wait = int(ctx.cfg.EVALUATION.get("num_steps_wait", 30))
    total = num_wait + args.steps
    frames: list[np.ndarray] = []
    try:
        env.reset()
        obs = env.set_init_state(
            list(ctx.task_suite.get_task_init_states(args.task_id))[args.init_idx]
        )
        if belt is not None:
            # Stationary belt + seat object on deck (do not follow the object).
            layout_conveyor_for_driver(
                belt,
                driver,
                traj,
                horizon_steps=float(args.steps),
            )
        if driver.target is None:
            driver.on_episode_start()
        for t in range(total):
            if t == num_wait:
                driver.begin_policy()
            action = bridge.dummy_action()
            obs, _, _, _ = env.step(action)
            if belt is not None and driver.motion_enabled:
                # Only scroll surface stripes; belt body stays planted.
                belt.sync_scroll(driver.drift_m)
            if t < num_wait:
                continue
            rgb = render_agent(env)
            frames.append(
                _annotate_frame(
                    rgb,
                    enabled=bool(args.caption),
                    benchmark="LIBERO",
                    task_label=f"{args.task_suite}/t{args.task_id}: {desc}",
                    trajectory=args.trajectory,
                    speed=args.speed,
                    drift_cm=driver.drift_m * 100.0,
                    step=t - num_wait,
                    total=args.steps,
                    extra=f"motive={motive}",
                )
            )
            if driver.escaped:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    out = Path(args.out_dir) / "dynamic_libero" / args.out_name
    _save_mp4(frames, out, args.fps)
    return {
        "benchmark": "Dynamic-LIBERO",
        "path": str(out.resolve()),
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "init_idx": args.init_idx,
        "description": desc,
        "trajectory": kind,
        "traj_complexity": getattr(args, "traj_complexity", "none"),
        "speed": args.speed,
        "frames": len(frames),
        "fps": args.fps,
        "duration_s": round(len(frames) / args.fps, 2),
        "escaped": bool(driver.escaped),
        "motive": motive,
        "movers": movers_on,
        "mover_language": lang_mode,
        "demo_toycar": movers_on,
        "conveyor": belt is not None,
        "pusher": pusher is not None,
        "place_pusher": place_pusher is not None,
        **driver.episode_stats(),
    }


def record_robotwin(args: argparse.Namespace) -> dict[str, Any]:
    from contextlib import nullcontext

    from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge
    from benchmarks.dynamic_robotwin.env.demo_assets import temporary_toycar_moving_box
    from benchmarks.dynamic_robotwin.env.driver import RealtimeRoboTwinDriver, wrap_env_with_driver
    from benchmarks.dynamic_robotwin.env.dynamics import build_dynamic_config, resolve_secondary_rails
    from benchmarks.dynamic_robotwin.env.physics import _hold_qpos_action
    from benchmarks.dynamic_robotwin.env.task_registry import (
        default_instruction,
        get_task_meta,
        resolve_traj_kwargs,
    )

    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    # Prefer task defaults for axis unless user overrode via CLI defaults carefully:
    # when axis is the argparse default "x", still let task defaults win for demos.
    use_toycar = getattr(args, "demo_toycar", None)
    if use_toycar is None:
        use_toycar = True
    use_toycar = bool(use_toycar) and str(args.task_name) == "handover_block"
    traj_overrides: dict[str, Any] = {}
    if getattr(args, "axis_set", False):
        traj_overrides["axis"] = args.axis
        traj_overrides["direction"] = args.direction
        traj_overrides["toward_center"] = False
    traj_kw = resolve_traj_kwargs(args.task_name, traj_overrides)
    traj_kind = str(args.trajectory)
    if traj_kind == "sine":
        traj_kw.setdefault("amplitude", float(getattr(args, "amplitude", 0.04)))
        traj_kw.setdefault("wavelength", float(getattr(args, "wavelength", 0.20)))
        axis = str(traj_kw.get("axis", "y"))
        traj_kw.setdefault("lateral_axis", "x" if axis == "y" else "y")
    traj = build_trajectory(traj_kind, speed=args.speed, **traj_kw)
    args.trajectory = traj_kind
    tw_args = bridge.load_task_args(args.robotwin_root, task_config=args.task_config, task_name=args.task_name)
    tw_args["_robotwin_root"] = str(args.robotwin_root)
    env = bridge.create_task_env(args.task_name, args.robotwin_root)
    instr = default_instruction(args.task_name)
    meta = get_task_meta(args.task_name)
    modes = list(meta.get("dynamic_modes") or ["contact_trigger", "occlusion", "contact_chain"])
    if meta.get("stress") == "handoff" and "handoff" not in modes:
        modes.append("handoff")
    # Demo default: no transient occluder (avoids a flashing grey box in videos).
    # Opt in with --occlusion for paper-style occlusion demos.
    want_occlusion = bool(getattr(args, "occlusion", False))
    if not want_occlusion:
        modes = [m for m in modes if m != "occlusion"]
    # Demo: keep blue placement pad still (secondary_rails off). Eval still uses auto.
    sec_spec = "off" if use_toycar else "auto"
    if not args.no_dynamics:
        dyn = build_dynamic_config(
            modes,
            contact_switch="medium",
            occlusion=want_occlusion,
            occlusion_duty=0.55 if want_occlusion else 0.0,
            occlusion_period=24.0,
            contact_chain=True,
            secondary_rails=resolve_secondary_rails(args.task_name, sec_spec, modes=modes),
        )
    else:
        dyn = build_dynamic_config("none")
    release = float(args.release_radius) if args.release_radius != 0.06 else 0.12
    driver = RealtimeRoboTwinDriver(
        env,
        traj,
        task_name=args.task_name,
        release_radius=release,
        physics_per_tick=args.physics_per_tick,
        dynamic=dyn,
        align_heading=use_toycar,
        # 057_toycar GLB: length along +Y, Z-up.
        heading_mesh_forward="y" if use_toycar else "x",
        # Demo-only: let the toy car physically push scene props instead of
        # ghost-sliding through them (PROTOCOL.md §8.1 default stays off for
        # real benchmark runs; this flag only ever fires for --demo-toycar).
        rails_collide=use_toycar,
    )
    wrap_env_with_driver(env, driver)

    # Demo: swap moving red box → 057_toycar.
    asset_cm = temporary_toycar_moving_box() if use_toycar else nullcontext()

    frames: list[np.ndarray] = []
    try:
        with asset_cm:
            bridge.setup_episode(env, tw_args, seed=args.seed, instruction=instr, ep_num=0)
            driver.on_episode_start()
            driver.begin_policy()
            hold = _hold_qpos_action(env)
            for t in range(args.steps):
                env.take_action(hold, action_type="qpos")
                rgb = np.ascontiguousarray(env.get_obs()["observation"]["head_camera"]["rgb"])
                stats = driver.episode_dynamics_stats()
                extra = (
                    f"modes={','.join(dyn.modes)}  "
                    f"occ_ticks={stats.get('occlusion_active_ticks', 0):.0f}  "
                    f"sec_rails={stats.get('secondary_rails', False)}"
                )
                frames.append(
                    _annotate_frame(
                        rgb,
                        enabled=bool(args.caption),
                        benchmark="RoboTwin",
                        task_label=f"{args.task_name}: {instr}",
                        trajectory=args.trajectory,
                        speed=args.speed,
                        drift_cm=driver.drift_m * 100.0,
                        step=t,
                        total=args.steps,
                        extra=extra,
                    )
                )
                if driver.escaped:
                    break
    finally:
        bridge.close_task_env(env)

    out = Path(args.out_dir) / "dynamic_robotwin" / args.out_name
    _save_mp4(frames, out, args.fps)
    stats = driver.episode_dynamics_stats()
    return {
        "benchmark": "Dynamic-RoboTwin",
        "path": str(out.resolve()),
        "task_name": args.task_name,
        "instruction": instr,
        "trajectory": args.trajectory,
        "speed": args.speed,
        "demo_toycar": use_toycar,
        "align_heading": use_toycar,
        "frames": len(frames),
        "fps": args.fps,
        "duration_s": round(len(frames) / args.fps, 2),
        "escaped": bool(driver.escaped),
        "dynamic_modes": list(dyn.modes),
        "occlusion_active_ticks": stats.get("occlusion_active_ticks"),
        "secondary_rails": stats.get("secondary_rails"),
        "contact_chain_hops": stats.get("contact_chain_hops"),
        "rails_collide": stats.get("rails_collide"),
    }


# Keep demo escape bounds aligned with eval defaults (kitchen world frame).
from benchmarks.common.workspace import ROBOCASA_DEFAULT_AABB as _ROBOCASA_AABB


def _robocasa_rgb(obs: dict[str, Any], cam_key: str) -> np.ndarray:
    rgb = np.ascontiguousarray(obs[cam_key])
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    return rgb


def record_robocasa(args: argparse.Namespace) -> dict[str, Any]:
    from contextlib import nullcontext

    from benchmarks.realtime_robocasa.env import robocasa_bridge as bridge
    from benchmarks.realtime_robocasa.env.demo_assets import (
        seat_produce_beside_tray,
        temporary_arrange_tea_produce,
    )
    from benchmarks.realtime_robocasa.env.driver import RealtimeRoboCasaDriver, wrap_gym_env_with_driver
    from benchmarks.realtime_robocasa.env.dynamics import build_dynamic_config
    from benchmarks.realtime_robocasa.env.task_registry import get_task_meta
    from benchmarks.realtime_robocasa.viz.demo_narrative import (
        arrange_tea_demo_config,
        fire_visible_disturbance,
        three_shot_plan,
    )

    os.environ["ROBOCASA_ROOT"] = str(args.robocasa_root)
    bridge.ensure_runtime_env()
    narrative_early = str(getattr(args, "demo_narrative", "three_shot"))
    use_produce = (
        narrative_early == "three_shot"
        and str(args.task_name) == "ArrangeTea"
        and not bool(args.no_dynamics)
    )
    # Produce-roll demos freeze tray/teapot/mug rails; only fruit/veg rolls.
    demo_speed = 0.0 if use_produce else float(args.speed)
    if (
        not use_produce
        and narrative_early == "three_shot"
        and demo_speed > 0.004
    ):
        demo_speed = 0.003
    traj = build_trajectory(
        args.trajectory,
        speed=demo_speed,
        axis=args.axis,
        toward_center=True,
    )
    produce_cm = temporary_arrange_tea_produce() if use_produce else nullcontext([])
    with produce_cm as produce_names:
        env = bridge.create_gym_env(
            args.task_name,
            split=args.split,
            seed=args.seed,
            camera_height=args.render_res,
            camera_width=args.render_res,
            # Produce meshes live under objaverse; force that registry for demos.
            obj_registries=("objaverse", "lightwheel") if use_produce else None,
        )
        aabb = _ROBOCASA_AABB
        meta = get_task_meta(args.task_name)
        narrative = str(getattr(args, "demo_narrative", "three_shot"))
        if args.no_dynamics:
            dyn = build_dynamic_config("none")
            narrative = "flat"
        elif narrative == "three_shot" and str(args.task_name) == "ArrangeTea":
            dyn = arrange_tea_demo_config()
        else:
            modes = list(
                meta.get("dynamic_modes")
                or ["multi_object", "articulated", "container", "stage_conditioned"]
            )
            dyn = build_dynamic_config(modes, level="medium")
        release = float(args.release_radius) if args.release_radius != 0.06 else 0.10
        driver = RealtimeRoboCasaDriver(
            env,
            traj,
            task_name=args.task_name,
            release_radius=release,
            workspace_aabb=aabb,
            control_hz=bridge.CONTROL_FREQ,
            dynamic=dyn,
        )
        wrap_gym_env_with_driver(env, driver)
        hold = bridge.hold_action_dict()
        cam_key = "video.robot0_agentview_left"

        frames: list[np.ndarray] = []
        shot_meta: list[dict[str, Any]] = []
        visible_pulses: list[dict[str, Any]] = []
        try:
            obs, _ = env.reset(seed=args.seed)
            if use_produce:
                from benchmarks.realtime_robocasa.env.driver import _unwrap_robocasa_kitchen

                _, kitchen = _unwrap_robocasa_kitchen(env)
                seat_produce_beside_tray(kitchen)
            driver.begin_policy()
            instr = bridge.get_instruction(obs, fallback=args.task_name)

            def _grab_frame(step_i: int, total_i: int, shot: str) -> np.ndarray:
                stats = driver.episode_dynamics_stats() if hasattr(driver, "episode_dynamics_stats") else {}
                extra = (
                    f"shot={shot}  modes={','.join(dyn.modes)}  "
                    f"stage={stats.get('stage', '')}  "
                    f"events={len(stats.get('disturbance_events') or [])}"
                )
                return _annotate_frame(
                    _robocasa_rgb(obs, cam_key),
                    enabled=bool(args.caption),
                    benchmark="RoboCasa",
                    task_label=f"{args.task_name}: {instr}",
                    trajectory=args.trajectory,
                    speed=demo_speed,
                    drift_cm=driver.drift_m * 100.0,
                    step=step_i,
                    total=total_i,
                    extra=extra,
                )

            if narrative == "three_shot":
                plan = three_shot_plan(fps=int(args.fps))
                total = sum(n for _, n, _ in plan)
                t_global = 0
                for shot_i, (shot_id, n_frames, stage_hint) in enumerate(plan):
                    driver.schedule.update_stage(stage_hint)
                    if shot_id == "disturb":
                        visible_pulses.extend(fire_visible_disturbance(driver))
                    shot_frames: list[np.ndarray] = []
                    for _ in range(n_frames):
                        obs, _, terminated, truncated, info = env.step(hold)
                        shot_frames.append(_grab_frame(t_global, total, shot_id))
                        t_global += 1
                        if driver.escaped or terminated or truncated:
                            break
                    shot_meta.append(
                        {
                            "shot": shot_id,
                            "frames": len(shot_frames),
                            "stage_hint": stage_hint,
                            "visible_pulse": shot_id == "disturb",
                        }
                    )
                    frames.extend(shot_frames)
                    if driver.escaped or terminated or truncated:
                        break
            else:
                for t in range(args.steps):
                    obs, _, terminated, truncated, info = env.step(hold)
                    frames.append(_grab_frame(t, args.steps, "flat"))
                    if driver.escaped or terminated or truncated:
                        break
        finally:
            bridge.close_env(env)

        out = Path(args.out_dir) / "dynamic_robocasa" / args.out_name
        _save_mp4(frames, out, args.fps)
        stats = driver.episode_dynamics_stats() if hasattr(driver, "episode_dynamics_stats") else {}
        return {
            "benchmark": "Dynamic-RoboCasa",
            "path": str(out.resolve()),
            "task_name": args.task_name,
            "split": args.split,
            "trajectory": args.trajectory,
            "speed": demo_speed,
            "frames": len(frames),
            "fps": args.fps,
            "duration_s": round(len(frames) / args.fps, 2),
            "escaped": bool(driver.escaped),
            "dynamic_modes": list(dyn.modes),
            "stage": stats.get("stage"),
            "n_disturbance_events": len(stats.get("disturbance_events") or []),
            "demo_narrative": narrative,
            "shots": shot_meta,
            "visible_pulses": visible_pulses,
            "aux_objects": stats.get("aux_objects"),
            "container_objects": stats.get("container_objects"),
            "demo_produce": list(produce_names),
        }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benchmark",
        choices=["libero", "robotwin", "robocasa", "all"],
        required=True,
    )
    p.add_argument("--trajectory", default="linear")
    p.add_argument(
        "--traj-complexity",
        default="none",
        choices=["none", "easy", "medium", "hard", "chaotic"],
        help="LIBERO demo path difficulty (hard/chaotic = seeded random polyline)",
    )
    p.add_argument("--n-waypoints", type=int, default=5)
    p.add_argument("--extent", type=float, default=0.12)
    p.add_argument("--speed", type=float, default=0.006)
    p.add_argument("--axis", default="x")
    p.add_argument("--direction", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=200, help="Motion frames after reset/wait")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--render-res", type=int, default=512)
    p.add_argument("--release-radius", type=float, default=0.06)
    p.add_argument("--out-dir", default="evaluate_results/demo_videos")
    p.add_argument("--out-name", default="object_motion_demo.mp4")
    p.add_argument(
        "--no-dynamics",
        action="store_true",
        help="Disable suite-specific dynamics (rails-only baseline demo)",
    )
    p.add_argument(
        "--occlusion",
        action="store_true",
        help="RoboTwin: enable transient grey occluder in demos (default: off)",
    )
    p.add_argument(
        "--caption",
        action="store_true",
        help="Overlay text captions on frames (default: off, raw recording only)",
    )
    p.add_argument(
        "--no-caption",
        action="store_true",
        help="Explicitly disable captions (default behavior; kept for CLI clarity)",
    )
    # LIBERO
    p.add_argument("--task-suite", default="libero_object")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--movers",
        type=str,
        default="",
        help="LIBERO: on/off toy-car swap. Empty = follow --motive (pusher/conveyor⇒off, toycar⇒on).",
    )
    p.add_argument(
        "--mover-language",
        type=str,
        default="rewrite",
        choices=["rewrite", "keep"],
        help="LIBERO: rewrite task language to toy-car nouns (ignored for pusher/conveyor).",
    )
    p.add_argument(
        "--motive",
        type=str,
        default="auto",
        help="LIBERO motion motive: auto|pusher|carrier|conveyor|toycar|ghost (auto uses suite defaults).",
    )
    p.add_argument(
        "--rails-variant",
        type=str,
        default="auto",
        help="Who moves: auto|pick|place|both.",
    )
    p.add_argument(
        "--dynamic-task",
        type=str,
        default=None,
        help="Catalog id, e.g. libero_object.t00.place.",
    )
    p.add_argument(
        "--demo-toycar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Alias forcing toy-car movers on/off when set.",
    )
    # RoboTwin
    p.add_argument("--task-name", default=None)
    p.add_argument("--robotwin-root", default="/DATA/YuanZhen/FastWAM/third_party/RoboTwin")
    p.add_argument("--task-config", default="demo_clean")
    p.add_argument("--physics-per-tick", type=int, default=12)
    # RoboCasa
    p.add_argument("--robocasa-root", default="/DATA/YuanZhen/RoboCasa/robocasa")
    p.add_argument("--split", default="target")
    p.add_argument(
        "--demo-narrative",
        default="three_shot",
        choices=["three_shot", "flat"],
        help="RoboCasa: three_shot=setup/produce-roll/aftermath (default); flat=single take",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    # Track whether --axis was explicitly intended; demos prefer task defaults.
    args.axis_set = "--axis" in sys.argv
    # Captions are off by default; --no-caption wins over --caption if both set.
    if args.no_caption:
        args.caption = False
    if args.task_name is None:
        args.task_name = {
            "libero": None,
            # Handoff task showcases secondary rails + paper stresses.
            "robotwin": "handover_block",
            # Composite long-horizon kitchen task (needs objaverse assets).
            "robocasa": "ArrangeTea",
            "all": None,
        }[args.benchmark if args.benchmark != "all" else "robotwin"]

    bench_map = {
        "libero": record_libero,
        "robotwin": record_robotwin,
        "robocasa": record_robocasa,
    }
    targets = list(bench_map) if args.benchmark == "all" else [args.benchmark]
    results: list[dict[str, Any]] = []
    t0 = time.time()
    for name in targets:
        print(f"\n=== recording {name} ===", flush=True)
        meta = bench_map[name](args)
        meta["no_caption"] = not bool(args.caption)
        results.append(meta)
        print(json.dumps(meta, indent=2), flush=True)

    meta_path = Path(args.out_dir) / "recording_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with prior suite entries when re-recording a subset.
    merged = list(results)
    if meta_path.is_file():
        try:
            prev = json.loads(meta_path.read_text())
            prev_results = prev.get("results") or []
            updated = {r.get("benchmark") for r in results}
            for old in prev_results:
                if old.get("benchmark") not in updated:
                    merged.append(old)
        except Exception:
            pass
    meta_path.write_text(
        json.dumps(
            {
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "no_caption": not bool(args.caption),
                "results": merged,
            },
            indent=2,
        )
    )
    print(f"\nDone in {time.time()-t0:.1f}s; meta -> {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
