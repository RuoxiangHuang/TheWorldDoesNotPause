"""Single-episode Real-Time RoboTwin rollout."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from benchmarks.common.executor import RealtimeExecutor
from benchmarks.common.latency import (
    LatencyCoupler,
    LatencyMode,
    LatencyTrace,
    episode_latency_fields,
    parse_latency_mode,
    time_policy_predict,
)
from benchmarks.common.policy.base import PolicyClient
from benchmarks.common.policy.factory import make_inprocess_policy
from benchmarks.common.protocol import (
    DEFAULT_BACKEND,
    DEFAULT_DEADLINE_MS,
    RealtimeBackend,
    blind_window_ratio,
    effective_replan_steps,
)
from benchmarks.common.world_wire import policy_predict_kwargs
from benchmarks.common.workspace import aabb_to_dict, resolve_workspace_aabb
from benchmarks.dynamic_libero.trajectories import build_trajectory

from ..env.driver import RealtimeRoboTwinDriver, wrap_env_with_driver
from ..env.dynamics import DynamicConfig, build_dynamic_config, resolve_secondary_rails
from ..env import robotwin_bridge as bridge
from ..env.physics import resolve_physics_per_tick
from ..env.task_registry import (
    DEFAULT_DYNAMIC_MODES,
    default_instruction,
    default_release_radius,
    get_task_meta,
    resolve_traj_kwargs,
)


def _robotwin_head_rgb(env) -> np.ndarray:
    obs = env.get_obs()
    rgb = obs["observation"]["head_camera"]["rgb"]
    return np.ascontiguousarray(rgb)


def _attach_robotwin_frame_sink(env, driver, frame_sink) -> None:
    """Record one head-camera frame after each thinking tick and action step."""
    orig_tick = driver.tick

    def tick(dt: float = 1.0, _orig=orig_tick):
        _orig(dt)
        frame_sink(_robotwin_head_rgb(env))

    driver.tick = tick  # type: ignore[method-assign]
    orig_take = env.take_action

    def take_action(action, action_type: str = "qpos", _orig=orig_take):
        out = _orig(action, action_type=action_type)
        frame_sink(_robotwin_head_rgb(env))
        return out

    env.take_action = take_action


def run_episode(
    task_name: str,
    seed: int,
    ctx: bridge.EpisodeContext,
    *,
    trajectory_kind: str = "linear",
    speed: float = 0.003,
    trajectory_kwargs: Optional[dict[str, Any]] = None,
    n_freeze: float = 0.0,
    release_radius: Optional[float] = None,
    physics_per_tick: Optional[int] = None,
    auto_calibrate_physics: bool = True,
    policy_hz: Optional[float] = None,
    robotwin_root=None,
    task_config: str = "demo_clean",
    instruction: Optional[str] = None,
    backend: RealtimeBackend | str = DEFAULT_BACKEND,
    policy: PolicyClient | None = None,
    replan_steps: int | None = None,
    allow_replan_override: bool = False,
    workspace_aabb: Optional[np.ndarray] = None,
    escape_check: bool = True,
    dynamic_modes: Optional[Sequence[str]] = None,
    contact_switch: str = "medium",
    occlusion: bool | None = None,
    occlusion_duty: float = 0.35,
    contact_chain: bool | None = None,
    secondary_rails: bool | str | None = None,
    dynamic: Optional[DynamicConfig] = None,
    language_mode: str = "keep",
    grasp_mode: str = "ghost",
    engage_alpha: float = 1.75,
    scene: Optional[Any] = None,
    latency_mode: str | LatencyMode = LatencyMode.MEASURED,
    replay_trace: LatencyTrace | None = None,
    deadline_ms: float = DEFAULT_DEADLINE_MS,
    delivery_delay_s: float = 0.0,
    max_steps: int | None = None,
    frame_sink: Optional[Any] = None,
    motion_scale: float | None = None,
) -> dict[str, Any]:
    from ..env.dynamic_tasks import resolve_eval_spec

    spec = resolve_eval_spec(task_name)
    env_task = spec.task_name
    if spec.trajectory_kind == "native":
        return _run_native_extra_episode(
            spec=spec,
            env_task=env_task,
            seed=seed,
            ctx=ctx,
            n_freeze=n_freeze,
            physics_per_tick=physics_per_tick,
            auto_calibrate_physics=auto_calibrate_physics,
            policy_hz=policy_hz,
            robotwin_root=robotwin_root,
            task_config=task_config,
            instruction=instruction,
            backend=backend,
            policy=policy,
            replan_steps=replan_steps,
            allow_replan_override=allow_replan_override,
            language_mode=language_mode,
            scene=scene,
            latency_mode=latency_mode,
            replay_trace=replay_trace,
            deadline_ms=deadline_ms,
            delivery_delay_s=delivery_delay_s,
            motion_scale=motion_scale,
            speed=speed,
        )
    if isinstance(backend, str):
        backend = RealtimeBackend(backend)
    policy = policy or make_inprocess_policy("robotwin", ctx)
    reset_fn = getattr(policy, "reset", None)
    if callable(reset_fn):
        reset_fn()
    rsteps = effective_replan_steps(
        "robotwin",
        replan_steps if replan_steps is not None else getattr(ctx, "replan_steps", 8),
        allow_override=allow_replan_override,
    )

    root = robotwin_root or bridge.DEFAULT_ROBOTWIN_ROOT
    traj_kw = resolve_traj_kwargs(env_task, dict(trajectory_kwargs or {}))
    if spec.trajectory_kind == "curve" and spec.curve_id:
        if trajectory_kind in ("linear", "", "default"):
            trajectory_kind = "curve"
        traj_kw.setdefault("curve_id", spec.curve_id)
    if spec.heading_deg is not None and trajectory_kind in ("linear", "", "default"):
        traj_kw["heading_deg"] = float(spec.heading_deg)
        traj_kw.setdefault("toward_center", True)
    release = (
        float(release_radius)
        if release_radius is not None
        else default_release_radius(env_task)
    )
    from benchmarks.common.language import apply_language_mode, parse_language_mode
    from benchmarks.common.outcomes import outcome_fields
    from benchmarks.common.scene import (
        SceneCondition,
        effective_language_mode,
        resolve_scene_condition,
        trajectory_speed_for_condition,
    )

    if scene is None:
        scene = resolve_scene_condition(
            speed=float(speed),
            catalog_job=bool(spec.spawn_pushers),
        )
    elif not isinstance(scene, SceneCondition):
        raise TypeError("scene must be a SceneCondition")
    n_freeze = scene.apply_n_freeze(float(n_freeze))
    traj_speed = trajectory_speed_for_condition(scene, float(speed))
    trajectory = build_trajectory(trajectory_kind, speed=traj_speed, **traj_kw)
    meta = get_task_meta(env_task)
    if dynamic is None:
        modes = list(dynamic_modes) if dynamic_modes is not None else list(DEFAULT_DYNAMIC_MODES)
        # Auto-enable handoff mode metadata for stress=handoff tasks.
        if meta.get("stress") == "handoff" and "handoff" not in modes:
            modes.append("handoff")
        sec = resolve_secondary_rails(env_task, secondary_rails, modes=modes)
        driving = scene.advances_rails()
        dynamic = build_dynamic_config(
            modes,
            contact_switch=contact_switch if driving else "off",
            occlusion=occlusion,
            occlusion_duty=occlusion_duty,
            contact_chain=contact_chain if driving else False,
            secondary_rails=sec if driving else False,
        )
    args = bridge.load_task_args(root, task_config=task_config, task_name=env_task)
    args["_robotwin_root"] = str(root)
    env = bridge.create_task_env(env_task, root)
    from ..env.task_registry import instruction_noun

    base_instr = instruction or spec.language or default_instruction(env_task)
    effective_lang = effective_language_mode(parse_language_mode(language_mode), scene)
    instr = apply_language_mode(
        base_instr,
        effective_lang,
        noun=instruction_noun(env_task),
    )
    hz = float(policy_hz if policy_hz is not None else getattr(ctx, "policy_hz", 20.0))
    aabb = resolve_workspace_aabb(
        "robotwin",
        task_name=env_task,
        override=workspace_aabb,
        escape_check=escape_check,
    )
    driver = RealtimeRoboTwinDriver(
        env,
        trajectory,
        task_name=env_task,
        release_radius=release,
        physics_per_tick=12,
        policy_hz=hz,
        workspace_aabb=aabb,
        dynamic=dynamic,
        grasp_mode=grasp_mode,
        engage_alpha=engage_alpha,
        rails_variant=spec.rails_variant,
        spawn_pushers=bool(spec.spawn_pushers) and scene.modifies_layout(),
        place_attr=spec.place_attr,
        scene=scene,
    )
    wrap_env_with_driver(env, driver)

    freeze_ticks = float(n_freeze)
    mode = parse_latency_mode(latency_mode)
    coupler = LatencyCoupler(
        mode=mode,
        control_hz=hz,
        deadline_ms=float(deadline_ms),
        constant_n_freeze=float(freeze_ticks),
        replay_trace=replay_trace,
    )
    coupler.trace.meta.update(
        {
            "trajectory": trajectory_kind,
            "speed": float(speed),
            "backend": backend.value,
            "task_name": spec.spec,
            "scene": scene.to_dict(),
        }
    )
    execu = RealtimeExecutor(replan_steps=rsteps, backend=backend)
    escaped = False
    success = False
    steps = 0
    step_lim = 0
    physics_source = "default"

    try:
        bridge.setup_episode(env, args, seed=seed, instruction=instr, ep_num=0)
        driver.on_episode_start()
        ppt, physics_source = resolve_physics_per_tick(
            env,
            explicit=physics_per_tick,
            auto_calibrate=auto_calibrate_physics,
        )
        driver.set_physics_per_tick(ppt, source=physics_source)
        driver.begin_policy()
        obs = env.get_obs()
        if frame_sink is not None:
            _attach_robotwin_frame_sink(env, driver, frame_sink)
            frame_sink(_robotwin_head_rgb(env))
        step_cap = int(env.step_lim if max_steps is None else min(int(env.step_lim), int(max_steps)))

        def _async_step(action):
            nonlocal escaped, obs
            env.take_action(np.asarray(action, dtype=np.float32), action_type="qpos")
            if driver.escaped:
                escaped = True

        while env.take_action_cnt < step_cap and not env.eval_success:
            if execu.needs_replan():
                replan_idx = int(execu.n_replans)
                predict_kw = policy_predict_kwargs(policy, driver, obs)
                from benchmarks.common.cache_log import diff_snapshots, snapshot_policy

                cache_before = snapshot_policy(policy)
                t_obs_s = float(getattr(driver, "policy_t", 0.0)) / float(hz)
                if mode == LatencyMode.MEASURED:
                    chunk, lat_s = time_policy_predict(policy, obs, instr, **predict_kw)
                    n_delay = coupler.resolve_and_record(
                        replan_idx,
                        measured_latency_s=lat_s,
                        delivery_delay_s=delivery_delay_s,
                        t_obs_s=t_obs_s,
                        cache=diff_snapshots(cache_before, snapshot_policy(policy)),
                    )
                else:
                    chunk = policy.predict(obs, instr, **predict_kw)
                    n_delay = coupler.resolve_and_record(
                        replan_idx,
                        delivery_delay_s=delivery_delay_s,
                        t_obs_s=t_obs_s,
                        cache=diff_snapshots(cache_before, snapshot_policy(policy)),
                    )
                freeze_ticks = float(n_delay)
                execu.on_replan(
                    chunk,
                    driver,
                    freeze_ticks,
                    async_step_fn=_async_step if backend == RealtimeBackend.ASYNC else None,
                    max_ticks=float(env.step_lim),
                )
                if driver.escaped:
                    escaped = True
                    break
                if execu.open_loop_break(freeze_ticks):
                    break

            if execu.chunk is None:
                break
            action = execu.next_action()
            env.take_action(np.asarray(action, dtype=np.float32), action_type="qpos")
            if driver.escaped:
                escaped = True
                break
            if env.eval_success:
                break
            if execu.needs_replan():
                obs = env.get_obs()

        success = bool(env.eval_success) and not escaped
        steps = int(env.take_action_cnt)
        step_lim = int(env.step_lim)
    finally:
        bridge.close_task_env(env)

    hold_snap = driver.grasp_hold.snapshot()
    held = hold_snap.get("grasp_hold")
    approached = bool(driver.ever_approached or driver.released)
    if success:
        fail_stage = None
    elif escaped:
        fail_stage = "escape"
    elif steps >= step_lim:
        fail_stage = "timeout"
    elif not approached:
        fail_stage = "approach"
    elif held is not True:
        fail_stage = "grasp"
    else:
        fail_stage = "place"

    lat_fields = episode_latency_fields(coupler)
    mean_n = lat_fields.get("n_delay_mean")
    bwr = blind_window_ratio(
        float(mean_n) if mean_n is not None else float(freeze_ticks),
        rsteps,
    )
    release_vel = (
        float(np.linalg.norm(trajectory.velocity_at(trajectory.t, hz=hz)))
        if driver.released
        else 0.0
    )
    out = {
        "escaped": bool(escaped),
        "fail_stage": fail_stage,
        "min_tcp_dist": (
            None if driver.min_tcp_dist == float("inf") else float(driver.min_tcp_dist)
        ),
        "max_held_streak": int(driver.max_held_streak),
        "ever_approached": bool(approached),
        "steps": steps,
        "n_replans": int(execu.n_replans),
        "n_freeze": float(freeze_ticks) if math.isfinite(freeze_ticks) else None,
        "applied_freeze": float(execu.applied_freeze_total),
        "stale_executed": int(execu.stale_executed_total),
        "stale_hold_ticks": float(execu.stale_hold_ticks_total),
        "chunk_len_mean": (
            execu.chunk_len_sum / execu.n_len_samples if execu.n_len_samples else None
        ),
        "leftover_len_mean": (
            execu.leftover_len_sum / execu.n_len_samples if execu.n_len_samples else None
        ),
        "replan_steps": int(rsteps),
        "blind_window_ratio": bwr,
        "backend": backend.value,
        "drift_m": float(driver.drift_m),
        "release_radius": float(release),
        "rails_release_velocity_m_s": release_vel,
        "release_handoff_m_s": release_vel,
        "physics_per_tick": int(driver.physics_per_tick),
        "physics_per_tick_source": driver.physics_per_tick_source,
        "policy_hz": float(hz),
        "workspace_aabb": aabb_to_dict(aabb),
        "escape_check": bool(escape_check),
        "trajectory": trajectory_kind,
        "speed": float(speed),
        "layout_speed": float(scene.layout_speed),
        "traj_speed": float(traj_speed),
        "task_name": spec.spec,
        "env_task": env_task,
        "rails_variant": spec.rails_variant,
        "spawn_pushers": spec.spawn_pushers,
        "apparatus_visible": bool(scene.installs_visual_actor()),
        "seed": int(seed),
        "traj_kwargs": traj_kw,
        "bimanual": bool(meta.get("bimanual")),
        "stress": meta.get("stress"),
        "language_mode": effective_lang,
        "instruction": instr,
        **scene.to_dict(),
        **outcome_fields(
            rails_released=bool(driver.released),
            grasp_hold=held,
            task_success=success,
            aux_events=list(driver.aux.events),
            grasp_hold_extra=hold_snap,
        ),
    }
    out.update(driver.episode_dynamics_stats())
    out.update(lat_fields)
    out["fail_stage"] = fail_stage
    out["delivery_delay_s"] = float(delivery_delay_s)
    out["ever_approached"] = bool(approached)
    # Legacy gripper-close streak. Not grasp_hold and not task success.
    out["grasped_held"] = bool(driver.grasped_held())
    return out


def _run_native_extra_episode(
    *,
    spec,
    env_task: str,
    seed: int,
    ctx: bridge.EpisodeContext,
    n_freeze: float,
    physics_per_tick: int | None,
    auto_calibrate_physics: bool,
    policy_hz: float | None,
    robotwin_root,
    task_config: str,
    instruction: Optional[str],
    backend: RealtimeBackend | str,
    policy: PolicyClient | None,
    replan_steps: int | None,
    allow_replan_override: bool,
    language_mode: str,
    scene: Any = None,
    latency_mode: str | LatencyMode = LatencyMode.MEASURED,
    replay_trace: LatencyTrace | None = None,
    deadline_ms: float = DEFAULT_DEADLINE_MS,
    delivery_delay_s: float = 0.0,
    motion_scale: float | None = None,
    speed: float = 0.0,
) -> dict[str, Any]:
    from benchmarks.common.language import apply_language_mode, parse_language_mode
    from benchmarks.common.outcomes import outcome_fields
    from benchmarks.common.scene import SceneCondition, resolve_scene_condition

    from ..env.extra_driver import ExtraNativeDriver
    from ..env.extra_motion import install_extra_motion

    if isinstance(backend, str):
        backend = RealtimeBackend(backend)
    policy = policy or make_inprocess_policy("robotwin", ctx)
    reset_fn = getattr(policy, "reset", None)
    if callable(reset_fn):
        reset_fn()
    rsteps = effective_replan_steps(
        "robotwin",
        replan_steps if replan_steps is not None else getattr(ctx, "replan_steps", 8),
        allow_override=allow_replan_override,
    )
    root = robotwin_root or bridge.DEFAULT_ROBOTWIN_ROOT
    hz = float(policy_hz if policy_hz is not None else getattr(ctx, "policy_hz", 20.0))
    scale = 1.0 if motion_scale is None else float(motion_scale)
    if scene is None:
        scene = resolve_scene_condition(speed=float(speed), catalog_job=False)
    elif not isinstance(scene, SceneCondition):
        raise TypeError("scene must be a SceneCondition")
    base_instr = instruction or spec.language or default_instruction(env_task)
    instr = apply_language_mode(base_instr, parse_language_mode(language_mode))
    args = bridge.load_task_args(root, task_config=task_config, task_name=env_task)
    args["_robotwin_root"] = str(root)
    env = bridge.create_task_env(env_task, root)
    driver = ExtraNativeDriver(env, env_task, policy_hz=hz)
    mode = parse_latency_mode(latency_mode)
    constant_freeze = scene.apply_n_freeze(float(n_freeze))
    coupler = LatencyCoupler(
        mode=mode,
        control_hz=hz,
        deadline_ms=float(deadline_ms),
        constant_n_freeze=float(constant_freeze),
        replay_trace=replay_trace,
    )
    freeze_ticks = float(constant_freeze)
    execu = RealtimeExecutor(replan_steps=rsteps, backend=backend)
    escaped = False
    success = False
    steps = 0
    physics_source = "default"
    profile = None
    try:
        bridge.setup_episode(env, args, seed=seed, instruction=instr, ep_num=0)
        profile = install_extra_motion(env, env_task, seed=seed, motion_scale=scale)
        if hasattr(env, "reset_flight"):
            env.reset_flight()
        driver.on_episode_start()
        ppt, physics_source = resolve_physics_per_tick(
            env,
            explicit=physics_per_tick,
            auto_calibrate=auto_calibrate_physics,
        )
        driver.set_physics_per_tick(ppt, source=physics_source)
        driver.begin_policy()
        obs = env.get_obs()
        miss_horizon = int(round(float(profile.horizon_s) * hz))
        lift_budget = int(round(2.0 * hz))

        def _async_step(action):
            nonlocal escaped, obs
            env.take_action(np.asarray(action, dtype=np.float32), action_type="qpos")

        def _freeze_for_replan(replan_idx: int):
            predict_kw = policy_predict_kwargs(policy, driver, obs)
            t_obs_s = float(getattr(driver, "policy_t", 0.0)) / float(hz)
            if mode == LatencyMode.MEASURED:
                chunk, lat_s = time_policy_predict(policy, obs, instr, **predict_kw)
                n_delay = coupler.resolve_and_record(
                    replan_idx,
                    measured_latency_s=lat_s,
                    delivery_delay_s=delivery_delay_s,
                    t_obs_s=t_obs_s,
                )
            else:
                chunk = policy.predict(obs, instr, **predict_kw)
                n_delay = coupler.resolve_and_record(
                    replan_idx,
                    delivery_delay_s=delivery_delay_s,
                    t_obs_s=t_obs_s,
                )
            if not scene.thinking_advances_world():
                n_delay = 0.0
            return chunk, float(n_delay)

        while env.take_action_cnt < env.step_lim and not env.eval_success:
            if driver.caught:
                if int(env.take_action_cnt) >= miss_horizon + lift_budget:
                    break
            elif int(env.take_action_cnt) >= miss_horizon:
                break
            if execu.needs_replan():
                chunk, freeze_ticks = _freeze_for_replan(int(execu.n_replans))
                execu.on_replan(
                    chunk,
                    driver,
                    freeze_ticks,
                    async_step_fn=_async_step if backend == RealtimeBackend.ASYNC else None,
                    max_ticks=float(env.step_lim),
                )
                if execu.open_loop_break(freeze_ticks):
                    break
            if execu.chunk is None:
                break
            action = execu.next_action()
            env.take_action(np.asarray(action, dtype=np.float32), action_type="qpos")
            if env.eval_success:
                break
            if execu.needs_replan():
                obs = env.get_obs()
        success = bool(env.eval_success)
        steps = int(env.take_action_cnt)
    finally:
        bridge.close_task_env(env)

    lat_fields = episode_latency_fields(coupler)
    mean_n = lat_fields.get("n_delay_mean")
    bwr = blind_window_ratio(
        float(mean_n) if mean_n is not None else float(freeze_ticks),
        rsteps,
    )
    aux = list(getattr(driver.aux, "events", []) or [])
    catch_t = float(profile.catch_t) if profile is not None else None
    return {
        "escaped": False,
        "steps": steps,
        "n_replans": int(execu.n_replans),
        "n_freeze": float(freeze_ticks) if math.isfinite(freeze_ticks) else None,
        "applied_freeze": float(execu.applied_freeze_total),
        "stale_executed": int(execu.stale_executed_total),
        "stale_hold_ticks": float(execu.stale_hold_ticks_total),
        "chunk_len_mean": (
            execu.chunk_len_sum / execu.n_len_samples if execu.n_len_samples else None
        ),
        "leftover_len_mean": (
            execu.leftover_len_sum / execu.n_len_samples if execu.n_len_samples else None
        ),
        "replan_steps": int(rsteps),
        "blind_window_ratio": bwr,
        "backend": backend.value,
        "protocol": "real_time_v2",
        "drift_m": 0.0,
        "physics_per_tick": int(driver.physics_per_tick),
        "physics_per_tick_source": driver.physics_per_tick_source,
        "policy_hz": float(hz),
        "trajectory": "native",
        "speed": float(scale),
        "motion_scale": float(scale),
        "motion_kind": None if profile is None else profile.kind,
        "task_name": spec.spec,
        "env_task": env_task,
        "seed": int(seed),
        "language_mode": language_mode,
        "instruction": instr,
        "caught": bool(driver.caught),
        "min_tcp_dist": float(driver.min_tcp_dist) if math.isfinite(driver.min_tcp_dist) else None,
        "last_grip": float(driver.last_grip),
        "catch_t": catch_t,
        **scene.to_dict(),
        **lat_fields,
        **outcome_fields(
            rails_released=False,
            grasp_hold=None,
            task_success=success,
            aux_events=aux,
        ),
    }
