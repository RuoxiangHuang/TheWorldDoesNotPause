"""Single-episode Real-Time rollout (shared by paired SR and sweeps)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

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
from benchmarks.common.workspace import (
    aabb_to_dict,
    infer_libero_arena,
    resolve_workspace_aabb,
)

from ..env.driver import RealtimeDriver, wrap_env_with_driver
from ..env import libero_bridge as bridge
from ..trajectories import build_trajectory
from ..trajectories.smooth_turn import (
    default_smooth_turn_kwargs,
    is_smooth_turn,
    mix_episode_seed,
)
from .turn_reaction import maybe_turn_tracker


def _target_to_eef(obs: Any, driver: Any) -> Optional[np.ndarray]:
    if not isinstance(obs, dict) or getattr(driver, "target", None) is None:
        return None
    rel = obs.get(f"{driver.target}_to_robot0_eef_pos")
    if rel is None:
        return None
    return np.asarray(rel, dtype=np.float64)


def _note_turn_step(tracker: Any, driver: Any, obs: Any) -> None:
    if tracker is None:
        return
    dist = None
    try:
        dist = driver._eef_dist(obs) if isinstance(obs, dict) else None
    except Exception:
        dist = None
    tracker.on_step(
        float(driver.policy_t),
        _target_to_eef(obs, driver),
        dist,
        released=bool(driver.released),
    )


def _predict_kwargs(policy: Any, driver: Any, obs: Any | None = None) -> dict[str, Any]:
    from benchmarks.common.world_wire import policy_predict_kwargs

    return policy_predict_kwargs(policy, driver, obs)


@torch.no_grad()
def run_episode(
    task,
    init_state,
    ctx: bridge.EpisodeContext,
    *,
    trajectory_kind: str = "linear",
    speed: float = 0.003,
    trajectory_kwargs: Optional[dict[str, Any]] = None,
    n_freeze: float = 0.0,
    release_radius: float = 0.06,
    max_steps: int = 400,
    render_res: Optional[int] = None,
    backend: RealtimeBackend | str = DEFAULT_BACKEND,
    policy: PolicyClient | None = None,
    replan_steps: int | None = None,
    allow_replan_override: bool = False,
    task_suite_name: Optional[str] = None,
    workspace_aabb: Optional[np.ndarray] = None,
    escape_check: bool = True,
    latency_mode: str | LatencyMode = LatencyMode.MEASURED,
    replay_trace: LatencyTrace | None = None,
    latency_trace_path: str | Path | None = None,
    deadline_ms: float = DEFAULT_DEADLINE_MS,
    task_id: int = 0,
    movers_enabled: bool = False,
    mover_language: str = "keep",
    language_mode: str = "keep",
    target_registry: bool = True,
    grasp_mode: str = "ghost",
    engage_alpha: float = 1.75,
    place_freeze: str | bool = "both",
    release_kick: Optional[bool] = None,
    scene: Optional[Any] = None,
    motive: str = "auto",
    rails_variant: str = "pick",
    place_target_name: Optional[str] = None,
    dynamic_task_id: Optional[str] = None,
    init_id: int = 0,
    delivery_delay_s: float = 0.0,
    frame_sink: Optional[Any] = None,
) -> dict[str, Any]:
    """Run one Real-Time episode with per-replan latency coupling.

    ``latency_mode``:
      - ``measured``: time each ``predict``; that L_c advances the world
      - ``replay``: actions from ``predict``, n_delay from ``replay_trace``
      - ``constant``: legacy single ``n_freeze`` for every replan

    ``language_mode``:
      - ``keep``: training instruction (primary)
      - ``motion``: B-track — append motion suffix; goal clause unchanged
    """
    if isinstance(backend, str):
        backend = RealtimeBackend(backend)
    mode = parse_latency_mode(latency_mode)
    policy = policy or make_inprocess_policy("libero", ctx)
    reset_fn = getattr(policy, "reset", None)
    if callable(reset_fn):
        reset_fn()
    rsteps = effective_replan_steps(
        "libero",
        replan_steps or ctx.cfg.EVALUATION.get("replan_steps"),
        allow_override=allow_replan_override,
    )

    traj_kw = dict(trajectory_kwargs or {})
    dyn = None
    if dynamic_task_id:
        from ..env.dynamic_tasks import get_dynamic_task

        dyn = get_dynamic_task(dynamic_task_id)
        rails_variant = dyn.variant
        if place_target_name is None:
            place_target_name = dyn.place_target
        kind, spec_kw = dyn.traj_spec()
        if is_smooth_turn(kind) or getattr(dyn, "track", "main") == "react":
            trajectory_kind = str(kind)
            traj_kw = {**spec_kw, **traj_kw}
        elif is_smooth_turn(trajectory_kind):
            pass
        elif kind != "linear" or spec_kw:
            trajectory_kind = kind
            traj_kw = {**spec_kw, **traj_kw} if not spec_kw.get("waypoints") else spec_kw
        if dyn.motive:
            motive = dyn.motive
    if is_smooth_turn(trajectory_kind):
        for k, v in default_smooth_turn_kwargs().items():
            traj_kw.setdefault(k, v)
        if "event_seed" not in traj_kw:
            base_seed = 0
            try:
                base_seed = int(ctx.cfg.get("seed", 0) or 0)
            except Exception:
                base_seed = 0
            traj_kw["event_seed"] = mix_episode_seed(
                base_seed, int(task_id), int(init_id)
            )
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
            catalog_job=dynamic_task_id is not None or dyn is not None,
        )
    elif not isinstance(scene, SceneCondition):
        raise TypeError("scene must be a SceneCondition")
    n_freeze = scene.apply_n_freeze(float(n_freeze))
    traj_speed = trajectory_speed_for_condition(scene, float(speed))
    trajectory = build_trajectory(trajectory_kind, speed=traj_speed, **traj_kw)
    resolution = int(render_res or bridge.get_env_resolution())
    suite = task_suite_name or str(ctx.cfg.EVALUATION.get("task_suite_name", "libero_object"))
    from dataclasses import replace

    from benchmarks.common.language import apply_language_mode, parse_language_mode

    from ..env.dynamic_tasks import (
        default_place_target,
        parse_rails_variant,
    )
    from ..env.motives import keeps_original_mesh, resolve_motive
    from ..env.movers import (
        mover_assets,
        parse_mover_language,
        resolve_rails_spec,
        rewrite_language,
    )

    variant = parse_rails_variant(rails_variant, default="pick")
    raw_variant = str(rails_variant or "auto").strip().lower()
    motive_resolved = resolve_motive(suite, motive)
    # CLI ``--motive carrier`` without an explicit variant still means dual motion.
    # Catalog / ``--rails-variant pick`` keeps the bowl on the flatbed only.
    if (
        motive_resolved == "carrier"
        and dyn is None
        and raw_variant in ("", "auto", "default")
    ):
        variant = "both"
    if place_target_name is None:
        place_target_name = default_place_target(suite, task_id)
    # Pusher / conveyor keep the training mesh. Never combine with movers=on.
    if motive_resolved in ("pusher", "carrier", "conveyor"):
        movers_enabled = False
    appearance_mode = parse_mover_language(mover_language)
    if keeps_original_mesh(motive_resolved):
        appearance_mode = "keep"
    lang_mode = parse_language_mode(language_mode)
    rails_spec = resolve_rails_spec(
        suite,
        task_id,
        movers_enabled=movers_enabled,
        target_registry=target_registry,
    )
    with mover_assets(suite, task_id, enabled=movers_enabled) as asset_spec:
        env, desc = bridge.get_libero_env(task, resolution, ctx.cfg.get("seed"))
    pusher = None
    place_pusher = None
    belt = None
    if scene.uses_layout_actor() and motive_resolved in ("pusher", "carrier"):
        from ..env.pusher import install_cars, make_pusher_scene

        pusher, place_pusher = make_pusher_scene(motive_resolved, variant)
        cars = [c for c in (pusher, place_pusher) if c is not None]
        if cars and scene.installs_visual_actor():
            install_cars(env, cars)
    elif scene.uses_layout_actor() and motive_resolved == "conveyor":
        from ..env.conveyor import ConveyorBelt

        belt = ConveyorBelt()
        if scene.installs_visual_actor():
            belt.install(env)
    # Prefer registry spec for driver; asset_spec equals it when swap is on.
    mover_spec = asset_spec if asset_spec is not None else rails_spec
    if motive_resolved == "conveyor" and mover_spec is not None:
        mover_spec = replace(
            mover_spec,
            motion="conveyor",
            height_offset=0.0,
            noun=mover_spec.category.replace("_", " "),
        )
    if movers_enabled and appearance_mode == "rewrite" and mover_spec is not None:
        desc = rewrite_language(desc, mover_spec)
    # B-track motion suffix (after optional appearance noun rewrite).
    noun = None
    if mover_spec is not None:
        noun = mover_spec.noun if (movers_enabled and appearance_mode == "rewrite") else mover_spec.category
    if dyn is not None and str(getattr(dyn, "language_policy", "keep")) == "scene":
        desc = str(dyn.language)
    effective_lang = effective_language_mode(lang_mode, scene)
    desc = apply_language_mode(
        desc, effective_lang, noun=noun, motive=motive_resolved
    )
    aabb = resolve_workspace_aabb(
        "libero",
        suite_name=suite,
        task_id=task_id,
        arena=infer_libero_arena(env),
        override=workspace_aabb,
        escape_check=escape_check,
    )
    control_hz = float(ctx.control_freq)
    driver = RealtimeDriver(
        env,
        trajectory,
        release_radius=release_radius,
        workspace_aabb=aabb,
        control_hz=control_hz,
        mover=mover_spec,
        grasp_mode=grasp_mode,
        engage_alpha=engage_alpha,
        pusher=pusher,
        place_pusher=place_pusher,
        rails_variant=variant,
        place_target_name=place_target_name,
        clear_path=bool(scene.modifies_layout() and (dyn is None or getattr(dyn, "clear_path", True))),
        place_freeze=place_freeze,
        release_kick=release_kick,
        scene=scene,
    )
    wrap_env_with_driver(env, driver)
    if frame_sink is not None:
        from ..viz.annotate import render_agent

        def _emit_frame() -> None:
            if not driver.motion_enabled:
                return
            frame_sink(render_agent(env))

        _orig_tick = driver.tick

        def _tick(dt: float = 1.0, _orig=_orig_tick):
            _orig(dt)
            _emit_frame()

        driver.tick = _tick  # type: ignore[method-assign]
        _orig_step = env.step

        def _step(action, _orig=_orig_step):
            out = _orig(action)
            _emit_frame()
            return out

        env.step = _step

    num_steps_wait = int(ctx.cfg.EVALUATION.get("num_steps_wait", 30))
    coupler = LatencyCoupler(
        mode=mode,
        control_hz=control_hz,
        deadline_ms=float(deadline_ms),
        constant_n_freeze=float(n_freeze),
        replay_trace=replay_trace,
    )
    coupler.trace.meta.update(
        {
            "trajectory": trajectory_kind,
            "speed": float(speed),
            "backend": backend.value,
            "task_suite": suite,
            "movers": bool(movers_enabled),
            "mover_language": appearance_mode,
            "language_mode": effective_lang,
            "target_registry": bool(target_registry),
            "grasp_mode": driver.grasp_mode,
            "engage_alpha": float(engage_alpha),
            "motive": motive_resolved,
            "rails_variant": variant,
            "dynamic_task_id": None if dyn is None else dyn.id,
            "path_name": None if dyn is None else dyn.path_name,
            "scene": scene.to_dict(),
        }
    )
    execu = RealtimeExecutor(replan_steps=rsteps, backend=backend)

    env.reset()
    obs = env.set_init_state(init_state)
    if belt is not None:
        from ..env.conveyor import layout_conveyor_for_driver

        layout_conveyor_for_driver(
            belt, driver, trajectory, horizon_steps=float(max_steps)
        )
    done = False
    escaped = False
    t = 0
    current_n_delay = 0.0
    current_replan_idx = -1
    tracker = maybe_turn_tracker(trajectory, control_hz)

    def _async_step(action):
        # Stale-tail actions belong to the previous chunk; do not attribute their
        # observation age to the newly timed replan (age is recorded for the new
        # chunk only, after thinking returns).
        nonlocal obs, done, escaped, t
        obs, _, done, info = env.step(action.tolist() if hasattr(action, "tolist") else action)
        t += 1
        _note_turn_step(tracker, driver, obs)
        if isinstance(info, dict) and info.get("escaped"):
            escaped = True
        if driver.escaped:
            escaped = True

    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(bridge.dummy_action())
                t += 1
                continue

            if not driver.motion_enabled:
                driver.begin_policy()
                if frame_sink is not None:
                    from ..viz.annotate import render_agent

                    frame_sink(render_agent(env))

            if execu.needs_replan():
                replan_idx = int(execu.n_replans)  # next index (on_replan increments)
                if tracker is not None:
                    tracker.on_replan_obs(
                        float(driver.policy_t),
                        rails_active=bool(driver.rails_active),
                    )
                from benchmarks.common.cache_log import diff_snapshots, snapshot_policy

                cache_before = snapshot_policy(policy)
                t_obs_s = float(driver.policy_t) / float(control_hz)
                if mode == LatencyMode.MEASURED:
                    chunk, lat_s = time_policy_predict(
                        policy, obs, desc, **_predict_kwargs(policy, driver, obs)
                    )
                    n_delay = coupler.resolve_and_record(
                        replan_idx,
                        measured_latency_s=lat_s,
                        delivery_delay_s=delivery_delay_s,
                        t_obs_s=t_obs_s,
                        cache=diff_snapshots(cache_before, snapshot_policy(policy)),
                    )
                else:
                    chunk = policy.predict(obs, desc, **_predict_kwargs(policy, driver, obs))
                    n_delay = coupler.resolve_and_record(
                        replan_idx,
                        delivery_delay_s=delivery_delay_s,
                        t_obs_s=t_obs_s,
                        cache=diff_snapshots(cache_before, snapshot_policy(policy)),
                    )
                current_n_delay = float(n_delay)
                current_replan_idx = replan_idx
                execu.on_replan(
                    chunk,
                    driver,
                    current_n_delay,
                    async_step_fn=_async_step if backend == RealtimeBackend.ASYNC else None,
                    max_ticks=float(max_steps),
                )
                if tracker is not None:
                    tracker.on_takeover(float(driver.policy_t))
                if execu.open_loop_break(current_n_delay):
                    escaped = bool(driver.escaped)
                    break

            action = execu.next_action()
            # next_action increments idx; age uses j = idx-1 within the new chunk.
            j = int(execu.idx) - 1
            obs, _, done, info = env.step(action.tolist() if hasattr(action, "tolist") else action)
            t += 1
            _note_turn_step(tracker, driver, obs)
            if current_replan_idx >= 0:
                coupler.on_action_applied(current_replan_idx, j, current_n_delay)
            if isinstance(info, dict) and info.get("escaped"):
                escaped = True
            if driver.escaped:
                escaped = True
            if done or escaped:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    lat_fields = episode_latency_fields(coupler)
    mean_n = lat_fields.get("n_delay_mean")
    bwr = blind_window_ratio(
        float(mean_n) if mean_n is not None else float(n_freeze),
        rsteps,
    )
    if latency_trace_path is not None:
        coupler.trace.save(latency_trace_path)

    task_success = bool(done) and not escaped
    hold_snap = driver.grasp_hold.snapshot()
    out = {
        "success": task_success,
        "done": bool(done),
        "escaped": bool(escaped),
        "steps": int(t),
        "n_replans": int(execu.n_replans),
        "applied_freeze": float(execu.applied_freeze_total),
        "stale_executed": int(execu.stale_executed_total),
        "stale_hold_ticks": float(execu.stale_hold_ticks_total),
        "replan_steps": int(rsteps),
        "blind_window_ratio": bwr,
        "backend": backend.value,
        "drift_m": float(driver.drift_m),
        "release_handoff_m_s": (
            float(np.linalg.norm(trajectory.velocity_at(trajectory.t, hz=control_hz)))
            if driver.released
            else 0.0
        ),
        "workspace_aabb": aabb_to_dict(aabb),
        "escape_check": bool(escape_check),
        "trajectory": trajectory_kind,
        "speed": float(speed),
        "layout_speed": float(scene.layout_speed),
        "traj_speed": float(traj_speed),
        "control_hz": float(control_hz),
        "deadline_ms": float(deadline_ms),
        "movers": bool(movers_enabled),
        "mover_language": appearance_mode,
        "language_mode": effective_lang,
        "target_registry": bool(target_registry),
        "task_description": desc,
        "motive": motive_resolved,
        "rails_variant": variant,
        "dynamic_task_id": None if dyn is None else dyn.id,
        "path_name": None if dyn is None else dyn.path_name,
        "eval_split": None if dyn is None else getattr(dyn, "eval_split", None),
        "language_policy": None if dyn is None else getattr(dyn, "language_policy", None),
        "pusher": pusher is not None,
        "place_pusher": place_pusher is not None,
        "conveyor": belt is not None,
        "apparatus_visible": bool(scene.installs_visual_actor()),
        "init_id": int(init_id),
        "place_success": task_success,
        **scene.to_dict(),
        **outcome_fields(
            rails_released=bool(driver.released),
            grasp_hold=hold_snap.get("grasp_hold"),
            task_success=task_success,
            aux_events=list(driver.aux.events),
            grasp_hold_extra=hold_snap,
        ),
    }
    if mover_spec is not None:
        out["mover_target"] = mover_spec.target
        out["mover_category"] = mover_spec.category
    out.update(driver.episode_stats())
    out.update(lat_fields)
    if dyn is not None:
        out["track"] = getattr(dyn, "track", "main")
        out["aligned_main_id"] = dyn.aligned_main_id
        if dyn.event_type is not None:
            out.setdefault("event_type", dyn.event_type)
    experienced = None
    if getattr(trajectory, "event_t", None) is not None:
        experienced = bool(getattr(driver, "event_experienced", False))
        out["event_experienced"] = experienced
    if tracker is not None:
        out.update(
            tracker.summary(
                event_experienced=True if experienced is None else bool(experienced)
            )
        )
    if latency_trace_path is not None:
        out["latency_trace_path"] = str(Path(latency_trace_path).resolve())
    out["delivery_delay_s"] = float(delivery_delay_s)
    if task_success:
        fail_stage = None
    elif escaped:
        fail_stage = "escape"
    elif int(t) >= int(max_steps):
        fail_stage = "timeout"
    elif out.get("grasp_hold") is True:
        fail_stage = "place"
    elif out.get("engage_t") is None and not out.get("rails_released"):
        fail_stage = "approach"
    else:
        fail_stage = "grasp"
    out["fail_stage"] = fail_stage
    return out
