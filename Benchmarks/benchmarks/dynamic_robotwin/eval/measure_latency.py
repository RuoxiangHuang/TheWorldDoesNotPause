"""Varying-observation latency measurement for Dynamic-RoboTwin."""

from __future__ import annotations

import time

import numpy as np
import torch

from ..env import robotwin_bridge as bridge

# Re-export shared freeze helpers (same math as LIBERO; policy_hz plays the role of f_ctrl).
from benchmarks.dynamic_libero.eval.measure_latency import (  # noqa: F401
    OPEN_LOOP_LATENCY_S,
    apply_thinking_freeze,
    freeze_segments,
    freeze_ticks_from_latency,
    n_freeze_from_latency,
)


def measure_latency_varying(
    task_name: str,
    ctx: bridge.EpisodeContext,
    *,
    robotwin_root=None,
    task_config: str = "demo_clean",
    seed: int = 0,
    k: int = 8,
    warmup: int = 3,
) -> tuple[float, list[float]]:
    """Median replan latency with changing observations between timed calls.

    Requires ``k > warmup`` so compile / cache warmups are never folded into the
    reported median.
    """
    if int(k) <= int(warmup):
        raise ValueError(
            f"latency sample count k={k} must be > warmup={warmup} "
            "(need at least one post-warmup timed sample; "
            "compile/cache warmups would otherwise pollute the median)."
        )
    from ..env.dynamic_tasks import resolve_eval_spec

    spec = resolve_eval_spec(task_name)
    env_task = spec.task_name
    root = robotwin_root or bridge.DEFAULT_ROBOTWIN_ROOT
    args = bridge.load_task_args(root, task_config=task_config, task_name=env_task)
    args["_robotwin_root"] = str(root)
    env = bridge.create_task_env(env_task, root)
    instruction = spec.language
    lats: list[float] = []
    try:
        bridge.setup_episode(env, args, seed=seed, instruction=instruction, ep_num=0)
        obs = env.get_obs()
        for _ in range(int(k)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            chunk = bridge.predict_action_chunk(obs, instruction, ctx)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lats.append(time.perf_counter() - t0)
            # Advance world a little so the next observation differs.
            for a in chunk[: min(3, len(chunk))]:
                env.take_action(a, action_type="qpos")
                if env.eval_success or env.take_action_cnt >= env.step_lim:
                    break
            obs = env.get_obs()
            if env.eval_success or env.take_action_cnt >= env.step_lim:
                break
    finally:
        bridge.close_task_env(env)

    if len(lats) <= int(warmup):
        raise RuntimeError(
            f"collected only {len(lats)} latency samples but warmup={warmup}; "
            "episode ended too early during measurement — increase k or use a longer task."
        )
    use = lats[int(warmup) :]
    med = float(np.median(use))
    return med, lats
