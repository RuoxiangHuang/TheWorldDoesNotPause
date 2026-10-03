"""Latency measurement helpers for openpi + Real-Time benchmarks."""

from __future__ import annotations

import time

import numpy as np
import torch

from fasterpi.realtime.loader import OpenPIRealtimePolicy


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_libero_latency(
    rt_policy: OpenPIRealtimePolicy,
    task,
    *,
    suite,
    task_id: int = 0,
    seed: int = 0,
    k: int = 8,
    warmup: int = 3,
) -> tuple[float, list[float]]:
    """Median replan latency with varying LIBERO observations."""
    if int(k) <= int(warmup):
        raise ValueError(f"k={k} must be > warmup={warmup}")

    from benchmarks.realtime_libero.env import libero_bridge as bridge

    bridge.ensure_runtime_env()
    resolution = bridge.get_env_resolution()
    env, desc = bridge.get_libero_env(task, resolution, seed)
    lats: list[float] = []
    try:
        env.reset()
        inits = list(suite.get_task_init_states(task_id))
        obs = env.set_init_state(inits[0])
        for _ in range(30):
            obs, _, _, _ = env.step(bridge.dummy_action())
        for _ in range(int(k)):
            _sync()
            t0 = time.perf_counter()
            chunk = rt_policy.predict(obs, desc)
            _sync()
            lats.append(time.perf_counter() - t0)
            for a in chunk[: min(5, len(chunk))]:
                obs, _, done, _ = env.step(a.tolist())
                if done:
                    break
    finally:
        try:
            env.close()
        except Exception:
            pass
    use = lats[int(warmup) :]
    if not use:
        raise RuntimeError("no post-warmup latency samples collected")
    return float(np.median(use)), lats


def measure_robotwin_latency(
    rt_policy: OpenPIRealtimePolicy,
    task_name: str,
    instruction: str,
    *,
    robotwin_root,
    task_config: str = "demo_clean",
    seed: int = 0,
    k: int = 8,
    warmup: int = 3,
) -> tuple[float, list[float]]:
    if int(k) <= int(warmup):
        raise ValueError(f"k={k} must be > warmup={warmup}")

    from benchmarks.realtime_robotwin.env import robotwin_bridge as bridge

    bridge.ensure_runtime_env()
    args = bridge.load_task_args(robotwin_root, task_config=task_config, task_name=task_name)
    args["_robotwin_root"] = str(robotwin_root)
    env = bridge.create_task_env(task_name, robotwin_root)
    lats: list[float] = []
    try:
        bridge.setup_episode(env, args, seed=seed, instruction=instruction, ep_num=0)
        obs = env.get_obs()
        for _ in range(int(k)):
            _sync()
            t0 = time.perf_counter()
            chunk = rt_policy.predict(obs, instruction)
            _sync()
            lats.append(time.perf_counter() - t0)
            for a in chunk[: min(4, len(chunk))]:
                env.take_action(np.asarray(a, dtype=np.float32), action_type="qpos")
            obs = env.get_obs()
    finally:
        bridge.close_task_env(env)
    use = lats[int(warmup) :]
    if not use:
        raise RuntimeError("no post-warmup latency samples collected")
    return float(np.median(use)), lats
