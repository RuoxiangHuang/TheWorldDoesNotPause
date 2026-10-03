"""Measure replan latency against a remote Real-Time HTTP policy server."""

from __future__ import annotations

import time

import numpy as np
import torch

from benchmarks.common.policy.factory import make_policy_client


def measure_http_latency_libero(
    policy_url: str,
    task,
    *,
    suite,
    task_id: int = 0,
    seed: int = 0,
    k: int = 8,
    warmup: int = 3,
) -> tuple[float, list[float]]:
    from benchmarks.dynamic_libero.env import libero_bridge as bridge

    if int(k) <= int(warmup):
        raise ValueError(f"k={k} must be > warmup={warmup}")

    bridge.ensure_runtime_env()
    policy = make_policy_client("libero", policy_url=policy_url)
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
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            chunk = policy.predict(obs, desc)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lats.append(time.perf_counter() - t0)
            for a in chunk[: min(5, len(chunk))]:
                obs, _, done, _ = env.step(a.tolist() if hasattr(a, "tolist") else a)
                if done:
                    break
    finally:
        try:
            env.close()
        except Exception:
            pass
        policy.close()
    use = lats[int(warmup) :]
    if not use:
        raise RuntimeError("no post-warmup latency samples")
    return float(np.median(use)), lats


def measure_http_latency_robocasa(
    policy_url: str,
    task_name: str,
    *,
    split: str = "target",
    seed: int = 0,
    k: int = 8,
    warmup: int = 3,
    robocasa_root=None,
    codec: str | None = None,
) -> tuple[float, list[float]]:
    from benchmarks.realtime_robocasa.env import robocasa_bridge as bridge
    from benchmarks.realtime_robocasa.env.task_registry import default_instruction

    if int(k) <= int(warmup):
        raise ValueError(f"k={k} must be > warmup={warmup}")

    bridge.ensure_runtime_env()
    policy = make_policy_client("robocasa", policy_url=policy_url, codec=codec)
    instr = default_instruction(task_name)
    env = bridge.create_gym_env(
        task_name, split=split, seed=seed, robocasa_root=robocasa_root
    )
    lats: list[float] = []
    try:
        obs, _ = env.reset(seed=int(seed))
        instr = bridge.get_instruction(obs, fallback=instr)
        for _ in range(int(k)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            chunk = policy.predict(obs, instr)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lats.append(time.perf_counter() - t0)
            for a in chunk[: min(5, len(chunk))]:
                flat = np.asarray(a, dtype=np.float32).reshape(-1)
                if flat.size != 12:
                    raise ValueError(f"expected 12-D RoboCasa action, got {flat.shape}")
                obs, _, terminated, truncated, _ = env.step(bridge.flat_action_to_dict(flat))
                if terminated or truncated:
                    obs, _ = env.reset()
                    break
    finally:
        try:
            env.close()
        except Exception:
            pass
        policy.close()
    use = lats[int(warmup) :]
    if not use:
        raise RuntimeError("no post-warmup latency samples")
    return float(np.median(use)), lats


def measure_http_latency_robotwin(
    policy_url: str,
    task_name: str,
    instruction: str,
    *,
    robotwin_root,
    task_config: str = "demo_clean",
    seed: int = 0,
    k: int = 8,
    warmup: int = 3,
) -> tuple[float, list[float]]:
    from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge
    from benchmarks.dynamic_robotwin.env.dynamic_tasks import resolve_eval_spec

    if int(k) <= int(warmup):
        raise ValueError(f"k={k} must be > warmup={warmup}")

    spec = resolve_eval_spec(task_name)
    env_task = spec.task_name
    bridge.ensure_runtime_env()
    policy = make_policy_client("robotwin", policy_url=policy_url)
    args = bridge.load_task_args(robotwin_root, task_config=task_config, task_name=env_task)
    args["_robotwin_root"] = str(robotwin_root)
    env = bridge.create_task_env(env_task, robotwin_root)
    lats: list[float] = []
    try:
        bridge.setup_episode(env, args, seed=seed, instruction=instruction, ep_num=0)
        obs = env.get_obs()
        for _ in range(int(k)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            chunk = policy.predict(obs, instruction)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lats.append(time.perf_counter() - t0)
            for a in chunk[: min(4, len(chunk))]:
                try:
                    env.take_action(np.asarray(a, dtype=np.float32), action_type="qpos")
                except Exception:
                    # Some official envs raise in check_success (e.g. missing arm_tag).
                    # Keep latency samples; do not abort the whole paired run.
                    break
            try:
                obs = env.get_obs()
            except Exception:
                break
    finally:
        bridge.close_task_env(env)
        policy.close()
    use = lats[int(warmup) :]
    if not use:
        raise RuntimeError("no post-warmup latency samples")
    return float(np.median(use)), lats
