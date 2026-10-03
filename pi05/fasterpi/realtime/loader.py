"""Load openpi policies with FasterPI stacks for Real-Time benchmarks."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

import fasterpi.stack as A
from fasterpi.realtime.adapters import libero_obs_to_openpi, robotwin_obs_to_openpi


def _batch_to_torch(obj: Any, device: str):
    if isinstance(obj, dict):
        return {k: _batch_to_torch(v, device) for k, v in obj.items()}
    t = obj if isinstance(obj, torch.Tensor) else torch.as_tensor(obj)
    if t.device.type != "cuda":
        t = t.to(device, non_blocking=True)
    return t.unsqueeze(0)


def _coerce_world_state(ws: Any, driver: Any, obs: Any) -> Any:
    from fasterpi.world_state import WorldState

    if ws is None and driver is not None:
        return WorldState.from_driver(driver, obs)
    if isinstance(ws, dict):
        keys = (
            "phase",
            "object_speed",
            "object_disp",
            "eef_dist",
            "engage_radius",
            "released",
            "control_hz",
        )
        return WorldState(**{k: ws[k] for k in keys if k in ws})
    return ws


def _apply_num_steps(policy: Any, num_steps: int | None) -> None:
    if num_steps is None:
        return
    n = int(num_steps)
    policy._sample_kwargs = dict(getattr(policy, "_sample_kwargs", None) or {})
    policy._sample_kwargs["num_steps"] = n


def _infer_actions_fast(policy: Any, element: dict) -> np.ndarray:
    """Skip jax.tree + extra numpy copies in openpi Policy.infer."""
    from openpi.models import model as _model_mod

    inputs = policy._input_transform(element)
    device = policy._pytorch_device
    batched = _batch_to_torch(inputs, device)
    observation = _model_mod.Observation.from_dict(batched)
    sample_kw = dict(getattr(policy, "_sample_kwargs", None) or {})
    actions = policy._sample_actions(device, observation, **sample_kw)
    acts = np.asarray(actions[0].detach().cpu(), dtype=np.float32)
    state = inputs["state"]
    if isinstance(state, torch.Tensor):
        state = np.asarray(state.detach().cpu())
    else:
        state = np.asarray(state)
    out = policy._output_transform({"state": state, "actions": acts})
    return np.asarray(out["actions"], dtype=np.float32)


def chunk_return_len(n_raw: int, replan_steps: int, keep_full_chunk: bool) -> int:
    """How many generated actions to hand the executor.

    Replan cadence stays ``replan_steps``. ``keep_full_chunk`` only decides
    whether leftover tail past that cadence is returned for async thinking.
    """
    n_raw = int(n_raw)
    if keep_full_chunk:
        return n_raw
    return min(int(replan_steps), n_raw)


@dataclass
class OpenPIRealtimePolicy:
    """PolicyClient-compatible wrapper around openpi ``Policy.infer``."""

    policy: Any
    benchmark: str
    replan_steps: int = 10
    resize: int = 224
    label: str = "openpi"
    keep_full_chunk: bool = False
    num_steps: int = 10

    def predict(self, obs: Any, instruction: str, **kwargs: Any) -> np.ndarray:
        from fasterpi.world_state import world_scope

        ws = _coerce_world_state(kwargs.get("world_state"), kwargs.get("driver"), obs)
        if self.benchmark == "libero":
            host = bool(getattr(self.policy._model, "_fasterpi_host", False))
            element = libero_obs_to_openpi(
                obs, instruction, resize=self.resize, skip_resize=host
            )
        elif self.benchmark == "robotwin":
            element = robotwin_obs_to_openpi(obs, instruction)
            host = bool(getattr(self.policy._model, "_fasterpi_host", False))
        else:
            raise ValueError(f"unsupported benchmark {self.benchmark!r}")
        with world_scope(ws):
            if host:
                actions = _infer_actions_fast(self.policy, element)
            else:
                actions = np.asarray(self.policy.infer(element)["actions"], dtype=np.float32)
        n_lock = int(self.replan_steps)
        model = getattr(self.policy, "_model", None)
        if (
            ws is not None
            and bool(getattr(model, "_fasterpi_adaptive_replan", False))
        ):
            n_lock = int(ws.recommended_replan_steps(locked=n_lock))
        n_raw = int(len(actions))
        n = chunk_return_len(n_raw, n_lock, bool(self.keep_full_chunk))
        self.last_n_raw = n_raw
        self.last_n_used = n
        logged = int(getattr(self, "_logged_horizon", 0))
        if logged < 3:
            print(
                f"[openpi] n_raw={n_raw} n_used={n} replan_steps={n_lock} "
                f"keep_full={bool(self.keep_full_chunk)} label={self.label}",
                flush=True,
            )
            self._logged_horizon = logged + 1
        return actions[:n]

    def reset(self) -> None:
        model = getattr(self.policy, "_model", None)
        fn = getattr(model, "_fasterpi_reset_caches", None)
        if callable(fn):
            fn()

    def close(self) -> None:
        return


def _parse_dirs(spec: str) -> list[str]:
    spec = A.resolve_speedup_spec(spec or "eager")
    if spec in ("", "none", "eager"):
        return []
    if spec == "builtin":
        return []  # keep openpi default compile
    # Accept ``compile,chunk_residual_cache,vision`` or ``compile+step_cache``.
    parts: list[str] = []
    for chunk in spec.replace("+", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def install_speedup(policy: Any, speedup_dirs: str) -> str:
    """Apply the π₀.₅ PACE stack; return a short mode label.

    ``pace`` / ``ours`` / ``fasterpi`` install compile + residual cache + step
    cache + compiled SigLIP
    (``compile+chunk_residual_cache+step_cache+compile_prefix``). ``baseline``
    restores eager Euler.
    """
    model = policy._model
    raw = (speedup_dirs or "eager").strip()
    key = raw.lower()
    spec = A.resolve_speedup_spec(raw)
    if spec in ("", "none", "eager"):
        A.restore_eager(model)
        _apply_pi_inference_mode(model, "baseline")
        policy._sample_actions = model.sample_actions
        return "baseline" if key == "baseline" else "eager"
    if spec == "builtin":
        return "builtin"
    dirs = _parse_dirs(spec)
    kw = {}
    if raw in A.WORLD_BLIND_CACHE_ALIASES:
        kw = dict(A.WORLD_BLIND_CACHE_KWARGS)
        print(f"[fasterpi] world-blind Cache profile spec={raw!r} dirs={dirs} {kw}", flush=True)
    elif raw in ("clira", "fasterpi_clira"):
        kw = dict(A.CLIRA_PREFIX_KWARGS)
        print(f"[fasterpi] CLIRA prefix-gate profile {kw}", flush=True)
    A.install_stack(model, dirs, policy=policy, **kw)
    if key in ("ours", "fasterpi", "pace"):
        required = ("compile", "chunk_residual_cache", "step_cache", "compile_prefix")
        missing = [name for name in required if name not in dirs]
        if missing:
            raise RuntimeError(
                "π PACE must include compile, chunk_residual_cache, step_cache, "
                "and compile_prefix. "
                f"missing={missing} dirs={dirs}"
            )
        print(
            "[fasterpi] PACE locked: compile+chunk_residual_cache+step_cache+compile_prefix "
            "(compile runs inside the residual expert body)",
            flush=True,
        )
    _apply_pi_inference_mode(model, "pace")
    return "+".join(dirs)


def _apply_pi_inference_mode(model: Any, mode: str) -> None:
    model._inference_mode = mode


def load_libero_policy(
    *,
    ckpt: str,
    norm_assets: str,
    norm_id: str = "physical-intelligence/libero",
    config_name: str = "pi05_libero",
    device: str = "cuda",
    speedup_dirs: str = "eager",
    num_steps: int = 10,
) -> OpenPIRealtimePolicy:
    from openpi.training import checkpoints as _checkpoints
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    cfg = _config.get_config(config_name)
    norm_stats = _checkpoints.load_norm_stats(norm_assets, norm_id)
    policy = _policy_config.create_trained_policy(
        cfg, ckpt, norm_stats=norm_stats, pytorch_device=device
    )
    label = install_speedup(policy, speedup_dirs)
    _apply_num_steps(policy, num_steps)
    return OpenPIRealtimePolicy(
        policy=policy,
        benchmark="libero",
        replan_steps=10,
        label=label if int(num_steps) == 10 else f"{label}_nfe{int(num_steps)}",
        num_steps=int(num_steps),
    )


def load_robotwin_policy(
    *,
    ckpt: str,
    norm_id: str | None = None,
    config_name: str = "pi05_aloha",
    adapt_to_pi: bool = True,
    action_horizon: int = 32,
    device: str = "cuda",
    speedup_dirs: str = "eager",
    replan_steps: int = 8,
    num_steps: int = 10,
) -> OpenPIRealtimePolicy:
    """Load a RoboTwin OpenPI policy.

    Default matches ``motus-robotics/pi0.5_robotwin2``: π₀.₅, action_horizon=32,
    ``adapt_to_pi=True``, norm stats under
    ``assets/pi0.5_clean_randomize_joint_training``.
    """
    from pathlib import Path

    from openpi.training import checkpoints as _checkpoints
    from openpi.training import config as _config
    from openpi.training.config import AssetsConfig, LeRobotAlohaDataConfig
    from openpi.policies import policy_config as _policy_config

    _patch_nonstrict_safetensors()
    asset_id = norm_id or "pi0.5_clean_randomize_joint_training"
    base = _config.get_config(config_name)
    model = dataclasses.replace(base.model, action_horizon=int(action_horizon))
    cfg = dataclasses.replace(
        base,
        model=model,
        data=LeRobotAlohaDataConfig(
            adapt_to_pi=adapt_to_pi,
            assets=AssetsConfig(asset_id=asset_id),
        ),
    )
    ckpt_path = Path(ckpt)
    assets_under_ckpt = ckpt_path / "assets"
    if (assets_under_ckpt / asset_id / "norm_stats.json").is_file():
        norm_stats = _checkpoints.load_norm_stats(str(assets_under_ckpt), asset_id)
    else:
        norm_stats = _checkpoints.load_norm_stats(str(ckpt_path), asset_id)
    policy = _policy_config.create_trained_policy(
        cfg, str(ckpt_path), norm_stats=norm_stats, pytorch_device=device
    )
    label = install_speedup(policy, speedup_dirs)
    _apply_num_steps(policy, num_steps)
    return OpenPIRealtimePolicy(
        policy=policy,
        benchmark="robotwin",
        replan_steps=replan_steps,
        label=label if int(num_steps) == 10 else f"{label}_nfe{int(num_steps)}",
        num_steps=int(num_steps),
    )


def _patch_nonstrict_safetensors() -> None:
    import safetensors.torch as st

    if getattr(st, "_fasterpi_nonstrict", False):
        return
    orig = st.load_model

    def load_model(model, filename, **kw):
        kw.setdefault("strict", False)
        return orig(model, filename, **kw)

    st.load_model = load_model
    st._fasterpi_nonstrict = True


def warmup_policy(rt_policy: OpenPIRealtimePolicy, *, n: int = 4) -> None:
    if rt_policy.benchmark == "libero":
        dummy_obs = {
            "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float64),
            "robot0_gripper_qpos": np.zeros(2),
        }
        for _ in range(n):
            rt_policy.predict(dummy_obs, "warmup")
    else:
        dummy = {
            "observation": {
                "head_camera": {"rgb": np.zeros((256, 256, 3), dtype=np.uint8)},
                "right_camera": {"rgb": np.zeros((256, 256, 3), dtype=np.uint8)},
                "left_camera": {"rgb": np.zeros((256, 256, 3), dtype=np.uint8)},
            },
            "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
        }
        for _ in range(n):
            rt_policy.predict(dummy, "warmup")
