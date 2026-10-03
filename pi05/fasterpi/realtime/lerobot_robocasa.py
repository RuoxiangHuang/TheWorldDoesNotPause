"""LeRobot π₀.₅ RoboCasa policy wrapper for Dynamic-RoboCasa (scheme A).

Loads community checkpoints such as ``pepijn223/pi05_robocasa_full`` and exposes
a ``predict(obs, instruction) -> (T, 12)`` API compatible with
``realtime_common`` HTTP / in-process clients.

Acceleration presets for scheme A:
  - ``eager``: plain PyTorch inference
  - ``compile``: ``torch.compile`` on the inner model (reduce-overhead)

Known env quirks (openwam / lerobot 0.5.2):
  - transformers 5.x ``create_causal_mask`` may reject kwargs lerobot still passes
    → monkeypatch ``masking_utils`` and ``lerobot.policies.pi_gemma``.
  - Some stacks expect flat ``vision_tower.*`` while the pepijn ckpt uses
    ``vision_tower.vision_model.*`` → adaptive remap when loading weights.
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# RoboCasa365 / pepijn LeRobot layout (matches starVLA + convert_hdf5_lerobot).
STATE_KEYS = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)
IMAGE_KEYS = (
    ("observation.images.robot0_agentview_left", "video.robot0_agentview_left"),
    ("observation.images.robot0_agentview_right", "video.robot0_agentview_right"),
    ("observation.images.robot0_eye_in_hand", "video.robot0_eye_in_hand"),
)

# pepijn / official LeRobot RoboCasa flat action (see lerobot.envs.robocasa.convert_action):
#   base_motion(4) + control_mode(1) + eef_pos(3) + eef_rot(3) + gripper(1)
# Dynamic-RoboCasa bridge / starVLA / RoboCasa HDF5 convert_action expect:
#   eef_pos(3) + eef_rot(3) + gripper(1) + base_motion(4) + control_mode(1)
ACTION_LAYOUT_LEROBOT = "lerobot_base_first"
ACTION_LAYOUT_BRIDGE = "bridge_eef_first"


def remap_lerobot_action_to_bridge12(actions: np.ndarray) -> np.ndarray:
    """Reorder pepijn/LeRobot 12-D actions into bridge/starVLA eef-first layout."""
    arr = np.asarray(actions, dtype=np.float32)
    if arr.shape[-1] < 12:
        raise ValueError(f"expected trailing dim >= 12, got {arr.shape}")
    out = np.empty(arr.shape[:-1] + (12,), dtype=np.float32)
    out[..., 0:3] = arr[..., 5:8]  # eef_pos
    out[..., 3:6] = arr[..., 8:11]  # eef_rot
    out[..., 6:7] = arr[..., 11:12]  # gripper
    out[..., 7:11] = arr[..., 0:4]  # base_motion
    out[..., 11:12] = arr[..., 4:5]  # control_mode
    return out


_CAUSAL_MASK_PATCHED = False


def _patch_create_causal_mask() -> None:
    """Drop unsupported kwargs for transformers 5.x create_causal_mask."""
    global _CAUSAL_MASK_PATCHED
    if _CAUSAL_MASK_PATCHED:
        return
    import transformers.masking_utils as mu

    orig = mu.create_causal_mask
    allowed = set(inspect.signature(orig).parameters)

    def _patched(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return orig(*args, **kwargs)

    mu.create_causal_mask = _patched
    try:
        import lerobot.policies.pi_gemma as pg

        pg.create_causal_mask = _patched
    except Exception:
        pass
    _CAUSAL_MASK_PATCHED = True


def _remap_state_dict(
    sd: dict[str, torch.Tensor],
    *,
    model_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Adapt vision_tower nesting + ensure ``model.`` prefix."""
    sample = next(
        (k for k in model_keys if "vision_tower" in k and "patch_embedding" in k),
        "",
    )
    want_flat = bool(sample) and "vision_tower.vision_model." not in sample
    fixed: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        if want_flat:
            nk = nk.replace("vision_tower.vision_model.", "vision_tower.")
        if not nk.startswith("model."):
            nk = "model." + nk
        fixed[nk] = v
    return fixed


def _to_chw_float01(img: Any) -> torch.Tensor:
    arr = np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected HxWxC or CxHxW image, got {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        chw = arr
    else:
        chw = np.transpose(arr, (2, 0, 1))
    t = torch.from_numpy(np.ascontiguousarray(chw))
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    else:
        t = t.float()
        if float(t.max()) > 1.5:
            t = t / 255.0
    return t


def _pack_state(obs: dict) -> torch.Tensor:
    parts = []
    for k in STATE_KEYS:
        if k not in obs:
            raise KeyError(f"RoboCasa obs missing state key {k!r}; have={sorted(obs)[:20]}")
        v = np.asarray(obs[k], dtype=np.float32).reshape(-1)
        parts.append(v)
    state = np.concatenate(parts, axis=0)
    if state.shape[0] != 16:
        raise ValueError(f"expected 16-D state, got {state.shape}")
    return torch.from_numpy(state.copy())


def robocasa_obs_to_lerobot_batch(obs: dict, instruction: str) -> dict[str, Any]:
    """Map RoboCasa gym obs → LeRobot PI05 preprocessor batch (no batch dim)."""
    batch: dict[str, Any] = {"task": str(instruction), "observation.state": _pack_state(obs)}
    for dst, src in IMAGE_KEYS:
        if src not in obs:
            raise KeyError(f"RoboCasa obs missing image key {src!r}")
        batch[dst] = _to_chw_float01(obs[src])
    return batch


@dataclass
class LeRobotRoboCasaPolicy:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: str = "cuda"
    replan_steps: int = 10
    label: str = "lerobot_pi05_robocasa"
    _compiled: bool = False

    def predict(self, obs: Any, instruction: str, **kwargs: Any) -> np.ndarray:
        del kwargs
        batch = robocasa_obs_to_lerobot_batch(obs, instruction)
        proc = self.preprocessor(batch)
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(proc)
            # postprocessor expects PolicyAction (= torch.Tensor), not a dict.
            acts = self.postprocessor(actions)
        if isinstance(acts, dict):
            acts = acts.get("action", acts)
        arr = acts.detach().float().cpu().numpy() if torch.is_tensor(acts) else np.asarray(acts)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[None, :]
        n = min(int(self.replan_steps), int(arr.shape[0]))
        # Model emits LeRobot base-first 12-D; bridge.flat_action_to_dict is eef-first.
        return remap_lerobot_action_to_bridge12(arr[:n, :12])

    def close(self) -> None:
        return


def _apply_accel(policy: Any, accel: str) -> str:
    accel = (accel or "eager").strip().lower()
    if accel in ("", "eager", "off", "none"):
        return "eager"
    if accel in ("compile", "torch.compile", "builtin"):
        target = getattr(policy, "model", policy)
        policy.model = torch.compile(target, mode="reduce-overhead")
        return "compile"
    raise ValueError(f"unsupported scheme-A accel {accel!r}; use eager|compile")


def _point_preprocessor_at_local_tokenizer(ckpt: Path, tok: Path) -> None:
    if not tok.is_dir():
        return
    pre_json = ckpt / "policy_preprocessor.json"
    if not pre_json.is_file():
        return
    data = json.loads(pre_json.read_text())
    changed = False
    for step in data.get("steps", []):
        if step.get("registry_name") == "tokenizer_processor":
            cfg = step.setdefault("config", {})
            if cfg.get("tokenizer_name") != str(tok):
                cfg["tokenizer_name"] = str(tok)
                changed = True
    if changed:
        pre_json.write_text(json.dumps(data, indent=2) + "\n")


def _load_weights_with_remap(pol: Any, ckpt: Path) -> None:
    from safetensors.torch import load_file

    weight_path = ckpt / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"missing weights at {weight_path}")
    raw = load_file(str(weight_path))
    fixed = _remap_state_dict(raw, model_keys=set(pol.state_dict().keys()))
    missing, unexpected = pol.load_state_dict(fixed, strict=False)
    vision_miss = [k for k in missing if "vision_tower" in k]
    if vision_miss or unexpected:
        raise RuntimeError(
            f"state_dict mismatch after vision remap: missing={len(missing)} "
            f"(vision_miss={len(vision_miss)}) unexpected={len(unexpected)} "
            f"miss_sample={missing[:3]} unexp_sample={unexpected[:3]}"
        )


def load_lerobot_robocasa_policy(
    *,
    ckpt: str | Path,
    device: str = "cuda",
    accel: str = "eager",
    replan_steps: int = 10,
    local_files_only: bool = True,
    tokenizer_path: str | Path | None = None,
) -> LeRobotRoboCasaPolicy:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _patch_create_causal_mask()

    ckpt = Path(ckpt)
    tok = Path(
        tokenizer_path
        or os.environ.get("PALIGEMMA_TOKENIZER", "")
        or (ckpt.parent / "paligemma_tokenizer")
    )
    _point_preprocessor_at_local_tokenizer(ckpt, tok)

    # Prefer empty construct + remapped load so vision keys are verified.
    cfg = PreTrainedConfig.from_pretrained(str(ckpt), local_files_only=local_files_only)
    pol = PI05Policy(cfg)
    _load_weights_with_remap(pol, ckpt)
    pol.to(device).eval()

    pre_overrides = None
    if tok.is_dir():
        pre_overrides = {"tokenizer_processor": {"tokenizer_name": str(tok)}}
    pre, post = make_pre_post_processors(
        pol.config,
        pretrained_path=str(ckpt),
        preprocessor_overrides=pre_overrides,
    )
    label = _apply_accel(pol, accel)
    return LeRobotRoboCasaPolicy(
        policy=pol,
        preprocessor=pre,
        postprocessor=post,
        device=device,
        replan_steps=replan_steps,
        label=label,
        _compiled=(label == "compile"),
    )


def warmup_lerobot_policy(rt: LeRobotRoboCasaPolicy, *, n: int = 3) -> None:
    dummy = {
        "video.robot0_agentview_left": np.zeros((256, 256, 3), dtype=np.uint8),
        "video.robot0_agentview_right": np.zeros((256, 256, 3), dtype=np.uint8),
        "video.robot0_eye_in_hand": np.zeros((256, 256, 3), dtype=np.uint8),
        "state.base_position": np.zeros(3, dtype=np.float32),
        "state.base_rotation": np.array([0, 0, 0, 1], dtype=np.float32),
        "state.end_effector_position_relative": np.zeros(3, dtype=np.float32),
        "state.end_effector_rotation_relative": np.array([0, 0, 0, 1], dtype=np.float32),
        "state.gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    for _ in range(n):
        rt.predict(dummy, "warmup pick and place")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
