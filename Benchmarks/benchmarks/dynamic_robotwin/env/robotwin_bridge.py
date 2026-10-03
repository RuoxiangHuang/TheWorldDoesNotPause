"""FasterWAM + native RoboTwin bridge for Dynamic-RoboTwin.

Does **not** import DOMINO or call ``inject_latency_drift``. Env creation uses
RoboTwin ``demo_clean`` configs under ``ROBOTWIN_ROOT``.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from benchmarks.repo_paths import checkpoints_root, fastwam_root, repo_root

DEFAULT_ROBOTWIN_ROOT = Path(
    os.environ.get("ROBOTWIN_ROOT", "/DATA/YuanZhen/FastWAM/third_party/RoboTwin")
)
DEFAULT_TASK = "robotwin_uncond_3cam_384_1e-4"
DEFAULT_CKPT = str(checkpoints_root() / "fastwam_release/robotwin_uncond_3cam_384.pt")
DEFAULT_STATS = str(
    checkpoints_root() / "fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
)


def ensure_runtime_env() -> None:
    if not os.environ.get("DIFFSYNTH_MODEL_BASE_PATH"):
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(checkpoints_root())


def _ensure_paths(robotwin_root: Path) -> None:
    for p in (
        str(fastwam_root() / "src"),
        str(fastwam_root()),
        str(repo_root() / "Benchmarks"),
        str(robotwin_root),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)
    # RoboTwin scripts expect CWD-relative imports for description utils.
    desc = str(robotwin_root / "description" / "utils")
    if desc not in sys.path:
        sys.path.insert(0, desc)


@dataclass
class EpisodeContext:
    model: Any
    processor: Any
    action_horizon: int
    replan_steps: int
    num_inference_steps: int
    num_video_frames: int
    device: str
    policy_hz: float = 20.0
    seed: Optional[int] = None
    text_cfg_scale: float = 1.0
    negative_prompt: str = ""
    rand_device: str = "cpu"
    tiled: bool = False
    sigma_shift: Optional[float] = None
    task_cfg_name: str = DEFAULT_TASK
    _accepts_num_video_frames: bool = field(default=True, repr=False)


def install_accel(model, preset: str, task: str):
    from fasterwam.accel import AccelConfig, AccelStack, get_stack

    existing = get_stack(model)
    if existing is not None:
        existing.uninstall()
    if preset == "off":
        cfg = {"enabled": False}
    else:
        from fasterwam.utils.config_resolvers import register_default_resolvers

        register_default_resolvers()
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(fastwam_root() / "configs"), version_base="1.3"):
            train_cfg = compose(
                config_name="train",
                overrides=[f"task={task}", f"accel={preset}", "model.load_text_encoder=true"],
            )
        cfg = AccelConfig.from_any(train_cfg.model.accel, apply_env=False).to_dict()
    stack = AccelStack(model, AccelConfig.from_any(cfg, apply_env=False)).install()
    if str(preset) in {"ours", "pace"}:
        from fasterwam.accel import require_pace_stack

        require_pace_stack(stack)
    return stack


def load_policy(
    checkpoint: str = DEFAULT_CKPT,
    stats: Optional[str] = None,
    task: str = DEFAULT_TASK,
    device: str = "cuda:0",
    num_inference_steps: int = 4,
    replan_steps: int = 8,
    seed: int = 0,
    accel: str = "off",
    policy_hz: float = 20.0,
) -> EpisodeContext:
    """Load FasterWAM RoboTwin policy and install an accel preset."""
    ensure_runtime_env()
    stats = stats or DEFAULT_STATS
    from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fasterwam.runtime import prepare_for_inference
    from fasterwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(fastwam_root() / "configs"), version_base="1.3"):
        train_cfg = compose(
            config_name="train",
            overrides=[f"task={task}", "accel=off", "model.load_text_encoder=true"],
        )
    model = instantiate(train_cfg.model, model_dtype=torch.bfloat16, device=device)
    model, _ = prepare_for_inference(model, checkpoint_path=checkpoint, accel={"enabled": False})
    install_accel(model, accel, task)

    processor = instantiate(train_cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(stats))
    action_horizon = int(train_cfg.data.train.num_frames) - 1
    ratio = int(train_cfg.data.train.get("action_video_freq_ratio", 4))
    num_video_frames = (action_horizon // ratio) + 1
    accepts = "num_video_frames" in inspect.signature(model.infer_action).parameters
    return EpisodeContext(
        model=model,
        processor=processor,
        action_horizon=action_horizon,
        replan_steps=int(replan_steps),
        num_inference_steps=int(num_inference_steps),
        num_video_frames=num_video_frames,
        device=device,
        policy_hz=float(policy_hz),
        seed=seed,
        task_cfg_name=task,
        _accepts_num_video_frames=accepts,
    )


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(image.astype(np.uint8), mode="RGB").resize(size_wh, resample=Image.BILINEAR),
        dtype=np.uint8,
    )


def build_image_tensor(observation: dict, model) -> torch.Tensor:
    obs = observation["observation"]
    head = _resize_rgb(obs["head_camera"]["rgb"], (320, 256))
    left = _resize_rgb(obs["left_camera"]["rgb"], (160, 128))
    right = _resize_rgb(obs["right_camera"]["rgb"], (160, 128))
    image = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
        device=model.device, dtype=model.torch_dtype
    )
    return tensor * (2.0 / 255.0) - 1.0


def _normalize_state(processor, state: np.ndarray) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    key = state_meta[0]["key"]
    batch = {"state": {key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
    batch = processor.action_state_transform(batch)
    batch = processor.normalizer.forward(batch)
    return batch["state"][key]


def _denormalize_action(processor, action: torch.Tensor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    action_meta = processor.shape_meta["action"]
    normalizer = processor.normalizer.normalizers["action"][action_meta[0]["key"]]
    return normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()


@torch.no_grad()
def predict_action_chunk(observation: dict, instruction: str, ctx: EpisodeContext) -> np.ndarray:
    from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

    image = build_image_tensor(observation, ctx.model)
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    proprio = _normalize_state(ctx.processor, state)
    kwargs = {
        "prompt": DEFAULT_PROMPT.format(task=instruction),
        "input_image": image,
        "action_horizon": ctx.action_horizon,
        "proprio": proprio,
        "negative_prompt": ctx.negative_prompt,
        "text_cfg_scale": ctx.text_cfg_scale,
        "num_inference_steps": ctx.num_inference_steps,
        "sigma_shift": ctx.sigma_shift,
        "seed": ctx.seed,
        "rand_device": ctx.rand_device,
        "tiled": ctx.tiled,
    }
    if ctx._accepts_num_video_frames:
        kwargs["num_video_frames"] = ctx.num_video_frames
    pred = ctx.model.infer_action(**kwargs)
    return _denormalize_action(ctx.processor, pred["action"])[0]


def resolve_robotwin_env_name(task_name: str) -> str:
    """Map catalog ids like ``place_phone_stand.both`` to the RoboTwin env class name.

    Bare whitelist names (``place_empty_cup``) are returned unchanged.
    """
    from benchmarks.dynamic_robotwin.env.dynamic_tasks import resolve_eval_spec

    return resolve_eval_spec(task_name).task_name


def load_task_args(
    robotwin_root: Path | str = DEFAULT_ROBOTWIN_ROOT,
    task_config: str = "demo_clean",
    task_name: str = "place_empty_cup",
) -> dict[str, Any]:
    """Load and resolve RoboTwin task YAML (embodiment / camera paths)."""
    env_name = resolve_robotwin_env_name(task_name)
    root = Path(robotwin_root).resolve()
    _ensure_paths(root)
    cwd = os.getcwd()
    try:
        os.chdir(root)
        from envs import CONFIGS_PATH

        with open(root / "task_config" / f"{task_config}.yml", "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        def get_embodiment_file(emb_type):
            robot_file = embodiment_types[emb_type]["file_path"]
            if robot_file is None:
                raise RuntimeError(f"No embodiment file for {emb_type}")
            return robot_file

        camera_cfg_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")
        with open(camera_cfg_path, "r", encoding="utf-8") as f:
            camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        emb = args.get("embodiment")
        if len(emb) == 1:
            args["left_robot_file"] = get_embodiment_file(emb[0])
            args["right_robot_file"] = get_embodiment_file(emb[0])
            args["dual_arm_embodied"] = True
        elif len(emb) == 3:
            args["left_robot_file"] = get_embodiment_file(emb[0])
            args["right_robot_file"] = get_embodiment_file(emb[1])
            args["embodiment_dis"] = emb[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be 1 or 3")

        with open(os.path.join(args["left_robot_file"], "config.yml"), "r", encoding="utf-8") as f:
            args["left_embodiment_config"] = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(os.path.join(args["right_robot_file"], "config.yml"), "r", encoding="utf-8") as f:
            args["right_embodiment_config"] = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = camera_config[head_camera_type]["h"]
        args["head_camera_w"] = camera_config[head_camera_type]["w"]
        args["task_name"] = env_name
        args["task_config"] = task_config
        args["eval_mode"] = True
        args["eval_video_log"] = False
        args["render_freq"] = 0
        return args
    finally:
        os.chdir(cwd)


def create_task_env(task_name: str, robotwin_root: Path | str = DEFAULT_ROBOTWIN_ROOT):
    env_name = resolve_robotwin_env_name(task_name)
    root = Path(robotwin_root).resolve()
    _ensure_paths(root)
    cwd = os.getcwd()
    try:
        os.chdir(root)
        mod = importlib.import_module(f"envs.{env_name}")
        cls = getattr(mod, env_name)
        return cls()
    finally:
        os.chdir(cwd)


_PICK_ACTOR_ATTRS = (
    "object",
    "can",
    "cup",
    "phone",
    "stapler",
    "mouse",
    "fan",
    "pillbottle",
    "bread",
    "container",
)


def ensure_play_once_eval_attrs(task_env) -> None:
    """Fill fields that official ``check_success`` reads from ``play_once``.

    Policy eval never calls ``play_once``. ``place_object_scale`` (and a few
    siblings) set ``arm_tag`` only there, so ``check_success`` raises
    ``AttributeError`` and the episode is scored as an error. Use the same
    spawn-side rule as the demo: right arm if the pick object is at +x.
    """
    if getattr(task_env, "arm_tag", None) is not None:
        return
    actor = None
    for name in _PICK_ACTOR_ATTRS:
        actor = getattr(task_env, name, None)
        if actor is not None:
            break
    side = "right"
    if actor is not None:
        try:
            side = "right" if float(actor.get_pose().p[0]) > 0.0 else "left"
        except Exception:
            side = "right"
    arm = side
    try:
        from envs.utils import ArmTag

        arm = ArmTag(side)
    except Exception:
        pass
    task_env.arm_tag = arm


def setup_episode(
    task_env,
    args: dict[str, Any],
    seed: int,
    instruction: str,
    ep_num: int = 0,
):
    """Call setup_demo + set instruction. Must be run with CWD=robotwin_root or paths set."""
    root = Path(args.get("_robotwin_root", DEFAULT_ROBOTWIN_ROOT)).resolve()
    _ensure_paths(root)
    cwd = os.getcwd()
    try:
        os.chdir(root)
        task_env.setup_demo(now_ep_num=ep_num, seed=int(seed), is_test=True, **args)
        task_env.set_instruction(instruction=instruction)
        ensure_play_once_eval_attrs(task_env)
        return task_env
    finally:
        os.chdir(cwd)


def close_task_env(task_env) -> None:
    try:
        task_env.close_env(clear_cache=True)
    except Exception:
        try:
            task_env.close()
        except Exception:
            pass
