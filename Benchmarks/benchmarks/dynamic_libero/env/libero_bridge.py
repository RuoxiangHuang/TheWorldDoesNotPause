"""LIBERO env + policy bridge for Dynamic-LIBERO.

The public API is self-contained. Environment construction and obs→action
currently reuse FastWAM's battle-tested `eval_libero_single` helpers (optional
dependency path). The RealtimeDriver and trajectories live entirely in this
package and do **not** import FastWAM's `dynamic_libero_sr`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from benchmarks.repo_paths import checkpoints_root, fastwam_root, repo_root

LEGACY_FASTWAM_ROOT = Path(os.environ.get("FASTWAM_ROOT", "/DATA/YuanZhen/FastWAM"))


def ensure_runtime_env() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not os.environ.get("DIFFSYNTH_MODEL_BASE_PATH"):
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(checkpoints_root())


def _ensure_fastwam_on_path() -> None:
    fw = fastwam_root()
    for p in (
        str(fw / "src"),
        str(fw),
        str(repo_root() / "Benchmarks"),
        str(LEGACY_FASTWAM_ROOT),
        str(LEGACY_FASTWAM_ROOT / "experiments" / "speedup"),
        str(LEGACY_FASTWAM_ROOT / "experiments" / "libero"),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


def _import_eval_libero_single():
    """Import FastWAM eval helpers; tolerate already-registered OmegaConf resolvers."""
    from omegaconf import OmegaConf

    _ensure_fastwam_on_path()
    # FasterWAM already registers `eval`; FastWAM's module re-registers without replace.
    _orig = OmegaConf.register_new_resolver

    def _register(name, resolver, *args, replace=False, **kwargs):
        return _orig(name, resolver, *args, replace=True, **kwargs)

    OmegaConf.register_new_resolver = _register  # type: ignore[assignment]
    try:
        import experiments.libero.eval_libero_single as E
    finally:
        OmegaConf.register_new_resolver = _orig  # type: ignore[assignment]
    return E


class _CfgNode:
    """Attribute + ``.get()`` bag. Stands in for OmegaConf on the native (no-FastWAM) path."""

    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class EpisodeContext:
    model: Any
    processor: Any
    cfg: Any
    action_horizon: int
    vw: int
    vh: int
    device: str
    task_suite: Any
    control_freq: int = 20


NATIVE_CONTROL_FREQ = 20
NATIVE_REPLAN_STEPS = 10
NATIVE_NUM_STEPS_WAIT = 30
NATIVE_ACTION_HORIZON = 10
NATIVE_VIDEO_SIZE = 224


def load_eval_cfg(
    task: str,
    ckpt: str,
    stats: str,
    task_suite_name: str,
    task_id: int,
    seed: int = 7,
    num_inference_steps: Optional[int] = None,
):
    """Load FastWAM sim_libero cfg (EVALUATION block) with FasterWAM overrides."""
    from omegaconf import OmegaConf

    _ensure_fastwam_on_path()
    from common import model_loader as fw_loader

    cfg = fw_loader.load_cfg(task, ckpt, config_name="sim_libero")
    OmegaConf.set_struct(cfg, False)
    cfg.seed = seed
    cfg.EVALUATION.task_suite_name = task_suite_name
    cfg.EVALUATION.task_id = task_id
    cfg.EVALUATION.dataset_stats_path = stats
    cfg.EVALUATION.output_dir = "/tmp/dynamic_libero"
    if num_inference_steps is not None:
        cfg.EVALUATION.num_inference_steps = int(num_inference_steps)
    return cfg


def load_fasterwam_policy(ckpt: str, task: str, device: str = "cuda"):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
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
    model, _ = prepare_for_inference(model, checkpoint_path=ckpt, accel={"enabled": False})
    return model


def install_accel(model, preset: str, task: str):
    from fasterwam.accel import AccelConfig, AccelStack, get_stack

    raw = str(preset or "off")
    existing = get_stack(model)
    if existing is not None:
        existing.uninstall()
    if raw == "off":
        cfg = {"enabled": False}
    else:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from fasterwam.utils.config_resolvers import register_default_resolvers

        register_default_resolvers()
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(fastwam_root() / "configs"), version_base="1.3"):
            train_cfg = compose(
                config_name="train",
                overrides=[f"task={task}", f"accel={raw}", "model.load_text_encoder=true"],
            )
        cfg = AccelConfig.from_any(train_cfg.model.accel, apply_env=False).to_dict()
    stack = AccelStack(model, AccelConfig.from_any(cfg, apply_env=False)).install()
    if raw in {"ours", "pace"}:
        from fasterwam.accel import require_pace_stack

        require_pace_stack(stack)
    return stack


def build_episode_context(
    task: str,
    ckpt: str,
    stats: str,
    task_suite_name: str,
    task_id: int,
    device: str = "cuda",
    seed: int = 7,
) -> EpisodeContext:
    from hydra.utils import instantiate

    _ensure_fastwam_on_path()
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from libero.libero import benchmark

    cfg = load_eval_cfg(task, ckpt, stats, task_suite_name, task_id, seed=seed)
    model = load_fasterwam_policy(ckpt, task, device=device)
    dataset_stats = load_dataset_stats_from_json(stats)
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    action_horizon = int(cfg.data.train.num_frames) - 1
    vh, vw = (int(x) for x in cfg.data.train.get("video_size", [224, 224]))
    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    return EpisodeContext(
        model=model,
        processor=processor,
        cfg=cfg,
        action_horizon=action_horizon,
        vw=vw,
        vh=vh,
        device=device,
        task_suite=suite,
        control_freq=int(cfg.EVALUATION.get("control_freq", 20)),
    )


def _get_libero_env_native(task, resolution, seed):
    """Create LIBERO env without importing FastWAM eval helpers."""
    import pathlib

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _dummy_action_native():
    return [0, 0, 0, 0, 0, 0, -1]


LIBERO_ENV_RESOLUTION = 256


def get_libero_env(task, resolution, seed):
    """Create a plain LIBERO env (no FastWAM dynamic monkeypatches)."""
    try:
        return _get_libero_env_native(task, resolution, seed)
    except Exception:
        E = _import_eval_libero_single()
        return E.get_libero_env(task, resolution, seed)


def predict_action_chunk(obs, task_description, ctx: EpisodeContext):
    E = _import_eval_libero_single()
    return E._predict_action_chunk(
        obs=obs,
        task_description=task_description,
        model=ctx.model,
        processor=ctx.processor,
        cfg=ctx.cfg,
        action_horizon=ctx.action_horizon,
        input_w=ctx.vw,
        input_h=ctx.vh,
        model_device=ctx.device,
    )


def dummy_action():
    try:
        return _dummy_action_native()
    except Exception:
        E = _import_eval_libero_single()
        return E.get_libero_dummy_action()


def get_env_resolution():
    return LIBERO_ENV_RESOLUTION


DEFAULT_TASK = "libero_uncond_2cam224_1e-4"
DEFAULT_CKPT = str(checkpoints_root() / "fastwam_release/libero_uncond_2cam224.pt")
DEFAULT_STATS = str(
    checkpoints_root() / "fastwam_release/libero_uncond_2cam224_dataset_stats.json"
)


def load_native_env_context(
    task_suite_name: str = "libero_object",
    task_id: int = 0,
    seed: int = 0,
    *,
    device: str = "cpu",
) -> EpisodeContext:
    """LIBERO suite + protocol knobs. No FastWAM hydra / processor / GPU model."""
    ensure_runtime_env()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    cfg = _CfgNode(
        seed=int(seed),
        EVALUATION=_CfgNode(
            task_suite_name=task_suite_name,
            task_id=int(task_id),
            control_freq=NATIVE_CONTROL_FREQ,
            replan_steps=NATIVE_REPLAN_STEPS,
            num_steps_wait=NATIVE_NUM_STEPS_WAIT,
        ),
    )
    return EpisodeContext(
        model=None,
        processor=None,
        cfg=cfg,
        action_horizon=NATIVE_ACTION_HORIZON,
        vw=NATIVE_VIDEO_SIZE,
        vh=NATIVE_VIDEO_SIZE,
        device=device,
        task_suite=suite,
        control_freq=NATIVE_CONTROL_FREQ,
    )


def load_env_context(
    stats: Optional[str] = None,
    task: str = DEFAULT_TASK,
    task_suite_name: str = "libero_object",
    task_id: int = 0,
    seed: int = 0,
    num_inference_steps: Optional[int] = 4,
) -> EpisodeContext:
    """Env-only context for in-process OpenPI / HTTP clients (no FastWAM hydra)."""
    del stats, task, num_inference_steps  # FastWAM-only knobs; native path ignores them
    return load_native_env_context(
        task_suite_name=task_suite_name, task_id=task_id, seed=seed
    )


def load_policy(
    checkpoint: str = DEFAULT_CKPT,
    stats: Optional[str] = None,
    task: str = DEFAULT_TASK,
    task_suite_name: str = "libero_object",
    task_id: int = 0,
    seed: int = 0,
    device: str = "cuda:0",
    num_inference_steps: Optional[int] = 4,
    accel: str = "off",
) -> EpisodeContext:
    """Load FasterWAM + processor + LIBERO suite and install an accel preset."""
    ensure_runtime_env()
    stats = stats or DEFAULT_STATS
    ctx = build_episode_context(
        task=task,
        ckpt=checkpoint,
        stats=stats,
        task_suite_name=task_suite_name,
        task_id=task_id,
        device=device,
        seed=seed,
    )
    if num_inference_steps is not None:
        ctx.cfg.EVALUATION.num_inference_steps = int(num_inference_steps)
    install_accel(ctx.model, accel, task)
    return ctx
