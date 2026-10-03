"""Assembly of the acceleration controllers onto a live model.

Install order is not a preference, it is a correctness constraint, so it lives
here rather than at the call site — see `AccelStack.INSTALL_ORDER`.
"""

from __future__ import annotations

from typing import Any, Iterator

from ..utils.logging_config import get_logger
from .base import Controller
from .chunk_residual_cache import ChunkResidualCacheController
from .compile_ctrl import CompileController
from .config import AccelConfig
from .obs_pipeline import ObsPipelineController
from .step_cache import StepCacheController
from .text_cache import TextCacheController
from .vae_cache import VaeCacheController
from .video_token_cache import VideoTokenCacheController
from .world_state import clear_video_token_state, ensure_video_token_state

logger = get_logger(__name__)

MODEL_ATTR = "_fasterwam_accel"
_VIDEO_TOKEN_STATE_CONTROLLERS = frozenset({"video_token_cache", "chunk_residual_cache"})


class AccelStack:
    """The installed set of controllers for one model instance."""

    # `compile` must wrap the real model forwards before any Python gate.
    # Encoder caches (video_token_cache, vae_cache) and obs_pipeline sit outside
    # compile so skip / refresh decisions do not invalidate CUDA graphs.
    # `obs_pipeline` wraps encode outermost so a prefetch hit short-circuits both
    # vae_cache and the real VAE.
    # `chunk_residual_cache` wraps predict outside compile so residual hits never
    # enter the CUDA-graph path; `step_cache` is last so held denoise steps
    # never reach residual read/write.
    INSTALL_ORDER = (
        "compile",
        "text_cache",
        "video_token_cache",
        "vae_cache",
        "obs_pipeline",
        "chunk_residual_cache",
        "step_cache",
    )

    _FACTORIES = {
        "compile": CompileController,
        "text_cache": TextCacheController,
        "video_token_cache": VideoTokenCacheController,
        "vae_cache": VaeCacheController,
        "obs_pipeline": ObsPipelineController,
        "chunk_residual_cache": ChunkResidualCacheController,
        "step_cache": StepCacheController,
    }

    def __init__(self, model: Any, config: AccelConfig | None = None):
        self.model = model
        self.config = config or AccelConfig()
        self.controllers: dict[str, Controller] = {}

    def install(self) -> "AccelStack":
        if self.controllers:
            return self
        if not self.config.enabled:
            logger.info("FasterWAM acceleration disabled; running the reference path.")
            return self

        existing = getattr(self.model, MODEL_ATTR, None)
        if existing is not None and existing is not self and existing.controllers:
            raise RuntimeError(
                "This model already has an acceleration stack installed. "
                "Call `.uninstall()` on it before installing another."
            )

        for name in self.INSTALL_ORDER:
            section = getattr(self.config, name)
            if not section.active:
                continue
            factory = self._FACTORIES[name]
            missing = factory.missing_requirements(self.model, section)
            if missing:
                logger.warning(
                    "Skipping %s: %s does not expose %s.",
                    name,
                    type(self.model).__name__,
                    ", ".join(missing),
                )
                continue
            if name in _VIDEO_TOKEN_STATE_CONTROLLERS:
                controller = factory(
                    self.model,
                    section,
                    video_state=ensure_video_token_state(self.model),
                )
            else:
                controller = factory(self.model, section)
            controller.install()
            self.controllers[name] = controller

        setattr(self.model, MODEL_ATTR, self)
        active = ", ".join(sorted(self.controllers)) or "nothing"
        print(f"[fasterwam] stack active: {active}", flush=True)
        if self.config.verbose:
            print(f"[fasterwam] {self.config.summary()}", flush=True)
            logger.info("FasterWAM acceleration active: %s", active)
        return self

    def reset(self) -> None:
        """Drop cross-episode cached state on every installed controller.

        Call at each episode boundary. Approximate caches (video K/V, VAE,
        chunk residual, step hold, obs prefetch) must not see the previous
        episode's last replan; exact text memoization is left intact.
        """
        for name in reversed(self.INSTALL_ORDER):
            controller = self.controllers.get(name)
            if controller is not None:
                controller.reset()

    def uninstall(self) -> None:
        had_video_state = any(
            name in _VIDEO_TOKEN_STATE_CONTROLLERS for name in self.controllers
        )
        # Reverse order: the outermost patch must come off first.
        for name in reversed(self.INSTALL_ORDER):
            controller = self.controllers.pop(name, None)
            if controller is not None:
                controller.uninstall()
        if getattr(self.model, MODEL_ATTR, None) is self:
            try:
                delattr(self.model, MODEL_ATTR)
            except AttributeError:
                pass
        if had_video_state:
            clear_video_token_state(self.model)

    @property
    def active(self) -> bool:
        return bool(self.controllers)

    def stats(self) -> dict[str, Any]:
        return {name: c.stats() for name, c in self.controllers.items()}

    def __enter__(self) -> "AccelStack":
        return self.install()

    def __exit__(self, *exc: Any) -> None:
        self.uninstall()

    def __iter__(self) -> Iterator[Controller]:
        return iter(self.controllers.values())

    def __repr__(self) -> str:
        return f"<AccelStack {sorted(self.controllers)}>"


OURS_REQUIRED = ("compile", "text_cache", "step_cache")


def require_pace_stack(stack: AccelStack) -> None:
    """Fail closed unless FastWAM PACE installed compile plus the caches.

    ``accel=pace`` is the paper name. ``accel=ours`` is the locked alias.
    """
    have = set(stack.controllers)
    missing = [name for name in OURS_REQUIRED if name not in have]
    if missing:
        raise RuntimeError(
            "PACE (accel=pace / accel=ours) must install compile, text_cache, "
            f"and step_cache together. missing={missing} installed={sorted(have)}"
        )
    print(
        "[fasterwam] PACE locked: compile + text_cache + step_cache",
        flush=True,
    )


require_ours_stack = require_pace_stack


def accelerate(model: Any, config: Any = None, *, apply_env: bool = True) -> AccelStack:
    """Install the acceleration stack on `model` and return it.

    `config` may be None (defaults), a mapping / OmegaConf block, or an
    `AccelConfig`. Environment variables (`FASTERWAM_ACCEL*`) are applied on top
    unless `apply_env=False`.

    Acceleration targets the inference path only. Installing on a model in
    training mode is refused rather than silently corrupting gradients.
    """
    resolved = AccelConfig.from_any(config, apply_env=apply_env)
    if resolved.active and getattr(model, "training", False):
        raise RuntimeError(
            "Refusing to install the acceleration stack on a model in training mode. "
            "Call `model.eval()` first, or pass an accel config with `enabled: false`."
        )
    return AccelStack(model, resolved).install()


def get_stack(model: Any) -> AccelStack | None:
    """Return the stack installed on `model`, if any."""
    return getattr(model, MODEL_ATTR, None)


def reset_inference_state(model: Any) -> None:
    """Reset the installed accel stack, if any. Safe no-op when disabled."""
    stack = get_stack(model)
    if stack is not None:
        stack.reset()
