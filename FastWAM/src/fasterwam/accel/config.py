"""Configuration for the FasterWAM inference-acceleration stack.

The stack has composable first-class controllers whose speedups multiply:

    text_cache            memoize the T5 prompt encode across replans   (exact)
    step_cache            hold the last velocity while the action latent (near-lossless)
                          has barely drifted -> fewer denoise steps
    compile               torch.compile the action step / video prefill  (lossless)
    video_token_cache     reuse / sparsely refresh video K/V across       (near-lossless)
                          consecutive replans
    vae_cache             reuse VAE latents when RGB barely moved         (near-lossless)
    obs_pipeline          async observation prefetch API for runners      (lossless)
    chunk_residual_cache  reuse DiT block residual R=h^L-h^0 across       (near-lossless)
                          consecutive chunks at the same denoise index

Defaults keep the measured FastWAM operating point (text + step + compile).
Named comparison modes: `accel=baseline` (off), `accel=pace` (paper name PACE;
`accel=ours` is the same alias of `default`).
`video_token_cache` is a core controller but off in `accel=default`; enable via
`accel=p0` (recommended encoder path), `accel=p1`, or `accel=full`.
`chunk_residual_cache` is off in `accel=default`; enable via `accel=action0` or
`accel=full`.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

ENV_PREFIX = "FASTERWAM_ACCEL"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"{name}: cannot interpret {value!r} as bool")


@dataclass
class CompileConfig:
    """torch.compile of the action denoise step and the video prefill.

    The action step pushes only ~32 tokens through a 30-layer 1B expert, which
    makes it launch-bound rather than compute-bound: eager spends ~69x the
    weight-bandwidth floor on kernel launch and Python dispatch. Compiling
    recovers that overhead losslessly.
    """

    enabled: bool = True
    mode: str = "reduce-overhead"
    predict: bool = True
    prefill: bool = True
    # Prefill is often gated by video_token_cache; CUDA graphs thrash when the
    # compiled prefill is only invoked intermittently. Override independently.
    prefill_mode: str | None = None
    # A data-dependent slice in `pre_dit` graph-breaks; without this the break
    # raises instead of falling back to eager for that one op.
    suppress_dynamo_errors: bool = True

    def __post_init__(self) -> None:
        self.enabled = _as_bool(self.enabled, "compile.enabled")
        self.predict = _as_bool(self.predict, "compile.predict")
        self.prefill = _as_bool(self.prefill, "compile.prefill")
        self.suppress_dynamo_errors = _as_bool(
            self.suppress_dynamo_errors, "compile.suppress_dynamo_errors"
        )
        allowed = ("reduce-overhead", "default", "max-autotune")
        if self.mode not in allowed:
            raise ValueError(f"compile.mode must be one of {allowed}, got {self.mode!r}")
        if self.prefill_mode is None:
            self.prefill_mode = self.mode
        if self.prefill_mode not in allowed:
            raise ValueError(
                f"compile.prefill_mode must be one of {allowed}, got {self.prefill_mode!r}"
            )

    @property
    def active(self) -> bool:
        return self.enabled and (self.predict or self.prefill)


@dataclass
class TextCacheConfig:
    """Memoize `encode_prompt` per prompt string.

    The task prompt is constant within an episode but re-encoded on every
    replan, so caching returns bit-identical embeddings.
    """

    enabled: bool = True
    max_entries: int = 64

    def __post_init__(self) -> None:
        self.enabled = _as_bool(self.enabled, "text_cache.enabled")
        self.max_entries = int(self.max_entries)
        if self.max_entries < 1:
            raise ValueError(f"text_cache.max_entries must be >= 1, got {self.max_entries}")

    @property
    def active(self) -> bool:
        return self.enabled


@dataclass
class StepCacheConfig:
    """Skip denoise steps whose velocity is still reusable.

    Gating is on the *input* action-latent drift rather than on the velocity
    itself: the scheduler updates the latent on every step, so the drift stays
    observable during a hold and the gate can react the instant the trajectory
    starts moving. An output-side velocity gate can only refresh on a compute
    and therefore runs stale exactly during the hold it is meant to guard.
    """

    enabled: bool = True
    # Cumulative relative-L1 input drift tolerated before recomputing.
    # 0.2-0.3 is the measured sweet spot (~50% of steps skipped).
    threshold: float = 0.20
    # Bounds how far the integration can run on one held velocity.
    max_consecutive_skips: int = 2
    # The early ODE steps have high curvature and must not be held.
    cold_steps: int = 2

    def __post_init__(self) -> None:
        self.enabled = _as_bool(self.enabled, "step_cache.enabled")
        self.threshold = float(self.threshold)
        self.max_consecutive_skips = int(self.max_consecutive_skips)
        self.cold_steps = int(self.cold_steps)
        if self.threshold < 0:
            raise ValueError(f"step_cache.threshold must be >= 0, got {self.threshold}")
        if self.max_consecutive_skips < 0:
            raise ValueError(
                f"step_cache.max_consecutive_skips must be >= 0, got {self.max_consecutive_skips}"
            )
        if self.cold_steps < 0:
            raise ValueError(f"step_cache.cold_steps must be >= 0, got {self.cold_steps}")

    @property
    def active(self) -> bool:
        return self.enabled and self.max_consecutive_skips > 0


@dataclass
class VideoTokenCacheConfig:
    """Reuse or sparsely refresh video K/V across consecutive replans.

    `full_reuse_threshold` gates a bit-identical cache hit on the previous
    prefill. Above that, tokens whose relative-L1 exceeds `token_threshold`
    are marked dynamic; if their fraction is at most `max_refresh_ratio`, the
    sparse MoT path recomputes only those tokens.
    """

    enabled: bool = False
    # Global relative-L1 on `video_tokens` below which the whole K/V is reused.
    full_reuse_threshold: float = 0.02
    # Per-token relative-L1 above which a token is treated as dynamic.
    token_threshold: float = 0.05
    # If more than this fraction of tokens are dynamic, fall back to full prefill.
    max_refresh_ratio: float = 0.40
    # First N replans always fully prefill (cold start / episode open).
    cold_replans: int = 1
    # Bound how many consecutive full-reuse hits are allowed before a refresh.
    max_consecutive_reuses: int = 4
    # Force a full prefill every N replans (0 disables).
    force_refresh_every: int = 8

    def __post_init__(self) -> None:
        self.enabled = _as_bool(self.enabled, "video_token_cache.enabled")
        self.full_reuse_threshold = float(self.full_reuse_threshold)
        self.token_threshold = float(self.token_threshold)
        self.max_refresh_ratio = float(self.max_refresh_ratio)
        self.cold_replans = int(self.cold_replans)
        self.max_consecutive_reuses = int(self.max_consecutive_reuses)
        self.force_refresh_every = int(self.force_refresh_every)
        if self.full_reuse_threshold < 0:
            raise ValueError(
                f"video_token_cache.full_reuse_threshold must be >= 0, got {self.full_reuse_threshold}"
            )
        if self.token_threshold < 0:
            raise ValueError(
                f"video_token_cache.token_threshold must be >= 0, got {self.token_threshold}"
            )
        if not 0.0 <= self.max_refresh_ratio <= 1.0:
            raise ValueError(
                f"video_token_cache.max_refresh_ratio must be in [0, 1], got {self.max_refresh_ratio}"
            )
        if self.cold_replans < 0:
            raise ValueError(
                f"video_token_cache.cold_replans must be >= 0, got {self.cold_replans}"
            )
        if self.max_consecutive_reuses < 0:
            raise ValueError(
                "video_token_cache.max_consecutive_reuses must be >= 0, "
                f"got {self.max_consecutive_reuses}"
            )
        if self.force_refresh_every < 0:
            raise ValueError(
                f"video_token_cache.force_refresh_every must be >= 0, got {self.force_refresh_every}"
            )

    @property
    def active(self) -> bool:
        return self.enabled


@dataclass
class VaeCacheConfig:
    """Reuse the previous VAE latent when the RGB observation barely moved."""

    enabled: bool = False
    threshold: float = 0.01
    cold_calls: int = 1
    max_consecutive_hits: int = 8
    force_refresh_every: int = 16

    def __post_init__(self) -> None:
        self.enabled = _as_bool(self.enabled, "vae_cache.enabled")
        self.threshold = float(self.threshold)
        self.cold_calls = int(self.cold_calls)
        self.max_consecutive_hits = int(self.max_consecutive_hits)
        self.force_refresh_every = int(self.force_refresh_every)
        if self.threshold < 0:
            raise ValueError(f"vae_cache.threshold must be >= 0, got {self.threshold}")
        if self.cold_calls < 0:
            raise ValueError(f"vae_cache.cold_calls must be >= 0, got {self.cold_calls}")
        if self.max_consecutive_hits < 0:
            raise ValueError(
                f"vae_cache.max_consecutive_hits must be >= 0, got {self.max_consecutive_hits}"
            )
        if self.force_refresh_every < 0:
            raise ValueError(
                f"vae_cache.force_refresh_every must be >= 0, got {self.force_refresh_every}"
            )

    @property
    def active(self) -> bool:
        return self.enabled


@dataclass
class ObsPipelineConfig:
    """Async observation prefetch API for overlapping encode with execution."""

    enabled: bool = False

    def __post_init__(self) -> None:
        self.enabled = _as_bool(self.enabled, "obs_pipeline.enabled")

    @property
    def active(self) -> bool:
        return self.enabled


@dataclass
class ChunkResidualCacheConfig:
    """Reuse DiT block residual R=h^L-h^0 across consecutive chunks at index k.

    `accel=action0` adds this cache on top of text, step, and compile.
    `accel=full` enables it together with the encoder caches.
    """

    enabled: bool = False
    # Condition-side relative L1 below which residual reuse is allowed.
    # Paper schedule: set very large (e.g. 1e9) so the gate never fires.
    condition_threshold: float = 0.05
    # Action-latent relative L1 vs the latent that produced R_k.
    action_latent_threshold: float = 0.15
    cold_chunks: int = 1
    # 0 = unlimited consecutive reuses (paper τ handles refresh instead).
    max_consecutive_reuses: int = 4
    force_refresh_every: int = 0  # 0 = off; paper τ when > 0
    # Only enable reuse for denoise steps with index >= warm_steps.
    warm_steps: int = 0
    # Inclusive last denoise index that may reuse R. -1 = no cap.
    # Paper Table 1/2: 6 → [0,6], 7 → [0,7] of a 10-step schedule.
    max_cache_step: int = -1
    # Always full-compute the last T steps of the current NFE.
    # 0 = off. RoboTwin eval is NFE=4, so k<=6 would otherwise cache every
    # step and drop the current frame from the action MoT. T=2 keeps the
    # large late flow steps on the live video while still reusing R_0, R_1.
    min_full_tail: int = 0

    def __post_init__(self) -> None:
        self.enabled = _as_bool(self.enabled, "chunk_residual_cache.enabled")
        self.condition_threshold = float(self.condition_threshold)
        self.action_latent_threshold = float(self.action_latent_threshold)
        self.cold_chunks = int(self.cold_chunks)
        self.max_consecutive_reuses = int(self.max_consecutive_reuses)
        self.force_refresh_every = int(self.force_refresh_every)
        self.warm_steps = int(self.warm_steps)
        self.max_cache_step = int(self.max_cache_step)
        self.min_full_tail = int(self.min_full_tail)
        if self.condition_threshold < 0:
            raise ValueError(
                f"chunk_residual_cache.condition_threshold must be >= 0, got {self.condition_threshold}"
            )
        if self.action_latent_threshold < 0:
            raise ValueError(
                "chunk_residual_cache.action_latent_threshold must be >= 0, "
                f"got {self.action_latent_threshold}"
            )
        if self.cold_chunks < 0:
            raise ValueError(
                f"chunk_residual_cache.cold_chunks must be >= 0, got {self.cold_chunks}"
            )
        if self.max_consecutive_reuses < 0:
            raise ValueError(
                "chunk_residual_cache.max_consecutive_reuses must be >= 0, "
                f"got {self.max_consecutive_reuses}"
            )
        if self.force_refresh_every < 0:
            raise ValueError(
                "chunk_residual_cache.force_refresh_every must be >= 0, "
                f"got {self.force_refresh_every}"
            )
        if self.warm_steps < 0:
            raise ValueError(
                f"chunk_residual_cache.warm_steps must be >= 0, got {self.warm_steps}"
            )
        if self.max_cache_step < -1:
            raise ValueError(
                f"chunk_residual_cache.max_cache_step must be >= -1, got {self.max_cache_step}"
            )
        if self.min_full_tail < 0:
            raise ValueError(
                f"chunk_residual_cache.min_full_tail must be >= 0, got {self.min_full_tail}"
            )

    @property
    def active(self) -> bool:
        return self.enabled


@dataclass
class AccelConfig:
    enabled: bool = True
    compile: CompileConfig = field(default_factory=CompileConfig)
    text_cache: TextCacheConfig = field(default_factory=TextCacheConfig)
    step_cache: StepCacheConfig = field(default_factory=StepCacheConfig)
    video_token_cache: VideoTokenCacheConfig = field(default_factory=VideoTokenCacheConfig)
    vae_cache: VaeCacheConfig = field(default_factory=VaeCacheConfig)
    obs_pipeline: ObsPipelineConfig = field(default_factory=ObsPipelineConfig)
    chunk_residual_cache: ChunkResidualCacheConfig = field(default_factory=ChunkResidualCacheConfig)
    verbose: bool = True

    _SECTIONS = (
        "compile",
        "text_cache",
        "step_cache",
        "video_token_cache",
        "vae_cache",
        "obs_pipeline",
        "chunk_residual_cache",
    )

    def __post_init__(self) -> None:
        self.enabled = _as_bool(self.enabled, "accel.enabled")
        self.verbose = _as_bool(self.verbose, "accel.verbose")
        for name, kind in (
            ("compile", CompileConfig),
            ("text_cache", TextCacheConfig),
            ("step_cache", StepCacheConfig),
            ("video_token_cache", VideoTokenCacheConfig),
            ("vae_cache", VaeCacheConfig),
            ("obs_pipeline", ObsPipelineConfig),
            ("chunk_residual_cache", ChunkResidualCacheConfig),
        ):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                setattr(self, name, kind(**dict(value)))
            elif not isinstance(value, kind):
                raise TypeError(f"accel.{name} must be a mapping or {kind.__name__}, got {type(value)}")
    @property
    def active(self) -> bool:
        """True when installing the stack would change anything."""
        return self.enabled and any(getattr(self, s).active for s in self._SECTIONS)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def disabled(cls) -> "AccelConfig":
        return cls(enabled=False)

    @classmethod
    def from_any(cls, value: Any = None, *, apply_env: bool = True) -> "AccelConfig":
        """Build a config from None / a mapping / an existing AccelConfig.

        Accepts OmegaConf containers so a hydra `accel:` block can be passed
        straight through. Environment variables are applied last so they can
        always override a config file at deployment time.
        """
        if isinstance(value, AccelConfig):
            # Deep-copy the sections: env overlays must not mutate the caller's
            # config, which is typically the one recorded on the model.
            cfg = cls(
                enabled=value.enabled,
                compile=CompileConfig(**asdict(value.compile)),
                text_cache=TextCacheConfig(**asdict(value.text_cache)),
                step_cache=StepCacheConfig(**asdict(value.step_cache)),
                video_token_cache=VideoTokenCacheConfig(**asdict(value.video_token_cache)),
                vae_cache=VaeCacheConfig(**asdict(value.vae_cache)),
                obs_pipeline=ObsPipelineConfig(**asdict(value.obs_pipeline)),
                chunk_residual_cache=ChunkResidualCacheConfig(
                    **asdict(value.chunk_residual_cache)
                ),
                verbose=value.verbose,
            )
        elif value is None:
            cfg = cls()
        else:
            cfg = cls(**_to_plain_dict(value))
        if apply_env:
            cfg.apply_env()
        return cfg

    def apply_env(self) -> "AccelConfig":
        """Overlay `FASTERWAM_ACCEL*` environment variables in place."""
        top = os.environ.get(ENV_PREFIX)
        if top is not None:
            self.enabled = _as_bool(top, ENV_PREFIX)
        for env_name, target, attr in (
            (f"{ENV_PREFIX}_VERBOSE", self, "verbose"),
            (f"{ENV_PREFIX}_COMPILE", self.compile, "enabled"),
            (f"{ENV_PREFIX}_COMPILE_PREDICT", self.compile, "predict"),
            (f"{ENV_PREFIX}_COMPILE_PREFILL", self.compile, "prefill"),
            (f"{ENV_PREFIX}_TEXT_CACHE", self.text_cache, "enabled"),
            (f"{ENV_PREFIX}_STEP_CACHE", self.step_cache, "enabled"),
            (f"{ENV_PREFIX}_VIDEO_TOKEN_CACHE", self.video_token_cache, "enabled"),
            (f"{ENV_PREFIX}_VAE_CACHE", self.vae_cache, "enabled"),
            (f"{ENV_PREFIX}_OBS_PIPELINE", self.obs_pipeline, "enabled"),
            (f"{ENV_PREFIX}_CHUNK_RESIDUAL_CACHE", self.chunk_residual_cache, "enabled"),
        ):
            raw = os.environ.get(env_name)
            if raw is not None:
                setattr(target, attr, _as_bool(raw, env_name))

        mode = os.environ.get(f"{ENV_PREFIX}_COMPILE_MODE")
        if mode is not None:
            self.compile.mode = mode
        for env_name, attr, cast in (
            (f"{ENV_PREFIX}_STEP_CACHE_THRESHOLD", "threshold", float),
            (f"{ENV_PREFIX}_STEP_CACHE_MAX_SKIP", "max_consecutive_skips", int),
            (f"{ENV_PREFIX}_STEP_CACHE_COLD_STEPS", "cold_steps", int),
        ):
            raw = os.environ.get(env_name)
            if raw is not None:
                setattr(self.step_cache, attr, cast(raw))
        for env_name, attr, cast in (
            (f"{ENV_PREFIX}_VIDEO_FULL_REUSE_THRESHOLD", "full_reuse_threshold", float),
            (f"{ENV_PREFIX}_VIDEO_TOKEN_THRESHOLD", "token_threshold", float),
            (f"{ENV_PREFIX}_VIDEO_MAX_REFRESH_RATIO", "max_refresh_ratio", float),
            (f"{ENV_PREFIX}_VIDEO_COLD_REPLANS", "cold_replans", int),
            (f"{ENV_PREFIX}_VIDEO_MAX_CONSECUTIVE_REUSES", "max_consecutive_reuses", int),
            (f"{ENV_PREFIX}_VIDEO_FORCE_REFRESH_EVERY", "force_refresh_every", int),
        ):
            raw = os.environ.get(env_name)
            if raw is not None:
                setattr(self.video_token_cache, attr, cast(raw))
        for env_name, attr, cast in (
            (f"{ENV_PREFIX}_VAE_CACHE_THRESHOLD", "threshold", float),
            (f"{ENV_PREFIX}_VAE_CACHE_COLD_CALLS", "cold_calls", int),
            (f"{ENV_PREFIX}_VAE_CACHE_MAX_HITS", "max_consecutive_hits", int),
            (f"{ENV_PREFIX}_VAE_CACHE_FORCE_REFRESH_EVERY", "force_refresh_every", int),
        ):
            raw = os.environ.get(env_name)
            if raw is not None:
                setattr(self.vae_cache, attr, cast(raw))
        for env_name, attr, cast in (
            (f"{ENV_PREFIX}_CHUNK_RESIDUAL_THRESHOLD", "condition_threshold", float),
            (f"{ENV_PREFIX}_CHUNK_RESIDUAL_ACTION_LATENT_THRESHOLD", "action_latent_threshold", float),
            (f"{ENV_PREFIX}_CHUNK_RESIDUAL_COLD_CHUNKS", "cold_chunks", int),
            (f"{ENV_PREFIX}_CHUNK_RESIDUAL_MAX_REUSES", "max_consecutive_reuses", int),
            (f"{ENV_PREFIX}_CHUNK_RESIDUAL_FORCE_REFRESH_EVERY", "force_refresh_every", int),
            (f"{ENV_PREFIX}_CHUNK_RESIDUAL_WARM_STEPS", "warm_steps", int),
            (f"{ENV_PREFIX}_CHUNK_RESIDUAL_MAX_CACHE_STEP", "max_cache_step", int),
            (f"{ENV_PREFIX}_CHUNK_RESIDUAL_MIN_FULL_TAIL", "min_full_tail", int),
        ):
            raw = os.environ.get(env_name)
            if raw is not None:
                setattr(self.chunk_residual_cache, attr, cast(raw))
        for section in self._SECTIONS:
            getattr(self, section).__post_init__()
        return self

    def summary(self) -> str:
        if not self.enabled:
            return "accel: disabled"
        parts = []
        if self.text_cache.active:
            parts.append("text_cache")
        if self.step_cache.active:
            parts.append(f"step_cache(th={self.step_cache.threshold:g})")
        if self.video_token_cache.active:
            parts.append(
                "video_token_cache("
                f"full={self.video_token_cache.full_reuse_threshold:g},"
                f"tok={self.video_token_cache.token_threshold:g})"
            )
        if self.vae_cache.active:
            parts.append(f"vae_cache(th={self.vae_cache.threshold:g})")
        if self.obs_pipeline.active:
            parts.append("obs_pipeline")
        if self.chunk_residual_cache.active:
            parts.append(
                "chunk_residual_cache("
                f"th={self.chunk_residual_cache.condition_threshold:g},"
                f"max={self.chunk_residual_cache.max_consecutive_reuses},"
                f"k<={self.chunk_residual_cache.max_cache_step},"
                f"tau={self.chunk_residual_cache.force_refresh_every},"
                f"tail={self.chunk_residual_cache.min_full_tail})"
            )
        if self.compile.active:
            targets = "+".join(
                t for t, on in (("predict", self.compile.predict), ("prefill", self.compile.prefill)) if on
            )
            parts.append(f"compile({self.compile.mode},{targets})")
        return "accel: " + (", ".join(parts) if parts else "nothing active")


def _to_plain_dict(value: Any) -> dict[str, Any]:
    """Convert a mapping-like config (incl. OmegaConf) into plain dicts."""
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    if not isinstance(value, Mapping):
        raise TypeError(f"accel config must be a mapping, got {type(value)}")
    out = dict(value)
    unknown = set(out) - {
        "enabled",
        "compile",
        "text_cache",
        "step_cache",
        "video_token_cache",
        "vae_cache",
        "obs_pipeline",
        "chunk_residual_cache",
        "verbose",
    }
    if unknown:
        raise ValueError(f"unknown accel config keys: {sorted(unknown)}")
    return out
