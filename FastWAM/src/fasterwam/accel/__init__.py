"""Inference acceleration for FasterWAM.

Training-free first-class controllers, composed multiplicatively:

    text_cache            exact         memoize the T5 prompt encode across replans
    step_cache            near-lossless hold the last velocity while the action
                                        latent has barely drifted
    compile               lossless      torch.compile the action step / video prefill
    video_token_cache     near-lossless reuse / sparsely refresh video K/V across
                                        consecutive replans (core; on via p0/p1/full)
    vae_cache             near-lossless reuse VAE latents when the RGB barely moved
    obs_pipeline          lossless      async observation prefetch API for runners
    chunk_residual_cache  near-lossless reuse DiT residual R=h^L-h^0 across chunks
                                        at the same denoise index (action0 / full)

Named comparison modes: `accel=baseline` | `accel=pace` (paper name PACE;
`accel=ours` is the same stack) | `accel=full`.

Typical use is implicit: `create_fastwam` records the stack from the `accel`
block of the model config; `prepare_for_inference` installs it.
"""

from .base import Controller
from .chunk_residual_cache import ChunkResidualCacheController
from .compile_ctrl import CompileController
from .config import (
    AccelConfig,
    ChunkResidualCacheConfig,
    CompileConfig,
    ObsPipelineConfig,
    StepCacheConfig,
    TextCacheConfig,
    VaeCacheConfig,
    VideoTokenCacheConfig,
)
from .obs_pipeline import ObsPipelineController
from .stack import (
    AccelStack,
    accelerate,
    get_stack,
    require_ours_stack,
    require_pace_stack,
    reset_inference_state,
)
from .step_cache import StepCacheController
from .text_cache import TextCacheController
from .vae_cache import VaeCacheController
from .video_token_cache import VideoTokenCacheController
from .world_state import VideoTokenState, get_video_token_state

__all__ = [
    "AccelConfig",
    "AccelStack",
    "ChunkResidualCacheConfig",
    "ChunkResidualCacheController",
    "CompileConfig",
    "CompileController",
    "Controller",
    "ObsPipelineConfig",
    "ObsPipelineController",
    "StepCacheConfig",
    "StepCacheController",
    "TextCacheConfig",
    "TextCacheController",
    "VaeCacheConfig",
    "VaeCacheController",
    "VideoTokenCacheConfig",
    "VideoTokenCacheController",
    "VideoTokenState",
    "accelerate",
    "get_stack",
    "get_video_token_state",
    "require_ours_stack",
    "require_pace_stack",
    "reset_inference_state",
]
