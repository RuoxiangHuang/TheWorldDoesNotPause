"""FasterPI — training-free accelerate stack for openpi pi0.5."""

from __future__ import annotations

_LAZY = {
    "DEFAULT_FASTERPI_SPEC",
    "OURS_NO_ADAPTIVE_NFE_SPEC",
    "OURS_NO_D1_SPEC",
    "OURS_NO_STEP_CACHE_SPEC",
    "OURS_NO_VISION_SPEC",
    "FASTERPI_V1_SPEC",
    "FASTERPI_V2_SPEC",
    "SPEEDUP_ALIASES",
    "VALID_DIRS",
    "AdaptiveNFEController",
    "ChunkResidualCacheController",
    "CompileController",
    "CompilePrefixController",
    "FuseLoopController",
    "SdpaController",
    "ObsPipeline",
    "PrefixKVCacheController",
    "SplitPrefixController",
    "StepCacheController",
    "TextCacheController",
    "VisionCacheController",
    "install_stack",
    "rel_l1",
    "resolve_speedup_spec",
    "restore_eager",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        from fasterpi import stack as _stack

        return getattr(_stack, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
