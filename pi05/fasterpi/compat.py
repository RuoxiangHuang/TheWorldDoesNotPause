"""Backward-compatible alias (`import fasterpi.compat as pi05_accel`). Prefer `import fasterpi`."""

from fasterpi.stack import *  # noqa: F401,F403
from fasterpi.stack import (  # noqa: F401
    DEFAULT_FASTERPI_SPEC,
    FASTERPI_V1_SPEC,
    FASTERPI_V2_SPEC,
    SPEEDUP_ALIASES,
    VALID_DIRS,
    ChunkResidualCacheController,
    CompileController,
    ObsPipeline,
    PrefixKVCacheController,
    StepCacheController,
    TextCacheController,
    VisionCacheController,
    install_stack,
    rel_l1,
    resolve_speedup_spec,
    restore_eager,
)
