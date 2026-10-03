"""Root shim: `import pi05_accel` → `fasterpi` (legacy path)."""

from fasterpi import *  # noqa: F401,F403
from fasterpi import (  # noqa: F401
    DEFAULT_FASTERPI_SPEC,
    FASTERPI_V1_SPEC,
    FASTERPI_V2_SPEC,
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
