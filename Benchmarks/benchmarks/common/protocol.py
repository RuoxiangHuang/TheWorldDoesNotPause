"""Protocol constants shared by Dynamic-LIBERO / RoboTwin."""

from __future__ import annotations

import math
from enum import Enum


PROTOCOL_VERSION = "real_time_v2"

# Paper default deadline D for miss rate Pr[L_c > D].
DEFAULT_DEADLINE_MS = 50.0


class RealtimeBackend(str, Enum):
    """How wall-clock inference latency couples into the simulator."""

    ASYNC = "async"
    """Protocol default: execute stale tail of previous chunk, else zero-order hold."""

    FREEZE = "freeze"
    """Ablation: robot holds during thinking; object keeps moving."""


# Paper / protocol default thinking semantics (prior chunk else ZOH).
DEFAULT_BACKEND = RealtimeBackend.ASYNC

# Locked replan horizon per benchmark family (prevents winning via longer chunks).
LOCKED_REPLAN_STEPS: dict[str, int] = {
    "libero": 10,
    "robotwin": 8,
    "robocasa": 10,
}


def effective_replan_steps(
    benchmark: str,
    requested: int | None,
    *,
    allow_override: bool = False,
) -> int:
    """Return protocol replan_steps, enforcing the locked default unless overridden."""
    locked = int(LOCKED_REPLAN_STEPS[benchmark])
    if requested is None:
        return locked
    req = int(requested)
    if req != locked and not allow_override:
        raise ValueError(
            f"replan_steps={req} differs from protocol lock {locked} for {benchmark!r}. "
            f"Pass allow_override=True to opt out (not comparable across methods)."
        )
    return req


def blind_window_ratio(n_freeze: float, replan_steps: int) -> float | None:
    """Fraction of the control cycle spent blind: L / (L + replan_window).

    Uses fractional ``n_freeze`` ticks. Returns ``None`` for open-loop (inf freeze).
    """
    n = float(n_freeze)
    r = int(replan_steps)
    if not math.isfinite(n):
        return None
    if n <= 0.0:
        return 0.0
    denom = n + float(max(r, 0))
    return float(n / denom) if denom > 0.0 else None
