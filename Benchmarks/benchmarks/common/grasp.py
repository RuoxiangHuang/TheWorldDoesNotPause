"""Hybrid grasp protocol: ghost (primary) vs engage/release variants (A/B).

Tracks
------
- ``ghost``: trajectory-first pin / ghost sliding until distance release (primary).
- ``hybrid_a``: enter engage early (contact allowed), skip velocity injection when engaged.
- ``hybrid_b``: hybrid_a + soft tracking toward trajectory during engage.
"""

from __future__ import annotations

import argparse
from enum import Enum
from typing import Any

PLACE_FREEZE_MODES = ("off", "both", "all")
GRASP_MODES = ("ghost", "hybrid_a", "hybrid_b")


def parse_place_freeze(value: Any) -> str:
    """``both`` (default): park receptacle only after catching a moving pick object.

    Place-only tasks keep the basket on rails so FastWAM cannot treat them as
    static pick-and-place. ``all`` stops the basket on EEF proximity (lets
    unaccelerated FastWAM hit the ceiling). ``off`` never parks.
    """
    if value is None:
        return "both"
    if value is True:
        return "both"
    if value is False:
        return "off"
    v = str(value).strip().lower()
    if v in ("", "default"):
        return "both"
    if v in ("1", "true", "on", "yes"):
        return "both"
    if v in ("0", "false", "no"):
        return "off"
    if v in PLACE_FREEZE_MODES:
        return v
    raise ValueError(
        f"place_freeze must be one of {PLACE_FREEZE_MODES}, got {value!r}"
    )

DEFAULT_ENGAGE_ALPHA = 1.75
DEFAULT_SOFT_KP = 4.0  # 1/s position error gain during hybrid_b engage


class RailsPhase(str, Enum):
    PURSUIT = "pursuit"
    ENGAGE = "engage"
    FREE = "free"


def parse_grasp_mode(value: str | None) -> str:
    v = (value or "ghost").strip().lower()
    if v not in GRASP_MODES:
        raise ValueError(f"grasp_mode must be one of {GRASP_MODES}, got {value!r}")
    return v


def engage_radius(release_radius: float, *, alpha: float = DEFAULT_ENGAGE_ALPHA) -> float:
    return float(release_radius) * float(alpha)


def soft_kp(
    dist: float,
    engage_r: float,
    *,
    kp_max: float = DEFAULT_SOFT_KP,
) -> float:
    """Linear decay: full kp at engage boundary, zero at release distance."""
    if engage_r <= 0:
        return 0.0
    ratio = max(0.0, min(1.0, float(dist) / float(engage_r)))
    return float(kp_max) * ratio


def should_inject_release_velocity(
    mode: str,
    *,
    phase: RailsPhase | str,
    had_contact: bool = False,
) -> bool:
    """Ghost always injects; hybrid skips when already engaged or had contact."""
    m = parse_grasp_mode(mode)
    if m == "ghost":
        return True
    ph = RailsPhase(phase) if isinstance(phase, str) else phase
    if ph == RailsPhase.ENGAGE or had_contact:
        return False
    return True


def grasp_stats(
    *,
    grasp_mode: str,
    rails_phase: str,
    engage_t: float | None = None,
    release_t: float | None = None,
    contact_before_release: bool = False,
    release_velocity_injected: bool = False,
    engage_radius_m: float | None = None,
) -> dict[str, Any]:
    return {
        "grasp_mode": parse_grasp_mode(grasp_mode),
        "rails_phase": str(rails_phase),
        "engage_t": engage_t,
        "release_t": release_t,
        "contact_before_release": bool(contact_before_release),
        "release_velocity_injected": bool(release_velocity_injected),
        "engage_radius_m": engage_radius_m,
    }


def add_grasp_cli_args(parser, *, default: str = "ghost") -> None:
    parser.add_argument(
        "--grasp-mode",
        type=str,
        default=default,
        choices=list(GRASP_MODES),
        help="ghost=primary pin/ghost; hybrid_a=early engage+contact; hybrid_b=soft track engage",
    )
    parser.add_argument(
        "--engage-alpha",
        type=float,
        default=DEFAULT_ENGAGE_ALPHA,
        help="engage_radius = release_radius * engage_alpha",
    )
    parser.add_argument(
        "--place-freeze",
        type=str,
        default="both",
        choices=list(PLACE_FREEZE_MODES),
        help="Park moving receptacle: both=after catching dual-motion pick "
        "(default); all=on EEF proximity (also boosts unaccelerated FastWAM); "
        "off=never.",
    )
    parser.add_argument(
        "--release-kick",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Inject trajectory velocity when pick rails unpin. "
        "Default: off for toy-car seating, on for ghost-pin.",
    )
