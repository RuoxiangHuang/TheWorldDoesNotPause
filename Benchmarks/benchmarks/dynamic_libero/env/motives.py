"""Motion-motive registry for Dynamic-LIBERO.

Motives explain *why* a rails target moves (visual + semantics), independent of
the latency-coupling protocol (speed × trajectory × n_freeze).

- ``pusher``: keep the original LIBERO mesh; a toy car rides the rails and
  kinematically pushes the pick target (default for object / libero_10).
- ``carrier``: pickup flatbed *carries* the pick target. Spatial catalog
  tasks are pick-only (bowl on the bed, plate still). CLI ``--motive carrier``
  without ``--rails-variant`` still also bumper-pushes the receptacle.
- ``conveyor``: keep the original mesh; plant a visible belt under the path
  (legacy demo; not the object-suite default).
- ``toycar``: swap the rails-target mesh for a self-propelled toy car
  (viz default for spatial / goal; eval only when ``movers=on``).
- ``ghost``: invisible rails pin (no extra actor). Used by eval when
  ``toycar`` is selected but movers stay off.
"""

from __future__ import annotations

from typing import Literal

Motive = Literal["toycar", "conveyor", "pusher", "carrier", "ghost"]

KNOWN_MOTIVES: tuple[Motive, ...] = ("toycar", "conveyor", "pusher", "carrier", "ghost")

# Default by suite (motion motive, not a random task_id split).
DEFAULT_MOTIVE_BY_SUITE: dict[str, Motive] = {
    "libero_spatial": "toycar",
    "libero_object": "pusher",
    "libero_goal": "toycar",
    "libero_10": "pusher",
}

MOTIVE_ALIASES: dict[str, Motive] = {
    "toy": "toycar",
    "toy_car": "toycar",
    "car": "toycar",
    "self": "toycar",
    "self_propelled": "toycar",
    "belt": "conveyor",
    "conveyor_belt": "conveyor",
    "push": "pusher",
    "bumper": "pusher",
    "toy_push": "pusher",
    "pusher_car": "pusher",
    "carry": "carrier",
    "flatbed": "carrier",
    "pickup": "carrier",
    "carrier_car": "carrier",
    "pin": "ghost",
    "rails": "ghost",
    "invisible": "ghost",
}


def normalize_motive(value: str | None) -> Motive | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in {"", "auto", "default", "none"}:
        return None
    if key in KNOWN_MOTIVES:
        return key  # type: ignore[return-value]
    if key in MOTIVE_ALIASES:
        return MOTIVE_ALIASES[key]
    raise ValueError(
        f"Unknown motive {value!r}. Expected toycar|conveyor|pusher|carrier|ghost|auto "
        f"(aliases: {sorted(MOTIVE_ALIASES)})"
    )


def resolve_motive(suite: str, override: str | None = None) -> Motive:
    """Resolve motive for a suite; ``override`` wins when not auto/None."""
    got = normalize_motive(override)
    if got is not None:
        return got
    suite = suite.strip()
    if suite not in DEFAULT_MOTIVE_BY_SUITE:
        raise KeyError(
            f"No default motive for suite {suite!r}. "
            f"Known: {sorted(DEFAULT_MOTIVE_BY_SUITE)}"
        )
    return DEFAULT_MOTIVE_BY_SUITE[suite]


def keeps_original_mesh(motive: Motive) -> bool:
    """True when the pick target is not replaced by a toy-car MJCF."""
    return motive in ("pusher", "carrier", "conveyor", "ghost")
