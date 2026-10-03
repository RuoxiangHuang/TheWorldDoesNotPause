"""Trajectory complexity presets for Dynamic-LIBERO.

``easy`` / ``medium`` / ``hard`` / ``chaotic`` select a path family and
parameter scale. ``none`` means “use explicit ``--trajectory`` + kwargs”.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# (kind, default kwargs). Speeds stay on the caller (``--speed``).
COMPLEXITY_PRESETS: dict[str, tuple[str, dict[str, Any]]] = {
    "easy": (
        "linear",
        {"axis": "x", "direction": 1.0},
    ),
    "medium": (
        "sine",
        {
            "axis": "x",
            "direction": 1.0,
            "amplitude": 0.035,
            "wavelength": 0.18,
            "lateral_axis": "y",
        },
    ),
    "hard": (
        "random",
        {"n_waypoints": 5, "extent": 0.12, "planar": True},
    ),
    "chaotic": (
        "random",
        {"n_waypoints": 8, "extent": 0.16, "planar": True},
    ),
}

COMPLEXITY_CHOICES = ("none", "easy", "medium", "hard", "chaotic")


def resolve_traj_complexity(
    level: Optional[str],
    *,
    trajectory: str = "linear",
    seed: Optional[int] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Return ``(kind, kwargs)`` for ``build_trajectory``.

    - ``level in {None, '', 'none'}`` → keep ``trajectory`` + ``overrides``
    - otherwise → preset kind/kwargs; ``overrides`` win on colliding keys;
      ``seed`` is injected for random families.
    """
    lvl = (level or "none").strip().lower()
    ov = dict(overrides or {})
    if lvl in ("", "none"):
        return trajectory, ov

    if lvl not in COMPLEXITY_PRESETS:
        raise ValueError(
            f"Unknown traj complexity {level!r}; expected one of {list(COMPLEXITY_CHOICES)}"
        )
    kind, base_kw = COMPLEXITY_PRESETS[lvl]
    kw = {**base_kw, **ov}
    if kind in ("random", "random_polyline") and seed is not None and "seed" not in ov:
        kw["seed"] = int(seed)
    return kind, kw
