from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .base import Trajectory

_AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def planar_unit(heading_deg: float) -> np.ndarray:
    """World XY unit vector. 0°=+x, 90°=+y (right-handed)."""
    th = np.deg2rad(float(heading_deg))
    return np.array([np.cos(th), np.sin(th), 0.0], dtype=np.float64)


def corridor_clearance(
    origin_xy: np.ndarray,
    heading_deg: float,
    point_xy: np.ndarray,
    *,
    travel: float = 0.40,
    rear: float = 0.06,
) -> float:
    """Lateral gap from ``point`` to the heading ray. Large if outside the travel window."""
    v = planar_unit(heading_deg)[:2]
    rel = np.asarray(point_xy, dtype=np.float64).reshape(2) - np.asarray(
        origin_xy, dtype=np.float64
    ).reshape(2)
    along = float(np.dot(rel, v))
    if along < -float(rear) or along > float(travel):
        return 1e3
    lat = float(np.linalg.norm(rel - along * v))
    return lat


def inward_heading_candidates(origin_xy: np.ndarray, *, step_deg: int = 15) -> list[float]:
    """Headings whose XY component points toward the workspace origin."""
    toward = -np.asarray(origin_xy, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(toward))
    out: list[float] = []
    for h in range(0, 360, int(step_deg)):
        v = planar_unit(float(h))[:2]
        if n < 1e-6 or float(np.dot(v, toward)) >= -1e-9:
            out.append(float(h))
    return out or [0.0]


def effective_heading_deg(traj: Any) -> Optional[float]:
    """Travel heading in world XY, folding ``direction`` into ``heading_deg``."""
    if getattr(traj, "heading_deg", None) is None:
        return None
    h = float(traj.heading_deg) % 360.0
    if float(getattr(traj, "direction", 1.0) or 1.0) < 0.0:
        h = (h + 180.0) % 360.0
    return h


def _heading_delta_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def deflect_linear_heading(
    traj: Any,
    origin_xy: np.ndarray,
    avoid_xy: np.ndarray,
    *,
    travel: float = 0.40,
    min_clearance: float = 0.16,
    candidates: Optional[Sequence[float]] = None,
) -> Optional[float]:
    """Rotate a planar linear heading so the corridor misses ``avoid_xy``.

    Uses the *travel* heading (``heading_deg`` with ``direction`` folded in).
    The chosen heading is written back with ``direction=+1`` so later layout
    / path-clear see the same ray the object actually follows.
    """
    cur = effective_heading_deg(traj)
    if cur is None:
        return None
    cand = [cur]
    if candidates is None:
        cand.extend(float(h) for h in range(0, 91, 5))
    else:
        cand.extend(float(h) for h in candidates)
    scored: list[tuple] = []
    seen: set[float] = set()
    for h in cand:
        h = float(h) % 360.0
        if h in seen:
            continue
        seen.add(h)
        lat = corridor_clearance(origin_xy, h, avoid_xy, travel=travel)
        # Prefer a miss, then more lateral gap, then stay close to the travel heading.
        scored.append((0 if lat >= float(min_clearance) else 1, -lat, _heading_delta_deg(h, cur), h))
    scored.sort()
    best = float(scored[0][3])
    if _heading_delta_deg(best, cur) > 0.51:
        traj.heading_deg = best
        traj.direction = 1.0
        rebuild = getattr(traj, "_rebuild_axis", None)
        if callable(rebuild):
            rebuild()
    return best


class LinearTrajectory(Trajectory):
    """Uniform planar (or axis) motion: p(t) = p0 + dir * speed * t.

    ``heading_deg`` (0=+x, 90=+y) is preferred when set so reconstructed
    catalogs can fan out instead of sharing a single world axis. ``axis`` is
    the fallback for older CLI / official-whitelist defaults.
    """

    name = "linear"

    def __init__(
        self,
        speed: float = 0.003,
        axis: str = "y",
        direction: float = 1.0,
        toward_center: bool = False,
        heading_deg: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(speed=speed, **kwargs)
        self.heading_deg = None if heading_deg is None else float(heading_deg)
        if self.heading_deg is None:
            if axis not in _AXES:
                raise ValueError(f"axis must be one of {list(_AXES)}, got {axis!r}")
            self.axis_name = axis
        else:
            self.axis_name = "planar"
        self.toward_center = bool(toward_center)
        self.direction = float(np.sign(direction) or 1.0)
        self._rebuild_axis()

    def _rebuild_axis(self) -> None:
        d = float(self.direction)
        if self.heading_deg is not None:
            self._axis = planar_unit(self.heading_deg) * d
        else:
            self._axis = _AXES[self.axis_name] * d

    def _reset_impl(self, rng) -> None:
        # Orient drift toward the workspace origin so edge-spawned objects
        # move inward (RoboTwin tables) instead of immediately off the rim.
        if not self.toward_center or self.base_pos is None:
            return
        if self.heading_deg is not None:
            away = np.asarray(self.base_pos, dtype=np.float64).reshape(3)[:2]
            if float(np.dot(self._axis[:2], away)) > 0.0:
                self.direction = -self.direction
                self._rebuild_axis()
            return
        idx = {"x": 0, "y": 1, "z": 2}[self.axis_name]
        self.direction = -1.0 if float(self.base_pos[idx]) > 0.0 else 1.0
        self._rebuild_axis()

    def xyz_at(self, t: float) -> np.ndarray:
        assert self.base_pos is not None
        return self.base_pos + self._axis * (self.speed * float(t))

    def velocity_at(self, t: float, hz: float = 20.0) -> np.ndarray:
        if abs(self.speed) < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return self._axis * (self.speed * float(hz))
