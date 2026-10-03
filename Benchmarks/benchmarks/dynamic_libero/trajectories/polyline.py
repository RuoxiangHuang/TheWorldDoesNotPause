from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import Trajectory


class PolylineTrajectory(Trajectory):
    """Piecewise-linear path through relative waypoints (metres from base_pos).

    `speed` is metres/tick along the polyline arc. When the end is reached, the
    object stays at the final waypoint (escape-friendly for long episodes).
    """

    name = "polyline"

    def __init__(
        self,
        speed: float = 0.003,
        waypoints: Sequence[Sequence[float]] | None = None,
        **kwargs,
    ):
        super().__init__(speed=speed, **kwargs)
        if waypoints is None:
            waypoints = [(0.0, 0.0, 0.0), (0.0, 0.12, 0.0), (0.08, 0.12, 0.0)]
        pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in waypoints]
        if len(pts) < 2:
            raise ValueError("polyline needs at least 2 waypoints")
        self.rel_waypoints = pts
        self._seg_lengths: list[float] = []
        self._cum: list[float] = [0.0]

    def _reset_impl(self, rng) -> None:
        self._seg_lengths = []
        self._cum = [0.0]
        for a, b in zip(self.rel_waypoints[:-1], self.rel_waypoints[1:]):
            L = float(np.linalg.norm(b - a))
            self._seg_lengths.append(max(L, 1e-9))
            self._cum.append(self._cum[-1] + self._seg_lengths[-1])

    def xyz_at(self, t: float) -> np.ndarray:
        assert self.base_pos is not None
        if not self._seg_lengths:
            self._reset_impl(None)
        s = min(max(self.speed * float(t), 0.0), self._cum[-1])
        # Find segment.
        for i, (c0, c1) in enumerate(zip(self._cum[:-1], self._cum[1:])):
            if s <= c1 + 1e-12:
                u = (s - c0) / self._seg_lengths[i]
                rel = (1 - u) * self.rel_waypoints[i] + u * self.rel_waypoints[i + 1]
                return self.base_pos + rel
        return self.base_pos + self.rel_waypoints[-1]

    def velocity_at(self, t: float, hz: float = 20.0) -> np.ndarray:
        if abs(self.speed) < 1e-12:
            return np.zeros(3, dtype=np.float64)
        if not self._seg_lengths:
            self._reset_impl(None)
        s = self.speed * float(t)
        if s >= self._cum[-1] - 1e-12:
            return np.zeros(3, dtype=np.float64)
        for i, (c0, c1) in enumerate(zip(self._cum[:-1], self._cum[1:])):
            if s <= c1 + 1e-12:
                seg = self.rel_waypoints[i + 1] - self.rel_waypoints[i]
                direction = seg / max(self._seg_lengths[i], 1e-9)
                return direction * (self.speed * float(hz))
        return np.zeros(3, dtype=np.float64)
