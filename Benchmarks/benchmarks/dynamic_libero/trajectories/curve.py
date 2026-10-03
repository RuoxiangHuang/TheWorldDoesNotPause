"""Smooth C1 curves for Dynamic-LIBERO irregular extras.

Positions are sampled from analytic curves, then followed at constant arc
speed so heading never jumps (no polyline corners).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import Trajectory


def _search_s(cum: np.ndarray, s: float) -> tuple[int, float]:
    s = float(min(max(s, 0.0), float(cum[-1])))
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = max(0, min(i, len(cum) - 2))
    span = float(cum[i + 1] - cum[i])
    u = 0.0 if span < 1e-12 else (s - float(cum[i])) / span
    return i, u


class SmoothCurveTrajectory(Trajectory):
    """Constant-speed tracking of a named smooth planar curve."""

    name = "curve"

    def __init__(
        self,
        speed: float = 0.003,
        curve_id: str = "s_wave",
        n_samples: int = 160,
        **kwargs: Any,
    ):
        super().__init__(speed=speed, **kwargs)
        self.curve_id = str(curve_id)
        self.n_samples = max(32, int(n_samples))
        self.rel_waypoints: list[np.ndarray] = [np.zeros(3), np.array([0.1, 0.0, 0.0])]
        self._pts = np.zeros((2, 3), dtype=np.float64)
        self._cum = np.array([0.0, 0.1], dtype=np.float64)

    def _reset_impl(self, rng) -> None:
        from benchmarks.dynamic_libero.env.irregular_paths import sample_curve

        pts = sample_curve(self.curve_id, n=self.n_samples)
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self._pts = pts
        self._cum = np.concatenate([[0.0], np.cumsum(d)])
        if float(self._cum[-1]) < 1e-6:
            self._cum[-1] = 1e-6
        self.rel_waypoints = [p.copy() for p in pts]

    def xyz_at(self, t: float) -> np.ndarray:
        assert self.base_pos is not None
        if self._pts.shape[0] < 2:
            self._reset_impl(None)
        s = min(max(self.speed * float(t), 0.0), float(self._cum[-1]))
        i, u = _search_s(self._cum, s)
        rel = (1.0 - u) * self._pts[i] + u * self._pts[i + 1]
        return self.base_pos + rel

    def velocity_at(self, t: float, hz: float = 20.0) -> np.ndarray:
        if abs(self.speed) < 1e-12:
            return np.zeros(3, dtype=np.float64)
        if self._pts.shape[0] < 2:
            self._reset_impl(None)
        s = min(max(self.speed * float(t), 0.0), float(self._cum[-1]))
        if s >= float(self._cum[-1]) - 1e-12:
            return np.zeros(3, dtype=np.float64)
        i, _ = _search_s(self._cum, s)
        seg = self._pts[i + 1] - self._pts[i]
        length = float(self._cum[i + 1] - self._cum[i])
        direction = seg / max(length, 1e-9)
        return direction * (self.speed * float(hz))
