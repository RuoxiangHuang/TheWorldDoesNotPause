from __future__ import annotations

import numpy as np

from .base import Trajectory


class CircleTrajectory(Trajectory):
    """Horizontal circle in XY.

    Prefer `omega` (rad/tick). If omitted, `omega = speed / radius` with
    `speed` interpreted as tangential metres/tick.
    """

    name = "circle"

    def __init__(
        self,
        speed: float = 0.003,
        radius: float = 0.08,
        omega: float | None = None,
        **kwargs,
    ):
        super().__init__(speed=speed, **kwargs)
        self.radius = float(radius)
        if self.radius <= 1e-6:
            raise ValueError("circle radius must be > 0")
        self.omega = float(omega) if omega is not None else (self.speed / self.radius)

    def xyz_at(self, t: float) -> np.ndarray:
        assert self.base_pos is not None
        ang = self.omega * float(t)
        # Centre is offset so t=0 sits at base_pos.
        centre = self.base_pos.copy()
        centre[0] -= self.radius
        out = centre.copy()
        out[0] += self.radius * np.cos(ang)
        out[1] += self.radius * np.sin(ang)
        return out

    def velocity_at(self, t: float, hz: float = 20.0) -> np.ndarray:
        if abs(self.omega) < 1e-12:
            return np.zeros(3, dtype=np.float64)
        ang = self.omega * float(t)
        tangent = np.array([-np.sin(ang), np.cos(ang), 0.0], dtype=np.float64)
        # |v| = radius * omega (m/tick) along tangent
        return tangent * (self.radius * self.omega * float(hz))
