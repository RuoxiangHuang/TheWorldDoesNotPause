from __future__ import annotations

import numpy as np

from .base import Trajectory

_AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


class SineTrajectory(Trajectory):
    """Advance along `axis` at `speed`, with sinusoidal offset on `lateral_axis`."""

    name = "sine"

    def __init__(
        self,
        speed: float = 0.003,
        axis: str = "y",
        lateral_axis: str = "x",
        amplitude: float = 0.04,
        wavelength: float = 0.20,
        direction: float = 1.0,
        toward_center: bool = False,
        **kwargs,
    ):
        super().__init__(speed=speed, **kwargs)
        if axis == lateral_axis:
            raise ValueError("axis and lateral_axis must differ")
        self.axis_name = axis
        self.lateral_name = lateral_axis
        self.amplitude = float(amplitude)
        self.wavelength = float(wavelength)
        self.toward_center = bool(toward_center)
        self.direction = float(np.sign(direction) or 1.0)
        self._fwd = _AXES[axis] * self.direction
        self._lat = _AXES[lateral_axis]

    def _reset_impl(self, rng) -> None:
        if self.toward_center and self.base_pos is not None:
            idx = {"x": 0, "y": 1, "z": 2}[self.axis_name]
            self.direction = -1.0 if float(self.base_pos[idx]) > 0.0 else 1.0
            self._fwd = _AXES[self.axis_name] * self.direction

    def xyz_at(self, t: float) -> np.ndarray:
        assert self.base_pos is not None
        t = float(t)
        along = self.speed * t
        phase = (2.0 * np.pi * along / max(self.wavelength, 1e-6))
        return self.base_pos + self._fwd * along + self._lat * (self.amplitude * np.sin(phase))

    def velocity_at(self, t: float, hz: float = 20.0) -> np.ndarray:
        if abs(self.speed) < 1e-12:
            return np.zeros(3, dtype=np.float64)
        t = float(t)
        along = self.speed * t
        wl = max(self.wavelength, 1e-6)
        phase = 2.0 * np.pi * along / wl
        d_along = self.speed  # m/tick
        lateral_rate = self.amplitude * np.cos(phase) * (2.0 * np.pi / wl) * d_along
        v_tick = self._fwd * d_along + self._lat * lateral_rate
        return v_tick * float(hz)
