from __future__ import annotations

import numpy as np

from .linear import LinearTrajectory


class StopAndGoTrajectory(LinearTrajectory):
    """Linear motion with periodic pauses (duty cycle in (0, 1]).

    Each period lasts `period_ticks` ticks; the object moves for
    `duty * period` then stops for the remainder. Stresses reaction time.
    """

    name = "stop_and_go"

    def __init__(
        self,
        speed: float = 0.003,
        axis: str = "y",
        direction: float = 1.0,
        period_ticks: float = 40.0,
        duty: float = 0.5,
        period: float | None = None,
        **kwargs,
    ):
        super().__init__(speed=speed, axis=axis, direction=direction, **kwargs)
        self.period_ticks = float(period if period is not None else period_ticks)
        self.duty = float(np.clip(duty, 1e-3, 1.0))

    def _moving_time(self, t: float) -> float:
        """Equivalent moving ticks accumulated up to wall time t."""
        t = max(float(t), 0.0)
        period = max(self.period_ticks, 1e-6)
        n_full, rem = divmod(t, period)
        move_per = period * self.duty
        return n_full * move_per + min(rem, move_per)

    def xyz_at(self, t: float) -> np.ndarray:
        return super().xyz_at(self._moving_time(t))

    def velocity_at(self, t: float, hz: float = 20.0) -> np.ndarray:
        if abs(self.speed) < 1e-12:
            return np.zeros(3, dtype=np.float64)
        eps = 1e-4
        mt0 = self._moving_time(t)
        mt1 = self._moving_time(t + eps)
        dm = (mt1 - mt0) / eps
        if dm <= 1e-12:
            return np.zeros(3, dtype=np.float64)
        return self._axis * (self.speed * dm * float(hz))
