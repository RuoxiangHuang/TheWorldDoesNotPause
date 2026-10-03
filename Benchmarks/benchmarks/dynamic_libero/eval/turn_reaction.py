"""Eval-only reaction chain for the smooth-turn diagnostic.

Ground-truth poses are used here and never fed to the policy. The chain is:

    event → first observation of the change → new-plan takeover → effective correction

Effective correction: after a plan that *saw* the event takes over, the
target–eef lateral error (along the turn normal) decreases for ``persist``
consecutive control ticks and drops at least ``drop_m``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from benchmarks.dynamic_libero.trajectories.smooth_turn import is_smooth_turn


def maybe_turn_tracker(trajectory: Any, control_hz: float) -> Optional["TurnReactionTracker"]:
    kind = str(getattr(trajectory, "name", "") or "")
    if not is_smooth_turn(kind):
        return None
    if getattr(trajectory, "event_t", None) is None:
        return None
    return TurnReactionTracker(trajectory, control_hz=control_hz)


class TurnReactionTracker:
    def __init__(
        self,
        trajectory: Any,
        *,
        control_hz: float,
        persist: int = 3,
        drop_m: float = 0.002,
    ):
        self.hz = float(control_hz)
        self.t_event = float(trajectory.event_t)
        self.n_hat = np.asarray(trajectory.turn_normal_xy(), dtype=np.float64).reshape(2)
        ln = float(np.linalg.norm(self.n_hat))
        self.n_hat = self.n_hat / ln if ln > 1e-12 else np.array([0.0, 1.0])
        self.persist = max(1, int(persist))
        self.drop_m = float(drop_m)
        self.first_obs_t: Optional[float] = None
        self.takeover_t: Optional[float] = None
        self.correction_t: Optional[float] = None
        self._awaiting_takeover = False
        self._post_takeover = False
        self._lat_at_takeover: Optional[float] = None
        self._dec_run = 0
        self._prev_lat: Optional[float] = None
        self.close_t: Optional[float] = None
        self.close_lateral_err_m: Optional[float] = None
        self.close_dist_m: Optional[float] = None
        self.min_lateral_err_m: Optional[float] = None

    def on_replan_obs(self, policy_t: float, *, rails_active: bool = True) -> None:
        if not rails_active:
            return
        t = float(policy_t)
        if self.first_obs_t is None and t + 1e-9 >= self.t_event:
            self.first_obs_t = t
            self._awaiting_takeover = True

    def on_takeover(self, policy_t: float) -> None:
        if self._awaiting_takeover and self.takeover_t is None:
            self.takeover_t = float(policy_t)
            self._awaiting_takeover = False
            self._post_takeover = True
            self._dec_run = 0
            self._prev_lat = None
            self._lat_at_takeover = None

    def on_step(
        self,
        policy_t: float,
        rel_xyz: Optional[np.ndarray],
        dist_m: Optional[float],
        *,
        released: bool,
    ) -> None:
        if rel_xyz is None:
            return
        rel = np.asarray(rel_xyz, dtype=np.float64).reshape(-1)
        if rel.size < 2:
            return
        lat = float(abs(float(rel[0]) * self.n_hat[0] + float(rel[1]) * self.n_hat[1]))
        if self.min_lateral_err_m is None or lat < self.min_lateral_err_m:
            self.min_lateral_err_m = lat
        t = float(policy_t)
        if self._post_takeover and self.correction_t is None:
            if self._lat_at_takeover is None:
                self._lat_at_takeover = lat
            if self._prev_lat is not None and lat < self._prev_lat - 1e-9:
                self._dec_run += 1
            else:
                self._dec_run = 0
            self._prev_lat = lat
            dropped = lat <= float(self._lat_at_takeover) - self.drop_m
            if self._dec_run >= self.persist and dropped:
                self.correction_t = t
        if released and self.close_t is None:
            self.close_t = t
            self.close_lateral_err_m = lat
            self.close_dist_m = None if dist_m is None else float(dist_m)

    def summary(self, *, event_experienced: bool = True) -> dict[str, Any]:
        def _dt(later: Optional[float], earlier: Optional[float]) -> Optional[float]:
            if later is None or earlier is None:
                return None
            return (float(later) - float(earlier)) / self.hz

        out = {
            "turn_first_obs_t": self.first_obs_t,
            "turn_takeover_t": self.takeover_t,
            "turn_correction_t": self.correction_t,
            "turn_event_to_first_obs_s": _dt(self.first_obs_t, self.t_event),
            "turn_first_obs_to_takeover_s": _dt(self.takeover_t, self.first_obs_t),
            "turn_event_to_correction_s": _dt(self.correction_t, self.t_event),
            "turn_event_to_takeover_s": _dt(self.takeover_t, self.t_event),
            "turn_close_t": self.close_t,
            "turn_close_lateral_err_m": self.close_lateral_err_m,
            "turn_close_dist_m": self.close_dist_m,
            "turn_min_lateral_err_m": self.min_lateral_err_m,
            "turn_lat_at_takeover_m": self._lat_at_takeover,
            "event_experienced": bool(event_experienced),
        }
        if not event_experienced:
            for key in (
                "turn_first_obs_t",
                "turn_takeover_t",
                "turn_correction_t",
                "turn_event_to_first_obs_s",
                "turn_first_obs_to_takeover_s",
                "turn_event_to_correction_s",
                "turn_event_to_takeover_s",
                "turn_lat_at_takeover_m",
            ):
                out[key] = None
        return out
