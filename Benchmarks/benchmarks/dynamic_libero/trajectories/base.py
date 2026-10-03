"""Trajectory interface for Dynamic-LIBERO."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

import numpy as np


class Trajectory(ABC):
    """Object path parameterized in control ticks.

    Positions are metres in the world frame. `speed` is metres per tick unless
    a subclass documents otherwise. Call `reset` once per episode before `step`.
    """

    name: str = "trajectory"

    def __init__(self, speed: float = 0.003, **kwargs: Any):
        self.speed = float(speed)
        self.t = 0.0
        self.base_pos: Optional[np.ndarray] = None
        self.base_quat: Optional[np.ndarray] = None
        self._extra = kwargs

    def reset(self, base_pos: np.ndarray, base_quat: Optional[np.ndarray] = None, rng=None) -> None:
        self.base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3).copy()
        if base_quat is None:
            self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            self.base_quat = np.asarray(base_quat, dtype=np.float64).reshape(4).copy()
        self.t = 0.0
        self._reset_impl(rng)

    def _reset_impl(self, rng) -> None:
        return

    @abstractmethod
    def xyz_at(self, t: float) -> np.ndarray:
        """World XYZ at tick time `t` (can be fractional)."""

    def quat_at(self, t: float) -> np.ndarray:
        """Orientation; default keeps the episode base quaternion."""
        assert self.base_quat is not None
        return self.base_quat.copy()

    def step(self, dt: float = 1.0) -> np.ndarray:
        self.t += float(dt)
        return self.xyz_at(self.t)

    def pose_at(self, t: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        if t is None:
            t = self.t
        return self.xyz_at(t), self.quat_at(t)

    def path_length(self, t: Optional[float] = None) -> float:
        """Approximate arc length from 0 to t (default: current)."""
        if t is None:
            t = self.t
        return abs(self.speed) * float(t)

    def velocity_at(self, t: float, hz: float = 20.0) -> np.ndarray:
        """Linear velocity in m/s at trajectory time ``t``.

        Subclasses may override; default uses a small numerical derivative of
        ``xyz_at`` and scales by ``hz`` (speed is metres per control tick).
        """
        t = float(t)
        eps = 1e-4
        p0 = self.xyz_at(t)
        p1 = self.xyz_at(t + eps)
        return (p1 - p0) / eps * float(hz)


def build_trajectory(kind: str, speed: float = 0.003, **kwargs: Any) -> Trajectory:
    from .circle import CircleTrajectory
    from .curve import SmoothCurveTrajectory
    from .linear import LinearTrajectory
    from .polyline import PolylineTrajectory
    from .random_polyline import RandomPolylineTrajectory
    from .sine import SineTrajectory
    from .smooth_turn import SmoothTurnTrajectory
    from .stop_and_go import StopAndGoTrajectory

    table = {
        "linear": LinearTrajectory,
        "circle": CircleTrajectory,
        "sine": SineTrajectory,
        "polyline": PolylineTrajectory,
        "stop_and_go": StopAndGoTrajectory,
        "random": RandomPolylineTrajectory,
        "random_polyline": RandomPolylineTrajectory,
        "curve": SmoothCurveTrajectory,
        "smooth": SmoothCurveTrajectory,
        "spline": SmoothCurveTrajectory,
        "smooth_turn": SmoothTurnTrajectory,
        "turn": SmoothTurnTrajectory,
    }
    if kind not in table:
        raise ValueError(f"Unknown trajectory {kind!r}; expected one of {sorted(table)}")
    return table[kind](speed=speed, **kwargs)


def clone_trajectory(traj: Trajectory, *, speed: Optional[float] = None) -> Trajectory:
    """Same path family/shape, optionally a different speed (place rails)."""
    spd = float(traj.speed if speed is None else speed)
    kind = str(getattr(traj, "name", "linear") or "linear")
    if kind in ("polyline", "random_polyline") and getattr(traj, "rel_waypoints", None):
        wps = [tuple(float(x) for x in np.asarray(p).reshape(3)) for p in traj.rel_waypoints]
        return build_trajectory("polyline", speed=spd, waypoints=wps)
    if kind in ("curve", "smooth", "spline"):
        return build_trajectory(
            "curve",
            speed=spd,
            curve_id=str(getattr(traj, "curve_id", "s_wave")),
            n_samples=int(getattr(traj, "n_samples", 160)),
        )
    if kind in ("smooth_turn", "turn"):
        from .smooth_turn import clone_smooth_turn_kwargs

        return build_trajectory("smooth_turn", speed=spd, **clone_smooth_turn_kwargs(traj))
    if kind == "linear":
        kw: dict = {
            "axis": str(getattr(traj, "axis_name", "x") or "x"),
            "direction": float(getattr(traj, "direction", 1.0) or 1.0),
            "toward_center": bool(getattr(traj, "toward_center", False)),
        }
        hd = getattr(traj, "heading_deg", None)
        if hd is not None:
            kw["heading_deg"] = float(hd)
            kw.pop("axis", None)
        return build_trajectory("linear", speed=spd, **kw)
    extra = dict(getattr(traj, "_extra", {}) or {})
    extra.pop("waypoints", None)
    extra.pop("speed", None)
    if kind == "random_polyline":
        kind = "random"
    return build_trajectory(kind, speed=spd, **extra)


def trajectory_from_mapping(cfg: Mapping[str, Any]) -> Trajectory:
    cfg = dict(cfg)
    kind = str(cfg.pop("type", cfg.pop("kind", "linear")))
    speed = float(cfg.pop("speed", 0.003))
    return build_trajectory(kind, speed=speed, **cfg)
