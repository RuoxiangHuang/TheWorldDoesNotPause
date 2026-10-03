"""World-clock smooth turn for the Dynamic-LIBERO reaction diagnostic.

The pick target travels inbound on a straight rail, then smoothly changes
heading at a **pre-sampled simulation time**. ``xyz_at(t)`` is a function of
trajectory time only. Event phase is drawn from an episode seed at ``reset``,
never from a replan or ``predict`` call.

This tests latest-position correction, not motion prediction. It is a
diagnostic slice: do not add it to the official ``linear`` / ``sine`` SR grid.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from .linear import LinearTrajectory

# Locked diagnostic slice (20 Hz). Not official working points.
DIAGNOSTIC_KIND = "smooth_turn"
DIAGNOSTIC_SPEED = 0.001  # m / control tick → 2 cm/s at 20 Hz
DIAGNOSTIC_REPLAN_STEPS = 10
DIAGNOSTIC_CONTROL_HZ = 20.0
DIAGNOSTIC_RAILS_VARIANT = "pick"
DIAGNOSTIC_MOTIVE = "ghost"
# Policy-clock window after begin_policy (ticks). Uniform sample; not a
# multiple of replan_steps.
DIAGNOSTIC_EVENT_T_LO = 20.0  # 1.0 s
DIAGNOSTIC_EVENT_T_HI = 60.0  # 3.0 s
DIAGNOSTIC_TURN_DURATION_TICKS = 12.0  # 0.6 s raised-cosine heading blend
DIAGNOSTIC_TURN_ANGLE_DEG = 75.0

SMOOTH_TURN_KINDS = ("smooth_turn", "turn")


def is_smooth_turn(kind: str | None) -> bool:
    return str(kind or "").strip().lower() in SMOOTH_TURN_KINDS


def mix_episode_seed(base_seed: int, task_id: int, init_id: int) -> int:
    """Stable 32-bit mix so (suite-task, init, eval seed) share one event.

    Must not use Python ``hash()`` (per-process salted). Off / Cache / delayed
    Cache must draw the same ``t_event`` for the same episode key.
    """

    def u32(x: int) -> int:
        return int(x) & 0xFFFFFFFF

    x = u32(int(base_seed) + 0x9E3779B9)
    x = u32((x ^ u32(int(task_id) * 0x85EBCA6B)) * 0xC2B2AE35)
    x = u32((x ^ u32((int(init_id) + 1) * 0x27D4EB2F)) * 0x165667B1)
    x ^= x >> 16
    return int(x)


def default_smooth_turn_kwargs(
    *,
    axis: str = "x",
    toward_center: bool = True,
    direction: float = 1.0,
    event_seed: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """CLI / eval defaults for the diagnostic slice (phase still sampled at reset)."""
    kw: dict[str, Any] = {
        "axis": axis,
        "toward_center": bool(toward_center),
        "event_t_lo": DIAGNOSTIC_EVENT_T_LO,
        "event_t_hi": DIAGNOSTIC_EVENT_T_HI,
        "turn_duration_ticks": DIAGNOSTIC_TURN_DURATION_TICKS,
        "turn_angle_deg": DIAGNOSTIC_TURN_ANGLE_DEG,
        "control_hz": DIAGNOSTIC_CONTROL_HZ,
    }
    if not toward_center:
        kw["direction"] = float(direction)
    if event_seed is not None:
        kw["event_seed"] = int(event_seed)
    if extra:
        kw.update(dict(extra))
    return kw


class SmoothTurnTrajectory(LinearTrajectory):
    """Straight inbound rail, then a C1 raised-cosine heading change.

    Trajectory time ``t`` is control ticks, advanced by ``driver.tick`` during
    thinking and by ``on_control_step`` during execution — the world clock.
    """

    name = "smooth_turn"

    def __init__(
        self,
        speed: float = DIAGNOSTIC_SPEED,
        axis: str = "x",
        direction: float = 1.0,
        toward_center: bool = True,
        heading_deg: Optional[float] = None,
        event_seed: int = 0,
        event_t: Optional[float] = None,
        event_t_lo: float = DIAGNOSTIC_EVENT_T_LO,
        event_t_hi: float = DIAGNOSTIC_EVENT_T_HI,
        turn_duration_ticks: float = DIAGNOSTIC_TURN_DURATION_TICKS,
        turn_angle_deg: float = DIAGNOSTIC_TURN_ANGLE_DEG,
        turn_sign: Optional[int] = None,
        control_hz: float = DIAGNOSTIC_CONTROL_HZ,
        n_arc: int = 96,
        **kwargs: Any,
    ):
        super().__init__(
            speed=speed,
            axis=axis,
            direction=direction,
            toward_center=toward_center,
            heading_deg=heading_deg,
            **kwargs,
        )
        self.name = "smooth_turn"
        self.event_type = "turn"
        self.event_seed = int(event_seed)
        self._event_t_fixed = None if event_t is None else float(event_t)
        lo, hi = float(event_t_lo), float(event_t_hi)
        if hi < lo:
            lo, hi = hi, lo
        self.event_t_lo = lo
        self.event_t_hi = hi
        self.turn_duration_ticks = max(float(turn_duration_ticks), 0.0)
        self.turn_angle_deg = float(turn_angle_deg)
        self._turn_sign_fixed = None if turn_sign is None else int(np.sign(turn_sign) or 1)
        self.control_hz = float(control_hz)
        self.n_arc = max(16, int(n_arc))
        self.event_t: Optional[float] = self._event_t_fixed
        self.turn_sign: int = 1 if self._turn_sign_fixed is None else int(self._turn_sign_fixed)
        self.theta0 = 0.0
        self.dtheta = 0.0
        self._arc_t = np.zeros(1, dtype=np.float64)
        self._arc_rel = np.zeros((1, 3), dtype=np.float64)
        self._out_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self._out_rel = np.zeros(3, dtype=np.float64)
        self._out_t0 = 0.0
        self._baked = False

    def _make_rng(self, rng) -> np.random.Generator:
        if rng is not None:
            if isinstance(rng, np.random.Generator):
                return rng
            return np.random.default_rng(int(rng))
        return np.random.default_rng(int(self.event_seed))

    def _reset_impl(self, rng) -> None:
        super()._reset_impl(rng)
        gen = self._make_rng(rng)
        if self._event_t_fixed is not None:
            self.event_t = float(self._event_t_fixed)
        else:
            lo, hi = self.event_t_lo, self.event_t_hi
            self.event_t = float(lo if hi <= lo else gen.uniform(lo, hi))
        if self._turn_sign_fixed is not None:
            self.turn_sign = int(self._turn_sign_fixed)
        else:
            self.turn_sign = int(gen.choice(np.array([-1, 1], dtype=np.int64)))
        self._bake_arc()

    def _inbound_xy(self) -> np.ndarray:
        u = np.asarray(self._axis, dtype=np.float64).reshape(3).copy()
        n = float(np.linalg.norm(u[:2]))
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u[0] /= n
        u[1] /= n
        u[2] = 0.0
        return u

    def _bake_arc(self) -> None:
        t0 = float(self.event_t if self.event_t is not None else 0.0)
        T = float(self.turn_duration_ticks)
        u = self._inbound_xy()
        self.theta0 = float(np.arctan2(u[1], u[0]))
        self.dtheta = float(self.turn_sign) * np.deg2rad(self.turn_angle_deg)
        self._out_t0 = t0 + T
        p0 = u * (self.speed * t0)
        if T <= 1e-12 or abs(self.speed) < 1e-12 or abs(self.dtheta) < 1e-12:
            self._arc_t = np.array([t0, t0], dtype=np.float64)
            self._arc_rel = np.stack([p0, p0], axis=0)
            self._out_axis = u.copy()
            self._out_rel = p0.copy()
            self._baked = True
            return
        n = self.n_arc
        taus = np.linspace(0.0, 1.0, n, dtype=np.float64)
        # Raised cosine: θ'(0)=θ'(1)=0 so heading rate does not jump.
        blend = 0.5 * (1.0 - np.cos(np.pi * taus))
        theta = self.theta0 + self.dtheta * blend
        rel = np.zeros((n, 3), dtype=np.float64)
        rel[0] = p0
        dt = T / float(n - 1)
        for i in range(1, n):
            h = 0.5 * (theta[i] + theta[i - 1])
            rel[i, 0] = rel[i - 1, 0] + self.speed * dt * float(np.cos(h))
            rel[i, 1] = rel[i - 1, 1] + self.speed * dt * float(np.sin(h))
        self._arc_t = t0 + taus * T
        self._arc_rel = rel
        self._out_axis = np.array(
            [float(np.cos(theta[-1])), float(np.sin(theta[-1])), 0.0],
            dtype=np.float64,
        )
        self._out_rel = rel[-1].copy()
        self._baked = True

    def heading_rad_at(self, t: float) -> float:
        t = float(t)
        t0 = float(self.event_t or 0.0)
        T = float(self.turn_duration_ticks)
        if t <= t0 or T <= 1e-12:
            return float(self.theta0)
        if t >= t0 + T:
            return float(self.theta0 + self.dtheta)
        tau = (t - t0) / T
        return float(self.theta0 + self.dtheta * 0.5 * (1.0 - np.cos(np.pi * tau)))

    def turn_normal_xy(self) -> np.ndarray:
        """Planar unit pointing toward the inside of the turn (eval only)."""
        u = self._inbound_xy()
        n = np.array([-u[1], u[0]], dtype=np.float64) * float(self.turn_sign)
        ln = float(np.linalg.norm(n))
        if ln < 1e-12:
            return np.array([0.0, 1.0], dtype=np.float64)
        return n / ln

    def xyz_at(self, t: float) -> np.ndarray:
        assert self.base_pos is not None
        t = float(t)
        if not self._baked:
            self._bake_arc()
        t0 = float(self.event_t or 0.0)
        if t <= t0 or abs(self.speed) < 1e-12:
            return self.base_pos + self._axis * (self.speed * t)
        if t >= self._out_t0:
            return self.base_pos + self._out_rel + self._out_axis * (
                self.speed * (t - self._out_t0)
            )
        rel = np.array(
            [
                np.interp(t, self._arc_t, self._arc_rel[:, 0]),
                np.interp(t, self._arc_t, self._arc_rel[:, 1]),
                np.interp(t, self._arc_t, self._arc_rel[:, 2]),
            ],
            dtype=np.float64,
        )
        return self.base_pos + rel

    def velocity_at(self, t: float, hz: float = 20.0) -> np.ndarray:
        if abs(self.speed) < 1e-12:
            return np.zeros(3, dtype=np.float64)
        h = self.heading_rad_at(t)
        return (
            np.array([np.cos(h), np.sin(h), 0.0], dtype=np.float64)
            * self.speed
            * float(hz)
        )

    def event_spec(self) -> dict[str, Any]:
        hz = float(self.control_hz)
        t_ev = None if self.event_t is None else float(self.event_t)
        mag = float(self.turn_angle_deg) * float(self.turn_sign)
        return {
            "event_type": str(getattr(self, "event_type", "turn") or "turn"),
            "event_time": t_ev,
            "event_time_s": None if t_ev is None else float(t_ev) / hz,
            "transition_duration": float(self.turn_duration_ticks),
            "transition_duration_s": float(self.turn_duration_ticks) / hz,
            "event_magnitude": mag,
            "event_seed": int(self.event_seed),
            "turn_event_seed": int(self.event_seed),
            "turn_event_t": t_ev,
            "turn_event_s": None if t_ev is None else float(t_ev) / hz,
            "turn_event_t_lo": float(self.event_t_lo),
            "turn_event_t_hi": float(self.event_t_hi),
            "turn_duration_ticks": float(self.turn_duration_ticks),
            "turn_duration_s": float(self.turn_duration_ticks) / hz,
            "turn_angle_deg": float(self.turn_angle_deg),
            "turn_sign": int(self.turn_sign),
            "turn_inbound_xy": self._inbound_xy()[:2].tolist(),
            "turn_normal_xy": self.turn_normal_xy().tolist(),
        }


def clone_smooth_turn_kwargs(traj: SmoothTurnTrajectory) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "axis": str(getattr(traj, "axis_name", "x") or "x"),
        "direction": float(getattr(traj, "direction", 1.0) or 1.0),
        "toward_center": bool(getattr(traj, "toward_center", False)),
        "event_seed": int(traj.event_seed),
        "event_t_lo": float(traj.event_t_lo),
        "event_t_hi": float(traj.event_t_hi),
        "turn_duration_ticks": float(traj.turn_duration_ticks),
        "turn_angle_deg": float(traj.turn_angle_deg),
        "control_hz": float(traj.control_hz),
        "n_arc": int(traj.n_arc),
    }
    hd = getattr(traj, "heading_deg", None)
    if hd is not None:
        kw["heading_deg"] = float(hd)
        kw.pop("axis", None)
    # After reset, pin sampled phase so a place-rail clone matches world time.
    if traj.event_t is not None:
        kw["event_t"] = float(traj.event_t)
        kw["turn_sign"] = int(traj.turn_sign)
    elif traj._turn_sign_fixed is not None:
        kw["turn_sign"] = int(traj._turn_sign_fixed)
    return kw
