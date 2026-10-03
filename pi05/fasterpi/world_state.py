"""World-state snapshot for FasterPI cache / schedule gates.

Gates are kinematics (phase, object speed, eef distance), not RGB L1.
When no snapshot is set, world-gated controllers refuse cache (conservative).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

_CURRENT: ContextVar[Optional["WorldState"]] = ContextVar("fasterpi_world_state", default=None)

# Object travel (m) allowed in one open-loop chunk before we cut replan_steps.
_OPEN_LOOP_BUDGET_M = 0.02
_SLOW_SPEED = 1e-4  # m / control tick (~2 mm/s at 20 Hz)
_FAST_SPEED = 5e-4


@dataclass
class WorldState:
    phase: str = "pursuit"  # pursuit | engage | free
    object_speed: float = 0.0  # metres per control tick
    object_disp: float = 0.0  # metres since last snapshot
    eef_dist: float | None = None
    engage_radius: float = 0.105
    released: bool = False
    control_hz: float = 20.0

    @property
    def slow_world(self) -> bool:
        return (
            self.released
            or self.phase in ("engage", "free")
            or float(self.object_speed) < _SLOW_SPEED
        )

    def cache_ok(self, kind: str) -> bool:
        """Whether a cache kind may hit. World-gate is extra permission (AND with RGB/state)."""
        kind = str(kind).strip().lower()
        if self.released or self.phase == "free":
            return True
        fast = float(self.object_speed) >= _FAST_SPEED and self.phase == "pursuit"
        near = self.eef_dist is not None and float(self.eef_dist) < float(self.engage_radius)
        if kind in ("step_cache", "d1"):
            return self.slow_world and not fast
        if kind in ("vision", "prefix_kv", "agentview"):
            if fast:
                return False
            return self.slow_world or near
        if kind in ("chunk_residual_cache", "c3", "wrist"):
            if fast and not near:
                return False
            return True
        return not fast

    def recommended_replan_steps(self, locked: int = 10, min_act: int = 3) -> int:
        locked = int(locked)
        if self.released or self.phase != "pursuit":
            return locked
        v = max(float(self.object_speed), 1e-9)
        raw = int(round(_OPEN_LOOP_BUDGET_M / v))
        return int(min(locked, max(int(min_act), raw)))

    def recommended_nfe(self, default: int = 10) -> int:
        default = int(default)
        if self.released or self.phase == "free":
            return min(default, 6)
        if self.phase == "engage":
            return min(default, 8)
        if float(self.object_speed) < _SLOW_SPEED:
            return min(default, 8)
        return default

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "object_speed": float(self.object_speed),
            "object_disp": float(self.object_disp),
            "eef_dist": None if self.eef_dist is None else float(self.eef_dist),
            "engage_radius": float(self.engage_radius),
            "released": bool(self.released),
            "control_hz": float(self.control_hz),
            "replan_steps": self.recommended_replan_steps(),
            "nfe": self.recommended_nfe(),
        }

    @classmethod
    def from_driver(cls, driver: Any, obs: Any | None = None) -> "WorldState":
        hz = float(getattr(driver, "control_hz", 20.0) or 20.0)
        traj = getattr(driver, "trajectory", None)
        speed = 0.0
        disp = 0.0
        if traj is not None:
            t = float(getattr(traj, "t", 0.0))
            # Trajectory.velocity_at is m/s; WorldState.object_speed is m/tick.
            v = np.asarray(traj.velocity_at(t, hz=hz), dtype=np.float64).reshape(-1)
            speed = float(np.linalg.norm(v)) / max(hz, 1e-6)
            pos = np.asarray(traj.xyz_at(t), dtype=np.float64).reshape(3)
            last = getattr(driver, "_fasterpi_last_obj_pos", None)
            if last is not None:
                disp = float(np.linalg.norm(pos - np.asarray(last, dtype=np.float64).reshape(3)))
            driver._fasterpi_last_obj_pos = pos.copy()
        dist = None
        fn = getattr(driver, "_eef_dist", None)
        if callable(fn) and obs is not None:
            try:
                dist = fn(obs)
            except Exception:
                dist = None
        phase = getattr(driver, "rails_phase", "pursuit")
        phase_s = str(getattr(phase, "value", phase) or "pursuit").lower()
        release_r = float(getattr(driver, "release_radius", 0.06) or 0.06)
        alpha = float(getattr(driver, "engage_alpha", 1.75) or 1.75)
        return cls(
            phase=phase_s,
            object_speed=speed,
            object_disp=disp,
            eef_dist=None if dist is None else float(dist),
            engage_radius=release_r * alpha,
            released=bool(getattr(driver, "released", False)),
            control_hz=hz,
        )


def get_world_state() -> WorldState | None:
    return _CURRENT.get()


def set_world_state(state: WorldState | None):
    return _CURRENT.set(state)


def reset_world_state(token) -> None:
    _CURRENT.reset(token)


@contextmanager
def world_scope(state: WorldState | None) -> Iterator[WorldState | None]:
    token = _CURRENT.set(state)
    try:
        yield state
    finally:
        _CURRENT.reset(token)
