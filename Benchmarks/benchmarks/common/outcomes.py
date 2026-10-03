"""Episode outcome fields: rails release ≠ grasp hold ≠ task success.

Missing historical fields must stay unknown (JSON null), never coerced to 0.
Gripper closure is neither necessary nor sufficient for grasp hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def unknown_bool(value: Any) -> Optional[bool]:
    """Read a tri-state flag. Absent / null → None, never False-by-default."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"", "none", "null", "unknown", "na", "n/a"}:
            return None
        if key in {"1", "true", "yes", "on"}:
            return True
        if key in {"0", "false", "no", "off"}:
            return False
    return None


def read_rails_released(ep: dict[str, Any]) -> Optional[bool]:
    if "rails_released" in ep:
        return unknown_bool(ep.get("rails_released"))
    if "released" in ep:
        return unknown_bool(ep.get("released"))
    return None


def read_grasp_hold(ep: dict[str, Any]) -> Optional[bool]:
    """True grasp-hold. Do not fall back to rails release or gripper close."""
    if "grasp_hold" in ep:
        return unknown_bool(ep.get("grasp_hold"))
    return None


def read_task_success(ep: dict[str, Any]) -> Optional[bool]:
    if "task_success" in ep:
        return unknown_bool(ep.get("task_success"))
    if "success" in ep:
        return unknown_bool(ep.get("success"))
    return None


def read_aux_events(ep: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    raw = ep.get("aux_events")
    if raw is None:
        return None
    if isinstance(raw, list):
        return list(raw)
    return None


# ---------------------------------------------------------------------------
# Live tracker (LIBERO / RoboTwin)
# ---------------------------------------------------------------------------

# 0.5 s at 20 Hz. Lift threshold is 2 cm above the seated support.
DEFAULT_HOLD_TICKS = 10
DEFAULT_LIFT_M = 0.02


@dataclass
class GraspHoldTracker:
    """Platform-fixed grasp-hold: lift + sustained near-hand tracking.

    Gripper closing may corroborate but never grants hold by itself.
    """

    hold_radius_m: float
    lift_m: float = DEFAULT_LIFT_M
    min_ticks: int = DEFAULT_HOLD_TICKS
    seated_z: Optional[float] = None
    streak: int = 0
    max_streak: int = 0
    lift_max_m: float = 0.0
    hold: bool = False
    reason: str = "never_near_and_lifted"
    samples: int = 0

    def set_seated_z(self, z: float | None) -> None:
        if z is None:
            return
        self.seated_z = float(z)

    def update(
        self,
        *,
        dist_m: float | None,
        object_z: float | None,
        dt_ticks: float = 1.0,
    ) -> None:
        self.samples += 1
        if dist_m is None or object_z is None or self.seated_z is None:
            self.streak = 0
            return
        lift = float(object_z) - float(self.seated_z)
        if lift > self.lift_max_m:
            self.lift_max_m = float(lift)
        near = float(dist_m) < float(self.hold_radius_m)
        lifted = lift >= float(self.lift_m)
        if near and lifted:
            self.streak += max(1, int(round(float(dt_ticks))))
            if self.streak > self.max_streak:
                self.max_streak = int(self.streak)
        else:
            self.streak = 0
        if self.max_streak >= int(self.min_ticks):
            self.hold = True
            self.reason = "lift_and_follow"
        elif self.samples > 0:
            if self.lift_max_m < float(self.lift_m):
                self.reason = "no_lift"
            elif self.max_streak == 0:
                self.reason = "no_near_while_lifted"
            else:
                self.reason = "follow_too_short"

    def snapshot(self) -> dict[str, Any]:
        hold: Optional[bool]
        if self.samples <= 0 or self.seated_z is None:
            hold = None
        else:
            hold = bool(self.hold)
        return {
            "grasp_hold": hold,
            "grasp_hold_reason": None if hold is None else self.reason,
            "grasp_hold_streak_max": int(self.max_streak),
            "grasp_hold_lift_max_m": float(self.lift_max_m),
            "grasp_hold_seated_z": None if self.seated_z is None else float(self.seated_z),
            "grasp_hold_radius_m": float(self.hold_radius_m),
            "grasp_hold_min_ticks": int(self.min_ticks),
        }


@dataclass
class AuxLog:
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, kind: str, *, t: float | None = None, **extra: Any) -> None:
        row: dict[str, Any] = {"kind": str(kind)}
        if t is not None:
            row["t"] = float(t)
        for key, val in extra.items():
            if val is not None:
                row[key] = val
        self.events.append(row)

    def triggered(self, kind: str) -> bool:
        return any(e.get("kind") == kind for e in self.events)


def outcome_fields(
    *,
    rails_released: bool,
    grasp_hold: Optional[bool],
    task_success: bool,
    aux_events: Optional[list[dict[str, Any]]] = None,
    grasp_hold_extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rails_released": bool(rails_released),
        "grasp_hold": None if grasp_hold is None else bool(grasp_hold),
        "task_success": bool(task_success),
        "aux_events": list(aux_events or []),
        # Legacy aliases: released = rails constraint; grasp_success is unknown
        # unless the new grasp_hold field is present. Never copy rails→grasp.
        "released": bool(rails_released),
        "grasp_success": None if grasp_hold is None else bool(grasp_hold),
        "success": bool(task_success),
    }
    if grasp_hold_extra:
        out.update(grasp_hold_extra)
    return out


def summarize_tri_state(values: list[Optional[bool]]) -> dict[str, Any]:
    known = [v for v in values if v is not None]
    n = len(values)
    n_unknown = n - len(known)
    n_true = sum(1 for v in known if v)
    return {
        "n": n,
        "n_known": len(known),
        "n_unknown": n_unknown,
        "n_true": n_true,
        "rate_known": (n_true / len(known)) if known else None,
    }
