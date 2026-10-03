"""JSON-safe world-state payload for HTTP policy servers.

FastWAM eval processes may not import FasterPI. The dict matches
``fasterpi.world_state.WorldState`` fields (not the extra as_dict keys).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def world_state_payload(driver: Any, obs: Any | None = None) -> dict[str, Any]:
    hz = float(getattr(driver, "control_hz", 20.0) or 20.0)
    traj = getattr(driver, "trajectory", None)
    speed = 0.0
    disp = 0.0
    if traj is not None:
        t = float(getattr(traj, "t", 0.0))
        try:
            v = np.asarray(traj.velocity_at(t, hz=hz), dtype=np.float64).reshape(-1)
            speed = float(np.linalg.norm(v)) / max(hz, 1e-6)
        except Exception:
            speed = 0.0
        try:
            pos = np.asarray(traj.xyz_at(t), dtype=np.float64).reshape(3)
            last = getattr(driver, "_fasterpi_last_obj_pos", None)
            if last is not None:
                disp = float(np.linalg.norm(pos - np.asarray(last, dtype=np.float64).reshape(3)))
            driver._fasterpi_last_obj_pos = pos.copy()
        except Exception:
            disp = 0.0
    dist: Optional[float] = None
    fn = getattr(driver, "_eef_dist", None)
    if callable(fn) and obs is not None:
        try:
            d = fn(obs)
            dist = None if d is None else float(d)
        except Exception:
            dist = None
    phase = getattr(driver, "rails_phase", "pursuit")
    phase_s = str(getattr(phase, "value", phase) or "pursuit").lower()
    release_r = float(getattr(driver, "release_radius", 0.06) or 0.06)
    alpha = float(getattr(driver, "engage_alpha", 1.75) or 1.75)
    return {
        "phase": phase_s,
        "object_speed": float(speed),
        "object_disp": float(disp),
        "eef_dist": dist,
        "engage_radius": float(release_r) * float(alpha),
        "released": bool(getattr(driver, "released", False)),
        "control_hz": float(hz),
    }


def policy_predict_kwargs(policy: Any, driver: Any, obs: Any | None = None) -> dict[str, Any]:
    from .policy.http_client import HttpPolicyClient

    if isinstance(policy, HttpPolicyClient):
        try:
            return {"world_state": world_state_payload(driver, obs)}
        except Exception:
            return {}
    return {"driver": driver}
