"""Driven-car path for extra ``pursue_toycar``.

Base-subset toy cars ride rails: constant arc speed, yaw = velocity heading,
linear or one catalog curve, mesh-swap of a soup can / 057_toycar.

This path is a steered vehicle: slalom, a tight U-turn, speed drops in
curvature, then an escape toward the camera-near edge. Front-wheel steer
and wheel spin come from bicycle-model kinematics, not from rails.
"""

from __future__ import annotations

import numpy as np

WHEEL_R = 0.0130
WHEELBASE = 0.070
IDLE = 0.05
DRIVE = 3.40
V_CRUISE = 0.24
V_MIN = 0.06
STEER_MAX = 0.55

# Ground-center waypoints (m). Irregular slalom with a fake-out and a
# tight hook so the chase is not a single smooth arc.
WAYPOINTS = np.array(
    [
        [0.20, -0.18],
        [0.10, -0.04],
        [0.16, -0.14],
        [0.02, -0.12],
        [-0.06, 0.02],
        [0.05, 0.08],
        [-0.08, 0.06],
        [-0.16, -0.08],
        [-0.06, -0.14],
        [-0.18, -0.02],
        [-0.10, 0.10],
        [0.04, 0.14],
    ],
    dtype=np.float64,
)

# Left-arm intercept target used by the grasp demo.
INTERCEPT_XY = np.array([-0.16, -0.03], dtype=np.float64)


def _catmull(pts: np.ndarray, n_per: int = 24) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64)
    ext = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for k in range(n_per):
            t = k / float(n_per)
            t2, t3 = t * t, t * t * t
            out.append(
                0.5
                * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
            )
    out.append(p[-1])
    return np.asarray(out, dtype=np.float64)


def _arc_table(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    tang = np.gradient(xy, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    heading = np.arctan2(tang[:, 1], tang[:, 0])
    heading = np.unwrap(heading)
    ds = np.maximum(np.gradient(s), 1e-4)
    dhead = np.gradient(heading) / ds
    kappa = np.clip(np.abs(dhead), 0.0, 18.0)
    return s, heading, kappa


def _speed_profile(kappa: np.ndarray) -> np.ndarray:
    k = kappa / (float(np.max(kappa)) + 1e-9)
    return V_CRUISE - (V_CRUISE - V_MIN) * np.clip(k, 0.0, 1.0)


class DrivePath:
    def __init__(self) -> None:
        self.xy = _catmull(WAYPOINTS, n_per=28)
        self.s, self.heading, self.kappa = _arc_table(self.xy)
        self.v = _speed_profile(self.kappa)
        dt = np.diff(self.s, prepend=self.s[0]) / np.maximum(self.v, V_MIN)
        dt[0] = 0.0
        self.t_drive = np.cumsum(dt)
        scale = DRIVE / max(float(self.t_drive[-1]), 1e-6)
        self.t_drive = self.t_drive * scale
        self.v = self.v / scale
        self.length = float(self.s[-1])
        self.dhead = np.gradient(self.heading)
        self.intercept_t = self._nearest_t(INTERCEPT_XY)

    def _nearest_t(self, xy: np.ndarray) -> float:
        d = np.linalg.norm(self.xy - np.asarray(xy, dtype=np.float64).reshape(1, 2), axis=1)
        i = int(np.argmin(d))
        return float(self.t_drive[i])

    def sample(self, t: float) -> dict:
        """Body state at wall-clock time (includes idle)."""
        td = float(t) - IDLE
        if td <= 0.0:
            i = 0
            s = 0.0
            v = 0.0
            steer = 0.0
            spin = 0.0
        else:
            td = min(td, float(self.t_drive[-1]))
            i = int(np.searchsorted(self.t_drive, td, side="right") - 1)
            i = max(0, min(i, len(self.xy) - 1))
            s = float(self.s[i])
            v = float(self.v[i])
            dh = float(self.dhead[i])
            steer = float(
                np.clip(np.arctan(WHEELBASE * float(self.kappa[i])) * np.sign(dh + 1e-12), -STEER_MAX, STEER_MAX)
            )
            spin = s / WHEEL_R
        yaw = float(self.heading[i])
        return {
            "xy": self.xy[i].copy(),
            "yaw": yaw,
            "speed": v,
            "s": s,
            "kappa": float(self.kappa[i]),
            "steer": steer,
            "spin": spin,
            "i": i,
        }


PATH = DrivePath()
INTERCEPT_T = float(PATH.intercept_t + IDLE)
TOTAL = IDLE + DRIVE

HUBS = (
    np.array([0.5 * WHEELBASE, 0.5 * 0.044, WHEEL_R], dtype=np.float64),
    np.array([0.5 * WHEELBASE, -0.5 * 0.044, WHEEL_R], dtype=np.float64),
    np.array([-0.5 * WHEELBASE, 0.5 * 0.044, WHEEL_R], dtype=np.float64),
    np.array([-0.5 * WHEELBASE, -0.5 * 0.044, WHEEL_R], dtype=np.float64),
)
FRONT = (True, True, False, False)


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


def _roty(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[0, 0] = c
    T[0, 2] = s
    T[2, 0] = -s
    T[2, 2] = c
    return T


def _transl(p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(p, dtype=np.float64).reshape(3)
    return T


def body_matrix(state: dict, table_z: float) -> np.ndarray:
    xy = np.asarray(state["xy"], dtype=np.float64)
    return _transl([xy[0], xy[1], table_z]) @ _rotz(float(state["yaw"]))


def wheel_matrix(state: dict, hub: np.ndarray, table_z: float, *, front: bool) -> np.ndarray:
    steer = float(state["steer"]) if front else 0.0
    spin = float(state["spin"])
    return body_matrix(state, table_z) @ _transl(hub) @ _rotz(steer) @ _roty(spin)


def cabin_world(state: dict, table_z: float) -> np.ndarray:
    T = body_matrix(state, table_z)
    local = np.array([0.000, 0.0, 0.028, 1.0])
    return (T @ local)[:3]

