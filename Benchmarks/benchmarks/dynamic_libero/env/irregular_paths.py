"""Hand-authored smooth curves for the 20 Dynamic-LIBERO irregular extras.

Each libero_object skill gets two C1 planar curves (pick target / basket).
Shapes are analytic (sine / arc / clothoid / lissajous / …), not polylines,
so heading never snaps.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

Variant = Literal["pick", "place"]
Waypoint = tuple[float, float, float]

_N_SRC = 400


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(np.maximum(d, 1e-12))])
    total = float(cum[-1])
    if total < 1e-8:
        out = np.zeros((n, 3), dtype=np.float64)
        out[:, 0] = np.linspace(0.0, 0.05, n)
        return out
    s = np.linspace(0.0, total, int(n))
    out = np.empty((int(n), 3), dtype=np.float64)
    for k, sk in enumerate(s):
        i = int(np.searchsorted(cum, sk, side="right") - 1)
        i = max(0, min(i, len(cum) - 2))
        span = float(cum[i + 1] - cum[i])
        u = 0.0 if span < 1e-12 else (float(sk) - float(cum[i])) / span
        out[k] = (1.0 - u) * pts[i] + u * pts[i + 1]
    out[0] = 0.0
    return out


def _xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pts = np.stack([x - x[0], y - y[0], np.zeros_like(x)], axis=1)
    return pts


def _sine(*, length: float, amp: float, waves: float, n: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, _N_SRC)
    return _resample(_xy(length * u, amp * np.sin(waves * np.pi * u)), n)


def _arc(*, radius: float, angle: float, sign: float, n: int) -> np.ndarray:
    th = np.linspace(0.0, float(angle), _N_SRC)
    x = radius * np.sin(th)
    y = sign * radius * (1.0 - np.cos(th))
    return _resample(_xy(x, y), n)


def _ellipse(*, a: float, b: float, angle: float, sign: float, n: int) -> np.ndarray:
    th = np.linspace(0.0, float(angle), _N_SRC)
    x = a * np.sin(th)
    y = sign * b * (1.0 - np.cos(th))
    return _resample(_xy(x, y), n)


def _clothoid(*, length: float, sharpness: float, sign: float, n: int) -> np.ndarray:
    n0 = _N_SRC
    ds = float(length) / float(n0 - 1)
    pts = np.zeros((n0, 3), dtype=np.float64)
    th = 0.0
    x = 0.0
    y = 0.0
    for i in range(1, n0):
        s = i * ds
        th = sign * 0.5 * sharpness * s * s
        x += ds * np.cos(th)
        y += ds * np.sin(th)
        pts[i] = (x, y, 0.0)
    return _resample(pts, n)


def _lissajous(*, ax: float, ay: float, wx: float, wy: float, n: int, u_end: float = 1.0) -> np.ndarray:
    u = np.linspace(0.0, float(u_end), _N_SRC)
    x = ax * np.sin(wx * np.pi * u)
    y = ay * np.sin(wy * np.pi * u)
    return _resample(_xy(x, y), n)


def _spiral(*, turns: float, r0: float, r1: float, sign: float, n: int) -> np.ndarray:
    th = np.linspace(0.0, float(turns) * 2.0 * np.pi, _N_SRC)
    r = np.linspace(float(r0), float(r1), _N_SRC)
    x = r * np.cos(th) - r[0]
    y = sign * r * np.sin(th)
    return _resample(_xy(x, y), n)


def _rotated_sine(*, length: float, amp: float, waves: float, yaw: float, n: int) -> np.ndarray:
    pts = _sine(length=length, amp=amp, waves=waves, n=_N_SRC)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return _resample(pts @ rot.T, n)


def _chirp(*, length: float, amp: float, n: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, _N_SRC)
    phase = np.pi * u * (1.0 + 0.65 * u)
    return _resample(_xy(length * u, amp * np.sin(phase)), n)


def _limaçon(*, a: float, b: float, sign: float, n: int) -> np.ndarray:
    th = np.linspace(0.0, 1.35 * np.pi, _N_SRC)
    r = a + b * np.cos(th)
    x = r * np.sin(th)
    y = sign * r * (1.0 - np.cos(th))
    return _resample(_xy(x, y), n)


_GENERATORS: dict[str, Callable[[int], np.ndarray]] = {
    "s_wave": lambda n: _sine(length=0.30, amp=0.055, waves=1.0, n=n),
    "arc_right": lambda n: _arc(radius=0.16, angle=1.85, sign=1.0, n=n),
    "double_s": lambda n: _sine(length=0.32, amp=0.042, waves=2.0, n=n),
    "arc_left": lambda n: _arc(radius=0.15, angle=2.05, sign=-1.0, n=n),
    "shallow_s": lambda n: _sine(length=0.30, amp=0.028, waves=1.0, n=n),
    "wide_ellipse": lambda n: _ellipse(a=0.18, b=0.09, angle=2.4, sign=1.0, n=n),
    # Official t03.pick.irregular (replaces length=0.34 / sharpness=18).
    "clothoid_right": lambda n: _clothoid(length=0.26, sharpness=9.0, sign=1.0, n=n),
    "clothoid_left": lambda n: _clothoid(length=0.33, sharpness=20.0, sign=-1.0, n=n),
    "liss_1_2": lambda n: _lissajous(ax=0.16, ay=0.055, wx=1.0, wy=2.0, n=n, u_end=0.85),
    "ellipse_left": lambda n: _ellipse(a=0.16, b=0.10, angle=2.15, sign=-1.0, n=n),
    "soft_spiral": lambda n: _spiral(turns=0.42, r0=0.055, r1=0.13, sign=1.0, n=n),
    "comma": lambda n: _limaçon(a=0.09, b=0.05, sign=1.0, n=n),
    "diag_s": lambda n: _rotated_sine(length=0.30, amp=0.048, waves=1.0, yaw=0.55, n=n),
    "bowl": lambda n: _sine(length=0.30, amp=-0.05, waves=1.0, n=n),
    "long_arc": lambda n: _arc(radius=0.22, angle=1.25, sign=1.0, n=n),
    "side_s": lambda n: _rotated_sine(length=0.28, amp=0.05, waves=1.15, yaw=1.15, n=n),
    "chirp_s": lambda n: _chirp(length=0.32, amp=0.046, n=n),
    "bean": lambda n: _limaçon(a=0.08, b=0.045, sign=-1.0, n=n),
    "tight_s": lambda n: _sine(length=0.28, amp=0.038, waves=2.4, n=n),
    "soft_spiral_ccw": lambda n: _spiral(turns=0.38, r0=0.06, r1=0.125, sign=-1.0, n=n),
}

IRREGULAR_CURVES: dict[tuple[int, Variant], str] = {
    (0, "pick"): "s_wave",
    (0, "place"): "arc_right",
    (1, "pick"): "double_s",
    (1, "place"): "arc_left",
    (2, "pick"): "shallow_s",
    (2, "place"): "wide_ellipse",
    (3, "pick"): "clothoid_right",
    (3, "place"): "clothoid_left",
    (4, "pick"): "liss_1_2",
    (4, "place"): "ellipse_left",
    (5, "pick"): "soft_spiral",
    (5, "place"): "comma",
    (6, "pick"): "diag_s",
    (6, "place"): "bowl",
    (7, "pick"): "long_arc",
    (7, "place"): "side_s",
    (8, "pick"): "chirp_s",
    (8, "place"): "bean",
    (9, "pick"): "tight_s",
    (9, "place"): "soft_spiral_ccw",
}


def sample_curve(curve_id: str, n: int = 160) -> np.ndarray:
    if curve_id not in _GENERATORS:
        raise KeyError(f"Unknown curve_id {curve_id!r}. Known: {sorted(_GENERATORS)}")
    pts = _GENERATORS[curve_id](int(n))
    return np.asarray(pts, dtype=np.float64)


def irregular_curve_id(task_id: int, variant: Variant) -> str:
    key = (int(task_id), variant)
    if key not in IRREGULAR_CURVES:
        raise KeyError(f"No irregular curve for task_id={task_id} variant={variant}")
    return IRREGULAR_CURVES[key]


def irregular_path_name(task_id: int, variant: Variant) -> str:
    return irregular_curve_id(task_id, variant)


def irregular_waypoints(task_id: int, variant: Variant, n: int = 24) -> tuple[Waypoint, ...]:
    """Dense samples of the smooth curve (for clearance / tests, not sharp corners)."""
    pts = sample_curve(irregular_curve_id(task_id, variant), n=n)
    return tuple((float(p[0]), float(p[1]), float(p[2])) for p in pts)
