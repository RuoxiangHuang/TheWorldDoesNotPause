"""Seeded random polyline — episode-varying paths for Dynamic-LIBERO."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .polyline import PolylineTrajectory


class RandomPolylineTrajectory(PolylineTrajectory):
    """Piecewise-linear path with waypoints sampled at each ``reset``.

    Starts at the object base pose (relative origin), then walks randomly in
    the table plane within ``extent``. Paths are deterministic given
    ``seed`` + ``base_pos`` (same init → same path).
    """

    name = "random_polyline"

    def __init__(
        self,
        speed: float = 0.003,
        n_waypoints: int = 5,
        extent: float = 0.12,
        seed: Optional[int] = None,
        planar: bool = True,
        z_jitter: float = 0.0,
        **kwargs,
    ):
        # Placeholder waypoints; replaced in ``_reset_impl``.
        dummy = [(0.0, 0.0, 0.0), (float(extent), 0.0, 0.0)]
        super().__init__(speed=speed, waypoints=dummy, **kwargs)
        self.n_waypoints = max(2, int(n_waypoints))
        self.extent = float(max(extent, 1e-3))
        self.seed = None if seed is None else int(seed)
        self.planar = bool(planar)
        self.z_jitter = float(max(0.0, z_jitter))

    def _make_rng(self, rng) -> np.random.Generator:
        if rng is not None:
            return rng
        if self.seed is None or self.base_pos is None:
            return np.random.default_rng()
        # Mix seed with base pose so different inits get different paths.
        p = np.round(self.base_pos, decimals=4)
        ss = np.random.SeedSequence(
            [self.seed & 0xFFFFFFFF, int(abs(p[0]) * 1e4), int(abs(p[1]) * 1e4)]
        )
        return np.random.default_rng(ss)

    def _reset_impl(self, rng) -> None:
        rng = self._make_rng(rng)
        pts = [np.zeros(3, dtype=np.float64)]
        pos = np.zeros(3, dtype=np.float64)
        # Random walk with soft clamp to a square of side 2*extent.
        step = self.extent / max(self.n_waypoints - 1, 1)
        for _ in range(self.n_waypoints - 1):
            ang = float(rng.uniform(0.0, 2.0 * np.pi))
            length = float(rng.uniform(0.4 * step, 1.4 * step))
            delta = np.array(
                [length * np.cos(ang), length * np.sin(ang), 0.0],
                dtype=np.float64,
            )
            if not self.planar and self.z_jitter > 0.0:
                delta[2] = float(rng.uniform(-self.z_jitter, self.z_jitter))
            pos = pos + delta
            pos[0] = float(np.clip(pos[0], -self.extent, self.extent))
            pos[1] = float(np.clip(pos[1], -self.extent, self.extent))
            if self.planar:
                pos[2] = 0.0
            else:
                pos[2] = float(np.clip(pos[2], -self.z_jitter, self.z_jitter))
            pts.append(pos.copy())
        self.rel_waypoints = pts
        super()._reset_impl(rng)
