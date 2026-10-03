"""Shared metrics for CLIRA experiment 1: δ vs reduced-budget action error."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.float().reshape(-1)
    n = torch.linalg.vector_norm(x).clamp(min=eps)
    return x / n


def mean_pool(tokens: torch.Tensor) -> torch.Tensor:
    t = tokens.float()
    if t.ndim == 3:
        t = t[0]
    return t.mean(dim=0)


def cosine_delta(phi_a: torch.Tensor, phi_b: torch.Tensor) -> float:
    a = l2_normalize(phi_a)
    b = l2_normalize(phi_b)
    sim = float(torch.dot(a, b).clamp(-1.0, 1.0).item())
    return 1.0 - sim


def relative_l1(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = b.abs().mean().clamp(min=eps)
    return float((a - b).abs().mean().item() / float(denom.item()))


def mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()) ** 2).mean().item())


def camera_phis_horizontal(
    tokens: torch.Tensor,
    *,
    latent_h: int,
    latent_w: int,
    patch_h: int,
    patch_w: int,
    n_cam: int = 2,
) -> list[torch.Tensor]:
    """Mean-pool tokens on a width-split concatenated frame (agentview | wrist)."""
    t = tokens.float()
    if t.ndim == 3:
        t = t[0]
    th = int(latent_h) // int(patch_h)
    tw = int(latent_w) // int(patch_w)
    if th * tw != int(t.shape[0]) or tw % n_cam != 0:
        return [mean_pool(t)]
    grid = t.reshape(th, tw, t.shape[-1])
    w = tw // n_cam
    out = []
    for i in range(n_cam):
        sl = grid[:, i * w : (i + 1) * w].reshape(-1, t.shape[-1])
        out.append(sl.mean(dim=0))
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def _rank(vals: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                ranks[order[t]] = avg
            i = j + 1
        return ranks

    rx, ry = _rank(xs), _rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def decile_bins(
    xs: Iterable[float], ys: Iterable[float]
) -> list[dict[str, float | int]]:
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
    if len(pts) < 10:
        return []
    pts.sort(key=lambda t: t[0])
    n = len(pts)
    bins: list[dict[str, float | int]] = []
    for b in range(10):
        lo = b * n // 10
        hi = (b + 1) * n // 10
        chunk = pts[lo:hi]
        if not chunk:
            continue
        ys_c = [y for _, y in chunk]
        xs_c = [x for x, _ in chunk]
        bins.append(
            {
                "bin": b,
                "delta_lo": min(xs_c),
                "delta_hi": max(xs_c),
                "delta_mean": sum(xs_c) / len(xs_c),
                "err_mean": sum(ys_c) / len(ys_c),
                "err_p90": sorted(ys_c)[int(0.9 * (len(ys_c) - 1))],
                "n": len(chunk),
            }
        )
    return bins


def bin_means_monotone(bins: list[dict], *, min_rise: float = 0.0) -> bool:
    if len(bins) < 3:
        return False
    means = [float(b["err_mean"]) for b in bins]
    rises = sum(1 for i in range(1, len(means)) if means[i] >= means[i - 1] - min_rise)
    return rises >= (len(means) - 1) * 0.6
