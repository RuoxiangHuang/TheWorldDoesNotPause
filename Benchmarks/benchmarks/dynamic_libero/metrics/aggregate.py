"""Aggregate SR tables, AUC-SR / v50 curves, and bootstrap CIs for paper assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


def aggregate_sr_table(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Build nested SR[accel][trajectory][speed] and paired deltas."""
    nested: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    meta: dict[str, dict[str, float]] = {}
    for c in cells:
        nested[c["accel"]][c["trajectory"]][str(c["speed"])] = float(c["sr"])
        meta[c["accel"]] = {
            "latency_ms": float(c.get("latency_ms", 0.0)),
            "n_freeze": float(c.get("n_freeze", 0)),
            "blind_window_ratio": c.get("blind_window_ratio"),
            "replan_steps": c.get("replan_steps"),
            "backend": c.get("backend"),
        }

    deltas: dict[str, dict[str, float]] = defaultdict(dict)
    if "off" in nested and "full" in nested:
        for traj, speeds in nested["off"].items():
            for sp, sr_off in speeds.items():
                sr_full = nested["full"].get(traj, {}).get(sp)
                if sr_full is not None:
                    deltas[traj][sp] = float(sr_full) - float(sr_off)

    return {
        "sr": {a: {t: dict(s) for t, s in trajs.items()} for a, trajs in nested.items()},
        "latency": meta,
        "delta_full_minus_off": {t: dict(s) for t, s in deltas.items()},
    }


def _sorted_speed_curve(points: dict[str, float] | dict[float, float]) -> list[tuple[float, float]]:
    items = [(float(k), float(v)) for k, v in points.items()]
    items.sort(key=lambda x: x[0])
    return items


def auc_sr(speeds: Sequence[float], srs: Sequence[float]) -> float:
    """Trapezoidal area under SR(speed). Speeds must be sorted ascending."""
    if len(speeds) < 2 or len(speeds) != len(srs):
        return float("nan")
    area = 0.0
    for i in range(len(speeds) - 1):
        dx = float(speeds[i + 1]) - float(speeds[i])
        if dx <= 0:
            continue
        area += 0.5 * (float(srs[i]) + float(srs[i + 1])) * dx
    return float(area)


def normalize_auc_sr(
    speeds: Sequence[float],
    srs: Sequence[float],
    *,
    v_min: float | None = None,
    v_max: float | None = None,
) -> float:
    """AUC divided by speed span → mean SR under the curve on [v_min, v_max]."""
    xs = [float(x) for x in speeds]
    ys = [float(y) for y in srs]
    if len(xs) < 2:
        return float("nan")
    lo = float(v_min) if v_min is not None else xs[0]
    hi = float(v_max) if v_max is not None else xs[-1]
    span = hi - lo
    if span <= 0:
        return float("nan")
    return auc_sr(xs, ys) / span


def v50(speeds: Sequence[float], srs: Sequence[float], *, threshold: float = 0.5) -> float | None:
    """Smallest speed where SR drops to ``threshold`` via linear interpolation.

    Returns ``None`` if the curve never crosses the threshold (always above or
    always below without a downward crossing).
    """
    curve = list(zip([float(s) for s in speeds], [float(r) for r in srs]))
    if len(curve) < 2:
        return None
    curve.sort(key=lambda x: x[0])
    # Already at/below at the first point.
    if curve[0][1] <= threshold:
        return float(curve[0][0])
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if y0 >= threshold and y1 <= threshold:
            if y0 == y1:
                return float(x1)
            t = (y0 - threshold) / (y0 - y1)
            return float(x0 + t * (x1 - x0))
    return None


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean of Bernoulli / real values."""
    import numpy as np

    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        means[i] = float(sample.mean())
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {"mean": float(arr.mean()), "lo": lo, "hi": hi, "n": int(n)}


def curve_metrics(
    speed_to_sr: dict[str, float] | dict[float, float],
    *,
    v_min: float | None = None,
    v_max: float | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    curve = _sorted_speed_curve(speed_to_sr)
    speeds = [x for x, _ in curve]
    srs = [y for _, y in curve]
    return {
        "speeds": speeds,
        "sr": srs,
        "auc_sr": auc_sr(speeds, srs),
        "norm_auc_sr": normalize_auc_sr(speeds, srs, v_min=v_min, v_max=v_max),
        "v50": v50(speeds, srs, threshold=threshold),
    }


def enrich_table_with_curve_metrics(
    table: dict[str, Any],
    *,
    v_min: float = 0.0,
    v_max: float = 0.010,
) -> dict[str, Any]:
    """Attach AUC-SR / v50 under ``table['curves'][accel][trajectory]``."""
    curves: dict[str, dict[str, Any]] = {}
    for accel, trajs in table.get("sr", {}).items():
        curves[accel] = {}
        for traj, speeds in trajs.items():
            curves[accel][traj] = curve_metrics(speeds, v_min=v_min, v_max=v_max)
    out = dict(table)
    out["curves"] = curves
    # Paired ΔAUC if both off and full present.
    delta_auc: dict[str, float] = {}
    if "off" in curves and "full" in curves:
        for traj in curves["off"]:
            if traj in curves["full"]:
                a0 = curves["off"][traj].get("norm_auc_sr")
                a1 = curves["full"][traj].get("norm_auc_sr")
                if a0 is not None and a1 is not None and math.isfinite(a0) and math.isfinite(a1):
                    delta_auc[traj] = float(a1) - float(a0)
    out["delta_norm_auc_full_minus_off"] = delta_auc
    return out


def episode_success_ci(
    episodes: Sequence[dict[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    succ = [1.0 if bool(e.get("success")) else 0.0 for e in episodes]
    return bootstrap_ci(succ, n_boot=n_boot, seed=seed)


def write_curve_csv(cells: list[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "accel",
        "trajectory",
        "speed",
        "sr",
        "n",
        "latency_ms",
        "n_freeze",
        "blind_window_ratio",
        "replan_steps",
        "feedback_hz",
        "feedback_period_s",
        "backend",
        "pursuit_lag_m",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cells:
            w.writerow({k: c.get(k) for k in fields})


def write_summary_md(table: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dynamic-LIBERO — summary",
        "",
        "## Latency",
        "",
        "| accel | latency_ms | n_freeze |",
        "|---|---:|---:|",
    ]
    for accel, m in table.get("latency", {}).items():
        lines.append(
            f"| {accel} | {m.get('latency_ms', 0):.1f} | {float(m.get('n_freeze', 0)):.3f} |"
        )
    lines += ["", "## SR (accel × trajectory × speed)", ""]
    for accel, trajs in table.get("sr", {}).items():
        lines.append(f"### accel={accel}")
        lines.append("")
        lines.append("| trajectory | speed | SR |")
        lines.append("|---|---:|---:|")
        for traj, speeds in trajs.items():
            for sp, sr in sorted(speeds.items(), key=lambda x: float(x[0])):
                lines.append(f"| {traj} | {sp} | {sr:.3f} |")
        lines.append("")
    if table.get("delta_full_minus_off"):
        lines += ["## Paired ΔSR = SR(full) − SR(off)", ""]
        lines.append("| trajectory | speed | ΔSR |")
        lines.append("|---|---:|---:|")
        for traj, speeds in table["delta_full_minus_off"].items():
            for sp, d in sorted(speeds.items(), key=lambda x: float(x[0])):
                lines.append(f"| {traj} | {sp} | {d:+.3f} |")
        lines.append("")
    if table.get("curves"):
        lines += ["## Curve metrics (AUC-SR / v50)", ""]
        lines.append("| accel | trajectory | norm_AUC-SR | v50 |")
        lines.append("|---|---|---:|---:|")
        for accel, trajs in table["curves"].items():
            for traj, m in trajs.items():
                v50_s = "—" if m.get("v50") is None else f"{m['v50']:.4f}"
                nau = m.get("norm_auc_sr")
                nau_s = "nan" if nau is None or not math.isfinite(float(nau)) else f"{float(nau):.3f}"
                lines.append(f"| {accel} | {traj} | {nau_s} | {v50_s} |")
        lines.append("")
    if table.get("delta_norm_auc_full_minus_off"):
        lines += ["## Paired Δnorm_AUC = AUC(full) − AUC(off)", ""]
        lines.append("| trajectory | Δnorm_AUC |")
        lines.append("|---|---:|")
        for traj, d in table["delta_norm_auc_full_minus_off"].items():
            lines.append(f"| {traj} | {d:+.3f} |")
        lines.append("")
    path.write_text("\n".join(lines))


def _load_cells(paths: list[str]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for pat in paths:
        for p in sorted(Path().glob(pat) if any(c in pat for c in "*?[") else [Path(pat)]):
            if not p.is_file():
                continue
            data = json.loads(p.read_text())
            if "cells" in data:
                cells.extend(data["cells"])
            elif "sr" in data and "accel" in data:
                cells.append(data)
            elif "episodes" in data and "accel" not in data:
                # paired summary: skip or flatten later
                continue
            elif "episodes" in data:
                cells.append(data)
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Dynamic-LIBERO JSON results")
    ap.add_argument("--inputs", nargs="+", required=True, help="JSON paths or globs")
    ap.add_argument("--out", type=str, default="evaluate_results/dynamic_libero/SUMMARY.md")
    ap.add_argument("--csv", type=str, default="evaluate_results/dynamic_libero/sr_curves.csv")
    ap.add_argument("--v-min", type=float, default=0.0)
    ap.add_argument("--v-max", type=float, default=0.010)
    args = ap.parse_args()
    cells = _load_cells(args.inputs)
    table = enrich_table_with_curve_metrics(
        aggregate_sr_table(cells),
        v_min=args.v_min,
        v_max=args.v_max,
    )
    write_summary_md(table, Path(args.out))
    write_curve_csv(cells, Path(args.csv))
    out_json = Path(args.out).with_suffix(".json")
    out_json.write_text(json.dumps({"table": table, "n_cells": len(cells)}, indent=2))
    print(f"Wrote {args.out} and {args.csv} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
