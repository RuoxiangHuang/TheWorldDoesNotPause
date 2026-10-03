#!/usr/bin/env python3
"""Analyze coupling probe JSONL: δ_world vs err_res / err_hold."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def _rank(vals: list[float]) -> list[float]:
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


def _decile_bins(rows: list[dict], key_x: str, key_y: str) -> list[dict]:
    pts = [(float(r[key_x]), float(r[key_y])) for r in rows if r.get(key_y) is not None and math.isfinite(r[key_x])]
    if len(pts) < 10:
        return []
    pts.sort(key=lambda t: t[0])
    n = len(pts)
    bins = []
    for b in range(10):
        lo = b * n // 10
        hi = (b + 1) * n // 10
        chunk = pts[lo:hi]
        if not chunk:
            continue
        ys = [y for _, y in chunk]
        xs = [x for x, _ in chunk]
        bins.append(
            {
                "bin": b,
                "delta_lo": min(xs),
                "delta_hi": max(xs),
                "delta_mean": sum(xs) / len(xs),
                "err_mean": sum(ys) / len(ys),
                "err_p90": sorted(ys)[int(0.9 * (len(ys) - 1))],
                "n": len(chunk),
            }
        )
    return bins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="JSONL files or directories")
    parser.add_argument("--out", required=True)
    parser.add_argument("--warm-k", type=int, default=0, help="split k < warm_k vs k >= warm_k")
    args = parser.parse_args()

    paths: list[Path] = []
    for raw in args.inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.jsonl")))
        else:
            paths.append(p)
    rows = _load_rows(paths)

    finite_res = [r for r in rows if r.get("err_res") is not None and math.isfinite(r["delta_world"])]
    finite_hold = [r for r in rows if r.get("err_hold") is not None and math.isfinite(r["delta_world"])]

    report: dict[str, Any] = {
        "n_rows": len(rows),
        "n_err_res": len(finite_res),
        "n_err_hold": len(finite_hold),
        "spearman": {},
        "decile_err_res": _decile_bins(finite_res, "delta_world", "err_res"),
        "decile_err_hold": _decile_bins(finite_hold, "delta_world", "err_hold"),
        "by_k_half": {},
        "by_drift_level": {},
    }

    if finite_res:
        report["spearman"]["delta_world_vs_err_res"] = _spearman(
            [float(r["delta_world"]) for r in finite_res],
            [float(r["err_res"]) for r in finite_res],
        )
    if finite_hold:
        report["spearman"]["delta_world_vs_err_hold"] = _spearman(
            [float(r["delta_world"]) for r in finite_hold],
            [float(r["err_hold"]) for r in finite_hold],
        )
    lat = [r for r in finite_res if math.isfinite(r.get("latent_drift", float("inf")))]
    if lat:
        report["spearman"]["latent_drift_vs_err_res"] = _spearman(
            [float(r["latent_drift"]) for r in lat],
            [float(r["err_res"]) for r in lat],
        )

    warm_k = args.warm_k
    for label, subset in (
        ("early_k", [r for r in finite_res if int(r["k"]) < warm_k]),
        ("late_k", [r for r in finite_res if int(r["k"]) >= warm_k]),
    ):
        if subset:
            report["by_k_half"][label] = {
                "n": len(subset),
                "err_res_mean": sum(float(r["err_res"]) for r in subset) / len(subset),
                "spearman_delta_err": _spearman(
                    [float(r["delta_world"]) for r in subset],
                    [float(r["err_res"]) for r in subset],
                ),
            }

    by_drift: dict[float, list[float]] = defaultdict(list)
    for r in finite_res:
        by_drift[float(r["drift_level"])].append(float(r["err_res"]))
    for d, ys in sorted(by_drift.items()):
        report["by_drift_level"][str(d)] = {
            "n": len(ys),
            "err_res_mean": sum(ys) / len(ys),
            "err_res_p90": sorted(ys)[int(0.9 * (len(ys) - 1))] if ys else None,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    md = Path(str(out).replace(".json", ".md") if out.suffix == ".json" else out.with_suffix(".md"))
    lines = [
        "# Coupling probe analysis",
        "",
        f"Rows: {report['n_rows']} (err_res={report['n_err_res']}, err_hold={report['n_err_hold']})",
        "",
        "## Spearman ρ",
        "",
    ]
    for k, v in report["spearman"].items():
        lines.append(f"- **{k}**: {v:.4f}" if v is not None else f"- **{k}**: n/a")
    lines += ["", "## err_res by synthetic drift level", "", "| drift | n | mean | p90 |", "|---|---:|---:|---:|"]
    for d, m in report["by_drift_level"].items():
        lines.append(f"| {d} | {m['n']} | {m['err_res_mean']:.6f} | {m['err_res_p90']:.6f} |")
    lines += ["", "## δ_world decile bins → err_res", "", "| bin | δ mean | err mean | err p90 | n |", "|---:|---:|---:|---:|---:|"]
    for b in report["decile_err_res"]:
        lines.append(
            f"| {b['bin']} | {b['delta_mean']:.5f} | {b['err_mean']:.6f} | {b['err_p90']:.6f} | {b['n']} |"
        )
    md.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out} and {md}")


if __name__ == "__main__":
    main()
