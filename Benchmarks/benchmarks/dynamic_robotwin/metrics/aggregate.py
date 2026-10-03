"""Re-export dynamic_libero aggregation + RoboTwin curve metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.dynamic_libero.metrics.aggregate import (
    aggregate_sr_table,
    auc_sr,
    bootstrap_ci,
    curve_metrics,
    enrich_table_with_curve_metrics,
    episode_success_ci,
    normalize_auc_sr,
    v50,
    write_curve_csv,
    write_summary_md,
)

__all__ = [
    "aggregate_sr_table",
    "auc_sr",
    "bootstrap_ci",
    "curve_metrics",
    "enrich_table_with_curve_metrics",
    "episode_success_ci",
    "normalize_auc_sr",
    "v50",
    "write_curve_csv",
    "write_summary_md",
    "main",
]


def _load_cells(paths: list[str]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for pat in paths:
        matches = sorted(Path().glob(pat)) if any(c in pat for c in "*?[") else [Path(pat)]
        for p in matches:
            if not p.is_file():
                continue
            data = json.loads(p.read_text())
            if "cells" in data:
                cells.extend(data["cells"])
            elif {"accel", "trajectory", "speed", "sr"} <= set(data):
                cells.append(data)
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Dynamic-RoboTwin JSON")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", type=str, default="evaluate_results/dynamic_robotwin/SUMMARY.md")
    ap.add_argument("--csv", type=str, default="evaluate_results/dynamic_robotwin/sr_curves.csv")
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
    Path(args.out).with_suffix(".json").write_text(
        json.dumps({"table": table, "n_cells": len(cells)}, indent=2)
    )
    text = Path(args.out).read_text()
    Path(args.out).write_text(
        text.replace("Dynamic-LIBERO", "Dynamic-RoboTwin", 1)
    )
    print(f"Wrote {args.out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
