"""Build lightweight paper tables from existing sweep/paired JSON (no GPU)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.dynamic_libero.metrics.aggregate import (
    aggregate_sr_table,
    write_curve_csv,
    write_summary_md,
)


def _collect(root: Path) -> list[dict]:
    cells = []
    # Prefer aggregated sweep files to avoid double-counting cell_*.json.
    sweep_files = list(root.rglob("sr_sweep.json"))
    if sweep_files:
        for p in sweep_files:
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            if isinstance(data, dict) and "cells" in data:
                cells.extend(data["cells"])
        return cells
    for p in root.rglob("cell_*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and {"accel", "trajectory", "speed", "sr"} <= set(data):
            cells.append(data)
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="evaluate_results/dynamic_libero",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="evaluate_results/dynamic_libero/SUMMARY.md",
    )
    args = ap.parse_args()
    root = Path(args.root)
    cells = _collect(root)
    table = aggregate_sr_table(cells)
    write_summary_md(table, Path(args.out))
    write_curve_csv(cells, Path(args.out).with_name("sr_curves.csv"))
    Path(args.out).with_suffix(".json").write_text(
        json.dumps({"table": table, "n_cells": len(cells)}, indent=2)
    )
    # Mirror into paper/
    paper = Path(__file__).resolve().parent
    write_summary_md(table, paper / "SUMMARY_generated.md")
    print(f"cells={len(cells)} -> {args.out}")


if __name__ == "__main__":
    main()
