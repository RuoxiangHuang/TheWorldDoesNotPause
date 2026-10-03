#!/usr/bin/env python3
"""Aggregate baseline vs method SR on the reconstructed 60-task set.

Default labels are FastWAM (off) vs FasterWAM (full). Pass ``--baseline pi``
and ``--method fasterpi`` for the PI pairing.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _split(task_id: str) -> str:
    if ".irregular" in task_id:
        return "object_curve"
    if task_id.startswith("libero_spatial."):
        return "spatial_carrier"
    return "object_linear"


def _load_cells(root: Path) -> list[dict]:
    cells: list[dict] = []
    for p in sorted(root.rglob("sr_sweep.json")):
        payload = json.loads(p.read_text())
        cells.extend(payload.get("cells") or [])
    if not cells:
        for p in sorted(root.rglob("cell_*.json")):
            cells.append(json.loads(p.read_text()))
    return cells


def _sr(episodes: list[dict]) -> tuple[float, int]:
    if not episodes:
        return float("nan"), 0
    s = [1.0 if bool(e.get("success")) else 0.0 for e in episodes]
    return float(np.mean(s)), len(s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=str,
        default="evaluate_results/dynamic_libero/reconstructed_60_off_vs_full",
    )
    p.add_argument("--out", type=str, default="")
    p.add_argument("--baseline", type=str, default="off", help="Cell accel label for the slow stack")
    p.add_argument("--method", type=str, default="full", help="Cell accel label for the fast stack")
    p.add_argument("--baseline-name", type=str, default="FastWAM")
    p.add_argument("--method-name", type=str, default="FasterWAM")
    args = p.parse_args()
    root = Path(args.root)
    cells = _load_cells(root)
    if not cells:
        raise SystemExit(f"no sr_sweep/cell json under {root}")

    bucket: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    speeds: set[float] = set()
    tasks: set[str] = set()
    for cell in cells:
        accel = str(cell.get("accel") or "")
        speed = float(cell.get("speed"))
        speeds.add(speed)
        for e in cell.get("episodes") or []:
            tid = str(e.get("dynamic_task_id") or f"t{e.get('task_id')}")
            tasks.add(tid)
            bucket[(accel, speed, tid)].append(e)

    speed_list = sorted(speeds)
    task_list = sorted(tasks)
    off_k, full_k = args.baseline, args.method
    lines = [
        f"# {args.baseline_name} vs {args.method_name} — Dynamic-LIBERO reconstructed 60",
        "",
        f"{args.baseline_name} = `accel={off_k}`. {args.method_name} = `accel={full_k}`.",
        "Success = original LIBERO BDDL `done` and not escaped. Language/appearance `keep`.",
        "",
    ]

    lines += [
        "## Macro SR",
        "",
        f"| split | speed | n | {args.baseline_name} | {args.method_name} | ΔSR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("object_linear", "object_curve", "spatial_carrier", "all"):
        for speed in speed_list:
            off_eps, full_eps = [], []
            for tid in task_list:
                if split != "all" and _split(tid) != split:
                    continue
                off_eps.extend(bucket.get((off_k, speed, tid), []))
                full_eps.extend(bucket.get((full_k, speed, tid), []))
            sr_off, n_off = _sr(off_eps)
            sr_full, n_full = _sr(full_eps)
            n = min(n_off, n_full)
            delta = sr_full - sr_off if n else float("nan")
            lines.append(
                f"| {split} | {speed:g} | {n} | {sr_off:.3f} | {sr_full:.3f} | {delta:+.3f} |"
            )

    lines += ["", "## Per-task SR", ""]
    header = "| task | split |"
    sep = "|---|---|"
    for speed in speed_list:
        header += (
            f" {args.baseline_name}@{speed:g} | {args.method_name}@{speed:g} | Δ@{speed:g} |"
        )
        sep += "---:|---:|---:|"
    lines += [header, sep]
    for tid in task_list:
        row = f"| `{tid}` | {_split(tid)} |"
        for speed in speed_list:
            sr_off, _ = _sr(bucket.get((off_k, speed, tid), []))
            sr_full, _ = _sr(bucket.get((full_k, speed, tid), []))
            delta = sr_full - sr_off
            row += f" {sr_off:.2f} | {sr_full:.2f} | {delta:+.2f} |"
        lines.append(row)

    text = "\n".join(lines) + "\n"
    out = Path(args.out) if args.out else root / "COMPARISON.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
