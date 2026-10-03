#!/usr/bin/env python3
"""Summarize the reaction-track eval: reaction chain + SR, not a speed grid."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.common.outcomes import read_grasp_hold


REACTION_KEYS = (
    "turn_event_to_first_obs_s",
    "turn_first_obs_to_takeover_s",
    "turn_event_to_takeover_s",
    "turn_event_to_correction_s",
)
CLOSE_KEYS = (
    "turn_close_lateral_err_m",
    "turn_close_dist_m",
)
KEYS = REACTION_KEYS + CLOSE_KEYS


def _cells(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(root.glob("**/sr_sweep.json")):
        blob = json.loads(p.read_text())
        for cell in blob.get("cells") or []:
            row = dict(cell)
            row["_file"] = str(p)
            out.append(row)
    return out


def _median(xs: list[float]) -> float | None:
    xs = sorted(float(x) for x in xs if x is not None and not math.isnan(float(x)))
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    if n % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def _experienced(ep: dict[str, Any]) -> bool:
    # Old JSON without the flag still enters reaction stats.
    flag = ep.get("event_experienced")
    return True if flag is None else bool(flag)


def pool(cells: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        grouped[str(c.get("cell_label") or c.get("accel"))].append(c)
    out: dict[str, dict[str, Any]] = {}
    for accel, rows in grouped.items():
        succ: list[bool] = []
        grasp: list[bool] = []
        series: dict[str, list[float]] = {k: [] for k in KEYS}
        lats: list[float] = []
        n_event = 0
        n_missed = 0
        for r in rows:
            if r.get("latency_ms") is not None:
                lats.append(float(r["latency_ms"]))
            for ep in r.get("episodes") or []:
                succ.append(bool(ep.get("success")))
                hold = read_grasp_hold(ep)
                if hold is not None:
                    grasp.append(hold)
                saw = _experienced(ep)
                if saw:
                    n_event += 1
                    for k in KEYS:
                        v = ep.get(k)
                        if v is not None:
                            series[k].append(float(v))
                else:
                    n_missed += 1
                    for k in CLOSE_KEYS:
                        v = ep.get(k)
                        if v is not None:
                            series[k].append(float(v))
        n = len(succ)
        out[accel] = {
            "accel": accel,
            "n": n,
            "n_event": n_event,
            "n_missed": n_missed,
            "sr": (sum(succ) / n) if n else None,
            "wins": int(sum(succ)),
            "grasp_sr": (sum(grasp) / len(grasp)) if grasp else None,
            "latency_ms": _median(lats),
            **{k: _median(series[k]) for k in KEYS},
            "n_correction": len(series["turn_event_to_correction_s"]),
        }
    return out


def render(pooled: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Reaction-track diagnostic",
        "",
        "Aligned with main-track object pick (same objects / language / BDDL). "
        "Not the official linear/sine grid. Event times are world-clock; "
        "Cache can only shorten first-obs → takeover, not event → first obs. "
        "Episodes that grasped before the event keep full-task SR but are "
        "excluded from reaction timing.",
        "",
        "| accel | n | event | missed | SR | grasp SR | L ms | event→obs s | obs→takeover s | event→corr s | close lat mm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for accel, r in sorted(pooled.items()):
        lat_mm = r.get("turn_close_lateral_err_m")
        lat_mm_s = "—" if lat_mm is None else f"{1000.0 * float(lat_mm):.1f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    accel,
                    str(r["n"]),
                    str(r["n_event"]),
                    str(r["n_missed"]),
                    _fmt(r["sr"], 3),
                    _fmt(r["grasp_sr"], 3),
                    _fmt(r["latency_ms"], 1),
                    _fmt(r["turn_event_to_first_obs_s"]),
                    _fmt(r["turn_first_obs_to_takeover_s"]),
                    _fmt(r["turn_event_to_correction_s"]),
                    lat_mm_s,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze reaction-track JSON")
    p.add_argument("--inputs", type=str, required=True, help="Root with **/sr_sweep.json")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()
    root = Path(args.inputs)
    cells = _cells(root)
    if not cells:
        raise SystemExit(f"no sr_sweep.json under {root}")
    text = render(pool(cells))
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
