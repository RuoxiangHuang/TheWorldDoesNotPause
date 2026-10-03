#!/usr/bin/env python3
"""Rank Dynamic-LIBERO tasks by π₀.₅ Cache − eager ΔSR."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.common.outcomes import read_grasp_hold


def _cells(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    paths = list(root.glob("**/sr_sweep.json")) + list(root.glob("**/cell_*.json"))
    for p in sorted(paths):
        blob = json.loads(p.read_text())
        cells = blob.get("cells") if "cells" in blob else [blob]
        for cell in cells or []:
            if not isinstance(cell, dict) or "episodes" not in cell:
                continue
            key = (
                str(p.parent),
                str(cell.get("cell_label") or cell.get("accel") or ""),
                round(float(cell.get("speed") or 0), 6),
                str(cell.get("trajectory") or ""),
                int(cell.get("replan_steps") or 10),
                tuple(
                    str(ep.get("dynamic_task_id") or "")
                    for ep in (cell.get("episodes") or [])[:3]
                ),
            )
            if key in seen:
                continue
            seen.add(key)
            row = dict(cell)
            row["_file"] = str(p)
            out.append(row)
    return out


def _label(cell: dict[str, Any]) -> str:
    return str(cell.get("cell_label") or cell.get("accel") or "")


def _family(task_id: str) -> str:
    t = str(task_id)
    if t.endswith(".react"):
        return "react"
    if "carrier" in t:
        return "spatial"
    if t.endswith(".both"):
        return "both"
    if t.endswith(".place"):
        return "place"
    if t.endswith(".pick"):
        return "pick"
    return "other"


def pool(cells: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    grouped: dict[tuple, list[bool]] = defaultdict(list)
    grasp: dict[tuple, list[bool]] = defaultdict(list)
    experienced: dict[tuple, list[bool]] = defaultdict(list)
    lats: dict[tuple, list[float]] = defaultdict(list)
    for c in cells:
        lab = _label(c)
        try:
            speed = round(float(c.get("speed")), 6)
        except (TypeError, ValueError):
            continue
        lat = c.get("latency_ms")
        if lat is not None:
            lats[(lab, speed)].append(float(lat))
        for ep in c.get("episodes") or []:
            tid = str(ep.get("dynamic_task_id") or ep.get("task_id"))
            key = (lab, tid, speed)
            grouped[key].append(bool(ep.get("success")))
            hold = read_grasp_hold(ep)
            if hold is not None:
                grasp[key].append(hold)
            if ep.get("event_experienced") is True:
                experienced[key].append(bool(ep.get("success")))
    out: dict[tuple, dict[str, Any]] = {}
    for key, xs in grouped.items():
        n = len(xs)
        wins = int(sum(xs))
        g = grasp.get(key) or []
        exp = experienced.get(key) or []
        out[key] = {
            "label": key[0],
            "task": key[1],
            "speed": key[2],
            "family": _family(key[1]),
            "n": n,
            "wins": wins,
            "sr": wins / n if n else None,
            "grasp": (sum(g) / len(g)) if g else None,
            "n_experienced": len(exp),
            "sr_experienced": (sum(exp) / len(exp)) if exp else None,
            "latency_ms": (
                sum(lats[(key[0], key[2])]) / len(lats[(key[0], key[2])])
                if lats.get((key[0], key[2]))
                else None
            ),
        }
    return out


def render(pooled: dict[tuple, dict[str, Any]]) -> str:
    by_tv: dict[tuple[str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in pooled.values():
        by_tv[(r["task"], r["speed"])][r["label"]] = r

    ranked: list[dict[str, Any]] = []
    for (task, speed), labs in by_tv.items():
        off = labs.get("off") or labs.get("eager")
        cache = labs.get("cache")
        if not off or not cache or off["sr"] is None or cache["sr"] is None:
            continue
        d = cache["sr"] - off["sr"]
        ranked.append(
            {
                "task": task,
                "family": off["family"],
                "speed": speed,
                "n": off["n"],
                "off": off,
                "cache": cache,
                "dsr": d,
            }
        )
    ranked.sort(key=lambda x: (-x["dsr"], x["speed"], x["task"]))

    lines = [
        "# π₀.₅ Cache − eager task ranking",
        "",
        "Discovery pool (not official 60, not FastWAM pick slice): "
        "object place/both of t00/t01/t04/t05/t06, all 10 spatial carriers, "
        "react slice. async, r=10, N=5. Ceiling / floor called out.",
        "",
        "## Ranked by ΔSR",
        "",
        "| family | task | v | eager | Cache | ΔSR | note |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in ranked:
        note = ""
        if r["off"]["sr"] >= 0.8 and r["dsr"] <= 0.05:
            note = "ceiling"
        elif r["off"]["wins"] == 0 and r["cache"]["wins"] == 0:
            note = "floor"
        elif 0.2 <= r["off"]["sr"] <= 0.55 and r["dsr"] >= 0.2:
            note = "sweet"
        elif r["dsr"] >= 0.2:
            note = "gain"
        if r["family"] == "react":
            oe = r["off"].get("sr_experienced")
            ce = r["cache"].get("sr_experienced")
            if oe is not None and ce is not None:
                extra = f"experienced {oe:.2f}→{ce:.2f}"
                note = f"{note} {extra}".strip()
        lines.append(
            f"| {r['family']} | {r['task']} | {r['speed']:g} | "
            f"{r['off']['wins']}/{r['off']['n']} ({r['off']['sr']:.2f}) | "
            f"{r['cache']['wins']}/{r['cache']['n']} ({r['cache']['sr']:.2f}) | "
            f"{r['dsr']:+.2f} | {note} |"
        )

    best_by_task: dict[str, dict[str, Any]] = {}
    for r in ranked:
        prev = best_by_task.get(r["task"])
        if prev is None or r["dsr"] > prev["dsr"]:
            best_by_task[r["task"]] = r
    top = sorted(best_by_task.values(), key=lambda x: -x["dsr"])[:8]
    lines += [
        "",
        "## Strongest tasks (best ΔSR across speeds, n=5)",
        "",
        "These are discovery hits, not a post-hoc official table.",
        "",
    ]
    for r in top:
        if r["dsr"] <= 0:
            continue
        lines.append(
            f"- `{r['task']}` @ v={r['speed']:g}: "
            f"{r['off']['wins']}/{r['off']['n']} → {r['cache']['wins']}/{r['cache']['n']} "
            f"({r['dsr']:+.0%})"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", default="")
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
