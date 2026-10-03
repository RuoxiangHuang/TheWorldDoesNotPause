#!/usr/bin/env python3
"""Summarize π Cache-split (freeze, v=0) plus leftover/queue audit cells."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _median(xs: list[float]) -> float | None:
    xs = sorted(float(x) for x in xs if x is not None and not math.isnan(float(x)))
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _mean(xs: list[float]) -> float | None:
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _cells(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(root.glob("**/sr_sweep.json")):
        blob = json.loads(p.read_text())
        for cell in blob.get("cells") or []:
            row = dict(cell)
            row["_file"] = str(p)
            out.append(row)
    return out


def _label(cell: dict[str, Any]) -> str:
    return str(cell.get("cell_label") or cell.get("accel") or "")


def pool(cells: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        key = (
            _label(c),
            str(c.get("backend") or ""),
            str(c.get("trajectory") or "linear"),
            round(float(c.get("speed") or 0.0), 6),
        )
        grouped[key].append(c)
    out: dict[tuple, dict[str, Any]] = {}
    for key, rows in grouped.items():
        succ: list[bool] = []
        lats: list[float] = []
        chunk: list[float] = []
        leftover: list[float] = []
        stale: list[float] = []
        hold: list[float] = []
        replans: list[float] = []
        by_task: dict[str, list[bool]] = defaultdict(list)
        keyed: dict[tuple[str, int], bool] = {}
        for r in rows:
            if r.get("latency_ms") is not None:
                lats.append(float(r["latency_ms"]))
            for ep in r.get("episodes") or []:
                ok = bool(ep.get("success"))
                succ.append(ok)
                tid = str(ep.get("dynamic_task_id") or ep.get("task_name") or "")
                by_task[tid].append(ok)
                try:
                    sid = int(ep.get("seed"))
                except (TypeError, ValueError):
                    sid = -1
                keyed[(tid, sid)] = ok
                if ep.get("chunk_len_mean") is not None:
                    chunk.append(float(ep["chunk_len_mean"]))
                if ep.get("leftover_len_mean") is not None:
                    leftover.append(float(ep["leftover_len_mean"]))
                if ep.get("stale_executed") is not None:
                    stale.append(float(ep["stale_executed"]))
                if ep.get("stale_hold_ticks") is not None:
                    hold.append(float(ep["stale_hold_ticks"]))
                if ep.get("n_replans") is not None:
                    replans.append(float(ep["n_replans"]))
        n = len(succ)
        wins = int(sum(succ))
        out[key] = {
            "label": key[0],
            "backend": key[1],
            "trajectory": key[2],
            "speed": key[3],
            "n": n,
            "wins": wins,
            "sr": (wins / n) if n else None,
            "latency_ms": _median(lats),
            "chunk_len_mean": _mean(chunk),
            "leftover_len_mean": _mean(leftover),
            "stale_mean": _mean(stale),
            "hold_mean": _mean(hold),
            "n_replans_mean": _mean(replans),
            "keyed": keyed,
            "by_task": {
                t: {"wins": int(sum(xs)), "n": len(xs), "sr": sum(xs) / len(xs)}
                for t, xs in sorted(by_task.items())
            },
        }
    return out


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def _sr_frac(row: dict[str, Any]) -> str:
    if not row["n"]:
        return "—"
    return f"{row['wins']}/{row['n']} ({row['sr']:.3f})"


def render(pooled: dict[tuple, dict[str, Any]]) -> str:
    lines = [
        "# π Cache split on static Dynamic-RoboTwin pick",
        "",
        "Same five pick-slice skills × seeds 0–4 as the 14/25→5/25 drop. "
        "Freeze + catalog v=0 is stationary apparatus with thinking still "
        "coupled unless `--world-clock pause`. A remaining SR gap at pause "
        "or original-scene is approximation / implementation, not latency.",
        "",
        "## Freeze v=0 (Cache split)",
        "",
        "| cell | n | SR | L ms | chunk_len | leftover | stale | hold |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    freeze = [r for r in pooled.values() if r["backend"] == "freeze" and r["speed"] == 0.0]
    freeze.sort(key=lambda r: ["eager", "compile", "vision", "vision02", "c3", "cache", "off"].index(r["label"])
                if r["label"] in ("eager", "compile", "vision", "vision02", "c3", "cache", "off")
                else 99)
    eager = next((r for r in freeze if r["label"] == "eager"), None)
    for r in freeze:
        lines.append(
            f"| {r['label']} | {r['n']} | {_sr_frac(r)} | {_fmt(r['latency_ms'], 1)} | "
            f"{_fmt(r['chunk_len_mean'], 2)} | {_fmt(r['leftover_len_mean'], 2)} | "
            f"{_fmt(r['stale_mean'], 1)} | {_fmt(r['hold_mean'], 1)} |"
        )
    lines += ["", "### ΔSR vs eager (freeze v=0)", "", "| cell | eager | this | ΔSR |", "|---|---|---|---:|"]
    if eager:
        for r in freeze:
            if r["label"] == "eager" or r["sr"] is None or eager["sr"] is None:
                continue
            lines.append(
                f"| {r['label']} | {_sr_frac(eager)} | {_sr_frac(r)} | {r['sr'] - eager['sr']:+.3f} |"
            )
    tasks = sorted({t for r in freeze for t in r["by_task"]})
    if freeze and tasks:
        labels = [r["label"] for r in freeze]
        lines += ["", "### Per-task SR (freeze v=0)", "", "| task | " + " | ".join(labels) + " |"]
        lines.append("|" + "|".join(["---"] * (1 + len(labels))) + "|")
        for t in tasks:
            cells = []
            for r in freeze:
                info = r["by_task"].get(t)
                cells.append("—" if not info else f"{info['wins']}/{info['n']}")
            lines.append(f"| {t} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Queue / leftover audit (async, v=0.0003)",
        "",
        "If leftover≈0 and hold≈n_replans×n_freeze, the think window is ZOH. "
        "If stale≈n_replans×n_freeze, a leftover tail is covering thinking.",
        "",
        "| cell | backend | n | SR | L ms | chunk_len | leftover | stale | hold | n_replans |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    audit = [
        r
        for r in pooled.values()
        if not (r["backend"] == "freeze" and r["speed"] == 0.0)
    ]
    audit.sort(key=lambda r: (r["backend"], r["speed"], r["label"]))
    for r in audit:
        lines.append(
            f"| {r['label']} | {r['backend']} v={r['speed']} | {r['n']} | {_sr_frac(r)} | "
            f"{_fmt(r['latency_ms'], 1)} | {_fmt(r['chunk_len_mean'], 2)} | "
            f"{_fmt(r['leftover_len_mean'], 2)} | {_fmt(r['stale_mean'], 1)} | "
            f"{_fmt(r['hold_mean'], 1)} | {_fmt(r['n_replans_mean'], 1)} |"
        )
    lines += [
        "",
        "## How to read the split",
        "",
        "- compile drops, vision/C³ do not → numerical / graph issue.",
        "- vision drops, C³ does not → tighten or disable vision cache.",
        "- C³ drops, vision does not → less action reuse / restore some NFEs.",
        "- only the combo drops → interaction of the two caches.",
        "- all accelerated cells OK at freeze v=0 → the 14/25→5/25 drop was not freeze-static; look at async leftover.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()
    root = Path(args.inputs)
    text = render(pool(_cells(root)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
