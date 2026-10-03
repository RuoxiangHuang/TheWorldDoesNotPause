#!/usr/bin/env python3
"""Summarize π no-vision async 3-cell: eager / compile / compile+C³."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ORDER = ("eager", "compile", "c3")
SPEEDS = (0.0, 0.0003, 0.0006, 0.001)


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
        key = (_label(c), round(float(c.get("speed") or 0.0), 6))
        grouped[key].append(c)
    out: dict[tuple, dict[str, Any]] = {}
    for key, rows in grouped.items():
        succ: list[bool] = []
        lats: list[float] = []
        stale: list[float] = []
        hold: list[float] = []
        leftover: list[float] = []
        chunk: list[float] = []
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
                if ep.get("stale_executed") is not None:
                    stale.append(float(ep["stale_executed"]))
                if ep.get("stale_hold_ticks") is not None:
                    hold.append(float(ep["stale_hold_ticks"]))
                if ep.get("leftover_len_mean") is not None:
                    leftover.append(float(ep["leftover_len_mean"]))
                if ep.get("chunk_len_mean") is not None:
                    chunk.append(float(ep["chunk_len_mean"]))
        n = len(succ)
        wins = int(sum(succ))
        out[key] = {
            "label": key[0],
            "speed": key[1],
            "n": n,
            "wins": wins,
            "sr": (wins / n) if n else None,
            "latency_ms": _median(lats),
            "stale_mean": _mean(stale),
            "hold_mean": _mean(hold),
            "leftover_mean": _mean(leftover),
            "chunk_mean": _mean(chunk),
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


def _sr(row: dict[str, Any] | None) -> str:
    if not row or not row["n"]:
        return "—"
    return f"{row['wins']}/{row['n']} ({row['sr']:.3f})"


def _get(pooled: dict[tuple, dict[str, Any]], lab: str, spd: float) -> dict[str, Any] | None:
    return pooled.get((lab, round(spd, 6)))


def render(pooled: dict[tuple, dict[str, Any]]) -> str:
    lines = [
        "# π no-vision async (eager / compile / compile+C³)",
        "",
        "Vision cache off. Same pick_slice × seeds 0–4 as who-moves. "
        "async + measured L. Catalog v=0 is apparatus-hold and still injects "
        "`n_freeze` unless `--world-clock pause`. Original spawn is "
        "`--scene-preset original`. "
        "Delay-causal cells are v>0. Truncation and r=8 unchanged.",
        "",
        "## SR by speed",
        "",
        "| v | eager SR | compile SR | c3 SR | L eager | L compile | L c3 | "
        "Δ compile−eager | Δ c3−compile | Δ c3−eager |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for spd in SPEEDS:
        rows = {lab: _get(pooled, lab, spd) for lab in ORDER}
        def dsr(a: str, b: str) -> str:
            ra, rb = rows[a], rows[b]
            if not ra or not rb or ra["sr"] is None or rb["sr"] is None:
                return "—"
            return f"{rb['sr'] - ra['sr']:+.3f}"
        lines.append(
            f"| {spd:g} | {_sr(rows['eager'])} | {_sr(rows['compile'])} | {_sr(rows['c3'])} | "
            f"{_fmt(rows['eager']['latency_ms'] if rows['eager'] else None, 1)} | "
            f"{_fmt(rows['compile']['latency_ms'] if rows['compile'] else None, 1)} | "
            f"{_fmt(rows['c3']['latency_ms'] if rows['c3'] else None, 1)} | "
            f"{dsr('eager', 'compile')} | {dsr('compile', 'c3')} | {dsr('eager', 'c3')} |"
        )
    lines += [
        "",
        "Read: **Δ compile−eager** is the compile speedup. "
        "**Δ c3−compile** is extra C³ on top of compile. "
        "Do not credit C³ with compile's gain.",
        "",
        "## Queue (leftover should stay 0; think window is ZOH)",
        "",
        "| v | cell | chunk | leftover | stale | hold |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for spd in SPEEDS:
        for lab in ORDER:
            r = _get(pooled, lab, spd)
            if not r:
                continue
            lines.append(
                f"| {spd:g} | {lab} | {_fmt(r['chunk_mean'], 2)} | {_fmt(r['leftover_mean'], 2)} | "
                f"{_fmt(r['stale_mean'], 1)} | {_fmt(r['hold_mean'], 1)} |"
            )
    tasks = sorted({t for r in pooled.values() for t in r["by_task"]})
    if tasks:
        lines += ["", "## Per-task SR", ""]
        for spd in SPEEDS:
            present = [lab for lab in ORDER if _get(pooled, lab, spd)]
            if not present:
                continue
            lines += [f"### v={spd:g}", "", "| task | " + " | ".join(present) + " |"]
            lines.append("|" + "|".join(["---"] * (1 + len(present))) + "|")
            for t in tasks:
                cells = []
                for lab in present:
                    info = _get(pooled, lab, spd)["by_task"].get(t)  # type: ignore[index]
                    cells.append("—" if not info else f"{info['wins']}/{info['n']}")
                lines.append(f"| {t} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()
    text = render(pool(_cells(Path(args.inputs))))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
