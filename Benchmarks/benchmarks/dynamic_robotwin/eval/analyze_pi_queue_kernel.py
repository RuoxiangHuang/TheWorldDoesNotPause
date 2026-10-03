#!/usr/bin/env python3
"""2×2: compile vs compile+prefix × ZOH-8 vs leftover-32 at one async speed."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ORDER = ("compile_zoh", "compile_tail", "prefix_zoh", "prefix_tail")


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


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def _cells(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(root.glob("**/sr_sweep.json")):
        blob = json.loads(p.read_text())
        for cell in blob.get("cells") or []:
            row = dict(cell)
            row["_file"] = str(p)
            out.append(row)
    return out


def pool(cells: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        lab = str(c.get("cell_label") or c.get("accel") or "")
        grouped[lab].append(c)
    out: dict[str, dict[str, Any]] = {}
    for lab, rows in grouped.items():
        succ: list[bool] = []
        lats: list[float] = []
        leftover: list[float] = []
        chunk: list[float] = []
        stale: list[float] = []
        hold: list[float] = []
        keyed: dict[tuple[str, int], bool] = {}
        by_task: dict[str, list[bool]] = defaultdict(list)
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
                if ep.get("leftover_len_mean") is not None:
                    leftover.append(float(ep["leftover_len_mean"]))
                if ep.get("chunk_len_mean") is not None:
                    chunk.append(float(ep["chunk_len_mean"]))
                if ep.get("stale_executed") is not None:
                    stale.append(float(ep["stale_executed"]))
                if ep.get("stale_hold_ticks") is not None:
                    hold.append(float(ep["stale_hold_ticks"]))
        n = len(succ)
        wins = int(sum(succ))
        out[lab] = {
            "label": lab,
            "n": n,
            "wins": wins,
            "sr": (wins / n) if n else None,
            "latency_ms": _median(lats),
            "chunk_mean": _mean(chunk),
            "leftover_mean": _mean(leftover),
            "stale_mean": _mean(stale),
            "hold_mean": _mean(hold),
            "keyed": keyed,
            "by_task": {
                t: f"{int(sum(xs))}/{len(xs)}" for t, xs in sorted(by_task.items())
            },
        }
    return out


def _sr(r: dict[str, Any] | None) -> str:
    if not r or not r["n"]:
        return "—"
    return f"{r['wins']}/{r['n']} ({r['sr']:.3f})"


def render(pooled: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# π queue × kernel 2×2 (async, v=0.0003)",
        "",
        "Replan every 8 steps. Model still emits 32 / full NFE. "
        "ZOH = return 8 (leftover 0). Tail = return 32 (leftover 24 covers thinking).",
        "",
        "| cell | stack | queue | n | SR | L ms | chunk | leftover | stale | hold |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    meta = {
        "compile_zoh": ("compile", "ZOH-8"),
        "compile_tail": ("compile", "tail-32"),
        "prefix_zoh": ("compile+prefix", "ZOH-8"),
        "prefix_tail": ("compile+prefix", "tail-32"),
    }
    for lab in ORDER:
        r = pooled.get(lab)
        if not r:
            continue
        stack, queue = meta[lab]
        lines.append(
            f"| {lab} | {stack} | {queue} | {r['n']} | {_sr(r)} | "
            f"{_fmt(r['latency_ms'], 1)} | {_fmt(r['chunk_mean'], 2)} | "
            f"{_fmt(r['leftover_mean'], 2)} | {_fmt(r['stale_mean'], 1)} | "
            f"{_fmt(r['hold_mean'], 1)} |"
        )
    zoh_c = pooled.get("compile_zoh")
    tail_c = pooled.get("compile_tail")
    zoh_p = pooled.get("prefix_zoh")
    tail_p = pooled.get("prefix_tail")
    lines += ["", "## Contrasts", ""]
    def dsr(a, b, name):
        if not a or not b or a["sr"] is None or b["sr"] is None:
            return
        lines.append(f"- {name}: {b['sr'] - a['sr']:+.3f}  ({_sr(a)} → {_sr(b)})")
    dsr(zoh_c, tail_c, "queue on compile (tail − ZOH)")
    dsr(zoh_p, tail_p, "queue on prefix (tail − ZOH)")
    dsr(zoh_c, zoh_p, "prefix on ZOH (prefix − compile)")
    dsr(tail_c, tail_p, "prefix on tail (prefix − compile)")
    dsr(zoh_c, tail_p, "both vs compile-ZOH")
    tasks = sorted({t for r in pooled.values() for t in r["by_task"]})
    if tasks:
        present = [lab for lab in ORDER if lab in pooled]
        lines += ["", "## Per-task SR", "", "| task | " + " | ".join(present) + " |"]
        lines.append("|" + "|".join(["---"] * (1 + len(present))) + "|")
        for t in tasks:
            lines.append("| " + t + " | " + " | ".join(
                pooled[lab]["by_task"].get(t, "—") for lab in present
            ) + " |")
    return "\n".join(lines) + "\n"


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
