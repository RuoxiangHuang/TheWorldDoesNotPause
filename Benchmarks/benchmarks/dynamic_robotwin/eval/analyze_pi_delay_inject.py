#!/usr/bin/env python3
"""Fixed-L injection: same compile+prefix / ZOH-8, delay 0 / 114 / 280 ms."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ORDER = ("instant", "now", "slow")


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
        hold: list[float] = []
        stale: list[float] = []
        leftover: list[float] = []
        chunk: list[float] = []
        nf: list[float] = []
        keyed: dict[tuple[str, int], bool] = {}
        by_task: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            if r.get("latency_ms") is not None:
                lats.append(float(r["latency_ms"]))
            if r.get("n_freeze") is not None:
                nf.append(float(r["n_freeze"]))
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
                if ep.get("stale_hold_ticks") is not None:
                    hold.append(float(ep["stale_hold_ticks"]))
                if ep.get("stale_executed") is not None:
                    stale.append(float(ep["stale_executed"]))
                if ep.get("leftover_len_mean") is not None:
                    leftover.append(float(ep["leftover_len_mean"]))
                if ep.get("chunk_len_mean") is not None:
                    chunk.append(float(ep["chunk_len_mean"]))
        n = len(succ)
        wins = int(sum(succ))
        out[lab] = {
            "label": lab,
            "n": n,
            "wins": wins,
            "sr": (wins / n) if n else None,
            "latency_ms": _median(lats),
            "n_freeze": _median(nf),
            "hold_mean": _mean(hold),
            "stale_mean": _mean(stale),
            "leftover_mean": _mean(leftover),
            "chunk_mean": _mean(chunk),
            "keyed": keyed,
            "by_task": {t: f"{int(sum(xs))}/{len(xs)}" for t, xs in sorted(by_task.items())},
        }
    return out


def _sr(r: dict[str, Any] | None) -> str:
    if not r or not r["n"]:
        return "—"
    return f"{r['wins']}/{r['n']} ({r['sr']:.3f})"


def render(pooled: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# π delay injection (compile+prefix, ZOH-8, v=0.0003)",
        "",
        "Same stack / queue / seeds. Only the simulated think delay changes. "
        "0 ms still lets the object move during the 8 executed actions; it is a diagnostic, not a deployable score. "
        "280 ms is total L, not 114+280. Each replan uses the current observation.",
        "",
        "| cell | injected L | n | SR | n_freeze | leftover | stale | hold |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    inj = {"instant": "0 ms", "now": "114 ms", "slow": "280 ms"}
    for lab in ORDER:
        r = pooled.get(lab)
        if not r:
            continue
        lines.append(
            f"| {lab} | {inj.get(lab, '?')} | {r['n']} | {_sr(r)} | "
            f"{_fmt(r['n_freeze'], 2)} | {_fmt(r['leftover_mean'], 2)} | "
            f"{_fmt(r['stale_mean'], 1)} | {_fmt(r['hold_mean'], 1)} |"
        )
    inst, now, slow = pooled.get("instant"), pooled.get("now"), pooled.get("slow")
    lines += ["", "## ΔSR (paired same task×seed)", ""]

    def pair(a, b, name):
        if not a or not b:
            return
        keys = sorted(set(a["keyed"]) & set(b["keyed"]))
        only_a = sum(a["keyed"][k] and not b["keyed"][k] for k in keys)
        only_b = sum((not a["keyed"][k]) and b["keyed"][k] for k in keys)
        ds = ""
        if a["sr"] is not None and b["sr"] is not None:
            ds = f" ΔSR={b['sr'] - a['sr']:+.3f}"
        lines.append(
            f"- {name}: {_sr(a)} → {_sr(b)}{ds}; "
            f"only first ok={only_a}, only second ok={only_b}, n={len(keys)}"
        )

    pair(inst, now, "now − instant (cost of 114 ms wait)")
    pair(now, slow, "slow − now (cost of going 114 → 280 ms)")
    pair(inst, slow, "slow − instant (full delay cost)")
    tasks = sorted({t for r in pooled.values() for t in r["by_task"]})
    if tasks:
        present = [lab for lab in ORDER if lab in pooled]
        lines += ["", "## Per-task SR", "", "| task | " + " | ".join(present) + " |"]
        lines.append("|" + "|".join(["---"] * (1 + len(present))) + "|")
        for t in tasks:
            lines.append(
                "| " + t + " | " + " | ".join(pooled[lab]["by_task"].get(t, "—") for lab in present) + " |"
            )
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
