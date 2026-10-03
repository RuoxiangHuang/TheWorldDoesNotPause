#!/usr/bin/env python3
"""Confirm 280→114 ms on held-out seeds; also merge with delay-inject 0–4."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.common.outcomes import read_grasp_hold

STAGES = ("approach", "grasp", "place")


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


def pool(cells: list[dict[str, Any]], *, remap: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        lab = _label(c)
        if remap:
            lab = remap.get(lab, lab)
        grouped[lab].append(c)
    out: dict[str, dict[str, Any]] = {}
    for lab, rows in grouped.items():
        succ: list[bool] = []
        held: list[bool] = []
        stages: dict[str, int] = defaultdict(int)
        keyed_s: dict[tuple[str, int], bool] = {}
        keyed_h: dict[tuple[str, int], bool] = {}
        by_task: dict[str, list[bool]] = defaultdict(list)
        lats: list[float] = []
        for r in rows:
            if r.get("latency_ms") is not None:
                lats.append(float(r["latency_ms"]))
            for ep in r.get("episodes") or []:
                ok = bool(ep.get("success"))
                succ.append(ok)
                gh = read_grasp_hold(ep)
                if gh is not None:
                    held.append(gh)
                st = ep.get("fail_stage")
                if ok:
                    stages["success"] += 1
                elif st in STAGES:
                    stages[str(st)] += 1
                else:
                    stages["unknown"] += 1
                tid = str(ep.get("dynamic_task_id") or ep.get("task_name") or "")
                by_task[tid].append(ok)
                try:
                    sid = int(ep.get("seed"))
                except (TypeError, ValueError):
                    sid = -1
                keyed_s[(tid, sid)] = ok
                keyed_h[(tid, sid)] = gh
        n = len(succ)
        wins = int(sum(succ))
        hw = int(sum(held))
        out[lab] = {
            "label": lab,
            "n": n,
            "wins": wins,
            "sr": (wins / n) if n else None,
            "held_wins": hw,
            "held_sr": (hw / len(held)) if held else None,
            "held_n_known": len(held),
            "stages": dict(stages),
            "latency_ms": (sorted(lats)[len(lats) // 2] if lats else None),
            "keyed": keyed_s,
            "keyed_held": keyed_h,
            "by_task": {t: f"{int(sum(xs))}/{len(xs)}" for t, xs in sorted(by_task.items())},
        }
    return out


def _sr(r: dict[str, Any] | None, key: str = "sr", wins: str = "wins", nkey: str = "n") -> str:
    if not r:
        return "—"
    n = int(r.get(nkey) or 0)
    val = r.get(key)
    if val is None or n <= 0:
        return "unknown" if key == "held_sr" else "—"
    return f"{r[wins]}/{n} ({val:.3f})"


def _pair(a: dict[str, Any] | None, b: dict[str, Any] | None, keyed: str = "keyed") -> str:
    if not a or not b:
        return "—"
    keys = sorted(set(a[keyed]) & set(b[keyed]))
    only_a = sum(a[keyed][k] and not b[keyed][k] for k in keys)
    only_b = sum((not a[keyed][k]) and b[keyed][k] for k in keys)
    ds = ""
    ka = "sr" if keyed == "keyed" else "held_sr"
    nkey = "n" if keyed == "keyed" else "held_n_known"
    wkey = "wins" if keyed == "keyed" else "held_wins"
    if a[ka] is not None and b[ka] is not None:
        ds = f" Δ={b[ka] - a[ka]:+.3f}"
    return f"{_sr(a, ka, wkey, nkey)} → {_sr(b, ka, wkey, nkey)}{ds}; only slow={only_a} only fast={only_b} n={len(keys)}"


def _block(title: str, pooled: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| cell | n | task SR | grasped+held | approach fail | grasp fail | place fail | L ms |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for lab in ("now", "slow"):
        r = pooled.get(lab)
        if not r:
            continue
        st = r["stages"]
        lat = "—" if r["latency_ms"] is None else f"{r['latency_ms']:.1f}"
        lines.append(
            f"| {lab} | {r['n']} | {_sr(r)} | {_sr(r, 'held_sr', 'held_wins')} | "
            f"{st.get('approach', 0)} | {st.get('grasp', 0)} | {st.get('place', 0)} | {lat} |"
        )
    now, slow = pooled.get("now"), pooled.get("slow")
    lines += [
        "",
        f"- task SR slow→now: {_pair(slow, now)}",
        f"- grasped+held slow→now: {_pair(slow, now, 'keyed_held')}",
        "",
    ]
    tasks = sorted({t for r in pooled.values() for t in r["by_task"]})
    if tasks:
        present = [lab for lab in ("now", "slow") if lab in pooled]
        lines += ["| task | " + " | ".join(present) + " |", "|" + "|".join(["---"] * (1 + len(present))) + "|"]
        for t in tasks:
            lines.append("| " + t + " | " + " | ".join(pooled[lab]["by_task"].get(t, "—") for lab in present) + " |")
        lines.append("")
    return lines


def render(heldout: dict[str, dict[str, Any]], merged: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Confirm 280→114 ms (compile+prefix, ZOH-8, v=0.0003)",
        "",
        "Held-out seeds 5–14 were not used to pick the stack. "
        "Merged = held-out + delay-inject seeds 0–4 (same protocol).",
        "",
    ]
    lines += _block("Held-out only (seeds 5–14, n=50)", heldout)
    lines += _block("Merged with delay-inject 0–4 (n=75)", merged)
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--heldout", type=str, required=True)
    p.add_argument("--prior", type=str, default="")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()
    held = pool(_cells(Path(args.heldout)))
    merged_rows = list(_cells(Path(args.heldout)))
    if args.prior:
        merged_rows += [
            c
            for c in _cells(Path(args.prior))
            if _label(c) in ("now", "slow")
        ]
    text = render(held, pool(merged_rows))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
