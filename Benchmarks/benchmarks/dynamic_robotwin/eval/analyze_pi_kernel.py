#!/usr/bin/env python3
"""Summarize π kernel/CUDA-graph latency + freeze-v0 SR."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ORDER = ("compile", "prefix", "graph", "kernel")


def _median(xs: list[float]) -> float | None:
    xs = sorted(float(x) for x in xs if x is not None and not math.isnan(float(x)))
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


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
        keyed: dict[tuple[str, int], bool] = {}
        for r in rows:
            if r.get("latency_ms") is not None:
                lats.append(float(r["latency_ms"]))
            for ep in r.get("episodes") or []:
                ok = bool(ep.get("success"))
                succ.append(ok)
                tid = str(ep.get("dynamic_task_id") or ep.get("task_name") or "")
                try:
                    sid = int(ep.get("seed"))
                except (TypeError, ValueError):
                    sid = -1
                keyed[(tid, sid)] = ok
        n = len(succ)
        wins = int(sum(succ))
        out[lab] = {
            "label": lab,
            "n": n,
            "wins": wins,
            "sr": (wins / n) if n else None,
            "latency_ms": _median(lats),
            "keyed": keyed,
        }
    return out


def render(pooled: dict[str, dict[str, Any]]) -> str:
    base = pooled.get("compile")
    lines = [
        "# π kernel path: CUDA graphs + compiled SigLIP (no vision/C³)",
        "",
        "Same pick_slice × seeds 0–4, freeze + v=0. Images and NFE unchanged. "
        "`compile` is the current ~130 ms candidate (inductor graphs off). "
        "`graph` re-enables CUDA graphs on serial HTTP. "
        "`prefix` compiles/batches SigLIP. `kernel` is both.",
        "",
        "| cell | spec | n | SR | L ms | ΔL vs compile | ΔSR vs compile |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    specs = {
        "compile": "compile, graphs off",
        "prefix": "compile+compile_prefix, graphs off",
        "graph": "compile+graph",
        "kernel": "compile+compile_prefix+graph",
    }
    for lab in ORDER:
        r = pooled.get(lab)
        if not r:
            continue
        sr = "—" if not r["n"] else f"{r['wins']}/{r['n']} ({r['sr']:.3f})"
        dl = "—"
        ds = "—"
        if base and r["latency_ms"] is not None and base["latency_ms"] is not None:
            dl = f"{r['latency_ms'] - base['latency_ms']:+.1f}"
        if base and r["sr"] is not None and base["sr"] is not None:
            ds = f"{r['sr'] - base['sr']:+.3f}"
        lat = "—" if r["latency_ms"] is None else f"{r['latency_ms']:.1f}"
        lines.append(f"| {lab} | {specs.get(lab, lab)} | {r['n']} | {sr} | {lat} | {dl} | {ds} |")
    if base:
        lines += ["", "Paired vs compile (same task×seed):", ""]
        for lab in ORDER:
            if lab == "compile" or lab not in pooled:
                continue
            a, b = base["keyed"], pooled[lab]["keyed"]
            keys = sorted(set(a) & set(b))
            only_c = sum(a[k] and not b[k] for k in keys)
            only_o = sum((not a[k]) and b[k] for k in keys)
            lines.append(f"- {lab}: only compile ok={only_c}, only {lab} ok={only_o}, n={len(keys)}")
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
