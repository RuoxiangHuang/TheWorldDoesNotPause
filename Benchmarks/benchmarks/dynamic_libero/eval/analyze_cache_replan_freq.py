#!/usr/bin/env python3
"""Aggregate Cache replan-frequency ablation: SR vs L vs feedback Hz."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def _pool(cells: list[dict[str, Any]]) -> dict[tuple[str, int, float], dict[str, Any]]:
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        r = int(c.get("replan_steps") or 10)
        grouped[(str(c.get("accel")), r, float(c.get("speed")))].append(c)
    out: dict[tuple[str, int, float], dict[str, Any]] = {}
    for key, rows in grouped.items():
        succ: list[bool] = []
        lats: list[float] = []
        fbs: list[float] = []
        periods: list[float] = []
        n_replan: list[float] = []
        for r in rows:
            lats.append(float(r.get("latency_ms") or 0.0))
            if r.get("feedback_hz") is not None:
                fbs.append(float(r["feedback_hz"]))
            if r.get("feedback_period_s") is not None:
                periods.append(float(r["feedback_period_s"]))
            for ep in r.get("episodes") or []:
                succ.append(bool(ep.get("success")))
                if ep.get("n_replans") is not None:
                    n_replan.append(float(ep["n_replans"]))
        n = len(succ)
        out[key] = {
            "accel": key[0],
            "replan_steps": key[1],
            "speed": key[2],
            "n": n,
            "sr": (sum(succ) / n) if n else None,
            "wins": int(sum(succ)),
            "latency_ms": _median(lats),
            "feedback_hz": _median(fbs),
            "feedback_period_ms": (_median(periods) * 1000.0) if _median(periods) is not None else None,
            "n_replans_mean": (sum(n_replan) / len(n_replan)) if n_replan else None,
        }
    return out


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def render(title: str, pooled: dict[tuple[str, int, float], dict[str, Any]], accels: list[str]) -> str:
    replans = sorted({k[1] for k in pooled})
    speeds = sorted({k[2] for k in pooled})
    lines = [
        f"## {title}",
        "",
        "| accel | r | T_fb ms | Hz | " + " | ".join(f"v={s:g}" for s in speeds) + " | lat ms |",
        "|" + "---|" * (5 + len(speeds)),
    ]
    for a in accels:
        for r in replans:
            sr_bits = []
            lat = None
            hz = None
            tms = None
            for s in speeds:
                rec = pooled.get((a, r, s))
                if rec is None:
                    sr_bits.append("—")
                    continue
                sr_bits.append(f"{_fmt(rec['sr'])} ({rec['wins']}/{rec['n']})")
                lat = rec["latency_ms"]
                hz = rec["feedback_hz"]
                tms = rec["feedback_period_ms"]
            lines.append(
                f"| `{a}` | {r} | {_fmt(tms, 0)} | {_fmt(hz, 2)} | "
                + " | ".join(sr_bits)
                + f" | {_fmt(lat, 1)} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=str,
        default="/DATA/YuanZhen/FasterWAM/evaluate_results/iclr_8xh20/cache_replan_async",
    )
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()
    root = Path(args.root)
    fw = _pool(_cells(root / "fastwam"))
    pi = _pool(_cells(root / "pi"))
    text = "\n".join(
        [
            "# Cache replan-frequency ablation",
            "",
            "Model chunk length is unchanged. Only the executed prefix (`replan_steps`) varies.",
            r"Nominal feedback period \(T=L+r/20\) at 20 Hz. Main protocol remains \(r=10\).",
            "",
            render("FastWAM", fw, ["off", "default"]),
            render("π₀.₅", pi, ["eager", "cache"]),
            "",
        ]
    )
    print(text, end="")
    out = Path(args.out) if args.out else root / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    payload = {
        "fastwam": {f"{a}|r{r}|{s}": v for (a, r, s), v in fw.items()},
        "pi": {f"{a}|r{r}|{s}": v for (a, r, s), v in pi.items()},
    }
    (out.with_suffix(".json")).write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
