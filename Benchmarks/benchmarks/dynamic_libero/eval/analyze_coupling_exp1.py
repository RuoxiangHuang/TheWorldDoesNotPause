#!/usr/bin/env python3
"""Analyze CLIRA exp-1 JSONL: Spearman(δ, err) and decile bins.

Pass criterion (pre-registered): ρ(δ, err) > 0.3 and ≥3 samples/bin on the
primary metric `delta_cam_max` vs `err_nfe_cheap_l1` / `err_stale_l1`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from benchmarks.dynamic_libero.eval.coupling_math import (
    bin_means_monotone,
    decile_bins,
    spearman,
)

PRIMARY_DELTA = "delta_cam_max"
NFE_ERR = "err_nfe_cheap_l1"
STALE_ERR = "err_stale_l1"
RHO_PASS = 0.3


def _load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        if p.is_dir():
            files = sorted(p.glob("*.jsonl"))
        else:
            files = [p]
        for f in files:
            with f.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def _pairs(rows: list[dict], ykey: str, dkey: str = PRIMARY_DELTA) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for r in rows:
        x, y = r.get(dkey), r.get(ykey)
        if x is None or y is None:
            continue
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            continue
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def _block(rows: list[dict], ykey: str) -> dict:
    xs, ys = _pairs(rows, ykey)
    rho = spearman(xs, ys)
    bins = decile_bins(xs, ys)
    return {
        "n": len(xs),
        "rho": rho,
        "pass_rho": bool(rho is not None and rho > RHO_PASS),
        "pass_mono": bin_means_monotone(bins),
        "delta_mean": (sum(xs) / len(xs)) if xs else None,
        "err_mean": (sum(ys) / len(ys)) if ys else None,
        "bins": bins,
    }


def _summarize(rows: list[dict]) -> dict:
    by_bb = defaultdict(list)
    for r in rows:
        by_bb[r.get("backbone", "?")].append(r)
    out: dict = {"n_rows": len(rows), "rho_pass": RHO_PASS, "delta": PRIMARY_DELTA}
    for bb, rs in sorted(by_bb.items()):
        kinds = defaultdict(list)
        for r in rs:
            kinds[r.get("bin_kind", "?")].append(r)
        out[bb] = {
            "nfe_cheap": _block(rs, NFE_ERR),
            "nfe_mid": _block(rs, "err_nfe_mid_l1"),
            "stale": _block(rs, STALE_ERR),
            "by_kind": {
                k: {
                    "nfe_cheap": _block(v, NFE_ERR),
                    "stale": _block(v, STALE_ERR),
                }
                for k, v in sorted(kinds.items())
            },
        }
    return out


def _fmt_block(name: str, b: dict) -> str:
    rho = b.get("rho")
    rho_s = "na" if rho is None else f"{rho:.3f}"
    flag = "PASS" if b.get("pass_rho") else "FAIL"
    return (
        f"{name:12s} n={b['n']:5d}  ρ={rho_s:>7}  {flag}  "
        f"mono={int(bool(b.get('pass_mono')))}  "
        f"δ̄={b.get('delta_mean')}  err̄={b.get('err_mean')}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out-json", type=str, default="")
    args = p.parse_args()
    rows = _load([Path(x) for x in args.paths])
    summary = _summarize(rows)
    print(f"rows={summary['n_rows']}  pass if ρ({PRIMARY_DELTA}, err) > {RHO_PASS}")
    for bb, block in summary.items():
        if bb in ("n_rows", "rho_pass", "delta"):
            continue
        print(f"\n=== {bb} ===")
        print("  " + _fmt_block("B cheap NFE", block["nfe_cheap"]))
        print("  " + _fmt_block("B mid NFE", block["nfe_mid"]))
        print("  " + _fmt_block("A stale W", block["stale"]))
        for kind, kb in block["by_kind"].items():
            print(f"  [{kind}]")
            print("    " + _fmt_block("cheap", kb["nfe_cheap"]))
            print("    " + _fmt_block("stale", kb["stale"]))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
