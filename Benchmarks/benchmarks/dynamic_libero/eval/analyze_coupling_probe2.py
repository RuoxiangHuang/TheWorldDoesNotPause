#!/usr/bin/env python3
"""Analyze CLIRA probe-2 JSONL: cost, safe-reuse region, cache age, executor alignment.

Pre-registered gates (protocol.json):
  cost      skippable / probe > 1.5
  safety    exists τ with reuse_rate ≥ 0.20 and P(e>ε) ≤ 0.10
            ε = max(p90, 2×median) of consecutive full-compute relative L1
  age       ρ(δ_anchor, e_age) > 0.3 at age 1; report whether it holds at 2/4/8
  executor  median path-A vs path-B relative L1 < 0.05 or < 0.5ε
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.dynamic_libero.eval.coupling_math import decile_bins, spearman
from benchmarks.dynamic_libero.eval.coupling_probe2_common import AGES

RHO_PASS = 0.3
REUSE_MIN = 0.20
EXCEED_MAX = 0.10
COST_RATIO = 1.5
EXEC_ABS = 0.05


def _load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            with f.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def _finite(rows: list[dict], *keys: str) -> list[dict]:
    out = []
    for r in rows:
        ok = True
        for k in keys:
            v = r.get(k)
            if v is None or not math.isfinite(float(v)):
                ok = False
                break
        if ok:
            out.append(r)
    return out


def _vals(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(p * (len(ys) - 1)))))
    return ys[i]


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def _episode_key(r: dict) -> tuple:
    return (r.get("backbone"), r.get("task_id"), r.get("speed"), r.get("init_id"))


def _cluster_mean(rows: list[dict], fn) -> float | None:
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[_episode_key(r)].append(r)
    vals = []
    for g in groups.values():
        v = fn(g)
        if v is not None and math.isfinite(float(v)):
            vals.append(float(v))
    return _mean(vals)


def _epsilon(rows: list[dict]) -> dict:
    consec = _vals(rows, "err_full_consec_l1")
    med = _median(consec)
    p90 = _pct(consec, 0.9)
    eps = None
    if med is not None and p90 is not None:
        eps = max(float(p90), 2.0 * float(med))
    elif p90 is not None:
        eps = float(p90)
    return {
        "n": len(consec),
        "median": med,
        "p90": p90,
        "eps_l1": eps,
        "consec_mean": _mean(consec),
    }


def _safety(rows: list[dict], delta_k: str, err_k: str, eps: float | None) -> dict:
    pts = _finite(rows, delta_k, err_k)
    if not pts or eps is None:
        return {"n": len(pts), "eps": eps, "pass": False, "sweep": []}
    deltas = [float(r[delta_k]) for r in pts]
    sweep = []
    best = None
    for p in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        tau = _pct(deltas, p)
        if tau is None:
            continue
        allow = [r for r in pts if float(r[delta_k]) <= tau]
        rate = len(allow) / len(pts)
        exceed = sum(1 for r in allow if float(r[err_k]) > eps) / max(len(allow), 1)
        # episode-clustered exceed among allowed
        by_ep: dict[tuple, list] = defaultdict(list)
        for r in allow:
            by_ep[_episode_key(r)].append(float(r[err_k]) > eps)
        cl_ex = _mean([sum(v) / len(v) for v in by_ep.values()]) if by_ep else None
        rec = {
            "p": p,
            "tau": tau,
            "reuse_rate": rate,
            "p_exceed": exceed,
            "p_exceed_cluster": cl_ex,
            "n_allow": len(allow),
            "err_mean": _mean([float(r[err_k]) for r in allow]),
            "ok": bool(rate >= REUSE_MIN and exceed <= EXCEED_MAX),
        }
        sweep.append(rec)
        if rec["ok"] and (best is None or rec["reuse_rate"] > best["reuse_rate"]):
            best = rec
    return {
        "n": len(pts),
        "eps": eps,
        "pass": best is not None,
        "best": best,
        "sweep": sweep,
    }


def _rho_block(rows: list[dict], xk: str, yk: str) -> dict:
    pts = _finite(rows, xk, yk)
    xs = [float(r[xk]) for r in pts]
    ys = [float(r[yk]) for r in pts]
    rho = spearman(xs, ys)
    bins = decile_bins(xs, ys)
    per_ep = []
    groups: dict[tuple, list] = defaultdict(list)
    for r in pts:
        groups[_episode_key(r)].append(r)
    for g in groups.values():
        if len(g) < 4:
            continue
        rr = spearman([float(x[xk]) for x in g], [float(x[yk]) for x in g])
        if rr is not None:
            per_ep.append(rr)
    return {
        "n": len(pts),
        "rho": rho,
        "rho_episode_mean": _mean(per_ep) if per_ep else None,
        "n_episodes": len(per_ep),
        "pass_rho": bool(rho is not None and rho > RHO_PASS),
        "delta_mean": _mean(xs),
        "err_mean": _mean(ys),
        "err_p10": _pct(ys, 0.1),
        "err_p90": _pct(ys, 0.9),
        "bins": bins,
    }


def _cost_fastwam(rows: list[dict]) -> dict:
    vae, predit, prefill, action = (
        _vals(rows, "ms_vae"),
        _vals(rows, "ms_predit"),
        _vals(rows, "ms_prefill"),
        _vals(rows, "ms_action"),
    )
    n = min(len(vae), len(predit), len(prefill), len(action))
    if n == 0:
        return {"n": 0, "pass": False}
    probe = [vae[i] + predit[i] for i in range(n)]
    skip = prefill[:n]
    total = [probe[i] + skip[i] + action[i] for i in range(n)]
    ratio = [_mean(skip) / _mean(probe)] if _mean(probe) else [None]
    r = None if not probe or _mean(probe) == 0 else _mean(skip) / _mean(probe)
    return {
        "n": n,
        "ms_vae": _mean(vae),
        "ms_predit": _mean(predit),
        "ms_prefill": _mean(prefill),
        "ms_action": _mean(action),
        "ms_probe_vae_predit": _mean(probe),
        "ratio_skip_over_probe": r,
        "frac_prefill_of_fwd": (_mean(skip) / _mean(total)) if _mean(total) else None,
        "pass": bool(r is not None and r > COST_RATIO),
        "note": "probe=VAE+pre_dit (current z_c); skip=MoT prefill",
    }


def _cost_pi(rows: list[dict]) -> dict:
    sig, pref, act = _vals(rows, "ms_siglip"), _vals(rows, "ms_prefix"), _vals(rows, "ms_action")
    n = min(len(sig), len(pref), len(act))
    if n == 0:
        return {"n": 0, "pass": False}
    r = None if _mean(sig[:n]) in (None, 0) else _mean(pref[:n]) / _mean(sig[:n])
    tot = _mean(sig[:n]) + _mean(pref[:n]) + _mean(act[:n])
    return {
        "n": n,
        "ms_siglip": _mean(sig),
        "ms_prefix": _mean(pref),
        "ms_action": _mean(act),
        "ratio_prefix_over_siglip": r,
        "frac_prefix_of_fwd": (_mean(pref[:n]) / tot) if tot else None,
        "pass": bool(r is not None and r > COST_RATIO),
        "note": "probe=SigLIP; skip=PaliGemma prefix",
    }


def _executor(rows: list[dict], key: str, eps: float | None) -> dict:
    xs = _vals(rows, key)
    med = _median(xs)
    lim = EXEC_ABS
    if eps is not None:
        lim = max(EXEC_ABS, 0.5 * float(eps))
    return {
        "n": len(xs),
        "median": med,
        "mean": _mean(xs),
        "p90": _pct(xs, 0.9),
        "limit": lim,
        "pass": bool(med is not None and med < lim),
    }


def _summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[str(r.get("backbone", "?"))].append(r)
    out: dict = {
        "n_rows": len(rows),
        "gates": {
            "rho": RHO_PASS,
            "reuse_min": REUSE_MIN,
            "exceed_max": EXCEED_MAX,
            "cost_ratio": COST_RATIO,
            "exec_abs": EXEC_ABS,
        },
    }
    for bb, rs in sorted(by.items()):
        eps_blk = _epsilon(rs)
        eps = eps_blk.get("eps_l1")
        if bb == "fastwam":
            cost = _cost_fastwam(rs)
            stale_k = "err_stale_kv_l1_age1"
            exec_age = _executor(rs, "err_exec_kv_vs_vis_l1_age1", eps)
            exec_now = _executor(rs, "err_vis_vs_prop_l1", eps)
            safety = _safety(rs, "delta_cam_age1", stale_k, eps)
        else:
            cost = _cost_pi(rs)
            stale_k = "err_stale_prefix_l1_age1"
            exec_age = _executor(rs, "err_exec_img_vs_prefix_l1_age1", eps)
            exec_now = None
            safety = _safety(rs, "delta_cam_age1", stale_k, eps)
        ages = {}
        for age in AGES:
            dk, ek = f"delta_cam_age{age}", (
                f"err_stale_kv_l1_age{age}" if bb == "fastwam" else f"err_stale_prefix_l1_age{age}"
            )
            ages[str(age)] = _rho_block(rs, dk, ek)
        out[bb] = {
            "n": len(rs),
            "cost": cost,
            "epsilon": eps_blk,
            "safety_age1": safety,
            "age": ages,
            "executor_cached_paths": exec_age,
            "executor_current_vis_vs_prop": exec_now,
            "pass": {
                "cost": bool(cost.get("pass")),
                "safety": bool(safety.get("pass")),
                "age_rho": bool(ages.get("1", {}).get("pass_rho")),
                "executor": bool(exec_age.get("pass")),
            },
        }
        if bb == "fastwam":
            out[bb]["pass"]["executor_current"] = bool((exec_now or {}).get("pass"))
    return out


def _fmt_pass(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _print_bb(name: str, b: dict) -> None:
    p = b["pass"]
    print(f"\n=== {name}  n={b['n']} ===")
    c = b["cost"]
    print(
        f"  [1 cost] {_fmt_pass(p['cost'])}  {c.get('note')}  "
        f"ratio={c.get('ratio_skip_over_probe') or c.get('ratio_prefix_over_siglip')}  {c}"
    )
    e = b["epsilon"]
    s = b["safety_age1"]
    best = s.get("best")
    print(
        f"  [2 safety] {_fmt_pass(p['safety'])}  ε={e.get('eps_l1')}  "
        f"consec med/p90={e.get('median')}/{e.get('p90')}  best={best}"
    )
    if s.get("sweep"):
        print("       tau sweep (p, τ, reuse, P(e>ε)):")
        for rec in s["sweep"]:
            print(
                f"         p={rec['p']:.1f} τ={rec['tau']:.4f}  "
                f"reuse={rec['reuse_rate']:.3f}  exceed={rec['p_exceed']:.3f}  "
                f"{'OK' if rec['ok'] else 'no'}"
            )
    print(f"  [3 age] {_fmt_pass(p['age_rho'])}")
    for age, blk in b["age"].items():
        rho = blk.get("rho")
        rho_s = "na" if rho is None else f"{rho:.3f}"
        print(
            f"       age={age} n={blk['n']} ρ={rho_s}  "
            f"ep-mean={blk.get('rho_episode_mean')}  err̄={blk.get('err_mean')}  "
            f"{_fmt_pass(bool(blk.get('pass_rho')))}"
        )
    ex = b["executor_cached_paths"]
    print(
        f"  [4 executor] {_fmt_pass(p['executor'])}  "
        f"median={ex.get('median')} limit={ex.get('limit')} n={ex.get('n')}"
    )
    if b.get("executor_current_vis_vs_prop"):
        cur = b["executor_current_vis_vs_prop"]
        print(
            f"       current vis-KV vs proprio-KV  "
            f"{_fmt_pass(bool(cur.get('pass')))} median={cur.get('median')}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out-json", type=str, default="")
    args = p.parse_args()
    rows = _load([Path(x) for x in args.paths])
    summary = _summarize(rows)
    print(f"rows={summary['n_rows']}  gates={summary['gates']}")
    for k, v in summary.items():
        if k in ("n_rows", "gates"):
            continue
        _print_bb(k, v)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
