#!/usr/bin/env python3
"""Analyze perception probes A (see timely) and B (short history).

Pre-registered gates (protocol.json):
  A1 lag     capture_lag_ms == 0 in freeze dump; report infer_ms vs success
  A2 queue   ρ(object_speed, err_delay_d4) > 0.3 AND moving median > 2× static
             AND median err_d4 ≥ median err_d1 on moving
  A3 desync  max(med img_old_d4, med prop_old_d4) / med delay_d4 ≥ 0.25 (moving)
  A4 ahead   ρ(object_speed, err_ahead_d4) > 0.3
  B1 motion  ρ(delta_cos_d1, object_speed) > 0.3
  B2 mix     med(err_mix_k2) / med(err_ahead_d1) ≥ 0.25 on moving
  B3 linear  test R²_hist > R²_curr + 0.10 AND R²_hist > 0.20
             (ridge on even/odd episodes; predict object_speed from φ)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.dynamic_libero.eval.coupling_math import spearman
from benchmarks.dynamic_libero.eval.perception_probe_common import DELAYS

RHO_PASS = 0.3
RATIO_DESYNC = 0.25
RATIO_MIX = 0.25
R2_GAIN = 0.10
R2_MIN = 0.20
STATIC_MULT = 2.0


def _load_dir(path: Path) -> tuple[list[dict], np.ndarray | None]:
    rows: list[dict] = []
    phis = None
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    phi_chunks: list[np.ndarray] = []
    offset = 0
    for f in files:
        if f.suffix != ".jsonl":
            continue
        local: list[dict] = []
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    local.append(json.loads(line))
        npz = f.with_name(f.stem + "_phi.npz")
        chunk = None
        if npz.is_file():
            chunk = np.load(npz)["phi"]
            phi_chunks.append(np.asarray(chunk))
        for r in local:
            idx = r.get("phi_index")
            if idx is not None and chunk is not None:
                r["phi_index"] = int(idx) + offset
            rows.append(r)
        if chunk is not None:
            offset += int(chunk.shape[0])
    if phi_chunks:
        phis = np.concatenate(phi_chunks, axis=0)
    return rows, phis


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


def _median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _moving(rows: list[dict]) -> list[dict]:
    return [r for r in rows if float(r.get("speed") or 0.0) > 0.0]


def _static(rows: list[dict]) -> list[dict]:
    return [r for r in rows if float(r.get("speed") or 0.0) <= 0.0]


def _rho(rows: list[dict], xk: str, yk: str) -> float | None:
    pts = _finite(rows, xk, yk)
    if len(pts) < 8:
        return None
    return spearman([float(r[xk]) for r in pts], [float(r[yk]) for r in pts])


def _gate(ok: bool | None) -> str:
    if ok is None:
        return "NA"
    return "PASS" if ok else "FAIL"


def _ridge_r2(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, lam: float = 1e-1) -> float | None:
    if train.sum() < 20 or test.sum() < 10:
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xtr = np.concatenate([x[train], np.ones((int(train.sum()), 1))], axis=1)
    xtx = xtr.T @ xtr + lam * np.eye(xtr.shape[1])
    try:
        w = np.linalg.solve(xtx, xtr.T @ y[train])
    except np.linalg.LinAlgError:
        return None
    xte = np.concatenate([x[test], np.ones((int(test.sum()), 1))], axis=1)
    yhat = xte @ w
    yte = y[test]
    ss_res = float(((yte - yhat) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return None
    return 1.0 - ss_res / ss_tot


def _linear_probe(rows: list[dict], phis: np.ndarray | None) -> dict:
    if phis is None or len(phis) == 0:
        return {"status": "NA", "reason": "no phi sidecar"}
    by_ep: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("phi_index") is None:
            continue
        idx = int(r["phi_index"])
        if idx < 0 or idx >= len(phis):
            continue
        key = (r.get("backbone"), r.get("task_id"), r.get("speed"), r.get("init_id"))
        by_ep[key].append(r)
    xs_cur = []
    xs_hist = []
    ys = []
    ep_ids = []
    for ei, (key, grp) in enumerate(by_ep.items()):
        grp = sorted(grp, key=lambda r: int(r.get("replan") or 0))
        for i, r in enumerate(grp):
            if i < 2:
                continue
            if grp[i - 1].get("phi_index") is None or grp[i - 2].get("phi_index") is None:
                continue
            spd = r.get("object_speed")
            if spd is None or not math.isfinite(float(spd)):
                continue
            p0 = np.asarray(phis[int(r["phi_index"])], dtype=np.float32)
            p1 = np.asarray(phis[int(grp[i - 1]["phi_index"])], dtype=np.float32)
            p2 = np.asarray(phis[int(grp[i - 2]["phi_index"])], dtype=np.float32)
            xs_cur.append(p0)
            xs_hist.append(np.concatenate([p0, p0 - p1, p0 - p2], axis=0))
            ys.append(float(spd))
            ep_ids.append(ei)
    if len(ys) < 40:
        return {"status": "NA", "n": len(ys)}
    y = np.asarray(ys, dtype=np.float64)
    x_cur = np.stack(xs_cur, axis=0)
    x_hist = np.stack(xs_hist, axis=0)
    ep = np.asarray(ep_ids)
    train = (ep % 2) == 0
    test = ~train
    r2_cur = _ridge_r2(x_cur, y, train, test)
    r2_hist = _ridge_r2(x_hist, y, train, test)
    ok = (
        r2_cur is not None
        and r2_hist is not None
        and r2_hist > r2_cur + R2_GAIN
        and r2_hist > R2_MIN
    )
    return {
        "n": int(len(ys)),
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "r2_current": r2_cur,
        "r2_history": r2_hist,
        "status": _gate(ok),
        "gain": None if r2_cur is None or r2_hist is None else r2_hist - r2_cur,
    }


def analyze_backbone(name: str, rows: list[dict], phis: np.ndarray | None) -> dict:
    moving = _moving(rows)
    static = _static(rows)
    infer = _vals(rows, "infer_ms")
    cap = _vals(rows, "capture_lag_ms")
    succ = _vals([r for r in rows if r.get("success")], "infer_ms")
    fail = _vals([r for r in rows if not r.get("success")], "infer_ms")

    a1 = {
        "median_capture_lag_ms": _median(cap),
        "median_infer_ms": _median(infer),
        "median_infer_ms_success": _median(succ),
        "median_infer_ms_fail": _median(fail),
        "n": len(rows),
        "note": "freeze dump: obs at predict is already the latest sim frame",
        "status": _gate(bool(cap) and abs(_median(cap) or 0.0) < 1e-6),
    }

    rho_delay = _rho(moving, "object_speed", "err_delay_d4_l1")
    med_d4_m = _median(_vals(moving, "err_delay_d4_l1"))
    med_d4_s = _median(_vals(static, "err_delay_d4_l1"))
    med_d1_m = _median(_vals(moving, "err_delay_d1_l1"))
    speed_ok = rho_delay is not None and rho_delay > RHO_PASS
    split_ok = (
        med_d4_m is not None
        and med_d4_s is not None
        and med_d4_m > STATIC_MULT * max(med_d4_s, 1e-8)
    )
    mono_ok = med_d4_m is not None and med_d1_m is not None and med_d4_m >= med_d1_m
    a2 = {
        "rho_speed_delay_d4": rho_delay,
        "median_delay_d1_moving": med_d1_m,
        "median_delay_d2_moving": _median(_vals(moving, "err_delay_d2_l1")),
        "median_delay_d4_moving": med_d4_m,
        "median_delay_d4_static": med_d4_s,
        "status": _gate(speed_ok and split_ok and mono_ok),
    }

    med_img = _median(_vals(moving, "err_img_old_d4_l1"))
    med_prop = _median(_vals(moving, "err_prop_old_d4_l1"))
    den = med_d4_m if med_d4_m not in (None, 0.0) else None
    img_ratio = None if med_img is None or den is None else med_img / den
    prop_ratio = None if med_prop is None or den is None else med_prop / den
    desync_ok = False
    if img_ratio is not None and prop_ratio is not None:
        desync_ok = max(img_ratio, prop_ratio) >= RATIO_DESYNC
    a3 = {
        "median_img_old_d4_moving": med_img,
        "median_prop_old_d4_moving": med_prop,
        "img_over_delay": img_ratio,
        "prop_over_delay": prop_ratio,
        "dominant": None
        if med_img is None or med_prop is None
        else ("image" if med_img >= med_prop else "proprio"),
        "status": _gate(desync_ok if img_ratio is not None else None),
    }

    rho_ahead = _rho(moving, "object_speed", "err_ahead_d4_l1")
    a4 = {
        "rho_speed_ahead_d4": rho_ahead,
        "median_ahead_d1_moving": _median(_vals(moving, "err_ahead_d1_l1")),
        "median_ahead_d4_moving": _median(_vals(moving, "err_ahead_d4_l1")),
        "median_ahead_d4_static": _median(_vals(static, "err_ahead_d4_l1")),
        "status": _gate(None if rho_ahead is None else rho_ahead > RHO_PASS),
    }

    rho_d1 = _rho(rows, "delta_cos_d1", "object_speed")
    rho_rate = _rho(moving, "motion_rate_d4", "err_ahead_d1_l1")
    b1 = {
        "rho_delta_d1_speed": rho_d1,
        "rho_delta_d4_speed": _rho(rows, "delta_cos_d4", "object_speed"),
        "rho_motion_rate_d4_ahead_d1": rho_rate,
        "status": _gate(None if rho_d1 is None else rho_d1 > RHO_PASS),
    }

    med_mix = _median(_vals(moving, "err_mix_k2_l1"))
    med_ah1 = _median(_vals(moving, "err_ahead_d1_l1"))
    mix_ratio = None if med_mix is None or med_ah1 in (None, 0.0) else med_mix / med_ah1
    b2 = {
        "median_mix_k2_moving": med_mix,
        "median_ahead_d1_moving": med_ah1,
        "mix_over_ahead": mix_ratio,
        "status": _gate(None if mix_ratio is None else mix_ratio >= RATIO_MIX),
    }
    b3 = _linear_probe(rows, phis)

    return {
        "backbone": name,
        "n_rows": len(rows),
        "n_moving": len(moving),
        "n_static": len(static),
        "A1_lag": a1,
        "A2_queue": a2,
        "A3_desync": a3,
        "A4_lookahead": a4,
        "B1_visual_motion": b1,
        "B2_naive_mix": b2,
        "B3_linear_probe": b3,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fastwam", type=str)
    p.add_argument("pi", type=str)
    p.add_argument("--out-json", type=str, default="")
    args = p.parse_args()
    fw_rows, fw_phi = _load_dir(Path(args.fastwam))
    pi_rows, pi_phi = _load_dir(Path(args.pi))
    report = {
        "gates": {
            "rho": RHO_PASS,
            "desync_ratio": RATIO_DESYNC,
            "mix_ratio": RATIO_MIX,
            "r2_gain": R2_GAIN,
            "r2_min": R2_MIN,
            "delays": list(DELAYS),
        },
        "fastwam": analyze_backbone("fastwam", fw_rows, fw_phi),
        "pi": analyze_backbone("pi", pi_rows, pi_phi),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text)


if __name__ == "__main__":
    main()
