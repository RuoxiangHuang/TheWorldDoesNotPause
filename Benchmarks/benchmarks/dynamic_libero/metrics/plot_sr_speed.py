#!/usr/bin/env python3
"""Merge reconstructed-60 SR sweeps, overlay layout-replacement evals, plot SR–speed.

Base waves keep the 57 unchanged tasks. Dynamic-speed cells for
``RECONSTRUCTED_REPLACEMENTS`` come from the post-fix override roots
(v=0 is unchanged: no car / no path-clear at speed 0).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_BASE = [
    "evaluate_results/dynamic_libero/reconstructed_60_off_vs_full",
    "evaluate_results/dynamic_libero/reconstructed_60_off_vs_full_v0005",
    "evaluate_results/dynamic_libero/reconstructed_60_off_vs_full_v0003_0008",
    "evaluate_results/dynamic_libero/reconstructed_60_off_vs_full_v0002_0006",
]
DEFAULT_OVERRIDE = [
    "evaluate_results/dynamic_libero/three_dead_fix",
    "evaluate_results/dynamic_libero/three_dead_fix_rest",
]
SPEEDS = (0.0, 0.0002, 0.0003, 0.0005, 0.0006, 0.0008, 0.001, 0.003)
SPLITS = ("object_linear", "object_curve", "spatial_carrier")
SKILLS = ("pick", "place", "both", "curve_pick", "curve_place", "spatial_pick")


def _split(task_id: str) -> str:
    if ".irregular" in task_id:
        return "object_curve"
    if task_id.startswith("libero_spatial."):
        return "spatial_carrier"
    return "object_linear"


def _skill(task_id: str) -> str:
    parts = task_id.split(".")
    variant = parts[2] if len(parts) > 2 else "pick"
    if ".irregular" in task_id:
        return f"curve_{variant}"
    if task_id.startswith("libero_spatial."):
        return "spatial_pick"
    return variant


def _load_cells(root: Path) -> list[dict]:
    cells: list[dict] = []
    for p in sorted(root.rglob("sr_sweep.json")):
        cells.extend(json.loads(p.read_text()).get("cells") or [])
    return cells


def _episodes(cells: list[dict]) -> dict[tuple[str, float, str], list[dict]]:
    bucket: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    for cell in cells:
        accel = str(cell.get("accel") or "")
        speed = float(cell.get("speed"))
        for e in cell.get("episodes") or []:
            tid = str(e.get("dynamic_task_id") or f"t{e.get('task_id')}")
            bucket[(accel, speed, tid)].append(e)
    return bucket


def _sr(eps: list[dict]) -> tuple[float, int]:
    if not eps:
        return float("nan"), 0
    return float(np.mean([1.0 if e.get("success") else 0.0 for e in eps])), len(eps)


def _key_speed(speed: float) -> float:
    for s in SPEEDS:
        if abs(s - speed) < 1e-12:
            return s
    return speed


def merge_buckets(
    base: dict[tuple[str, float, str], list[dict]],
    override: dict[tuple[str, float, str], list[dict]],
    replacements: tuple[str, ...],
) -> dict[tuple[str, float, str], list[dict]]:
    out: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    for k, eps in base.items():
        accel, speed, tid = k
        speed = _key_speed(speed)
        out[(accel, speed, tid)] = list(eps)
    for (accel, speed, tid), eps in override.items():
        speed = _key_speed(speed)
        if tid not in replacements:
            continue
        if speed <= 0.0:
            continue
        out[(accel, speed, tid)] = list(eps)
    return out


def _macro_row(bucket, speed: float, split: str | None) -> dict:
    off, full = [], []
    tasks = sorted({k[2] for k in bucket})
    for tid in tasks:
        if split is not None and _split(tid) != split:
            continue
        off.extend(bucket.get(("off", speed, tid), []))
        full.extend(bucket.get(("full", speed, tid), []))
    sr_off, n_off = _sr(off)
    sr_full, n_full = _sr(full)
    n = min(n_off, n_full)
    return {
        "speed": speed,
        "off": sr_off,
        "full": sr_full,
        "n": n,
        "delta": (sr_full - sr_off) if n else float("nan"),
    }


def build_payload(bucket, replacements: tuple[str, ...], speedup: dict | None) -> dict:
    tasks = sorted({k[2] for k in bucket})
    grid = []
    for speed in SPEEDS:
        row: dict = {"speed": speed}
        all_row = _macro_row(bucket, speed, None)
        row["all_off"] = all_row["off"]
        row["all_full"] = all_row["full"]
        row["all_n"] = all_row["n"]
        row["all_delta"] = all_row["delta"]
        for split in SPLITS:
            r = _macro_row(bucket, speed, split)
            row[f"{split}_off"] = r["off"]
            row[f"{split}_full"] = r["full"]
            row[f"{split}_n"] = r["n"]
            row[f"{split}_delta"] = r["delta"]
        grid.append(row)

    skill_rows = []
    for skill in SKILLS:
        for speed in SPEEDS:
            off, full = [], []
            for tid in tasks:
                if _skill(tid) != skill:
                    continue
                off.extend(bucket.get(("off", speed, tid), []))
                full.extend(bucket.get(("full", speed, tid), []))
            sr_off, n_off = _sr(off)
            sr_full, n_full = _sr(full)
            skill_rows.append(
                {
                    "skill": skill,
                    "speed": speed,
                    "off": sr_off,
                    "full": sr_full,
                    "n": min(n_off, n_full),
                }
            )

    per_task = []
    for tid in tasks:
        rec = {"task": tid, "split": _split(tid), "skill": _skill(tid), "speeds": {}}
        for speed in SPEEDS:
            sr_off, n_off = _sr(bucket.get(("off", speed, tid), []))
            sr_full, n_full = _sr(bucket.get(("full", speed, tid), []))
            rec["speeds"][f"{speed:g}"] = {
                "off": sr_off,
                "full": sr_full,
                "n": min(n_off, n_full),
                "delta": sr_full - sr_off,
            }
        per_task.append(rec)

    replaced = []
    for tid in replacements:
        rec = {"task": tid, "speeds": {}}
        for speed in SPEEDS:
            if speed <= 0.0:
                continue
            sr_off, _ = _sr(bucket.get(("off", speed, tid), []))
            sr_full, _ = _sr(bucket.get(("full", speed, tid), []))
            rec["speeds"][f"{speed:g}"] = {"off": sr_off, "full": sr_full}
        replaced.append(rec)

    return {
        "protocol": (
            "real_time_v2 freeze movers=off language=keep N=5 reconstructed-60 "
            "+ layout replacements for t03.pick.irregular / t00.carrier / t05.carrier"
        ),
        "replacements": list(replacements),
        "speeds": list(SPEEDS),
        "grid": grid,
        "skill": skill_rows,
        "per_task": per_task,
        "replaced_tasks": replaced,
        "speedup": speedup or {},
    }


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# FastWAM vs FasterWAM — SR–Speed（含布局替换）",
        "",
        "三条原先动态 SR=0 的任务用微调后的实现替换后再统计。",
        "v=0 沿用原 sweep（speed=0 不装车、不清路径）。",
        "",
        "替换条目：",
        "",
    ]
    for tid in payload["replacements"]:
        lines.append(f"- `{tid}`")
    lines += [
        "",
        "## Macro",
        "",
        "| split | speed | n | FastWAM | FasterWAM | ΔSR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in (*SPLITS, "all"):
        for row in payload["grid"]:
            sp = row["speed"]
            lines.append(
                f"| {split} | {sp:g} | {row[f'{split}_n']} | "
                f"{row[f'{split}_off']:.3f} | {row[f'{split}_full']:.3f} | "
                f"{row[f'{split}_delta']:+.3f} |"
            )
    lines += [
        "",
        "## 替换三条（动态速度）",
        "",
        "| task | speed | FastWAM | FasterWAM |",
        "|---|---:|---:|---:|",
    ]
    for rec in payload["replaced_tasks"]:
        for sp, cell in rec["speeds"].items():
            lines.append(
                f"| `{rec['task']}` | {sp} | {cell['off']:.2f} | {cell['full']:.2f} |"
            )
    lines += [
        "",
        "## 技能 × 速度",
        "",
        "| skill | speed | n | FastWAM | FasterWAM | ΔSR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for rec in payload["skill"]:
        delta = rec["full"] - rec["off"]
        lines.append(
            f"| {rec['skill']} | {rec['speed']:g} | {rec['n']} | "
            f"{rec['off']:.3f} | {rec['full']:.3f} | {delta:+.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_png(payload: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = payload["grid"]
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160}
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), gridspec_kw={"width_ratios": [1.4, 1]})
    xplot = [r["speed"] * 1e4 for r in grid]
    ax = axes[0]
    ax.plot(
        xplot,
        [r["all_off"] * 100 for r in grid],
        "o-",
        color="#4a5568",
        lw=2,
        ms=6.5,
        label="FastWAM (off)",
    )
    ax.plot(
        xplot,
        [r["all_full"] * 100 for r in grid],
        "s-",
        color="#2b6cb0",
        lw=2,
        ms=6.5,
        label="FasterWAM (full)",
    )
    peak = max(grid, key=lambda r: r["all_delta"] if r["speed"] > 0 else -1)
    ax.annotate(
        f"Δ{peak['all_delta']*100:+.1f} pp",
        xy=(peak["speed"] * 1e4, peak["all_full"] * 100),
        xytext=(8, 10),
        textcoords="offset points",
        color="#2b6cb0",
        fontsize=9,
    )
    ax.set_xlabel("object speed v (×10⁻⁴ m / tick)")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("all 60 tasks")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    styles = [
        ("object_linear", "#2b6cb0", "linear"),
        ("object_curve", "#2f855a", "curve"),
        ("spatial_carrier", "#c05621", "spatial"),
    ]
    for key, color, lab in styles:
        ax.plot(
            xplot,
            [r[f"{key}_full"] * 100 for r in grid],
            "s-",
            color=color,
            lw=1.8,
            ms=5,
            label=f"FasterWAM {lab}",
        )
        ax.plot(
            xplot,
            [r[f"{key}_off"] * 100 for r in grid],
            "o--",
            color=color,
            lw=1.2,
            ms=4,
            alpha=0.55,
            label=f"FastWAM {lab}",
        )
    ax.set_xlabel("object speed v (×10⁻⁴ m / tick)")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("subset")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _require_overrides(bucket, replacements: tuple[str, ...]) -> None:
    missing = []
    for tid in replacements:
        for speed in SPEEDS:
            if speed <= 0.0:
                continue
            for accel in ("off", "full"):
                n = len(bucket.get((accel, speed, tid), []))
                if n < 5:
                    missing.append((tid, accel, speed, n))
    if missing:
        msg = "\n".join(f"  {t} {a} v={s:g} n={n}" for t, a, s, n in missing)
        raise SystemExit(f"override cells incomplete:\n{msg}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=str, default="/DATA/YuanZhen/FasterWAM")
    p.add_argument("--out-dir", type=str, default="")
    args = p.parse_args()
    repo = Path(args.repo)
    from benchmarks.dynamic_libero.env.dynamic_tasks import RECONSTRUCTED_REPLACEMENTS

    replacements = RECONSTRUCTED_REPLACEMENTS
    base_cells: list[dict] = []
    for rel in DEFAULT_BASE:
        base_cells.extend(_load_cells(repo / rel))
    ov_cells: list[dict] = []
    for rel in DEFAULT_OVERRIDE:
        ov_cells.extend(_load_cells(repo / rel))
    bucket = merge_buckets(_episodes(base_cells), _episodes(ov_cells), replacements)
    _require_overrides(bucket, replacements)

    old_json = repo / "evaluate_results/dynamic_libero/reconstructed_60_off_vs_full/sr_speed_curve.json"
    speedup = {}
    if old_json.is_file():
        speedup = json.loads(old_json.read_text()).get("speedup") or {}

    payload = build_payload(bucket, replacements, speedup)
    out_dir = Path(args.out_dir) if args.out_dir else old_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sr_speed_curve.json"
    png_path = out_dir / "sr_speed_curve.png"
    md_path = out_dir / "COMPARISON_speed_grid.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_png(payload, png_path)
    write_markdown(payload, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {md_path}")
    print("\nMacro all-60:")
    for row in payload["grid"]:
        print(
            f"  v={row['speed']:<8g} n={row['all_n']:3d}  "
            f"off={row['all_off']:.3f}  full={row['all_full']:.3f}  "
            f"Δ={row['all_delta']:+.3f}"
        )


if __name__ == "__main__":
    main()
