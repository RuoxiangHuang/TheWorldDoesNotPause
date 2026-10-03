#!/usr/bin/env python3
"""Pool latency-slice + aligned react-slice SR (off vs Cache)."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from benchmarks.common.outcomes import read_grasp_hold, read_rails_released


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


def _label(cell: dict[str, Any]) -> str:
    raw = str(cell.get("cell_label") or cell.get("accel") or "")
    return {
        "eager": "off",
        "pi": "off",
        "fasterpi": "cache",
    }.get(raw, raw)


def _traj(cell: dict[str, Any]) -> str:
    return str(cell.get("trajectory") or "")


def _speed(cell: dict[str, Any]) -> float:
    try:
        return float(cell.get("speed"))
    except (TypeError, ValueError):
        return -1.0


def pool(cells: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for c in cells:
        grouped[(_label(c), _traj(c), round(_speed(c), 6))].append(c)
    out: dict[tuple, dict[str, Any]] = {}
    for key, rows in grouped.items():
        succ: list[bool] = []
        grasp: list[bool] = []
        lats: list[float] = []
        exp_ok: list[bool] = []
        missed_ok: list[bool] = []
        by_task: dict[str, list[bool]] = defaultdict(list)
        keyed: dict[tuple[str, int], dict[str, Any]] = {}
        for r in rows:
            if r.get("latency_ms") is not None:
                lats.append(float(r["latency_ms"]))
            for ep in r.get("episodes") or []:
                ok = bool(ep.get("success"))
                succ.append(ok)
                hold = read_grasp_hold(ep)
                if hold is not None:
                    grasp.append(hold)
                tid = str(
                    ep.get("dynamic_task_id")
                    or ep.get("task_name")
                    or ep.get("task_id")
                )
                by_task[tid].append(ok)
                experienced = ep.get("event_experienced")
                if experienced is True:
                    exp_ok.append(ok)
                elif experienced is False:
                    missed_ok.append(ok)
                try:
                    iid = int(ep.get("init_id"))
                except (TypeError, ValueError):
                    iid = -1
                keyed[(tid, iid)] = {
                    "success": ok,
                    "event_experienced": experienced,
                    "grasp": read_grasp_hold(ep),
                }
        n = len(succ)
        wins = int(sum(succ))
        n_exp = len(exp_ok)
        n_miss = len(missed_ok)
        out[key] = {
            "label": key[0],
            "trajectory": key[1],
            "speed": key[2],
            "n": n,
            "wins": wins,
            "sr": (wins / n) if n else None,
            "grasp_sr": (sum(grasp) / len(grasp)) if grasp else None,
            "grasp_n_known": len(grasp),
            "latency_ms": _median(lats),
            "n_experienced": n_exp,
            "sr_experienced": (sum(exp_ok) / n_exp) if n_exp else None,
            "wins_experienced": int(sum(exp_ok)) if n_exp else 0,
            "n_missed": n_miss,
            "wins_missed": int(sum(missed_ok)) if n_miss else 0,
            "sr_missed": (sum(missed_ok) / n_miss) if n_miss else None,
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


def render(pooled: dict[tuple, dict[str, Any]]) -> str:
    tasks = sorted({t for r in pooled.values() for t in r["by_task"]})
    is_place = bool(tasks) and all(str(t).endswith(".place") for t in tasks)
    is_pick = bool(tasks) and all(
        str(t).endswith(".pick") and ".react" not in str(t) for t in tasks
    )
    is_rt = bool(tasks) and any(
        str(t).startswith(
            (
                "place_empty_cup",
                "place_can_basket",
                "move_can_pot",
                "place_container_plate",
                "place_object_stand",
            )
        )
        for t in tasks
    )
    if is_rt and is_place:
        blurb = (
            "Pre-registered Dynamic-RoboTwin **place** slice "
            "(cup/can-basket/can-pot/container-plate/object-stand). "
            "Moving receptacle, rails-only, async. Not official `all` / 30-id pool."
        )
    elif is_rt:
        blurb = (
            "Pre-registered Dynamic-RoboTwin **pick** slice "
            "(cup/can-basket/can-pot/container-plate/object-stand). "
            "Moving grasp target, rails-only, async. Not official `all` / 30-id pool."
        )
    elif is_place:
        blurb = (
            "Pre-registered **place** slice: same skills as the FastWAM pick "
            "slice (t00/t01/t04/t05/t06) with a moving basket. Linear. "
            "Not the official reconstructed 60."
        )
    elif is_pick:
        blurb = (
            "Pre-registered pick skills t00/t01/t04/t05/t06 × inits 0–4. "
            "Not the official reconstructed 60. Main = linear; react = same cups + one turn."
        )
    else:
        blurb = (
            "Latency-sensitive Dynamic-LIBERO slice × inits 0–4. "
            "Not the official reconstructed 60."
        )
    lines = [
        (
            "# Latency-sensitive Dynamic-RoboTwin slice"
            if is_rt
            else "# Latency-sensitive Dynamic-LIBERO slice"
        ),
        "",
        blurb,
        "",
        "## Cells",
        "",
        "| track | v | cell | n | SR | grasp | L ms |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    keys = sorted(pooled, key=lambda k: (k[1], k[2], k[0]))
    for k in keys:
        r = pooled[k]
        lines.append(
            "| "
            + " | ".join(
                [
                    r["trajectory"],
                    f"{r['speed']:g}",
                    r["label"],
                    str(r["n"]),
                    f"{r['wins']}/{r['n']} ({_fmt(r['sr'])})",
                    _fmt(r["grasp_sr"]),
                    _fmt(r["latency_ms"], 1),
                ]
            )
            + " |"
        )

    lines += ["", "## ΔSR (Cache − off)", "", "| track | v | off | Cache | ΔSR |", "|---|---:|---:|---:|---:|"]
    by_tv: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for k, r in pooled.items():
        by_tv[(r["trajectory"], r["speed"])][r["label"]] = r
    for (traj, speed), labs in sorted(by_tv.items()):
        off = labs.get("off")
        cache = labs.get("cache")
        if not off or not cache or off["sr"] is None or cache["sr"] is None:
            continue
        d = cache["sr"] - off["sr"]
        lines.append(
            f"| {traj} | {speed:g} | {_fmt(off['sr'])} | {_fmt(cache['sr'])} | {d:+.3f} |"
        )

    lines += [
        "",
        "## React: experienced vs grasped-before-event",
        "",
        "Full-task BDDL SR still counts missed events. Reaction claims use the "
        "experienced subset, or the paired subset where both stacks saw the turn.",
        "",
        "| cell | n | experienced | missed | SR all | SR experienced | SR missed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in keys:
        r = pooled[k]
        if r["trajectory"] != "smooth_turn":
            continue
        n_exp = int(r.get("n_experienced") or 0)
        n_miss = int(r.get("n_missed") or 0)
        lines.append(
            "| "
            + " | ".join(
                [
                    r["label"],
                    str(r["n"]),
                    str(n_exp),
                    str(n_miss),
                    f"{r['wins']}/{r['n']} ({_fmt(r['sr'])})",
                    (
                        f"{r['wins_experienced']}/{n_exp} ({_fmt(r['sr_experienced'])})"
                        if n_exp
                        else "—"
                    ),
                    (
                        f"{r['wins_missed']}/{n_miss} ({_fmt(r['sr_missed'])})"
                        if n_miss
                        else "—"
                    ),
                ]
            )
            + " |"
        )

    react_tv = [
        (traj, speed, labs)
        for (traj, speed), labs in sorted(by_tv.items())
        if traj == "smooth_turn" and "off" in labs and "cache" in labs
    ]
    if react_tv:
        lines += [
            "",
            "### Paired experienced (same task, init; both saw the turn)",
            "",
            "| v | n_paired | off | Cache | ΔSR |",
            "|---:|---:|---:|---:|---:|",
        ]
        for traj, speed, labs in react_tv:
            off_k = labs["off"].get("keyed") or {}
            cache_k = labs["cache"].get("keyed") or {}
            off_s: list[bool] = []
            cache_s: list[bool] = []
            for key in sorted(set(off_k) & set(cache_k)):
                a, b = off_k[key], cache_k[key]
                if a.get("event_experienced") is True and b.get("event_experienced") is True:
                    off_s.append(bool(a["success"]))
                    cache_s.append(bool(b["success"]))
            n = len(off_s)
            if not n:
                lines.append(f"| {speed:g} | 0 | — | — | — |")
                continue
            so = sum(off_s) / n
            sc = sum(cache_s) / n
            lines.append(
                f"| {speed:g} | {n} | {sum(off_s)}/{n} ({_fmt(so)}) | "
                f"{sum(cache_s)}/{n} ({_fmt(sc)}) | {sc - so:+.3f} |"
            )

    lines += ["", "## Per-task SR", ""]
    tasks = sorted(
        {
            t
            for r in pooled.values()
            for t in r["by_task"]
        }
    )
    if tasks:
        header = ["task"]
        cols = []
        for k in keys:
            r = pooled[k]
            header.append(f"{r['trajectory']} v={r['speed']:g} {r['label']}")
            cols.append(r)
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for t in tasks:
            row = [t]
            for r in cols:
                cell = r["by_task"].get(t)
                if cell is None:
                    row.append("—")
                else:
                    row.append(f"{cell['wins']}/{cell['n']}")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze latency-slice SR JSON")
    p.add_argument("--inputs", type=str, required=True)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()
    root = Path(args.inputs)
    cells = _cells(root)
    if not cells:
        raise SystemExit(f"no sr_sweep.json under {root}")
    text = render(pool(cells))
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
