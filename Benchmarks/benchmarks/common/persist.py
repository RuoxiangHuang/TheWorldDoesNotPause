"""Crash-safe dumps for long SR sweeps."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def dump_partial_cell(
    path: Path,
    cell: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    partial: bool = True,
) -> None:
    payload = dict(cell)
    payload["n"] = len(rows)
    payload["episodes"] = rows
    if rows:
        payload["sr"] = sum(1 for r in rows if r.get("success")) / len(rows)
    else:
        payload["sr"] = 0.0
    payload["partial"] = partial
    write_json_atomic(path, payload)
