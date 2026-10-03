"""CLI path bootstrap for the FasterWAM monorepo (importable without package path)."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_sys_path() -> Path:
    """Insert ``Benchmarks/`` + ``FastWAM/src`` on ``sys.path``; return repo root."""
    here = Path(__file__).resolve().parent
    repo = here.parent
    fastwam_src = repo / "FastWAM" / "src"
    if not fastwam_src.is_dir() or not (here / "benchmarks").is_dir():
        raise RuntimeError(f"Invalid monorepo layout under {repo}")
    for p in (str(here), str(fastwam_src)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def fastwam_root() -> Path:
    return repo_root() / "FastWAM"


def checkpoints_root() -> Path:
    return repo_root() / "checkpoints"
