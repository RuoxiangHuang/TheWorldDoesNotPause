"""Path helpers for Dynamic-LIBERO and Dynamic-RoboTwin."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Repo root for The World Does Not Pause (contains FastWAM/, Benchmarks/)."""
    here = Path(__file__).resolve()
    for anc in here.parents:
        if (anc / "FastWAM").is_dir() and (anc / "Benchmarks").is_dir():
            return anc
    return here.parents[2]


def fastwam_root() -> Path:
    return repo_root() / "FastWAM"


def benchmarks_root() -> Path:
    return repo_root() / "Benchmarks"


def checkpoints_root() -> Path:
    return repo_root() / "checkpoints"
