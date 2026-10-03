"""Repository path helpers for the FastWAM acceleration stack."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Repo root for The World Does Not Pause (contains FastWAM/, Benchmarks/)."""
    return Path(__file__).resolve().parents[1]


def fastwam_root() -> Path:
    return Path(__file__).resolve().parent


def benchmarks_root() -> Path:
    return repo_root() / "Benchmarks"


def checkpoints_root() -> Path:
    return repo_root() / "checkpoints"
