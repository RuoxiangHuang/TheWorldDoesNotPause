"""Shared CLI helpers for Dynamic-RoboTwin eval scripts."""

from __future__ import annotations

import argparse
from typing import Any

from benchmarks.common.protocol import DEFAULT_BACKEND, PROTOCOL_VERSION
from benchmarks.common.grasp import add_grasp_cli_args
from benchmarks.common.scene import add_scene_cli_args


def protocol_tag() -> str:
    return PROTOCOL_VERSION


def add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND.value,
        choices=["freeze", "async"],
        help="Protocol default is async (prior chunk else ZOH); freeze is ablation.",
    )
    p.add_argument("--allow-replan-override", action="store_true")
    add_grasp_cli_args(p)
    add_scene_cli_args(p)


def grasp_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "grasp_mode": getattr(args, "grasp_mode", "ghost"),
        "engage_alpha": float(getattr(args, "engage_alpha", 1.75)),
    }


def seeds_from_args(args: argparse.Namespace, *, default: str = "0,1,2,3,4") -> list[int]:
    raw = getattr(args, "seeds", None)
    if raw is None or str(raw).strip() == "":
        raw = default
    return [int(x) for x in str(raw).split(",") if x.strip() != ""]


def parse_pi_speedups(spec: str) -> list[tuple[str, str]]:
    """Parse ``off=eager,cache=cache`` (or a bare FasterPI spec) into cell labels."""
    pairs: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            label, s = item.split("=", 1)
            pairs.append((label.strip(), s.strip()))
        else:
            label = "off" if item in ("eager", "none", "") else "cache"
            pairs.append((label, item))
    return pairs


def stub_policy_ctx(*, replan_steps: int, policy_hz: float):
    """Minimal ctx when the policy is HTTP / in-process OpenPI, not FastWAM."""
    from dataclasses import dataclass

    @dataclass
    class _StubCtx:
        replan_steps: int
        policy_hz: float

    return _StubCtx(replan_steps=int(replan_steps), policy_hz=float(policy_hz))
