#!/usr/bin/env python3
"""Reaction-track pick eval — aligned with the main linear pick catalog.

Not part of the official linear/sine SR grid. Locks:

  20 Hz, replan_steps=10, speed=0.001 m/tick, rails=pick, motive=ghost,
  ``--dynamic-tasks react`` (same objects / language / BDDL as main pick).

The pick target goes inbound on the same heading as the aligned main-track
task, then turns at a world-clock time sampled from (task, init, seed).
Pass ``--dynamic-tasks`` yourself to run a subset.

Examples::

  python -m benchmarks.dynamic_libero.eval.run_smooth_turn \\
    --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \\
    --accels off,default --init-ids 0,1,2,3,4

  python -m benchmarks.dynamic_libero.eval.run_smooth_turn \\
    --checkpoint ... --accels default --fixed-latency 0.244
"""

from __future__ import annotations

import argparse
import sys

from benchmarks.dynamic_libero.trajectories.smooth_turn import (
    DIAGNOSTIC_CONTROL_HZ,
    DIAGNOSTIC_KIND,
    DIAGNOSTIC_MOTIVE,
    DIAGNOSTIC_RAILS_VARIANT,
    DIAGNOSTIC_REPLAN_STEPS,
    DIAGNOSTIC_SPEED,
)


def _has_flag(argv: list[str], names: tuple[str, ...]) -> bool:
    for a in argv:
        for n in names:
            if a == n or a.startswith(n + "="):
                return True
    return False


def _parse_passthrough(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=(
            "Reaction-track pick eval (not the official linear/sine grid). "
            "Locks 20 Hz, r=10, v=0.001, rails=pick, motive=ghost, "
            "and --dynamic-tasks react unless already set. "
            "Remaining flags are forwarded to run_sr "
            "(--checkpoint, --accels, --init-ids, --fixed-latency, ...)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.dynamic_libero.eval.run_smooth_turn "
            "--checkpoint CKPT --accels off,default\n"
            "  python -m benchmarks.dynamic_libero.eval.run_smooth_turn "
            "--checkpoint CKPT --accels default --fixed-latency 0.244\n"
            "  python -m benchmarks.dynamic_libero.eval.run_smooth_turn "
            "--checkpoint CKPT --dynamic-tasks libero_object.t00.pick.react\n"
        ),
        add_help=True,
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="evaluate_results/dynamic_libero/smooth_turn",
    )
    known, rest = p.parse_known_args(argv)
    return known, rest


def main(argv: list[str] | None = None) -> None:
    known, rest = _parse_passthrough(sys.argv[1:] if argv is None else argv)
    locked = [
        "--trajectories",
        DIAGNOSTIC_KIND,
        "--speeds",
        str(DIAGNOSTIC_SPEED),
        "--replan-steps",
        str(DIAGNOSTIC_REPLAN_STEPS),
        "--rails-variant",
        DIAGNOSTIC_RAILS_VARIANT,
        "--motive",
        DIAGNOSTIC_MOTIVE,
        "--out-dir",
        known.out_dir,
    ]
    if not _has_flag(rest, ("--dynamic-tasks", "--dynamic-task")):
        locked.extend(["--dynamic-tasks", "react"])
    # Locked flags last so they win over a mistaken --trajectories linear,sine.
    sys.argv = [sys.argv[0], *rest, *locked]
    print(
        "[react track] "
        f"traj={DIAGNOSTIC_KIND} v={DIAGNOSTIC_SPEED} r={DIAGNOSTIC_REPLAN_STEPS} "
        f"hz={DIAGNOSTIC_CONTROL_HZ} rails={DIAGNOSTIC_RAILS_VARIANT} "
        f"motive={DIAGNOSTIC_MOTIVE} (aligned with main pick; not official SR grid)",
        flush=True,
    )
    from benchmarks.dynamic_libero.eval.run_sr import main as sr_main

    sr_main()


if __name__ == "__main__":
    main()
