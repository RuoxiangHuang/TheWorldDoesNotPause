"""Shared CLI helpers for Dynamic-LIBERO eval scripts."""

from __future__ import annotations

import argparse
from typing import Any

from benchmarks.common.protocol import DEFAULT_BACKEND, PROTOCOL_VERSION
from benchmarks.common.grasp import add_grasp_cli_args, parse_place_freeze
from benchmarks.dynamic_libero.env.suite_registry import parse_init_ids, parse_task_ids
from benchmarks.dynamic_libero.trajectories.complexity import COMPLEXITY_CHOICES


def add_paired_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--task-ids", type=str, default="0", help="Comma-separated ids or 'all'")
    p.add_argument("--init-ids", type=str, default="0,1,2,3,4", help="Comma-separated init ids")
    p.add_argument("--accels", type=str, default="off,pace", help="Comma-separated accel presets")
    p.add_argument(
        "--trajectory",
        type=str,
        default="linear",
        help="Path family: linear|sine|circle|polyline|stop_and_go|random|smooth_turn "
        "(ignored when --traj-complexity is not none). "
        "smooth_turn is a diagnostic slice, not the official SR grid.",
    )
    p.add_argument(
        "--traj-complexity",
        type=str,
        default="none",
        choices=list(COMPLEXITY_CHOICES),
        help="Optional difficulty preset: easy=linear, medium=sine, "
        "hard/chaotic=seeded random polyline (overrides --trajectory)",
    )
    p.add_argument("--speed", type=float, default=0.003)
    p.add_argument("--axis", type=str, default="x")
    p.add_argument("--direction", type=float, default=1.0)
    p.add_argument(
        "--toward-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip rails so the target drifts toward the workspace origin (default on).",
    )
    p.add_argument("--radius", type=float, default=0.04)
    p.add_argument("--omega", type=float, default=0.03)
    p.add_argument("--amplitude", type=float, default=0.03)
    p.add_argument("--wavelength", type=float, default=0.20)
    p.add_argument("--n-waypoints", type=int, default=5, help="random polyline waypoint count")
    p.add_argument("--extent", type=float, default=0.12, help="random polyline half-span (m)")
    p.add_argument("--release-radius", type=float, default=0.06)
    p.add_argument(
        "--workspace-aabb",
        type=str,
        default=None,
        help="Override escape AABB as xmin,ymin,zmin,xmax,ymax,zmax (metres)",
    )
    p.add_argument("--no-escape-check", action="store_true")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--latency-k", type=int, default=8)
    p.add_argument("--latency-warmup", type=int, default=3)
    p.add_argument("--fixed-latency", type=float, default=None)
    p.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND.value,
        choices=["freeze", "async"],
        help="Protocol default is async (prior chunk else ZOH); freeze is ablation.",
    )
    p.add_argument("--policy-url", type=str, default=None)
    p.add_argument("--replan-steps", type=int, default=None)
    p.add_argument("--allow-replan-override", action="store_true")
    p.add_argument(
        "--movers",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Optional toy-car asset swap (secondary track). Official primary track: off.",
    )
    p.add_argument(
        "--mover-language",
        type=str,
        default="keep",
        choices=["rewrite", "keep"],
        help="Language for mover nouns. Official primary track: keep.",
    )
    p.add_argument(
        "--language-mode",
        type=str,
        default="keep",
        choices=["keep", "motion", "scene"],
        help="keep=training instruction (primary); motion=B-track suffix; "
        "scene=rewrite stale spatial locatives (bowl on moving platform).",
    )
    p.add_argument(
        "--no-target-registry",
        action="store_true",
        help="Disable MoverSpec rails targeting (legacy BDDL inference). Not recommended.",
    )
    add_grasp_cli_args(p)
    add_motive_arg(p)
    add_dynamic_task_args(p)
    add_smooth_turn_args(p)
    from benchmarks.common.scene import add_scene_cli_args

    add_scene_cli_args(p)


def add_motive_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--motive",
        type=str,
        default="auto",
        help="Motion motive: auto|pusher|carrier|conveyor|toycar|ghost. "
        "auto: pusher (toy car on rails, original mesh) for libero_object/"
        "libero_10; spatial/goal stay ghost-pin unless movers=on. "
        "carrier: flatbed carries the pick target and a second car pushes the basket. "
        "Does not swap the pick mesh.",
    )


def motive_from_args(args: argparse.Namespace) -> str:
    return str(getattr(args, "motive", "auto") or "auto")


def add_smooth_turn_args(p: argparse.ArgumentParser) -> None:
    """Optional pins for the smooth-turn diagnostic. Default: sample at reset."""
    p.add_argument(
        "--turn-event-t",
        type=float,
        default=None,
        help="Pin turn event to this policy-clock tick. Default: Uniform from episode seed.",
    )
    p.add_argument(
        "--turn-event-t-lo",
        type=float,
        default=None,
        help="Lower bound (ticks after begin_policy) when sampling t_event. Slice default 30.",
    )
    p.add_argument(
        "--turn-event-t-hi",
        type=float,
        default=None,
        help="Upper bound (ticks) when sampling t_event. Slice default 50.",
    )
    p.add_argument(
        "--turn-angle-deg",
        type=float,
        default=None,
        help="Override turn angle (deg). Diagnostic default is 75.",
    )
    p.add_argument(
        "--turn-duration-ticks",
        type=float,
        default=None,
        help="Override raised-cosine turn duration in control ticks (20 Hz → 12 = 0.6 s).",
    )
    p.add_argument(
        "--turn-sign",
        type=int,
        default=None,
        choices=[-1, 1],
        help="Pin turn side. Default: sampled from episode seed.",
    )


def add_dynamic_task_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--rails-variant",
        type=str,
        default="auto",
        help="Who moves: auto|pick|place|both. auto follows --dynamic-task or pick.",
    )
    p.add_argument(
        "--dynamic-task",
        type=str,
        default=None,
        help="Catalog id, e.g. libero_object.t00.place (overrides suite/task/variant).",
    )
    p.add_argument(
        "--dynamic-tasks",
        type=str,
        default="off",
        help=        "Eval set: off|seed|irregular|spatial|react|latency|place_slice|"
        "react_slice|main|extension|reconstructed (60)|ready|ids. "
        "main = object identity tasks (linear+irregular); "
        "spatial/extension = carriers with scene-matched instructions; "
        "latency = 5 object picks; "
        "place_slice = the same five with a moving basket; "
        "react_slice = the same five plus one world-clock turn.",
    )


def rails_variant_from_args(args: argparse.Namespace) -> str:
    return str(getattr(args, "rails_variant", "auto") or "auto")


def dynamic_eval_units(args: argparse.Namespace) -> list:
    """Resolve catalog jobs. Empty list → use legacy --task-ids."""
    from benchmarks.dynamic_libero.env.dynamic_tasks import (
        get_dynamic_task,
        parse_dynamic_tasks,
    )

    single = getattr(args, "dynamic_task", None)
    if single:
        return [get_dynamic_task(single)]
    return parse_dynamic_tasks(getattr(args, "dynamic_tasks", "off"))


def movers_enabled_from_args(args: argparse.Namespace) -> bool:
    from benchmarks.dynamic_libero.env.movers import parse_movers_flag

    return parse_movers_flag(getattr(args, "movers", "off"))


def mover_language_from_args(args: argparse.Namespace) -> str:
    from benchmarks.dynamic_libero.env.movers import parse_mover_language

    return parse_mover_language(getattr(args, "mover_language", "keep"))


def language_mode_from_args(args: argparse.Namespace) -> str:
    from benchmarks.common.language import parse_language_mode

    return parse_language_mode(getattr(args, "language_mode", "keep"))


def target_registry_from_args(args: argparse.Namespace) -> bool:
    """Official default: always use per-task MoverSpec for rails targets."""
    if getattr(args, "no_target_registry", False):
        return False
    return bool(getattr(args, "target_registry", True))


def grasp_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    kick = getattr(args, "release_kick", None)
    return {
        "grasp_mode": getattr(args, "grasp_mode", "ghost"),
        "engage_alpha": float(getattr(args, "engage_alpha", 1.75)),
        "place_freeze": parse_place_freeze(getattr(args, "place_freeze", "both")),
        "release_kick": None if kick is None else bool(kick),
    }


def protocol_tag() -> str:
    return PROTOCOL_VERSION


def traj_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Low-level kwargs for an explicit ``--trajectory`` kind (no complexity)."""
    kind = args.trajectory
    toward = bool(getattr(args, "toward_center", True))
    if kind == "linear":
        if toward:
            return {"axis": args.axis, "toward_center": True}
        return {"axis": args.axis, "direction": args.direction, "toward_center": False}
    if kind == "circle":
        return {"radius": args.radius, "omega": args.omega}
    if kind == "sine":
        kw = {
            "axis": args.axis,
            "amplitude": args.amplitude,
            "wavelength": args.wavelength,
            "toward_center": toward,
        }
        if not toward:
            kw["direction"] = args.direction
        return kw
    if kind == "stop_and_go":
        if toward:
            return {"axis": args.axis, "toward_center": True}
        return {"axis": args.axis, "direction": args.direction, "toward_center": False}
    if kind == "polyline":
        return {
            "waypoints": [[0, 0, 0], [0.08, 0, 0], [0.08, 0.06, 0], [0.0, 0.06, 0]],
        }
    if kind in ("random", "random_polyline"):
        return {
            "n_waypoints": int(getattr(args, "n_waypoints", 5)),
            "extent": float(getattr(args, "extent", 0.12)),
            "seed": int(getattr(args, "seed", 0)),
            "planar": True,
        }
    if kind in ("smooth_turn", "turn"):
        from benchmarks.dynamic_libero.trajectories.smooth_turn import (
            default_smooth_turn_kwargs,
        )

        extra: dict[str, Any] = {}
        if getattr(args, "turn_event_t", None) is not None:
            extra["event_t"] = float(args.turn_event_t)
        if getattr(args, "turn_event_t_lo", None) is not None:
            extra["event_t_lo"] = float(args.turn_event_t_lo)
        if getattr(args, "turn_event_t_hi", None) is not None:
            extra["event_t_hi"] = float(args.turn_event_t_hi)
        if getattr(args, "turn_angle_deg", None) is not None:
            extra["turn_angle_deg"] = float(args.turn_angle_deg)
        if getattr(args, "turn_duration_ticks", None) is not None:
            extra["turn_duration_ticks"] = float(args.turn_duration_ticks)
        if getattr(args, "turn_sign", None) is not None:
            extra["turn_sign"] = int(args.turn_sign)
        kw = default_smooth_turn_kwargs(
            axis=str(getattr(args, "axis", "x")),
            toward_center=toward,
            direction=float(getattr(args, "direction", 1.0)),
            extra=extra,
        )
        # Diagnostic Uniform(20,60) is a fallback. Do not stamp it here, or it
        # overwrites catalog slice windows (30–50) in rollout merge.
        if getattr(args, "turn_event_t_lo", None) is None:
            kw.pop("event_t_lo", None)
        if getattr(args, "turn_event_t_hi", None) is None:
            kw.pop("event_t_hi", None)
        return kw
    return {}


def resolve_trajectory_from_args(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Apply ``--traj-complexity`` (if any) then return ``(kind, kwargs)``."""
    from benchmarks.dynamic_libero.trajectories.complexity import resolve_traj_complexity

    level = getattr(args, "traj_complexity", "none") or "none"
    if level == "none":
        overrides = traj_kwargs_from_args(args)
    else:
        # Optional fine-tuning on top of the preset.
        overrides = {}
        if getattr(args, "n_waypoints", 5) != 5:
            overrides["n_waypoints"] = int(args.n_waypoints)
        if getattr(args, "extent", 0.12) != 0.12:
            overrides["extent"] = float(args.extent)
        if level == "medium":
            if getattr(args, "amplitude", 0.03) != 0.03:
                overrides["amplitude"] = float(args.amplitude)
            if getattr(args, "wavelength", 0.20) != 0.20:
                overrides["wavelength"] = float(args.wavelength)
    kind, kw = resolve_traj_complexity(
        level,
        trajectory=str(args.trajectory),
        seed=int(getattr(args, "seed", 0)),
        overrides=overrides,
    )
    # Stash resolved kind so callers/logging can use it.
    args.trajectory = kind
    return kind, kw


def resolve_task_and_init_ids(task_ids_spec: str, init_ids_spec: str, suite_name: str):
    return parse_task_ids(task_ids_spec, suite_name), parse_init_ids(init_ids_spec)
