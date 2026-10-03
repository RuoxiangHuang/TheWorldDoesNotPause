"""Scene / apparatus / rails / world-clock axes, independent of commanded speed.

Four axes
---------
- ``scene``: ``original`` (official spawn, no layout rewrite) vs ``modified``
  (path clear, seating / lift, optional actor).
- ``apparatus``: ``off`` | ``visible`` | ``hidden``. Hidden keeps the same
  seating, corridor, and support height as visible, but does not inject the
  car / belt mesh.
- ``rails``: ``off`` (free physics) | ``hold`` (constrain at t=0) | ``drive``
  (advance the trajectory). Hold still constrains; original→hold therefore
  includes the full layout cost, not only appearance.
- ``world_clock``: ``pause`` (thinking does not advance the world) vs
  ``realtime`` (inference wait is world time). Independent of speed.

Hold and drive of the same modified scene MUST share initial pose, path
clearing, support height, and release rules. Commanded ``speed`` is the
drive / layout speed; it no longer decides whether to install the apparatus.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Optional

SceneKind = Literal["original", "modified"]
ApparatusKind = Literal["off", "visible", "hidden"]
RailsKind = Literal["off", "hold", "drive"]
WorldClockKind = Literal["pause", "realtime"]
PresentationMode = Literal["demo", "test"]

SCENES: tuple[str, ...] = ("original", "modified")
APPARATUS: tuple[str, ...] = ("off", "visible", "hidden")
RAILS: tuple[str, ...] = ("off", "hold", "drive")
WORLD_CLOCKS: tuple[str, ...] = ("pause", "realtime")

# Corridor / seating reference used when the episode itself does not drive.
DEFAULT_LAYOUT_SPEED = 0.0006

SCENE_PRESETS: tuple[str, ...] = (
    "auto",
    "original",
    "apparatus_hold",
    "apparatus_drive",
    "ghost_hold",
    "ghost_drive",
)


@dataclass(frozen=True)
class SceneCondition:
    scene: SceneKind = "modified"
    apparatus: ApparatusKind = "visible"
    rails: RailsKind = "drive"
    world_clock: WorldClockKind = "realtime"
    layout_speed: float = DEFAULT_LAYOUT_SPEED

    def modifies_layout(self) -> bool:
        return self.scene == "modified"

    def installs_visual_actor(self) -> bool:
        return self.modifies_layout() and self.apparatus == "visible"

    def uses_layout_actor(self) -> bool:
        """Seating / corridor oracles exist even when the mesh is hidden."""
        return self.modifies_layout() and self.apparatus in ("visible", "hidden")

    def constrains_rails(self) -> bool:
        return self.rails in ("hold", "drive")

    def advances_rails(self) -> bool:
        return self.rails == "drive"

    def thinking_advances_world(self) -> bool:
        return self.world_clock == "realtime"

    def apply_n_freeze(self, n_freeze: float) -> float:
        if not self.thinking_advances_world():
            return 0.0
        return float(n_freeze)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_scene(value: str | None, *, default: SceneKind = "modified") -> SceneKind:
    if value is None or str(value).strip() == "":
        return default
    v = str(value).strip().lower()
    if v not in SCENES:
        raise ValueError(f"scene must be one of {SCENES}, got {value!r}")
    return v  # type: ignore[return-value]


def parse_apparatus(value: str | None, *, default: ApparatusKind = "visible") -> ApparatusKind:
    if value is None or str(value).strip() == "":
        return default
    v = str(value).strip().lower()
    aliases = {
        "off": "off",
        "none": "off",
        "visible": "visible",
        "on": "visible",
        "shown": "visible",
        "hidden": "hidden",
        "invisible": "hidden",
        "ghost": "hidden",
    }
    if v not in aliases:
        raise ValueError(f"apparatus must be off|visible|hidden, got {value!r}")
    return aliases[v]  # type: ignore[return-value]


def parse_rails(value: str | None, *, default: RailsKind = "drive") -> RailsKind:
    if value is None or str(value).strip() == "":
        return default
    v = str(value).strip().lower()
    aliases = {
        "off": "off",
        "none": "off",
        "free": "off",
        "hold": "hold",
        "still": "hold",
        "stationary": "hold",
        "static": "hold",
        "drive": "drive",
        "move": "drive",
        "moving": "drive",
    }
    if v not in aliases:
        raise ValueError(f"rails must be off|hold|drive, got {value!r}")
    return aliases[v]  # type: ignore[return-value]


def parse_world_clock(value: str | None, *, default: WorldClockKind = "realtime") -> WorldClockKind:
    if value is None or str(value).strip() == "":
        return default
    v = str(value).strip().lower()
    aliases = {
        "pause": "pause",
        "frozen": "pause",
        "upstream": "pause",
        "realtime": "realtime",
        "real-time": "realtime",
        "live": "realtime",
    }
    if v not in aliases:
        raise ValueError(f"world-clock must be pause|realtime, got {value!r}")
    return aliases[v]  # type: ignore[return-value]


def parse_presentation(value: str | None) -> PresentationMode | None:
    """``demo`` draws the bumper car. ``test`` keeps ghost rails and hides the mesh.

    Aliases: ``show`` / ``visible`` → demo; ``ghost`` / ``ghost_rail`` / ``hidden`` → test.
    ``None`` leaves the preset's apparatus unchanged (catalog ``auto`` stays visible).
    """
    if value is None or str(value).strip() == "":
        return None
    v = str(value).strip().lower().replace("-", "_")
    aliases = {
        "demo": "demo",
        "show": "demo",
        "visible": "demo",
        "test": "test",
        "ghost": "test",
        "ghost_rail": "test",
        "ghost_rails": "test",
        "hidden": "test",
    }
    if v not in aliases:
        raise ValueError(
            "presentation must be demo|test (aliases show, ghost, ghost_rail), "
            f"got {value!r}"
        )
    return aliases[v]  # type: ignore[return-value]


def apply_presentation(
    cond: SceneCondition,
    presentation: str | None,
    *,
    apparatus_explicit: bool = False,
) -> SceneCondition:
    """Force bumper visibility. Original scenes stay apparatus-off.

    Explicit ``--apparatus`` wins, so an ablation can still override the mode.
    ``test`` is ghost rails: same seating, corridor, and constraint, no car mesh.
    ``demo`` installs the visible bumper car.
    """
    mode = parse_presentation(presentation)
    if mode is None or apparatus_explicit or cond.scene != "modified":
        return cond
    apparatus: ApparatusKind = "visible" if mode == "demo" else "hidden"
    if cond.apparatus == apparatus:
        return cond
    return replace(cond, apparatus=apparatus)


def condition_label(preset: str, cond: SceneCondition) -> str:
    """File label from the resolved axes, so demo/test do not overwrite each other."""
    canon = {
        ("original", "off", "off"): "original",
        ("modified", "visible", "hold"): "apparatus_hold",
        ("modified", "visible", "drive"): "apparatus_drive",
        ("modified", "hidden", "hold"): "ghost_hold",
        ("modified", "hidden", "drive"): "ghost_drive",
    }.get((cond.scene, cond.apparatus, cond.rails))
    if preset == "auto":
        return canon or "auto"
    if canon is not None and preset in SCENE_PRESETS and preset != canon:
        return canon
    return preset


def parse_scene_preset(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        return "auto"
    v = str(value).strip().lower()
    if v not in SCENE_PRESETS:
        raise ValueError(f"scene-preset must be one of {SCENE_PRESETS}, got {value!r}")
    return v


def parse_scene_preset_list(spec: str | None) -> list[str]:
    token = str(spec or "auto").strip()
    if not token:
        return ["auto"]
    out: list[str] = []
    for part in token.split(","):
        key = parse_scene_preset(part)
        if key not in out:
            out.append(key)
    return out or ["auto"]


def _layout_speed(speed: float, override: float | None) -> float:
    if override is not None:
        return float(override)
    if abs(float(speed)) > 1e-12:
        return float(speed)
    return float(DEFAULT_LAYOUT_SPEED)


def scene_from_preset(
    preset: str,
    *,
    speed: float = 0.0,
    layout_speed: float | None = None,
) -> SceneCondition:
    key = parse_scene_preset(preset)
    layout = _layout_speed(speed, layout_speed)
    if key == "original":
        return SceneCondition(
            scene="original",
            apparatus="off",
            rails="off",
            world_clock="realtime",
            layout_speed=layout,
        )
    if key == "apparatus_hold":
        return SceneCondition(
            scene="modified",
            apparatus="visible",
            rails="hold",
            world_clock="realtime",
            layout_speed=layout,
        )
    if key == "apparatus_drive":
        return SceneCondition(
            scene="modified",
            apparatus="visible",
            rails="drive",
            world_clock="realtime",
            layout_speed=layout,
        )
    if key == "ghost_hold":
        return SceneCondition(
            scene="modified",
            apparatus="hidden",
            rails="hold",
            world_clock="realtime",
            layout_speed=layout,
        )
    if key == "ghost_drive":
        return SceneCondition(
            scene="modified",
            apparatus="hidden",
            rails="drive",
            world_clock="realtime",
            layout_speed=layout,
        )
    raise ValueError(f"auto preset must go through resolve_scene_condition, got {preset!r}")


def resolve_scene_condition(
    *,
    preset: str | None = "auto",
    scene: str | None = None,
    apparatus: str | None = None,
    rails: str | None = None,
    world_clock: str | None = None,
    speed: float = 0.0,
    layout_speed: float | None = None,
    catalog_job: bool = False,
    flags_explicit: bool = False,
    presentation: str | None = None,
    apparatus_explicit: bool = False,
) -> SceneCondition:
    """Resolve axes. Catalog jobs never use speed to decide installation.

    ``auto`` (no explicit flags), presentation unset:
    - catalog: modified + visible, rails hold iff speed==0 else drive,
      world-clock realtime (thinking still advances the robot / world clock).
    - no catalog: original when speed==0, else modified+visible+drive.

    ``presentation=demo`` forces the bumper mesh on modified scenes.
    ``presentation=test`` forces ghost rails (hidden mesh, same constraint).
    """
    layout = _layout_speed(speed, layout_speed)
    key = parse_scene_preset(preset)
    if key != "auto":
        cond = scene_from_preset(key, speed=speed, layout_speed=layout)
        cond = _overlay_flags(
            cond,
            scene=scene,
            apparatus=apparatus,
            rails=rails,
            world_clock=world_clock,
            layout_speed=layout,
            flags_explicit=flags_explicit,
        )
    elif flags_explicit or any(x is not None for x in (scene, apparatus, rails, world_clock)):
        base_scene = parse_scene(scene, default="modified" if catalog_job else (
            "original" if abs(float(speed)) <= 1e-12 else "modified"
        ))
        if base_scene == "original":
            default_app, default_rails = "off", "off"
        else:
            default_app = "visible"
            default_rails = "hold" if abs(float(speed)) <= 1e-12 else "drive"
        cond = SceneCondition(
            scene=base_scene,
            apparatus=parse_apparatus(apparatus, default=default_app),  # type: ignore[arg-type]
            rails=parse_rails(rails, default=default_rails),  # type: ignore[arg-type]
            world_clock=parse_world_clock(world_clock, default="realtime"),
            layout_speed=layout,
        )
    elif catalog_job:
        cond = SceneCondition(
            scene="modified",
            apparatus="visible",
            rails="hold" if abs(float(speed)) <= 1e-12 else "drive",
            world_clock="realtime",
            layout_speed=layout,
        )
    elif abs(float(speed)) <= 1e-12:
        cond = SceneCondition(
            scene="original",
            apparatus="off",
            rails="off",
            world_clock="realtime",
            layout_speed=layout,
        )
    else:
        cond = SceneCondition(
            scene="modified",
            apparatus="visible",
            rails="drive",
            world_clock="realtime",
            layout_speed=layout,
        )
    return apply_presentation(
        cond,
        presentation,
        apparatus_explicit=apparatus_explicit or apparatus is not None,
    )


def _overlay_flags(
    cond: SceneCondition,
    *,
    scene: str | None,
    apparatus: str | None,
    rails: str | None,
    world_clock: str | None,
    layout_speed: float,
    flags_explicit: bool,
) -> SceneCondition:
    if not flags_explicit and all(x is None for x in (scene, apparatus, rails, world_clock)):
        return cond
    return SceneCondition(
        scene=parse_scene(scene, default=cond.scene),
        apparatus=parse_apparatus(apparatus, default=cond.apparatus),
        rails=parse_rails(rails, default=cond.rails),
        world_clock=parse_world_clock(world_clock, default=cond.world_clock),
        layout_speed=float(layout_speed),
    )


def effective_language_mode(language_mode: str, cond: SceneCondition) -> str:
    """Do not advertise motion when rails are not driving."""
    mode = str(language_mode or "keep").strip().lower()
    if mode == "motion" and not cond.advances_rails():
        return "keep"
    return mode


def add_scene_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--presentation",
        type=str,
        default=None,
        help="demo: visible bumper car. test: ghost rails, mesh hidden "
        "(aliases ghost, ghost_rail). Unset keeps the preset; catalog auto "
        "stays visible. Explicit --apparatus overrides the mode.",
    )
    parser.add_argument(
        "--scene-preset",
        type=str,
        default="auto",
        help="auto, or comma-separated original|apparatus_hold|apparatus_drive|"
        "ghost_hold|ghost_drive. Hold/drive share layout; speed is not used to "
        "decide whether to install the apparatus.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        choices=list(SCENES),
        help="original=official spawn; modified=path-clear / seating / optional actor.",
    )
    parser.add_argument(
        "--apparatus",
        type=str,
        default=None,
        choices=list(APPARATUS),
        help="off|visible|hidden. hidden keeps seating and corridor without the mesh.",
    )
    parser.add_argument(
        "--rails",
        type=str,
        default=None,
        choices=list(RAILS),
        help="off=free; hold=constrain at t=0; drive=advance trajectory.",
    )
    parser.add_argument(
        "--world-clock",
        type=str,
        default=None,
        choices=list(WORLD_CLOCKS),
        help="pause=thinking does not advance the world; realtime=inference wait is world time.",
    )
    parser.add_argument(
        "--layout-speed",
        type=float,
        default=None,
        help="Corridor / seating reference (m/tick). Default: --speed, or "
        f"{DEFAULT_LAYOUT_SPEED} when speed is 0. Hold and drive must share this.",
    )


def scene_kwargs_from_args(
    args: argparse.Namespace,
    *,
    speed: float,
    catalog_job: bool,
    preset: str | None = None,
) -> SceneCondition:
    flags_explicit = any(
        getattr(args, name, None) is not None
        for name in ("scene", "apparatus", "rails", "world_clock")
    )
    return resolve_scene_condition(
        preset=preset if preset is not None else getattr(args, "scene_preset", "auto"),
        scene=getattr(args, "scene", None),
        apparatus=getattr(args, "apparatus", None),
        rails=getattr(args, "rails", None),
        world_clock=getattr(args, "world_clock", None),
        speed=float(speed),
        layout_speed=getattr(args, "layout_speed", None),
        catalog_job=bool(catalog_job),
        flags_explicit=flags_explicit,
        presentation=getattr(args, "presentation", None),
        apparatus_explicit=getattr(args, "apparatus", None) is not None,
    )


def iter_scene_conditions(
    args: argparse.Namespace,
    *,
    speed: float,
    catalog_job: bool,
) -> list[tuple[str, SceneCondition]]:
    presets = parse_scene_preset_list(getattr(args, "scene_preset", "auto"))
    out: list[tuple[str, SceneCondition]] = []
    for preset in presets:
        cond = scene_kwargs_from_args(
            args, speed=speed, catalog_job=catalog_job, preset=preset
        )
        out.append((condition_label(preset, cond), cond))
    return out


def trajectory_speed_for_condition(cond: SceneCondition, commanded: float) -> float:
    """Keep the layout corridor identical for hold vs drive.

    Hold does not step the trajectory, but ``Trajectory.speed`` still defines
    path-clear horizon and seating heading. Never zero it for hold.
    """
    if cond.scene == "original" and not cond.constrains_rails():
        return 0.0
    if cond.advances_rails() and abs(float(commanded)) > 1e-12:
        return float(commanded)
    return float(cond.layout_speed)


__all__ = [
    "DEFAULT_LAYOUT_SPEED",
    "SCENE_PRESETS",
    "SceneCondition",
    "add_scene_cli_args",
    "apply_presentation",
    "condition_label",
    "effective_language_mode",
    "parse_presentation",
    "iter_scene_conditions",
    "parse_scene_preset_list",
    "resolve_scene_condition",
    "scene_from_preset",
    "scene_kwargs_from_args",
    "trajectory_speed_for_condition",
]
