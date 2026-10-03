"""Default workspace AABBs for escape detection (Real-Time v1)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

# Metres in world frame: [[xmin, ymin, zmin], [xmax, ymax, zmax]]
# Slightly larger than the nominal table so edge drift triggers escape before
# objects fall off the sim entirely.

# LIBERO arenas do not share a table height. Floor (libero_object) free joints
# sit near z≈0.01; tabletop / kitchen sit near z=0.90; living-room ≈0.41;
# study ≈0.87. A single z∈[-0.02, 0.40] box marks every kitchen episode
# escaped=True on the first rails tick (1 mm of drift, n_replans=1).
_FLOOR_AABB = np.array([[-0.45, -0.45, -0.02], [0.45, 0.45, 0.40]], dtype=np.float64)
_TABLE_AABB = np.array([[-0.55, -0.65, 0.70], [0.55, 0.65, 1.35]], dtype=np.float64)
_LIVING_AABB = np.array([[-0.50, -0.90, 0.25], [0.50, 0.90, 0.75]], dtype=np.float64)
_STUDY_AABB = np.array([[-0.80, -0.65, 0.70], [0.50, 0.65, 1.30]], dtype=np.float64)
# Mixed suites (libero_10 / libero_90) when arena is unknown: never false-escape
# a valid spawn, still catch objects that fly far off the furniture.
_UNION_AABB = np.array([[-0.90, -0.90, -0.05], [0.90, 0.90, 1.40]], dtype=np.float64)

LIBERO_ARENA_AABB: dict[str, np.ndarray] = {
    "floor": _FLOOR_AABB,
    "table": _TABLE_AABB,
    "kitchen": _TABLE_AABB,
    "living_room": _LIVING_AABB,
    "coffee_table": _LIVING_AABB,
    "study": _STUDY_AABB,
    "union": _UNION_AABB,
}

LIBERO_SUITE_DEFAULT_ARENA: dict[str, str] = {
    "libero_object": "floor",
    "libero_spatial": "table",
    "libero_goal": "table",
    "libero_10": "union",
    "libero_90": "union",
}

# Official libero_10 order (libero_suite_task_map). Kitchen/study must not use
# the living-room z band.
LIBERO_10_TASK_ARENA: dict[int, str] = {
    0: "living_room",
    1: "living_room",
    2: "kitchen",
    3: "kitchen",
    4: "living_room",
    5: "study",
    6: "living_room",
    7: "living_room",
    8: "kitchen",
    9: "kitchen",
}

# Suite-level convenience (homogeneous suites only). libero_10 is mixed;
# resolve_workspace_aabb prefers task_id / arena when provided.
LIBERO_SUITE_AABB: dict[str, np.ndarray] = {
    "libero_object": _FLOOR_AABB,
    "libero_spatial": _TABLE_AABB,
    "libero_goal": _TABLE_AABB,
    "libero_10": _UNION_AABB,
    "libero_90": _UNION_AABB,
}

LIBERO_DEFAULT_AABB = LIBERO_SUITE_AABB["libero_object"]

ROBOTWIN_TASK_AABB: dict[str, np.ndarray] = {
    "place_empty_cup": np.array([[-0.45, -0.38, 0.72], [0.45, 0.38, 1.08]], dtype=np.float64),
    "grab_roller": np.array([[-0.50, -0.40, 0.70], [0.50, 0.40, 1.10]], dtype=np.float64),
    "move_playingcard_away": np.array([[-0.45, -0.38, 0.72], [0.45, 0.38, 1.05]], dtype=np.float64),
    "move_pillbottle_pad": np.array([[-0.45, -0.38, 0.72], [0.45, 0.38, 1.08]], dtype=np.float64),
    "place_phone_stand": np.array([[-0.45, -0.38, 0.72], [0.45, 0.38, 1.08]], dtype=np.float64),
    "place_a2b_left": np.array([[-0.50, -0.40, 0.72], [0.50, 0.40, 1.08]], dtype=np.float64),
    "handover_block": np.array([[-0.50, -0.40, 0.72], [0.50, 0.40, 1.15]], dtype=np.float64),
    "handover_mic": np.array([[-0.50, -0.40, 0.70], [0.50, 0.40, 1.20]], dtype=np.float64),
}

ROBOTWIN_DEFAULT_AABB = ROBOTWIN_TASK_AABB["place_empty_cup"]

# RoboCasa kitchen scenes use world-frame fixture layouts whose free-joint
# pick targets commonly spawn near |x|,|y| ≈ 1.5–5 m (seed0 CounterToCabinet
# obj ≈ [1.54, -0.41, 0.95]). The previous ±1.4 table-scale box marked almost
# every episode escaped=True on the first rails tick. Match the demo bounds in
# common/viz/record_object_motion.py (_ROBOCASA_AABB).
_ROBOCASA_KITCHEN_AABB = np.array([[-6.0, -6.0, -0.5], [6.0, 6.0, 2.5]], dtype=np.float64)

ROBOCASA_TASK_AABB: dict[str, np.ndarray] = {
    "PickPlaceCounterToCabinet": _ROBOCASA_KITCHEN_AABB.copy(),
    "PickPlaceCabinetToCounter": _ROBOCASA_KITCHEN_AABB.copy(),
    "PickPlaceCounterToSink": _ROBOCASA_KITCHEN_AABB.copy(),
    "PickPlaceSinkToCounter": _ROBOCASA_KITCHEN_AABB.copy(),
    "PickPlaceCounterToMicrowave": _ROBOCASA_KITCHEN_AABB.copy(),
    "PickPlaceMicrowaveToCounter": _ROBOCASA_KITCHEN_AABB.copy(),
}

ROBOCASA_DEFAULT_AABB = ROBOCASA_TASK_AABB["PickPlaceCounterToCabinet"]


def parse_aabb(values: Sequence[float] | str) -> np.ndarray:
    """Parse ``xmin,ymin,zmin,xmax,ymax,zmax`` or a length-6 sequence."""
    if isinstance(values, str):
        parts = [float(x.strip()) for x in values.split(",") if x.strip()]
    else:
        parts = [float(x) for x in values]
    if len(parts) != 6:
        raise ValueError(f"workspace AABB needs 6 numbers, got {len(parts)}")
    lo = np.asarray(parts[:3], dtype=np.float64)
    hi = np.asarray(parts[3:], dtype=np.float64)
    return np.stack([lo, hi])


def expand_aabb_to_include(
    aabb: np.ndarray,
    xyz: Sequence[float],
    *,
    margin: float = 0.08,
) -> np.ndarray:
    """Grow ``aabb`` so ``xyz`` lies strictly inside (spawn-in-box safety)."""
    box = np.asarray(aabb, dtype=np.float64).copy()
    p = np.asarray(xyz, dtype=np.float64).reshape(3)
    pad = float(margin)
    box[0] = np.minimum(box[0], p - pad)
    box[1] = np.maximum(box[1], p + pad)
    return box


def arena_from_task_name(task_name: Optional[str]) -> Optional[str]:
    if not task_name:
        return None
    u = str(task_name).upper()
    if u.startswith("KITCHEN"):
        return "kitchen"
    if u.startswith("LIVING_ROOM"):
        return "living_room"
    if u.startswith("STUDY"):
        return "study"
    return None


def infer_libero_arena(env: Any) -> Optional[str]:
    """Read ``_arena_type`` off a LIBERO / robosuite env wrapper chain."""
    base = env
    for _ in range(6):
        t = getattr(base, "_arena_type", None)
        if t:
            key = str(t).strip().lower()
            if key in LIBERO_ARENA_AABB:
                return key
        nxt = getattr(base, "env", None)
        if nxt is None or nxt is base:
            break
        base = nxt
    return None


def resolve_libero_arena(
    *,
    suite_name: Optional[str] = None,
    task_id: Optional[int] = None,
    task_name: Optional[str] = None,
    arena: Optional[str] = None,
) -> str:
    if arena:
        key = str(arena).strip().lower()
        if key in LIBERO_ARENA_AABB:
            return key
    named = arena_from_task_name(task_name)
    if named:
        return named
    suite = (suite_name or "").strip()
    if suite in ("libero_10", "libero_long") and task_id is not None:
        mapped = LIBERO_10_TASK_ARENA.get(int(task_id))
        if mapped:
            return mapped
    if suite in LIBERO_SUITE_DEFAULT_ARENA:
        return LIBERO_SUITE_DEFAULT_ARENA[suite]
    return "floor"


def resolve_workspace_aabb(
    benchmark: str,
    *,
    suite_name: Optional[str] = None,
    task_name: Optional[str] = None,
    task_id: Optional[int] = None,
    arena: Optional[str] = None,
    override: Optional[np.ndarray | Sequence[float] | str] = None,
    escape_check: bool = True,
) -> Optional[np.ndarray]:
    """Return workspace AABB or ``None`` when escape checking is disabled."""
    if not escape_check:
        return None
    if override is not None:
        if isinstance(override, np.ndarray):
            return np.asarray(override, dtype=np.float64)
        return parse_aabb(override)

    benchmark = benchmark.lower()
    if benchmark == "libero":
        key = resolve_libero_arena(
            suite_name=suite_name,
            task_id=task_id,
            task_name=task_name,
            arena=arena,
        )
        if key in LIBERO_ARENA_AABB:
            return LIBERO_ARENA_AABB[key].copy()
        if suite_name and suite_name in LIBERO_SUITE_AABB:
            return LIBERO_SUITE_AABB[suite_name].copy()
        return LIBERO_DEFAULT_AABB.copy()
    if benchmark == "robotwin":
        if task_name and task_name in ROBOTWIN_TASK_AABB:
            return ROBOTWIN_TASK_AABB[task_name].copy()
        return ROBOTWIN_DEFAULT_AABB.copy()
    if benchmark == "robocasa":
        if task_name and task_name in ROBOCASA_TASK_AABB:
            return ROBOCASA_TASK_AABB[task_name].copy()
        return ROBOCASA_DEFAULT_AABB.copy()
    raise ValueError(f"unknown benchmark {benchmark!r}")


def aabb_to_dict(aabb: Optional[np.ndarray]) -> Optional[dict[str, Any]]:
    if aabb is None:
        return None
    aabb = np.asarray(aabb, dtype=np.float64)
    return {"lo": aabb[0].tolist(), "hi": aabb[1].tolist()}
