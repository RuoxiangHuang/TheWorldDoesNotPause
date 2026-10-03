"""v1 whitelist: RoboTwin tasks with dynamic rigid targets + bimanual metadata."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# task_name -> attribute on the Base_Task instance holding the movable Actor
TARGET_ATTR: Dict[str, str] = {
    "place_empty_cup": "cup",
    "grab_roller": "roller",
    "move_playingcard_away": "playingcards",
    "move_pillbottle_pad": "pillbottle",
    "place_phone_stand": "phone",
    "place_a2b_left": "object",
    # True arm-to-arm handoff tasks (RoboTwin native).
    "handover_block": "box",
    "handover_mic": "microphone",
}

# Optional receptacle for pick|place|both reconstructions (not a mesh swap).
PLACE_ATTR: Dict[str, str] = {
    "place_empty_cup": "coaster",
    "place_phone_stand": "stand",
    "move_pillbottle_pad": "pad",
    "place_a2b_left": "target_object",
    "place_a2b_right": "target_object",
    "place_object_basket": "basket",
    "place_container_plate": "plate",
    "place_can_basket": "basket",
    "place_object_stand": "displaystand",
    "place_object_scale": "scale",
    "place_bread_skillet": "skillet",
    "move_stapler_pad": "pad",
    "move_can_pot": "pot",
    "place_mouse_pad": "target",
    "place_fan": "pad",
    "handover_block": "target_box",
}

# Reconstruction-only pick actors. Not on the official v1 whitelist (`all`).
RECON_TARGET_ATTR: Dict[str, str] = {
    "place_container_plate": "container",
    "place_object_basket": "object",
    "place_can_basket": "can",
    "place_object_stand": "object",
    "place_object_scale": "object",
    "place_bread_skillet": "bread",
    "move_stapler_pad": "stapler",
    "place_a2b_right": "object",
    "move_can_pot": "can",
    "place_mouse_pad": "mouse",
    "place_fan": "fan",
}

DEFAULT_INSTRUCTIONS: Dict[str, str] = {
    "place_empty_cup": "Pick up the cup and place it onto the coaster.",
    "grab_roller": "Grab the roller with both arms and lift it up.",
    "move_playingcard_away": "Move the playing card away from the table center.",
    "move_pillbottle_pad": "Move the pill bottle onto the pad.",
    "place_phone_stand": "Pick up the phone and place it onto the stand.",
    "place_a2b_left": "Pick up the object and place it to the left of the target.",
    "handover_block": "Hand the block from one arm to the other and place it on the target.",
    "handover_mic": "Hand the microphone from one arm to the other and lift it.",
    "place_a2b_right": "Pick up the object and place it to the right of the target.",
    "place_container_plate": "Pick up the container and place it onto the plate.",
    "place_object_basket": "Pick up the object and place it into the basket.",
    "place_can_basket": "Pick up the can and place it into the basket.",
    "place_object_stand": "Pick up the object and place it onto the stand.",
    "place_object_scale": "Pick up the object and place it onto the scale.",
    "place_bread_skillet": "Pick up the bread and place it onto the skillet.",
    "move_stapler_pad": "Move the stapler onto the pad.",
    "move_can_pot": "Pick up the can and place it beside the pot.",
    "place_mouse_pad": "Pick up the mouse and place it onto the pad.",
    "place_fan": "Pick up the fan and place it onto the pad.",
}

# Short noun for B-track motion hints (must match what the policy sees).
INSTRUCTION_NOUNS: Dict[str, str] = {
    "place_empty_cup": "cup",
    "grab_roller": "roller",
    "move_playingcard_away": "playing card",
    "move_pillbottle_pad": "pill bottle",
    "place_phone_stand": "phone",
    "place_a2b_left": "object",
    "handover_block": "block",
    "handover_mic": "microphone",
}

# Per-task trajectory / release defaults. Axes prefer in-plane motion that
# keeps the target reachable; ``toward_center`` flips direction after reset so
# objects spawn near table edges drift inward rather than off the rim.
TASK_TRAJ_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "place_empty_cup": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "grab_roller": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.14,
    },
    "move_playingcard_away": {
        "axis": "x",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "move_pillbottle_pad": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "place_phone_stand": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "place_a2b_left": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "place_a2b_right": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "place_container_plate": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "place_object_basket": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "place_can_basket": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "place_object_stand": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "place_object_scale": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "place_bread_skillet": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "move_stapler_pad": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "move_can_pot": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "place_mouse_pad": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "place_fan": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.12,
    },
    "handover_block": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.14,
    },
    "handover_mic": {
        "axis": "y",
        "toward_center": True,
        "release_radius": 0.14,
    },
}

# Default when task has no override. TCP proximity (not move-group EE) needs
# ~0.12 m: successful static grasps reach ~0.096 m TCP distance.
DEFAULT_RELEASE_RADIUS = 0.12

# Task-level metadata for Dynamic-RoboTwin coordination stresses.
# dynamic_modes lists modes that are *recommended* for the task; CLI may enable
# additional modes on any whitelist task.
TASK_META: Dict[str, Dict[str, Any]] = {
    "place_empty_cup": {
        "bimanual": False,
        "roles": ("pick",),
        "dynamic_modes": ("contact_trigger", "occlusion"),
        "stress": "tracking",
    },
    "grab_roller": {
        "bimanual": True,
        "roles": ("left_grasp", "right_grasp"),
        "dynamic_modes": ("contact_trigger", "occlusion", "contact_chain"),
        "stress": "dual_grasp",
    },
    "move_playingcard_away": {
        "bimanual": False,
        "roles": ("pick",),
        "dynamic_modes": ("contact_trigger", "occlusion"),
        "stress": "tracking",
    },
    "move_pillbottle_pad": {
        "bimanual": False,
        "roles": ("pick",),
        "dynamic_modes": ("contact_trigger", "occlusion"),
        "stress": "tracking",
    },
    "place_phone_stand": {
        "bimanual": False,
        "roles": ("pick",),
        "dynamic_modes": ("contact_trigger", "occlusion"),
        "stress": "tracking",
    },
    "place_a2b_left": {
        "bimanual": False,
        "roles": ("pick",),
        "dynamic_modes": ("contact_trigger", "occlusion"),
        "stress": "tracking",
    },
    "handover_block": {
        "bimanual": True,
        "roles": ("grasp_arm", "place_arm"),
        "dynamic_modes": ("handoff", "contact_trigger", "occlusion", "contact_chain"),
        "stress": "handoff",
        "secondary_attr": "target_box",
    },
    "handover_mic": {
        "bimanual": True,
        "roles": ("grasp_arm", "handover_arm"),
        "dynamic_modes": ("handoff", "contact_trigger", "occlusion"),
        "stress": "handoff",
    },
}

V1_TASKS = tuple(TARGET_ATTR.keys())
HANDOFF_TASKS: Tuple[str, ...] = tuple(
    t for t, m in TASK_META.items() if m.get("stress") == "handoff"
)
BIMANUAL_TASKS: Tuple[str, ...] = tuple(
    t for t, m in TASK_META.items() if m.get("bimanual")
)

# Default Dynamic-RoboTwin mode mix (paper-aligned).
DEFAULT_DYNAMIC_MODES: Tuple[str, ...] = (
    "contact_trigger",
    "occlusion",
    "contact_chain",
)

CONTACT_SWITCH_LEVELS = ("off", "mild", "medium", "strong")


def resolve_target_attr(task_name: str) -> str:
    if task_name in TARGET_ATTR:
        return TARGET_ATTR[task_name]
    if task_name in RECON_TARGET_ATTR:
        return RECON_TARGET_ATTR[task_name]
    from .extra_tasks import EXTRA_TARGET_ATTR

    if task_name in EXTRA_TARGET_ATTR:
        return EXTRA_TARGET_ATTR[task_name]
    raise KeyError(
        f"Task {task_name!r} is not in the Dynamic-RoboTwin v1 whitelist, "
        f"reconstruction set, or extra catalog. Supported: "
        f"{sorted(list(TARGET_ATTR) + list(RECON_TARGET_ATTR) + list(EXTRA_TARGET_ATTR))}"
    )


def resolve_place_attr(task_name: str) -> Optional[str]:
    """Receptacle actor name, or None if this skill has no place rails."""
    return PLACE_ATTR.get(task_name)


def default_instruction(task_name: str) -> str:
    if task_name in DEFAULT_INSTRUCTIONS:
        return DEFAULT_INSTRUCTIONS[task_name]
    from .extra_tasks import extra_instruction

    extra = extra_instruction(task_name)
    if extra:
        return extra
    return task_name.replace("_", " ")


def instruction_noun(task_name: str) -> str:
    """Pick-target noun for B-track motion language."""
    if task_name in INSTRUCTION_NOUNS:
        return INSTRUCTION_NOUNS[task_name]
    attr = TARGET_ATTR.get(task_name) or RECON_TARGET_ATTR.get(task_name)
    if attr:
        return attr.replace("_", " ")
    return "target"


def default_release_radius(task_name: str) -> float:
    cfg = TASK_TRAJ_DEFAULTS.get(task_name, {})
    return float(cfg.get("release_radius", DEFAULT_RELEASE_RADIUS))


def resolve_traj_kwargs(task_name: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Merge task defaults with caller overrides (caller wins)."""
    base = dict(TASK_TRAJ_DEFAULTS.get(task_name, {}))
    # release_radius is not a trajectory ctor kw; strip it here.
    base.pop("release_radius", None)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                base[k] = v
    return base


def get_task_meta(task_name: str) -> Dict[str, Any]:
    """Return a copy of task metadata (with safe defaults for unknown keys)."""
    resolve_target_attr(task_name)  # whitelist, reconstruction, or extra
    meta = dict(TASK_META.get(task_name, {}))
    from .extra_tasks import is_extra, get_extra

    if is_extra(task_name):
        extra = get_extra(task_name)
        meta.setdefault("bimanual", False)
        meta.setdefault("roles", ("pick",))
        meta.setdefault("dynamic_modes", ())
        meta.setdefault("stress", extra.stress)
        meta.setdefault("secondary_attr", None)
        meta.setdefault("extra", True)
        return meta
    meta.setdefault("bimanual", False)
    meta.setdefault("roles", ("pick",))
    meta.setdefault("dynamic_modes", ("contact_trigger", "occlusion"))
    meta.setdefault("stress", "tracking")
    meta.setdefault("secondary_attr", None)
    return meta


def is_bimanual(task_name: str) -> bool:
    return bool(get_task_meta(task_name).get("bimanual"))


def parse_dynamic_modes(
    value: Optional[str | Sequence[str]],
    *,
    default: Sequence[str] = DEFAULT_DYNAMIC_MODES,
) -> List[str]:
    """Parse CLI / config dynamic-mode list.

    Accepts comma-separated string, sequence, ``\"none\"`` / ``\"off\"``, or
    ``None`` (→ default).
    """
    if value is None:
        return list(default)
    if isinstance(value, str):
        raw = [x.strip() for x in value.split(",") if x.strip()]
    else:
        raw = [str(x).strip() for x in value if str(x).strip()]
    if not raw or raw == ["none"] or raw == ["off"]:
        return []
    if "all" in raw:
        return list(DEFAULT_DYNAMIC_MODES) + ["handoff"]
    allowed = {
        "contact_trigger",
        "occlusion",
        "handoff",
        "contact_chain",
        "rails_release_velocity",  # always-on legacy; ignored as a mode flag
    }
    out: List[str] = []
    for m in raw:
        if m not in allowed:
            raise ValueError(
                f"Unknown dynamic mode {m!r}; expected one of {sorted(allowed)} or none/all"
            )
        if m == "rails_release_velocity":
            continue
        if m not in out:
            out.append(m)
    return out
