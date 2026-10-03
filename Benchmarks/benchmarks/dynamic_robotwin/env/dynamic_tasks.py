"""Dynamic-RoboTwin reconstruction catalog.

Official SR ``all`` is still the ghost-rails whitelist in ``task_registry``.
This module is the *visual* reconstruction suite: original RoboTwin meshes,
unchanged instruction / ``check_success``, plus a bumper car that rides the
rails behind whoever is moving.

One reconstruction batch: every catalogued skill uses the same 5-id pattern
(pick / place / both linear + pick/place irregular). CLI aliases ``recon``,
``seed``, and ``wave2`` all resolve to this catalog. Ready / pick_only / skip
in ``SKILL_FIT`` rank original skills that are not yet catalogued.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional, Sequence

RailsVariant = Literal["pick", "place", "both"]
SkillStatus = Literal["ready", "pick_only", "skip"]

PLACE_EMPTY_CUP_LANG = "Pick up the cup and place it onto the coaster."


@dataclass(frozen=True)
class DynamicRoboTwinTask:
    id: str
    task_name: str
    variant: RailsVariant
    language: str
    pick_attr: str
    place_attr: Optional[str]
    trajectory_kind: str = "linear"
    curve_id: Optional[str] = None
    spawn_pushers: bool = True
    note: str = ""
    heading_deg: Optional[float] = None
    eval_split: str = "main"  # main | extension
    language_policy: str = "keep"
    eligibility: str = "identity"

    def traj_spec(self) -> tuple[str, dict]:
        if self.trajectory_kind in ("curve", "smooth", "spline") and self.curve_id:
            return "curve", {"curve_id": self.curve_id}
        kw: dict = {}
        if self.heading_deg is not None:
            kw["heading_deg"] = float(self.heading_deg)
            kw["toward_center"] = True
        return str(self.trajectory_kind or "linear"), kw

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvalSpec:
    """Resolved eval unit: catalog id or official whitelist name."""

    spec: str
    task_name: str
    rails_variant: RailsVariant
    spawn_pushers: bool
    place_attr: Optional[str]
    language: str
    trajectory_kind: str = "linear"
    curve_id: Optional[str] = None
    heading_deg: Optional[float] = None


@dataclass(frozen=True)
class SkillFit:
    """Whether an original RoboTwin skill should get the cup-style pusher treatment."""

    task_name: str
    status: SkillStatus
    pick_attr: Optional[str]
    place_attr: Optional[str]
    variants: tuple[str, ...]
    why: str


@dataclass(frozen=True)
class ReconSkill:
    task_name: str
    pick_attr: str
    place_attr: str
    language: str
    pick_noun: str
    place_noun: str


# One reconstruction batch. Each skill expands to 5 catalog ids.
# Official instruction / check_success / original meshes unchanged.
# Linear headings fan around the table (0=+x, 90=+y). Official whitelist
# names still use TASK_TRAJ_DEFAULTS (mostly +y / toward_center).
RECON_LINEAR_HEADINGS_DEG: tuple[float, ...] = (
    0.0,    # place_empty_cup
    24.0,   # place_phone_stand
    48.0,   # place_container_plate
    72.0,   # place_object_basket
    96.0,   # place_can_basket
    120.0,  # place_object_stand
    144.0,  # place_object_scale
    168.0,  # place_bread_skillet
    192.0,  # move_pillbottle_pad
    216.0,  # move_stapler_pad
    240.0,  # place_a2b_left
    264.0,  # place_a2b_right
    288.0,  # move_can_pot
    312.0,  # place_mouse_pad
    336.0,  # place_fan
)

RECON_SKILLS: tuple[ReconSkill, ...] = (
    ReconSkill("place_empty_cup", "cup", "coaster", PLACE_EMPTY_CUP_LANG, "CUP", "COASTER"),
    ReconSkill("place_phone_stand", "phone", "stand", "Pick up the phone and place it onto the stand.", "PHONE", "STAND"),
    ReconSkill("place_container_plate", "container", "plate", "Pick up the container and place it onto the plate.", "CONTAINER", "PLATE"),
    ReconSkill("place_object_basket", "object", "basket", "Pick up the object and place it into the basket.", "OBJECT", "BASKET"),
    ReconSkill("place_can_basket", "can", "basket", "Pick up the can and place it into the basket.", "CAN", "BASKET"),
    ReconSkill("place_object_stand", "object", "displaystand", "Pick up the object and place it onto the stand.", "OBJECT", "STAND"),
    ReconSkill("place_object_scale", "object", "scale", "Pick up the object and place it onto the scale.", "OBJECT", "SCALE"),
    ReconSkill("place_bread_skillet", "bread", "skillet", "Pick up the bread and place it onto the skillet.", "BREAD", "SKILLET"),
    ReconSkill("move_pillbottle_pad", "pillbottle", "pad", "Move the pill bottle onto the pad.", "BOTTLE", "PAD"),
    ReconSkill("move_stapler_pad", "stapler", "pad", "Move the stapler onto the pad.", "STAPLER", "PAD"),
    ReconSkill("place_a2b_left", "object", "target_object", "Pick up the object and place it to the left of the target.", "OBJECT A", "OBJECT B"),
    ReconSkill("place_a2b_right", "object", "target_object", "Pick up the object and place it to the right of the target.", "OBJECT A", "OBJECT B"),
    ReconSkill("move_can_pot", "can", "pot", "Pick up the can and place it beside the pot.", "CAN", "POT"),
    ReconSkill("place_mouse_pad", "mouse", "target", "Pick up the mouse and place it onto the pad.", "MOUSE", "PAD"),
    ReconSkill("place_fan", "fan", "pad", "Pick up the fan and place it onto the pad.", "FAN", "PAD"),
)


def _recon_heading(skill: ReconSkill) -> float:
    try:
        idx = next(i for i, s in enumerate(RECON_SKILLS) if s.task_name == skill.task_name)
    except StopIteration:
        idx = 0
    return float(RECON_LINEAR_HEADINGS_DEG[idx % len(RECON_LINEAR_HEADINGS_DEG)])


def _skill_variant(
    skill: ReconSkill,
    variant: RailsVariant,
    *,
    kind: str = "linear",
    curve_id: str | None = None,
) -> DynamicRoboTwinTask:
    heading = _recon_heading(skill)
    if kind in ("curve", "irregular"):
        tid = f"{skill.task_name}.{variant}.irregular"
        kind = "curve"
        curve_id = curve_id or "s_wave"
        who = skill.pick_noun if variant == "pick" else skill.place_noun
        note = f"{who} on smooth curve ({curve_id}). Same check_success."
        heading_deg = None
    else:
        tid = f"{skill.task_name}.{variant}"
        note = (
            f"Same check_success. Dynamics: who the car pushes ({variant}). "
            f"Linear heading {heading:.0f}° (0=+x, 90=+y), toward_center."
        )
        heading_deg = heading
    eval_split = "extension" if kind == "curve" else "main"
    return DynamicRoboTwinTask(
        id=tid,
        task_name=skill.task_name,
        variant=variant,
        language=skill.language,
        pick_attr=skill.pick_attr,
        place_attr=skill.place_attr,
        trajectory_kind=kind,
        curve_id=curve_id,
        spawn_pushers=True,
        note=note,
        heading_deg=heading_deg,
        eval_split=eval_split,
        language_policy="keep",
        eligibility="identity",
    )


def _expand_skill(skill: ReconSkill) -> tuple[DynamicRoboTwinTask, ...]:
    return (
        _skill_variant(skill, "pick"),
        _skill_variant(skill, "place"),
        _skill_variant(skill, "both"),
        _skill_variant(skill, "pick", kind="irregular", curve_id="s_wave"),
        _skill_variant(skill, "place", kind="irregular", curve_id="s_wave"),
    )


CATALOG: tuple[DynamicRoboTwinTask, ...] = tuple(
    t for skill in RECON_SKILLS for t in _expand_skill(skill)
)
_BY_ID: dict[str, DynamicRoboTwinTask] = {t.id: t for t in CATALOG}


def get_dynamic_task(task_id: str) -> DynamicRoboTwinTask:
    key = str(task_id).strip()
    if key not in _BY_ID:
        raise KeyError(
            f"Unknown Dynamic-RoboTwin reconstruction {task_id!r}. "
            f"Unified batch: {len(CATALOG)} ids "
            f"({len(RECON_SKILLS)} skills × pick/place/both + irregular); "
            f"aliases: recon, seed, wave2"
        )
    return _BY_ID[key]


def list_dynamic_tasks() -> list[DynamicRoboTwinTask]:
    return list(CATALOG)


def recon_tasks() -> list[DynamicRoboTwinTask]:
    """Full Dynamic-RoboTwin reconstruction batch (one suite, no seed/wave split)."""
    return list(CATALOG)


def main_tasks() -> list[DynamicRoboTwinTask]:
    """Identity language + native check_success: linear pick/place/both.

    Irregular curves keep the same language and are the extension track
    (extra path difficulty, not a locative rewrite). Not filtered by ΔSR.
    """
    return [t for t in CATALOG if t.eval_split == "main"]


def extension_tasks() -> list[DynamicRoboTwinTask]:
    """Irregular-path reconstructions (same language / success as linear)."""
    return [t for t in CATALOG if t.eval_split == "extension"]


def cup_seed_tasks() -> list[DynamicRoboTwinTask]:
    """Backward-compat alias for the unified reconstruction batch."""
    return recon_tasks()


def wave2_tasks() -> list[DynamicRoboTwinTask]:
    """Backward-compat alias for the unified reconstruction batch."""
    return recon_tasks()


def recon_demo_tasks() -> list[DynamicRoboTwinTask]:
    """Compact demo set: both linear + pick/place irregular, every catalogued skill."""
    out: list[DynamicRoboTwinTask] = []
    for t in CATALOG:
        if t.variant == "both" and t.trajectory_kind == "linear":
            out.append(t)
        elif t.trajectory_kind == "curve":
            out.append(t)
    return out


def recon_both_tasks() -> list[DynamicRoboTwinTask]:
    """Both-linear only (standoff / contact iteration)."""
    return [
        t for t in CATALOG if t.variant == "both" and t.trajectory_kind == "linear"
    ]


# Diagnostic slice of identity skills (not ΔSR-filtered; not the main table).
# Main table is ``main`` = all linear pick/place/both reconstructions.
LATENCY_SLICE_SKILLS: tuple[str, ...] = (
    "place_empty_cup",
    "place_can_basket",
    "move_can_pot",
    "place_container_plate",
    "place_object_stand",
)


def latency_slice_tasks(*, variant: RailsVariant = "pick") -> list[DynamicRoboTwinTask]:
    """Same five skills as a FastWAM LIBERO-style latency slice.

    ``pick`` = moving grasp target (LIBERO FastWAM analog).
    ``place`` = moving receptacle (π LIBERO analog; test whether FastWAM
    also converts latency here).
    """
    wanted = {name: i for i, name in enumerate(LATENCY_SLICE_SKILLS)}
    out = [
        t
        for t in CATALOG
        if t.task_name in wanted
        and t.variant == variant
        and t.trajectory_kind == "linear"
    ]
    out.sort(key=lambda t: wanted[t.task_name])
    return out


def place_slice_tasks() -> list[DynamicRoboTwinTask]:
    return latency_slice_tasks(variant="place")


def resolve_eval_spec(spec: str) -> EvalSpec:
    """Map a CLI token to env task + rails + whether to spawn bumper cars.

    Dotted catalog ids (``place_empty_cup.pick``) spawn cars.
    Bare whitelist names (``place_empty_cup``) stay ghost rails.
    """
    key = str(spec).strip()
    if key in _BY_ID:
        task = _BY_ID[key]
        return EvalSpec(
            spec=task.id,
            task_name=task.task_name,
            rails_variant=task.variant,
            spawn_pushers=True,
            place_attr=task.place_attr,
            language=task.language,
            trajectory_kind=task.trajectory_kind,
            curve_id=task.curve_id,
            heading_deg=task.heading_deg,
        )
    from .extra_tasks import extra_instruction, get_extra, is_extra
    from .task_registry import default_instruction, resolve_place_attr, resolve_target_attr

    if is_extra(key):
        extra = get_extra(key)
        return EvalSpec(
            spec=extra.id,
            task_name=extra.task_name,
            rails_variant="pick",
            spawn_pushers=False,
            place_attr=None,
            language=extra.language or extra_instruction(extra.task_name) or extra.task_name,
            trajectory_kind="native",
        )

    resolve_target_attr(key)  # raises KeyError if not on the official whitelist
    return EvalSpec(
        spec=key,
        task_name=key,
        rails_variant="pick",
        spawn_pushers=False,
        place_attr=resolve_place_attr(key),
        language=default_instruction(key),
        trajectory_kind="linear",
        curve_id=None,
        heading_deg=None,
    )


def is_catalog_eval(spec: str) -> bool:
    """True when this token installs the reconstruction apparatus (not original ghost)."""
    try:
        return bool(resolve_eval_spec(spec).spawn_pushers)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Original RoboTwin skills: same recipe as the catalog (keep language + check_success,
# original meshes, extra bumper car). Catalogued skills are ``ready``; the rest
# are pick_only / skip.
# ---------------------------------------------------------------------------

def _fit(
    name: str,
    status: SkillStatus,
    *,
    pick: str | None,
    place: str | None,
    variants: tuple[str, ...],
    why: str,
) -> SkillFit:
    return SkillFit(name, status, pick, place, variants, why)


SKILL_FIT: tuple[SkillFit, ...] = (
    _fit(
        "place_empty_cup",
        "ready",
        pick="cup",
        place="coaster",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Catalogued: rigid cup + coaster on the table, planar push, official success unchanged.",
    ),
    # --- ready: pick-and-place, both actors free rigid bodies on the table ---
    _fit(
        "place_phone_stand",
        "ready",
        pick="phone",
        place="stand",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Same layout as the cup: grasp phone, set onto a free-standing stand.",
    ),
    _fit(
        "move_pillbottle_pad",
        "ready",
        pick="pillbottle",
        place="pad",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Bottle onto a pad. Pad is a box actor but still a planar receptacle.",
    ),
    _fit(
        "move_stapler_pad",
        "ready",
        pick="stapler",
        place="pad",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Same pad recipe as the pill bottle; stapler is a compact rigid body.",
    ),
    _fit(
        "place_container_plate",
        "ready",
        pick="container",
        place="plate",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Bowl/cup onto a plate — closest sibling of place_empty_cup.",
    ),
    _fit(
        "place_object_basket",
        "ready",
        pick="object",
        place="basket",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Single object into a basket; basket is large enough for a visible push.",
    ),
    _fit(
        "place_can_basket",
        "ready",
        pick="can",
        place="basket",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Can + basket; cylindrical pick target like the cup.",
    ),
    _fit(
        "place_bread_skillet",
        "ready",
        pick="bread",
        place="skillet",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="One bread onto a skillet; both sit on the table.",
    ),
    _fit(
        "place_object_stand",
        "ready",
        pick="object",
        place="displaystand",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Object onto a display stand; same pick/place split as the phone.",
    ),
    _fit(
        "place_object_scale",
        "ready",
        pick="object",
        place="scale",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Object onto a scale; scale is a wide, pushable receptacle.",
    ),
    _fit(
        "place_a2b_left",
        "ready",
        pick="object",
        place="target_object",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Already on the v1 whitelist; A beside B, both free rigid actors.",
    ),
    _fit(
        "place_a2b_right",
        "ready",
        pick="object",
        place="target_object",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Mirror of place_a2b_left.",
    ),
    _fit(
        "move_can_pot",
        "ready",
        pick="can",
        place="pot",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Can beside a kitchen pot; pot is a large tabletop URDF, good visual pusher target.",
    ),
    _fit(
        "place_mouse_pad",
        "ready",
        pick="mouse",
        place="target",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Mouse onto a pad box. Pad is a colored box, not a mesh, but the recipe still holds.",
    ),
    _fit(
        "place_fan",
        "ready",
        pick="fan",
        place="pad",
        variants=("pick", "place", "both", "pick.irregular", "place.irregular"),
        why="Fan onto a pad; compact pick object, planar goal.",
    ),
    _fit(
        "place_shoe",
        "ready",
        pick="shoe",
        place="target_block",
        variants=("pick", "place"),
        why="One shoe onto a target block. Skip both until the block is confirmed kinematic-safe.",
    ),
    _fit(
        "place_bread_basket",
        "ready",
        pick="bread",
        place="breadbasket",
        variants=("place",),
        why="Basket is a great pusher target. Two breads make pick/both crowded — start with place-only.",
    ),
    # --- pick_only: one tabletop rigid body, no useful receptacle to push ---
    _fit(
        "grab_roller",
        "pick_only",
        pick="roller",
        place=None,
        variants=("pick",),
        why="Dual-arm lift of a long roller; no receptacle. Pick-rail car is enough.",
    ),
    _fit(
        "move_playingcard_away",
        "pick_only",
        pick="playingcards",
        place=None,
        variants=("pick",),
        why="No receptacle. Card is very thin — bumper contact will look weaker than the cup.",
    ),
    _fit(
        "lift_pot",
        "pick_only",
        pick="pot",
        place=None,
        variants=("pick",),
        why="Bimanual lift, no place target. Car can slide the pot; skip place/both.",
    ),
    _fit(
        "pick_dual_bottles",
        "pick_only",
        pick="bottle1",
        place=None,
        variants=("pick",),
        why="Two simultaneous grasps. One car cannot tell a clean story; do not add both-rails.",
    ),
    _fit(
        "pick_diverse_bottles",
        "pick_only",
        pick="bottle1",
        place=None,
        variants=("pick",),
        why="Same as pick_dual_bottles: bimanual two-object pick.",
    ),
    _fit(
        "place_dual_shoes",
        "pick_only",
        pick="left_shoe",
        place="shoe_box",
        variants=("place",),
        why="Two shoes into a box. Box-only (place) is viable; dual pick is not a single-car story.",
    ),
    _fit(
        "place_cans_plasticbox",
        "pick_only",
        pick="object1",
        place="plasticbox",
        variants=("place",),
        why="Two cans. Push the box, not both cans.",
    ),
    _fit(
        "place_burger_fries",
        "pick_only",
        pick="hamburg",
        place="tray",
        variants=("place",),
        why="Two foods onto a tray. Tray-only motion is the clean variant.",
    ),
    # --- skip: articulated, tool, click, hang, stack, in-hand, furniture ---
    _fit("handover_block", "skip", pick="box", place="target_box", variants=(), why="Already a handoff stress; a bumper car fights the pass."),
    _fit("handover_mic", "skip", pick="microphone", place=None, variants=(), why="Arm-to-arm handoff, not tabletop pushing."),
    _fit("beat_block_hammer", "skip", pick="hammer", place=None, variants=(), why="Tool use; success is a hit, not a place."),
    _fit("stamp_seal", "skip", pick="stamp", place=None, variants=(), why="Tool press onto a stamp pad."),
    _fit("press_stapler", "skip", pick="stapler", place=None, variants=(), why="In-place press, not pick-and-place."),
    _fit("scan_object", "skip", pick="scanner", place=None, variants=(), why="Tool scan; moving the scanner as a car is off-skill."),
    _fit("click_bell", "skip", pick=None, place=None, variants=(), why="Button click, not a free object to push."),
    _fit("click_alarmclock", "skip", pick=None, place=None, variants=(), why="Button click."),
    _fit("turn_switch", "skip", pick=None, place=None, variants=(), why="Articulated switch."),
    _fit("open_laptop", "skip", pick=None, place=None, variants=(), why="Articulated lid."),
    _fit("open_microwave", "skip", pick=None, place=None, variants=(), why="Articulated door."),
    _fit("put_object_cabinet", "skip", pick="object", place=None, variants=(), why="Insert into furniture / cabinet."),
    _fit("dump_bin_bigbin", "skip", pick=None, place=None, variants=(), why="Dump into a large bin — furniture scale."),
    _fit("put_bottles_dustbin", "skip", pick=None, place=None, variants=(), why="Drop into a dustbin, not a tabletop slide."),
    _fit("hanging_mug", "skip", pick="mug", place=None, variants=(), why="Hang on a rack; not planar pushing."),
    _fit("adjust_bottle", "skip", pick="bottle", place=None, variants=(), why="Orientation-only."),
    _fit("shake_bottle", "skip", pick="bottle", place=None, variants=(), why="In-hand shake after grasp."),
    _fit("shake_bottle_horizontally", "skip", pick="bottle", place=None, variants=(), why="In-hand shake."),
    _fit("rotate_qrcode", "skip", pick="qrcode", place=None, variants=(), why="Yaw/orientation skill."),
    _fit("stack_blocks_two", "skip", pick=None, place=None, variants=(), why="Precision stack; a moving base breaks the stack."),
    _fit("stack_blocks_three", "skip", pick=None, place=None, variants=(), why="Precision stack."),
    _fit("stack_bowls_two", "skip", pick=None, place=None, variants=(), why="Precision nest/stack."),
    _fit("stack_bowls_three", "skip", pick=None, place=None, variants=(), why="Precision nest/stack."),
    _fit("blocks_ranking_rgb", "skip", pick=None, place=None, variants=(), why="Multi-block ranking, not a single mover."),
    _fit("blocks_ranking_size", "skip", pick=None, place=None, variants=(), why="Multi-block ranking."),
)


def list_skill_fits(*, status: SkillStatus | Sequence[SkillStatus] | None = None) -> list[SkillFit]:
    if status is None:
        return list(SKILL_FIT)
    wanted = {status} if isinstance(status, str) else set(status)
    return [s for s in SKILL_FIT if s.status in wanted]
