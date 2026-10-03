"""Dynamic-LIBERO task catalog: one static LIBERO skill → rails variants.

The original BDDL goal and instruction stay unchanged. Dynamics are factored
into who moves:

- ``pick``: the grasp target rides rails (toy car pushes / ghost-pins it).
- ``place``: the receptacle rides rails; the grasp target stays put.
- ``both``: pick target and receptacle both move.

Eligibility is semantic, not accelerator ΔSR:

- **Main** (``eval_split=main``): object-suite identity language still holds
  (soup→basket linear and irregular). Report full-task SR here.
- **Extension** (``eval_split=extension``): spatial carriers whose official
  locatives are false after homing. Instruction is a public scene rule
  ("bowl on the moving platform"); BDDL stays bowl-on-plate.
- **React** (``track=react``): same object, instruction, and BDDL as the aligned
  main ``pick`` task, plus one pre-sampled world-clock motion event. Not part
  of the official 60.

Seed eval list is the alphabet-soup skill split into those three main tasks.
Other LIBERO skills are tagged ``ready`` / ``pick_only`` / ``skip`` so they can
be expanded the same way without changing training labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional, Sequence

RailsVariant = Literal["pick", "place", "both"]
TaskStatus = Literal[
    "seed", "ready", "pick_only", "skip", "irregular", "spatial", "react"
]
EvalTrack = Literal["main", "react"]
EventType = Literal["turn", "decel", "accel"]

RAILS_VARIANTS: tuple[RailsVariant, ...] = ("pick", "place", "both")

# Latency diagnostic (not the main table). Excluded for scene / success-condition
# legality, not because an accelerator already won:
# t03 BBQ: bottle geometry is ungraspable under official inits.
# t07 milk: grasp can succeed but BDDL place is unreliable on these inits.
# t08 pudding: react turn leaves the table AABB.
LATENCY_SLICE_TASK_IDS: tuple[int, ...] = (0, 1, 4, 5, 6)
# World-clock turn window for slice react tasks (20 Hz ticks after begin_policy).
SLICE_EVENT_T_LO = 30.0  # 1.5 s
SLICE_EVENT_T_HI = 50.0  # 2.5 s

# Original LIBERO-object instructions (training labels; never rewritten here).
_OBJECT_LANG: dict[int, str] = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese and place it in the basket",
    2: "pick up the salad dressing and place it in the basket",
    3: "pick up the bbq sauce and place it in the basket",
    4: "pick up the ketchup and place it in the basket",
    5: "pick up the tomato sauce and place it in the basket",
    6: "pick up the butter and place it in the basket",
    7: "pick up the milk and place it in the basket",
    8: "pick up the chocolate pudding and place it in the basket",
    9: "pick up the orange juice and place it in the basket",
}

# Official LIBERO-spatial instructions as returned by the env (training labels).
_SPATIAL_LANG: dict[int, str] = {
    0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
    1: "pick up the black bowl next to the ramekin and place it on the plate",
    2: "pick up the black bowl from table center and place it on the plate",
    3: "pick up the black bowl on the cookie box and place it on the plate",
    4: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    5: "pick up the black bowl on the ramekin and place it on the plate",
    6: "pick up the black bowl next to the cookie box and place it on the plate",
    7: "pick up the black bowl on the stove and place it on the plate",
    8: "pick up the black bowl next to the plate and place it on the plate",
    9: "pick up the black bowl on the wooden cabinet and place it on the plate",
}

_OBJECT_PICK: dict[int, str] = {
    0: "alphabet_soup_1",
    1: "cream_cheese_1",
    2: "salad_dressing_1",
    3: "bbq_sauce_1",
    4: "ketchup_1",
    5: "tomato_sauce_1",
    6: "butter_1",
    7: "milk_1",
    8: "chocolate_pudding_1",
    9: "orange_juice_1",
}


@dataclass(frozen=True)
class DynamicTask:
    """One Dynamic-LIBERO eval unit (static LIBERO skill × rails variant)."""

    id: str
    suite: str
    task_id: int
    variant: RailsVariant
    language: str
    pick_target: str
    place_target: Optional[str]
    status: TaskStatus = "ready"
    note: str = ""
    trajectory_kind: str = "linear"
    waypoints: Optional[tuple[tuple[float, float, float], ...]] = None
    path_name: str = "linear"
    motive: Optional[str] = None  # None → suite default (object=pusher, spatial=toycar)
    clear_path: bool = True  # False keeps other free objects at official init xy
    # Planar linear heading in degrees (0=+x, 90=+y). None → CLI / axis default.
    heading_deg: Optional[float] = None
    track: EvalTrack = "main"
    # React-track event template. ``event_time`` / ``event_seed`` are sampled at
    # episode reset (world clock), not stored here.
    event_type: Optional[EventType] = None
    transition_duration: Optional[float] = None  # control ticks
    event_magnitude: Optional[float] = None  # deg for turn; Δspeed later
    event_t_lo: Optional[float] = None
    event_t_hi: Optional[float] = None
    source_language: Optional[str] = None
    language_policy: str = "keep"  # keep | scene
    eval_split: str = "main"  # main | extension | react | diagnostic
    eligibility: str = "identity"

    @property
    def pick_motion(self) -> bool:
        return self.variant in ("pick", "both")

    @property
    def place_motion(self) -> bool:
        return self.variant in ("place", "both")

    @property
    def aligned_main_id(self) -> str:
        """Main-track catalog id this task shares objects / language / BDDL with."""
        if self.track == "react" or self.status == "react":
            return f"{self.suite}.t{self.task_id:02d}.{self.variant}"
        return self.id

    def traj_spec(self) -> tuple[str, dict]:
        """``(kind, kwargs)`` for ``build_trajectory``."""
        if self.trajectory_kind in ("curve", "smooth", "spline"):
            return "curve", {"curve_id": self.path_name}
        if self.waypoints:
            return "polyline", {"waypoints": [tuple(p) for p in self.waypoints]}
        if (
            self.track == "react"
            or self.status == "react"
            or self.trajectory_kind in ("smooth_turn", "turn")
        ):
            from benchmarks.dynamic_libero.trajectories.smooth_turn import (
                default_smooth_turn_kwargs,
            )

            extra: dict = {}
            if self.heading_deg is not None:
                extra["heading_deg"] = float(self.heading_deg)
            if self.transition_duration is not None:
                extra["turn_duration_ticks"] = float(self.transition_duration)
            if self.event_magnitude is not None:
                extra["turn_angle_deg"] = abs(float(self.event_magnitude))
            if self.event_t_lo is not None:
                extra["event_t_lo"] = float(self.event_t_lo)
            if self.event_t_hi is not None:
                extra["event_t_hi"] = float(self.event_t_hi)
            kw = default_smooth_turn_kwargs(
                toward_center=self.suite == "libero_object",
                extra=extra,
            )
            if "heading_deg" in kw:
                kw.pop("axis", None)
            return "smooth_turn", kw
        kw: dict = {}
        if self.heading_deg is not None:
            kw["heading_deg"] = float(self.heading_deg)
            # Object-suite floor is small; fan headings without this walk off the AABB.
            if self.suite == "libero_object":
                kw["toward_center"] = True
        return str(self.trajectory_kind or "linear"), kw

    def to_dict(self) -> dict:
        return asdict(self)


def parse_rails_variant(value: str | None, *, default: RailsVariant = "pick") -> RailsVariant:
    if value is None:
        return default
    key = str(value).strip().lower()
    if key in {"", "auto", "default"}:
        return default
    aliases = {
        "pick": "pick",
        "object": "pick",
        "grasp": "pick",
        "place": "place",
        "receptacle": "place",
        "basket": "place",
        "goal": "place",
        "both": "both",
        "dual": "both",
        "all": "both",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown rails variant {value!r}. Expected pick|place|both "
            f"(aliases: {sorted(aliases)})"
        )
    return aliases[key]  # type: ignore[return-value]


def _object_linear_heading(task_id: int, variant: RailsVariant) -> float:
    """Fan object-suite linear rails around the table (not a single +x).

    Robot-view +x looks vertical on the agent camera; spreading headings
    mixes depth, lateral, and diagonal slides. 12° variant offsets keep
    pick/place/both of the same skill visually distinct.
    """
    vi = {"pick": 0, "place": 1, "both": 2}[variant]
    return float((int(task_id) * 36 + vi * 12) % 360)


def _spatial_linear_heading(task_id: int) -> float:
    """Keep spatial carriers in the open table quadrant (avoid cabinet/stove).

    Kitchen cabinet occupies −y, stove −x. Headings stay in [5°, 85°].
    """
    table = (70.0, 20.0, 45.0, 85.0, 10.0, 55.0, 35.0, 75.0, 5.0, 50.0)
    return float(table[int(task_id) % 10])


def _object_task(task_id: int, variant: RailsVariant, *, status: TaskStatus) -> DynamicTask:
    lang = _OBJECT_LANG[task_id]
    pick = _OBJECT_PICK[task_id]
    heading = _object_linear_heading(task_id, variant)
    return DynamicTask(
        id=f"libero_object.t{task_id:02d}.{variant}",
        suite="libero_object",
        task_id=task_id,
        variant=variant,
        language=lang,
        pick_target=pick,
        place_target="basket_1",
        status=status,
        heading_deg=heading,
        note="Same BDDL In(object, basket). Dynamics: who rides the rails. "
        f"Linear heading {heading:.0f}° (0=+x, 90=+y).",
        language_policy="keep",
        eval_split="main",
        eligibility="identity",
    )


def _object_react_pick(task_id: int) -> DynamicTask:
    """Reaction extension: same skill as main linear pick, plus one turn event."""
    from benchmarks.dynamic_libero.trajectories.smooth_turn import (
        DIAGNOSTIC_TURN_ANGLE_DEG,
        DIAGNOSTIC_TURN_DURATION_TICKS,
    )

    main_status: TaskStatus = "seed" if int(task_id) == 0 else "ready"
    main = _object_task(int(task_id), "pick", status=main_status)
    in_slice = int(task_id) in LATENCY_SLICE_TASK_IDS
    note = (
        f"Reaction extension aligned with {main.id}: same object, instruction, "
        "and BDDL. Inbound heading matches the main linear pick, then one "
        "world-clock smooth turn. Not part of the official reconstructed 60."
    )
    if in_slice:
        note += (
            f" Latency slice: event Uniform({SLICE_EVENT_T_LO:.0f},{SLICE_EVENT_T_HI:.0f}) "
            "ticks (1.5–2.5 s) after begin_policy."
        )
    return DynamicTask(
        id=f"{main.id}.react",
        suite=main.suite,
        task_id=main.task_id,
        variant="pick",
        language=main.language,
        pick_target=main.pick_target,
        place_target=main.place_target,
        status="react",
        trajectory_kind="smooth_turn",
        path_name="smooth_turn",
        motive="ghost",
        heading_deg=main.heading_deg,
        track="react",
        eval_split="react",
        event_type="turn",
        transition_duration=float(DIAGNOSTIC_TURN_DURATION_TICKS),
        event_magnitude=float(DIAGNOSTIC_TURN_ANGLE_DEG),
        event_t_lo=SLICE_EVENT_T_LO if in_slice else None,
        event_t_hi=SLICE_EVENT_T_HI if in_slice else None,
        note=note,
    )


def _object_irregular(task_id: int, variant: RailsVariant) -> DynamicTask:
    from benchmarks.dynamic_libero.env.irregular_paths import irregular_curve_id

    if variant not in ("pick", "place"):
        raise ValueError("irregular extras are pick/place only")
    name = irregular_curve_id(task_id, variant)  # type: ignore[arg-type]
    who = "grasp target" if variant == "pick" else "basket"
    return DynamicTask(
        id=f"libero_object.t{task_id:02d}.{variant}.irregular",
        suite="libero_object",
        task_id=task_id,
        variant=variant,
        language=_OBJECT_LANG[task_id],
        pick_target=_OBJECT_PICK[task_id],
        place_target="basket_1",
        status="irregular",
        trajectory_kind="curve",
        waypoints=None,
        path_name=name,
        note=f"Smooth curve ({name}) on the {who}. Same BDDL. Distractors cleared off the path.",
        language_policy="keep",
        eval_split="main",
        eligibility="identity",
    )


# Official reconstructed-60 slots that replace the pre-fix layouts.
# Same IDs / BDDL / keep language; not dropped from the 60.
# t03 pick: mild clothoid_right (old 0.34 m / 59° yaw was dynamically unsolvable).
# t00 / t05 carrier: still plate is never path-cleared.
# t04 / t07 / t09 carrier: bowl+car homed onto the table (drawer / stove / cabinet
# spawn would plant the visual-only pickup inside the furniture).
RECONSTRUCTED_REPLACEMENTS: tuple[str, ...] = (
    "libero_object.t03.pick.irregular",
    "libero_spatial.t00.pick.carrier",
    "libero_spatial.t04.pick.carrier",
    "libero_spatial.t05.pick.carrier",
    "libero_spatial.t07.pick.carrier",
    "libero_spatial.t09.pick.carrier",
)


def _spatial_carrier(task_id: int) -> DynamicTask:
    """Bowl rides a flatbed. Extension track: language matches the live layout."""
    from benchmarks.common.language import SPATIAL_CARRIER_SCENE_INSTRUCTION

    heading = _spatial_linear_heading(task_id)
    source = _SPATIAL_LANG[task_id]
    return DynamicTask(
        id=f"libero_spatial.t{task_id:02d}.pick.carrier",
        suite="libero_spatial",
        task_id=task_id,
        variant="pick",
        language=SPATIAL_CARRIER_SCENE_INSTRUCTION,
        pick_target="akita_black_bowl_1",
        place_target="plate_1",
        status="spatial",
        trajectory_kind="linear",
        path_name="linear",
        motive="carrier",
        clear_path=True,
        heading_deg=heading,
        source_language=source,
        language_policy="scene",
        eval_split="extension",
        eligibility=(
            "locative_rewritten: official spawn locative is false after the bowl "
            "is homed onto the carrier; BDDL remains bowl-on-plate."
        ),
        note=(
            "Extension track. Instruction is generated by a fixed scene rule "
            f"(not the original locative {source!r}). "
            "Stationary plate_1 is never path-cleared. "
            "If the official spawn is in/on a fixture, the carrier is homed onto "
            f"the table before rails start. Linear heading {heading:.0f}°."
        ),
    )


def _catalog() -> tuple[DynamicTask, ...]:
    items: list[DynamicTask] = []
    for variant in RAILS_VARIANTS:
        items.append(_object_task(0, variant, status="seed"))
    for tid in range(1, 10):
        for variant in RAILS_VARIANTS:
            items.append(_object_task(tid, variant, status="ready"))
    for tid in range(10):
        items.append(_object_react_pick(tid))
    for tid in range(10):
        items.append(_object_irregular(tid, "pick"))
        items.append(_object_irregular(tid, "place"))
    for tid in range(10):
        items.append(_spatial_carrier(tid))
    items.extend(
        [
            DynamicTask(
                id="libero_spatial.t02.pick",
                suite="libero_spatial",
                task_id=2,
                variant="pick",
                language="pick up the black bowl from table center and place it on the plate",
                pick_target="akita_black_bowl_1",
                place_target="plate_1",
                status="ready",
                note="Spatial spawn relations are baked at t=0; success is still bowl-on-plate.",
            ),
            DynamicTask(
                id="libero_spatial.t02.place",
                suite="libero_spatial",
                task_id=2,
                variant="place",
                language="pick up the black bowl from table center and place it on the plate",
                pick_target="akita_black_bowl_1",
                place_target="plate_1",
                status="ready",
                note="Moving the plate after spawn weakens 'from table center' language but not BDDL.",
            ),
            DynamicTask(
                id="libero_spatial.t02.both",
                suite="libero_spatial",
                task_id=2,
                variant="both",
                language="pick up the black bowl from table center and place it on the plate",
                pick_target="akita_black_bowl_1",
                place_target="plate_1",
                status="ready",
                note="Bowl and plate both translate. Other spatial-suite tasks share this recipe.",
            ),
            DynamicTask(
                id="libero_goal.t06.pick",
                suite="libero_goal",
                task_id=6,
                variant="pick",
                language="put the cream cheese in the bowl",
                pick_target="cream_cheese_1",
                place_target="akita_black_bowl_1",
                status="ready",
            ),
            DynamicTask(
                id="libero_goal.t06.place",
                suite="libero_goal",
                task_id=6,
                variant="place",
                language="put the cream cheese in the bowl",
                pick_target="cream_cheese_1",
                place_target="akita_black_bowl_1",
                status="ready",
            ),
            DynamicTask(
                id="libero_goal.t06.both",
                suite="libero_goal",
                task_id=6,
                variant="both",
                language="put the cream cheese in the bowl",
                pick_target="cream_cheese_1",
                place_target="akita_black_bowl_1",
                status="ready",
            ),
            DynamicTask(
                id="libero_goal.t08.pick",
                suite="libero_goal",
                task_id=8,
                variant="pick",
                language="put the bowl on the plate",
                pick_target="akita_black_bowl_1",
                place_target="plate_1",
                status="ready",
            ),
            DynamicTask(
                id="libero_10.t00.pick",
                suite="libero_10",
                task_id=0,
                variant="pick",
                language="put both the alphabet soup and the tomato sauce in the basket",
                pick_target="alphabet_soup_1",
                place_target="basket_1",
                status="ready",
                note="Long-horizon: first pick moves; second pick stays. Dual-pick expansion is later.",
            ),
            DynamicTask(
                id="libero_10.t00.place",
                suite="libero_10",
                task_id=0,
                variant="place",
                language="put both the alphabet soup and the tomato sauce in the basket",
                pick_target="alphabet_soup_1",
                place_target="basket_1",
                status="ready",
            ),
            DynamicTask(
                id="libero_goal.t00.pick",
                suite="libero_goal",
                task_id=0,
                variant="pick",
                language="open the middle drawer of the cabinet",
                pick_target="akita_black_bowl_1",
                place_target=None,
                status="pick_only",
                note="No free-joint receptacle. Pick/disturbance rails only; cannot 1→3 split.",
            ),
            DynamicTask(
                id="libero_goal.t07.pick",
                suite="libero_goal",
                task_id=7,
                variant="pick",
                language="turn on the stove",
                pick_target="akita_black_bowl_1",
                place_target=None,
                status="skip",
                note="Articulated appliance; rails on a nearby object are only visual disturbance.",
            ),
        ]
    )
    return tuple(items)


CATALOG: tuple[DynamicTask, ...] = _catalog()
_BY_ID: dict[str, DynamicTask] = {t.id: t for t in CATALOG}


def get_dynamic_task(task_id: str) -> DynamicTask:
    key = str(task_id).strip()
    if key not in _BY_ID:
        raise KeyError(
            f"Unknown dynamic task {task_id!r}. Seed ids: "
            f"{[t.id for t in CATALOG if t.status == 'seed']}"
        )
    return _BY_ID[key]


def list_dynamic_tasks(
    *,
    status: TaskStatus | Sequence[TaskStatus] | None = None,
) -> list[DynamicTask]:
    if status is None:
        return list(CATALOG)
    wanted = {status} if isinstance(status, str) else set(status)
    return [t for t in CATALOG if t.status in wanted]


def seed_tasks() -> list[DynamicTask]:
    """Default reconstructed eval list: soup × {pick, place, both}."""
    return list_dynamic_tasks(status="seed")


def irregular_tasks() -> list[DynamicTask]:
    """20 extras: each libero_object skill × {pick, place} on a unique smooth curve."""
    return list_dynamic_tasks(status="irregular")


def spatial_tasks() -> list[DynamicTask]:
    """10 spatial skills: bowl on a linear flatbed, original prompts."""
    return list_dynamic_tasks(status="spatial")


def react_tasks() -> list[DynamicTask]:
    """10 object-suite pick tasks aligned 1:1 with the main linear pick catalog.

    Same cup, instruction, and BDDL as ``libero_object.tXX.pick``. Adds one
    world-clock smooth turn. Not in the reconstructed 60.
    """
    return list_dynamic_tasks(status="react")


def latency_slice_tasks() -> list[DynamicTask]:
    """Pre-registered pick slice for latency-sensitive SR (main track)."""
    wanted = set(LATENCY_SLICE_TASK_IDS)
    return [t for t in main_object_pick_tasks() if int(t.task_id) in wanted]


def place_slice_tasks() -> list[DynamicTask]:
    """Same five skills as ``latency_slice_tasks``, but the basket moves.

    FastWAM's latency slice is moving-cup pick. π₀.₅ already tracks those cups;
    the remaining latency bottleneck is depositing a static object into a
    basket that is driven for the whole episode (place-only never parks the
    receptacle). Not the official reconstructed 60. Not a post-hoc ΔSR filter.
    """
    wanted = set(LATENCY_SLICE_TASK_IDS)
    return [
        t
        for t in CATALOG
        if t.suite == "libero_object"
        and t.variant == "place"
        and t.track == "main"
        and t.trajectory_kind == "linear"
        and t.status in ("seed", "ready")
        and int(t.task_id) in wanted
    ]


def react_slice_tasks() -> list[DynamicTask]:
    """Same cups as ``latency_slice_tasks``, plus one world-clock turn."""
    wanted = set(LATENCY_SLICE_TASK_IDS)
    return [t for t in react_tasks() if int(t.task_id) in wanted]


def main_object_pick_tasks() -> list[DynamicTask]:
    """Main-track object-suite linear pick tasks (the react alignment set)."""
    return [
        t
        for t in CATALOG
        if t.suite == "libero_object"
        and t.variant == "pick"
        and t.track == "main"
        and t.trajectory_kind == "linear"
        and t.status in ("seed", "ready")
    ]


def main_track_tasks() -> list[DynamicTask]:
    """Main table: identity language still holds (object linear + irregular).

    Spatial carriers are extension-only: their official locatives are false
    after homing. Selection is semantic / scene-legal, not accelerator ΔSR.
    """
    return [
        t
        for t in CATALOG
        if t.eval_split == "main"
        and t.suite == "libero_object"
        and t.status in ("seed", "ready", "irregular")
    ]


def extension_tasks() -> list[DynamicTask]:
    """Spatial carriers and other locative-rewritten tasks."""
    return [t for t in CATALOG if t.eval_split == "extension"]


def reconstructed_tasks() -> list[DynamicTask]:
    """60-task set: main object 50 + spatial 10 extension (scene language).

    ``RECONSTRUCTED_REPLACEMENTS`` stay in this list (tuned layouts, same IDs).
    """
    tasks = main_track_tasks() + spatial_tasks()
    have = {t.id for t in tasks}
    missing = [i for i in RECONSTRUCTED_REPLACEMENTS if i not in have]
    if missing:
        raise RuntimeError(f"reconstructed 60 missing layout replacements: {missing}")
    return tasks


def parse_dynamic_tasks(spec: str | None) -> list[DynamicTask]:
    """Parse ``seed`` / ``latency`` / ``place_slice`` / ``react_slice`` / ``react`` / ``reconstructed`` / ids / ``off``."""
    if spec is None:
        return []
    token = str(spec).strip()
    if token.lower() in {"", "off", "none", "false"}:
        return []
    if token.lower() in {"seed", "seeds", "soup"}:
        return seed_tasks()
    if token.lower() in {"irregular", "polyline", "zigzag"}:
        return irregular_tasks()
    if token.lower() in {"spatial", "spatial-carrier", "bowl-car"}:
        return spatial_tasks()
    if token.lower() in {"react", "reaction", "react_pick"}:
        return react_tasks()
    if token.lower() in {
        "latency",
        "slice",
        "latency-slice",
        "latency_slice",
        "pick-slice",
    }:
        return latency_slice_tasks()
    if token.lower() in {
        "place_slice",
        "place-slice",
        "latency_place",
        "latency-place",
        "pi_place",
        "pi-place",
        "pi_latency",
    }:
        return place_slice_tasks()
    if token.lower() in {
        "react_slice",
        "react-slice",
        "latency-react",
        "slice_react",
    }:
        return react_slice_tasks()
    if token.lower() in {"reconstructed", "sixty", "60", "catalog-60"}:
        return reconstructed_tasks()
    if token.lower() in {"main", "main-track", "identity"}:
        return main_track_tasks()
    if token.lower() in {"extension", "extensions", "locative"}:
        return extension_tasks()
    if token.lower() in {"ready", "object"}:
        return list_dynamic_tasks(status=("seed", "ready"))
    if token.lower() in {"all"}:
        return list(CATALOG)
    out: list[DynamicTask] = []
    seen: set[str] = set()
    for part in token.split(","):
        key = part.strip()
        if not key:
            continue
        task = get_dynamic_task(key)
        if task.id not in seen:
            seen.add(task.id)
            out.append(task)
    return out


def default_place_target(suite: str, task_id: int) -> Optional[str]:
    """Free-joint receptacle for a static LIBERO skill, if one exists."""
    suite = suite.strip()
    if suite == "libero_object":
        return "basket_1"
    if suite == "libero_spatial":
        return "plate_1"
    if suite == "libero_goal":
        return {6: "akita_black_bowl_1", 8: "plate_1"}.get(int(task_id))
    if suite == "libero_10" and int(task_id) in {0, 1, 7}:
        return "basket_1"
    return None
