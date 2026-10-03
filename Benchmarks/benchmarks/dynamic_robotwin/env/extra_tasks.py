"""Appendix extra tasks: intercept / chase / physics, not the 95-task main table.

These five skills share the Dynamic-RoboTwin ``real_time_v2`` contract
(async stale-tail, measured latency, fractional ``n_freeze``, native
``check_success``) but they do **not** join:

- the 45 linear main-track tasks
- the 95-task paper suite
- reconstructed bumper-car ids (``recon``)

Motion is task-native (ballistic shuttlecock, hops, rolls, steered car,
PhysX scatter), not linear rails. See EXTRA.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional


ExtraKind = Literal["intercept", "chase", "physics_demo"]


@dataclass(frozen=True)
class ExtraTask:
    id: str
    task_name: str
    object_name: str
    object_attr: str
    language: str
    kind: ExtraKind
    ttc_s: float
    stress: str
    record_module: str
    cine: str
    note: str = ""
    eval_ready: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


CATALOG: tuple[ExtraTask, ...] = (
    ExtraTask(
        id="catch_shuttlecock",
        task_name="catch_shuttlecock",
        object_name="125_shuttlecock",
        object_attr="object",
        language="Catch the flying shuttlecock.",
        kind="intercept",
        ttc_s=0.55,
        stress="ballistic_intercept",
        record_module="benchmarks.dynamic_robotwin.viz.record_catch_shuttlecock",
        cine="evaluate_results/demo_videos/catch_shuttlecock/catch_shuttlecock_motion_cine.mp4",
        note="Cork-first parabola. Extra LoRA/eval TTC≈0.55s (bird flies from t=0); cine probe remains 0.36s.",
        eval_ready=True,
    ),
    ExtraTask(
        id="catch_bunny_toy",
        task_name="catch_bunny_toy",
        object_name="121_bunny_toy",
        object_attr="object",
        language="Catch the hopping rabbit toy.",
        kind="intercept",
        ttc_s=2.4,
        stress="hopping_intercept",
        record_module="benchmarks.dynamic_robotwin.viz.record_catch_bunny",
        cine="evaluate_results/demo_videos/catch_bunny_toy/catch_bunny_cine.mp4",
        note="Vinyl bunny hops across the table. Eval steps the hop ODE on the policy clock.",
        eval_ready=True,
    ),
    ExtraTask(
        id="stop_rolling_orange",
        task_name="stop_rolling_orange",
        object_name="122_orange_toy",
        object_attr="object",
        language="Stop the orange from rolling off.",
        kind="intercept",
        ttc_s=2.4,
        stress="rolling_intercept",
        record_module="benchmarks.dynamic_robotwin.viz.record_stop_rolling_orange",
        cine="evaluate_results/demo_videos/stop_rolling_orange/stop_rolling_orange_cine.mp4",
        note="Tangerine rolls toward the rim on a push-then-coast arc.",
        eval_ready=True,
    ),
    ExtraTask(
        id="pursue_toycar",
        task_name="pursue_toycar",
        object_name="123_diecast_car",
        object_attr="object",
        language="Chase the toy car.",
        kind="chase",
        ttc_s=3.0,
        stress="steered_chase",
        record_module="benchmarks.dynamic_robotwin.viz.record_pursue_toycar",
        cine="evaluate_results/demo_videos/pursue_toycar/pursue_toycar_cine.mp4",
        note="Die-cast coupe slalom and U-turn. Wheels follow the bicycle model.",
        eval_ready=True,
    ),
    ExtraTask(
        id="collide_pool_balls",
        task_name="collide_pool_balls",
        object_name="124_pool_ball",
        object_attr="object",
        language="Watch the pool balls collide.",
        kind="physics_demo",
        ttc_s=0.8,
        stress="physx_scatter",
        record_module="benchmarks.dynamic_robotwin.viz.record_collide_pool_balls",
        cine="evaluate_results/demo_videos/collide_pool_balls/collide_pool_balls_cine.mp4",
        note="PhysX cut shot. Success is both balls on the cloth and separated, not a grasp.",
        eval_ready=True,
    ),
)

_BY_ID = {t.id: t for t in CATALOG}
EXTRA_TASK_NAMES: tuple[str, ...] = tuple(t.task_name for t in CATALOG)
EXTRA_TARGET_ATTR: dict[str, str] = {t.task_name: t.object_attr for t in CATALOG}


def extra_tasks() -> list[ExtraTask]:
    return list(CATALOG)


def intercept_extras() -> list[ExtraTask]:
    return [t for t in CATALOG if t.kind == "intercept"]


def is_extra(spec: str) -> bool:
    key = str(spec).strip()
    return key in _BY_ID or key in EXTRA_TARGET_ATTR


def get_extra(spec: str) -> ExtraTask:
    key = str(spec).strip()
    if key in _BY_ID:
        return _BY_ID[key]
    for t in CATALOG:
        if t.task_name == key:
            return t
    raise KeyError(
        f"Unknown extra task {spec!r}. Catalog: {list(EXTRA_TASK_NAMES)}"
    )


def extra_instruction(task_name: str) -> Optional[str]:
    try:
        return get_extra(task_name).language
    except KeyError:
        return None
