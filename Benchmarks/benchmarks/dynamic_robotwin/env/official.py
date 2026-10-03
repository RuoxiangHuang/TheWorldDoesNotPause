"""Official Dynamic-RoboTwin evaluation contract (frozen for comparable runs).

Primary track keeps protocol ghost sliding (``rails_collide=False``) and the
paper-aligned dynamics mix from ``configs/default.yaml``. Demo-only
``rails_collide=True`` (toy-car physical pushes) is a secondary track, never
the official default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from benchmarks.common.protocol import (
    DEFAULT_BACKEND,
    DEFAULT_DEADLINE_MS,
    PROTOCOL_VERSION,
    RealtimeBackend,
)
from benchmarks.dynamic_robotwin.env.task_registry import (
    DEFAULT_DYNAMIC_MODES,
    DEFAULT_RELEASE_RADIUS,
    V1_TASKS,
)

# ---------------------------------------------------------------------------
# Locked contract
# ---------------------------------------------------------------------------

OFFICIAL_PROTOCOL = PROTOCOL_VERSION  # real_time_v2
OFFICIAL_BACKEND = DEFAULT_BACKEND  # async (prior chunk else ZOH)
OFFICIAL_REPLAN_STEPS = 8
OFFICIAL_POLICY_HZ = 20.0
OFFICIAL_RELEASE_RADIUS_M = DEFAULT_RELEASE_RADIUS  # 0.12; per-task overrides OK
OFFICIAL_DEADLINE_MS = DEFAULT_DEADLINE_MS

# Rails physics (primary track): trajectory-first ghost sliding.
OFFICIAL_RAILS_COLLIDE = False
OFFICIAL_LANGUAGE_MODE = "keep"  # primary; --language-mode motion for B-track

# Paper-aligned Dynamic-RoboTwin stresses (PROTOCOL §9).
OFFICIAL_DYNAMIC_MODES: tuple[str, ...] = DEFAULT_DYNAMIC_MODES
OFFICIAL_CONTACT_SWITCH = "medium"
OFFICIAL_OCCLUSION = True
OFFICIAL_CONTACT_CHAIN = True
OFFICIAL_SECONDARY_RAILS = "auto"

OFFICIAL_TASKS: tuple[str, ...] = V1_TASKS
OFFICIAL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
OFFICIAL_EPISODES_PER_TASK = len(OFFICIAL_SEEDS)

# Intensity grid (m / policy tick). Catalog auto maps v=0 → apparatus_hold.
# Original spawn is `--scene-preset original`.
OFFICIAL_SPEEDS: tuple[float, ...] = (
    0.0,
    0.0005,
    0.001,
    0.002,
    0.003,
    0.004,
    0.006,
    0.010,
)
OFFICIAL_V_MIN = 0.0
OFFICIAL_V_MAX = 0.010

OFFICIAL_TRAJECTORIES: tuple[str, ...] = ("linear", "sine")

OFFICIAL_WORKING_POINTS: tuple[dict[str, Any], ...] = (
    {"trajectory": "linear", "speed": 0.0},
    {"trajectory": "linear", "speed": 0.001},
    {"trajectory": "linear", "speed": 0.002},
    {"trajectory": "sine", "speed": 0.001},
)


@dataclass(frozen=True)
class OfficialContract:
    protocol: str = OFFICIAL_PROTOCOL
    backend: str = OFFICIAL_BACKEND.value
    replan_steps: int = OFFICIAL_REPLAN_STEPS
    policy_hz: float = OFFICIAL_POLICY_HZ
    release_radius_m: float = OFFICIAL_RELEASE_RADIUS_M
    deadline_ms: float = OFFICIAL_DEADLINE_MS
    rails_collide: bool = OFFICIAL_RAILS_COLLIDE
    language_mode: str = OFFICIAL_LANGUAGE_MODE
    dynamic_modes: tuple[str, ...] = OFFICIAL_DYNAMIC_MODES
    contact_switch: str = OFFICIAL_CONTACT_SWITCH
    occlusion: bool = OFFICIAL_OCCLUSION
    contact_chain: bool = OFFICIAL_CONTACT_CHAIN
    secondary_rails: str = OFFICIAL_SECONDARY_RAILS
    tasks: tuple[str, ...] = OFFICIAL_TASKS
    seeds: tuple[int, ...] = OFFICIAL_SEEDS
    episodes_per_task: int = OFFICIAL_EPISODES_PER_TASK
    speeds: tuple[float, ...] = OFFICIAL_SPEEDS
    v_min: float = OFFICIAL_V_MIN
    v_max: float = OFFICIAL_V_MAX
    trajectories: tuple[str, ...] = OFFICIAL_TRAJECTORIES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def n_episodes_macro(self) -> int:
        return len(self.tasks) * self.episodes_per_task

    def dynamics_cli(self) -> dict[str, str]:
        return {
            "dynamic_modes": ",".join(self.dynamic_modes),
            "contact_trigger": self.contact_switch,
            "occlusion": "on" if self.occlusion else "off",
            "contact_chain": "on" if self.contact_chain else "off",
            "secondary_rails": self.secondary_rails,
        }


OFFICIAL = OfficialContract()


def normalize_speed(
    speed: float, *, v_min: float = OFFICIAL_V_MIN, v_max: float = OFFICIAL_V_MAX
) -> float:
    """Map raw m/tick speed to normalized intensity in [0, 1]."""
    lo, hi = float(v_min), float(v_max)
    if hi <= lo:
        return 0.0
    x = (float(speed) - lo) / (hi - lo)
    return float(max(0.0, min(1.0, x)))


def apply_official_args(args: Any, *, force: bool = True) -> Any:
    """Overwrite argparse namespace fields with the official contract."""
    c = OFFICIAL
    dyn = c.dynamics_cli()
    mapping = {
        "backend": c.backend,
        "replan_steps": c.replan_steps,
        "policy_hz": c.policy_hz,
        "seeds": ",".join(str(s) for s in c.seeds),
        "dynamic_modes": dyn["dynamic_modes"],
        "contact_trigger": dyn["contact_trigger"],
        "occlusion": dyn["occlusion"],
        "contact_chain": dyn["contact_chain"],
        "secondary_rails": dyn["secondary_rails"],
        "language_mode": c.language_mode,
    }
    for key, val in mapping.items():
        if force or not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, val)
    if force or not getattr(args, "allow_replan_override", False):
        args.allow_replan_override = False
    args.rails_collide = c.rails_collide
    args.official_contract = c.to_dict()
    return args


def contract_banner() -> str:
    c = OFFICIAL
    return (
        f"[official] protocol={c.protocol} backend={c.backend} "
        f"replan={c.replan_steps} rails_collide={c.rails_collide} "
        f"modes={','.join(c.dynamic_modes)} contact={c.contact_switch} "
        f"N_seed={c.episodes_per_task} tasks={len(c.tasks)} "
        f"macro_episodes={c.n_episodes_macro()}"
    )


def validate_backend(backend: str | RealtimeBackend) -> RealtimeBackend:
    if isinstance(backend, str):
        backend = RealtimeBackend(backend)
    return backend


def parse_task_names(spec: str | None) -> list[str]:
    """Parse comma-separated names, ``all`` (official whitelist), or aliases.

    ``all`` stays the frozen v1 ghost-rails whitelist.
    ``recon`` (also ``seed`` / ``wave2``) is the unified bumper-car catalog.
    ``extra`` is the appendix intercept/chase catalog (not AUC).
    Dotted catalog ids are accepted alongside whitelist names.
    """
    from .dynamic_tasks import (
        extension_tasks,
        get_dynamic_task,
        latency_slice_tasks,
        main_tasks,
        recon_both_tasks,
        recon_demo_tasks,
        recon_tasks,
    )
    from .extra_tasks import EXTRA_TASK_NAMES, extra_tasks

    if spec is None or not str(spec).strip() or str(spec).strip().lower() == "all":
        return list(OFFICIAL_TASKS)
    token = str(spec).strip().lower()
    if token in (
        "recon",
        "dynamic",
        "catalog",
        "seed",
        "pusher-seed",
        "cup-seed",
        "wave2",
        "wave-2",
        "recon-wave2",
    ):
        return [t.id for t in recon_tasks()]
    if token in ("main", "main-track", "identity"):
        return [t.id for t in main_tasks()]
    if token in ("extension", "irregular", "curve"):
        return [t.id for t in extension_tasks()]
    if token in ("recon-demo", "wave2-demo"):
        return [t.id for t in recon_demo_tasks()]
    if token in ("recon-both", "wave2-both", "wave2_both"):
        return [t.id for t in recon_both_tasks()]
    if token in (
        "latency",
        "slice",
        "latency-slice",
        "latency_slice",
        "pick-slice",
        "pick_slice",
    ):
        return [t.id for t in latency_slice_tasks(variant="pick")]
    if token in (
        "place_slice",
        "place-slice",
        "latency_place",
        "latency-place",
    ):
        return [t.id for t in latency_slice_tasks(variant="place")]
    if token in ("extra", "extras", "appendix"):
        return [t.id for t in extra_tasks()]
    names = [x.strip() for x in str(spec).split(",") if x.strip()]
    unknown = []
    out: list[str] = []
    for n in names:
        if n in OFFICIAL_TASKS:
            out.append(n)
            continue
        from .task_registry import RECON_TARGET_ATTR

        if n in RECON_TARGET_ATTR:
            out.append(n)
            continue
        if n in EXTRA_TASK_NAMES:
            out.append(n)
            continue
        try:
            get_dynamic_task(n)
            out.append(n)
        except KeyError:
            unknown.append(n)
    if unknown:
        raise KeyError(
            f"Unknown RoboTwin task(s) {unknown}; "
            f"whitelist: {list(OFFICIAL_TASKS)}; "
            f"aliases: recon, recon-demo, recon-both, extra, latency, place_slice "
            f"(seed/wave2 are the same unified batch); "
            f"reconstructions: {len(recon_tasks())} ids; "
            f"extras: {list(EXTRA_TASK_NAMES)}"
        )
    return out
