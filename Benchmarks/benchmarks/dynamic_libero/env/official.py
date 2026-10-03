"""Official Dynamic-LIBERO evaluation contract (frozen for comparable runs).

Primary track keeps original LIBERO appearance and language so absolute SR
stays distributionally aligned with training. Rails targets are always taken
from the per-task registry (never inferred). Toy-car **asset swap** is an
optional secondary track (``movers=on``), not the official default. Object /
libero_10 default motive is ``pusher`` (a separate car actor; pick mesh stays).
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
from benchmarks.dynamic_libero.env.suite_registry import (
    DEFAULT_INIT_IDS,
    TRAINED_SUITES,
    suite_n_tasks,
)

# ---------------------------------------------------------------------------
# Locked contract
# ---------------------------------------------------------------------------

OFFICIAL_PROTOCOL = PROTOCOL_VERSION  # real_time_v2
OFFICIAL_BACKEND = DEFAULT_BACKEND  # async (prior chunk else ZOH)
OFFICIAL_REPLAN_STEPS = 10
OFFICIAL_CONTROL_HZ = 20.0
OFFICIAL_RELEASE_RADIUS_M = 0.06
OFFICIAL_MAX_STEPS = 400
OFFICIAL_DEADLINE_MS = DEFAULT_DEADLINE_MS

# Appearance / language (primary track).
OFFICIAL_MOVERS = False
OFFICIAL_MOVER_LANGUAGE = "keep"
OFFICIAL_LANGUAGE_MODE = "keep"  # primary; use --language-mode motion for B-track
# Always resolve rails target via MoverSpec registry (fixes BDDL inference bugs).
OFFICIAL_TARGET_REGISTRY = True

OFFICIAL_SUITES: tuple[str, ...] = TRAINED_SUITES
OFFICIAL_INIT_IDS: tuple[int, ...] = DEFAULT_INIT_IDS  # 0..4 → N=5 / task
OFFICIAL_EPISODES_PER_TASK = len(OFFICIAL_INIT_IDS)

# Intensity grid (m / control tick). Catalog auto maps v=0 → apparatus_hold
# (same layout as drive). Original spawn is `--scene-preset original`.
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
# Calibration anchors for normalized intensity \tilde v (within-suite).
OFFICIAL_V_MIN = 0.0
OFFICIAL_V_MAX = 0.010

# Required trajectories for an official suite report.
OFFICIAL_TRAJECTORIES: tuple[str, ...] = ("linear", "sine")
# smooth_turn is the reaction-track event path (PROTOCOL §13), not official.

# Working points highlighted in summaries (not the only points to run).
# Every point uses protocol-default async, including v=0 (object still;
# robot still executes the stale tail while thinking).
OFFICIAL_WORKING_POINTS: tuple[dict[str, Any], ...] = (
    {"trajectory": "linear", "speed": 0.0, "backend": "async"},
    {"trajectory": "linear", "speed": 0.0006, "backend": "async"},
    {"trajectory": "linear", "speed": 0.001, "backend": "async"},
    {"trajectory": "linear", "speed": 0.003, "backend": "async"},
    {"trajectory": "sine", "speed": 0.001, "backend": "async"},
)

# Paper latency-slice SR–speed curve. Denser than the frozen official 60 grid
# around the measured FastWAM off/Cache band. ``run_sr --speeds paper``.
# Backend is async at every speed, including v=0.
PAPER_SLICE_SPEEDS: tuple[float, ...] = (
    0.0,
    0.0003,
    0.0005,
    0.0006,
    0.0008,
    0.001,
    0.0015,
    0.002,
)


def parse_speed_list(spec: str | None, *, default: tuple[float, ...] | None = None) -> list[float]:
    """Parse ``--speeds``: ``official`` / ``paper`` aliases or comma-separated floats."""
    fallback = list(default if default is not None else OFFICIAL_SPEEDS)
    token = str(spec or "").strip()
    if token.lower() in {"", "official", "default", "grid"}:
        return fallback
    if token.lower() in {"paper", "slice", "latency", "working"}:
        return list(PAPER_SLICE_SPEEDS)
    return [float(x) for x in token.split(",") if x.strip() != ""]


@dataclass(frozen=True)
class OfficialContract:
    protocol: str = OFFICIAL_PROTOCOL
    backend: str = OFFICIAL_BACKEND.value
    replan_steps: int = OFFICIAL_REPLAN_STEPS
    control_hz: float = OFFICIAL_CONTROL_HZ
    release_radius_m: float = OFFICIAL_RELEASE_RADIUS_M
    max_steps: int = OFFICIAL_MAX_STEPS
    deadline_ms: float = OFFICIAL_DEADLINE_MS
    movers: bool = OFFICIAL_MOVERS
    mover_language: str = OFFICIAL_MOVER_LANGUAGE
    language_mode: str = OFFICIAL_LANGUAGE_MODE
    target_registry: bool = OFFICIAL_TARGET_REGISTRY
    suites: tuple[str, ...] = OFFICIAL_SUITES
    init_ids: tuple[int, ...] = OFFICIAL_INIT_IDS
    episodes_per_task: int = OFFICIAL_EPISODES_PER_TASK
    speeds: tuple[float, ...] = OFFICIAL_SPEEDS
    v_min: float = OFFICIAL_V_MIN
    v_max: float = OFFICIAL_V_MAX
    trajectories: tuple[str, ...] = OFFICIAL_TRAJECTORIES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def task_ids(self, suite: str) -> list[int]:
        return list(range(suite_n_tasks(suite)))

    def n_episodes_suite(self, suite: str) -> int:
        return suite_n_tasks(suite) * self.episodes_per_task

    def n_episodes_macro(self) -> int:
        return sum(self.n_episodes_suite(s) for s in self.suites)


OFFICIAL = OfficialContract()


def normalize_speed(speed: float, *, v_min: float = OFFICIAL_V_MIN, v_max: float = OFFICIAL_V_MAX) -> float:
    """Map raw m/tick speed to normalized intensity in [0, 1]."""
    lo, hi = float(v_min), float(v_max)
    if hi <= lo:
        return 0.0
    x = (float(speed) - lo) / (hi - lo)
    return float(max(0.0, min(1.0, x)))


def apply_official_args(args: Any, *, force: bool = True) -> Any:
    """Overwrite argparse namespace fields with the official contract.

    When ``force`` is False, only fill missing / default-like fields.
    """
    c = OFFICIAL
    mapping = {
        "backend": c.backend,
        "replan_steps": c.replan_steps,
        "release_radius": c.release_radius_m,
        "max_steps": c.max_steps,
        "deadline_ms": c.deadline_ms,
        "movers": "on" if c.movers else "off",
        "mover_language": c.mover_language,
        "language_mode": c.language_mode,
        "init_ids": ",".join(str(i) for i in c.init_ids),
    }
    for key, val in mapping.items():
        if force or not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, val)
    if force or not getattr(args, "allow_replan_override", False):
        args.allow_replan_override = False
    # Target registry is always on for official runs.
    args.target_registry = True
    args.official_contract = c.to_dict()
    return args


def contract_banner() -> str:
    c = OFFICIAL
    return (
        f"[official] protocol={c.protocol} backend={c.backend} "
        f"replan={c.replan_steps} movers={'on' if c.movers else 'off'} "
        f"target_registry={c.target_registry} "
        f"N_init={c.episodes_per_task} suites={len(c.suites)} "
        f"macro_episodes={c.n_episodes_macro()}"
    )


def validate_backend(backend: str | RealtimeBackend) -> RealtimeBackend:
    if isinstance(backend, str):
        backend = RealtimeBackend(backend)
    return backend
