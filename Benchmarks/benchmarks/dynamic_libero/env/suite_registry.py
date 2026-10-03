"""LIBERO suite registry for Dynamic-LIBERO full-suite eval."""

from __future__ import annotations

from typing import Iterable, Sequence

# Suites included in libero_uncond_2cam224 training data (configs/data/libero_2cam.yaml).
TRAINED_SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

# Full public LIBERO benchmark set (libero_90 is OOD for the release checkpoint).
ALL_SUITES: tuple[str, ...] = TRAINED_SUITES + ("libero_90",)

SUITE_N_TASKS: dict[str, int] = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}

SUITE_ALIASES: dict[str, str] = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
    "libero_long": "libero_10",
    "90": "libero_90",
}

DEFAULT_INIT_IDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def normalize_suite_name(name: str) -> str:
    key = name.strip()
    if key in SUITE_ALIASES:
        return SUITE_ALIASES[key]
    if key not in SUITE_N_TASKS:
        raise KeyError(
            f"Unknown LIBERO suite {name!r}. Supported: {sorted(SUITE_N_TASKS)} "
            f"(aliases: {sorted(SUITE_ALIASES)})"
        )
    return key


def is_trained_suite(suite_name: str) -> bool:
    return normalize_suite_name(suite_name) in TRAINED_SUITES


def suite_n_tasks(suite_name: str) -> int:
    return int(SUITE_N_TASKS[normalize_suite_name(suite_name)])


def resolve_suites(spec: str | Sequence[str]) -> list[str]:
    """Parse suite list from CLI.

    ``trained`` / ``all-trained`` → the four training suites.
    ``all`` → trained + libero_90 (OOD for release ckpt).
    Otherwise comma-separated suite names or aliases.
    """
    if isinstance(spec, str):
        token = spec.strip().lower()
        if token in {"trained", "all-trained", "train"}:
            return list(TRAINED_SUITES)
        if token in {"all", "*"}:
            return list(ALL_SUITES)
        parts = [x.strip() for x in spec.split(",") if x.strip()]
    else:
        parts = [str(x).strip() for x in spec if str(x).strip()]
    if not parts:
        raise ValueError("suite list is empty")
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        name = normalize_suite_name(p)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def parse_task_ids(spec: str, suite_name: str) -> list[int]:
    """Parse task ids; ``all`` / ``*`` expands to every task in the suite."""
    suite = normalize_suite_name(suite_name)
    token = spec.strip().lower()
    n = suite_n_tasks(suite)
    if token in {"all", "*"}:
        return list(range(n))
    ids = [int(x.strip()) for x in spec.split(",") if x.strip() != ""]
    for tid in ids:
        if tid < 0 or tid >= n:
            raise ValueError(f"task_id={tid} out of range for {suite!r} (n_tasks={n})")
    return ids


def parse_init_ids(spec: str) -> list[int]:
    token = spec.strip().lower()
    if token in {"all", "*"}:
        raise ValueError(
            "init_ids=all is not supported (50 inits/task); pass explicit ids, e.g. 0,1,2,3,4"
        )
    return [int(x.strip()) for x in spec.split(",") if x.strip() != ""]


def default_task_ids_for_suite(suite_name: str, *, full: bool = False) -> list[int]:
    """Smoke-friendly defaults: task 0 only, or all tasks when ``full=True``."""
    if full:
        return parse_task_ids("all", suite_name)
    return [0]
