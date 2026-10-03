"""RoboTwin policy-tick ↔ PhysX step calibration."""

from __future__ import annotations

import numpy as np

DEFAULT_PHYSICS_PER_TICK = 12


def _hold_qpos_action(task_env) -> np.ndarray:
    """Current joint targets as a qpos ``take_action`` vector."""
    robot = task_env.robot
    left = np.asarray(robot.get_left_arm_jointState(), dtype=np.float64)
    right = np.asarray(robot.get_right_arm_jointState(), dtype=np.float64)
    return np.concatenate([left, right]).astype(np.float32)


def calibrate_physics_per_tick(
    task_env,
    *,
    n_samples: int = 5,
) -> float:
    """Median ``scene.step`` count per ``take_action`` (hold-current-qpos).

    Call after ``setup_demo`` and driver hooks are installed so the count
    matches runtime (TOPP + pin hooks).
    """
    scene = task_env.scene
    orig_step = scene.step
    counts: list[int] = []

    def counting_step(*args, **kwargs):
        counting_step.n += 1  # type: ignore[attr-defined]
        return orig_step(*args, **kwargs)

    counting_step.n = 0  # type: ignore[attr-defined]

    try:
        scene.step = counting_step
        for _ in range(max(1, int(n_samples))):
            counting_step.n = 0  # type: ignore[attr-defined]
            action = _hold_qpos_action(task_env)
            task_env.take_action(action, action_type="qpos")
            n = int(counting_step.n)  # type: ignore[attr-defined]
            if n > 0:
                counts.append(n)
    finally:
        scene.step = orig_step

    if not counts:
        return float(DEFAULT_PHYSICS_PER_TICK)
    return float(np.median(counts))


def resolve_physics_per_tick(
    task_env,
    *,
    explicit: int | None = None,
    auto_calibrate: bool = True,
    n_samples: int = 5,
) -> tuple[int, str]:
    """Return ``(physics_per_tick, source)`` where source is ``explicit`` or ``calibrated``."""
    if explicit is not None and int(explicit) > 0:
        return int(explicit), "explicit"
    if auto_calibrate:
        measured = calibrate_physics_per_tick(task_env, n_samples=n_samples)
        return max(1, int(round(measured))), "calibrated"
    return int(DEFAULT_PHYSICS_PER_TICK), "default"
