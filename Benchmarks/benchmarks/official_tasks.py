"""The 95 paper tasks: 50 Dynamic-LIBERO and 45 Dynamic-RoboTwin.

Extensions (spatial carriers, irregular RoboTwin curves, reaction turns,
and the five native extra tasks) stay in the catalogs but are not part of
this count.
"""

from __future__ import annotations

from benchmarks.dynamic_libero.env.dynamic_tasks import main_track_tasks
from benchmarks.dynamic_robotwin.env.dynamic_tasks import main_tasks

N_DYNAMIC_LIBERO = 50
N_DYNAMIC_ROBOTWIN = 45
N_TASKS = N_DYNAMIC_LIBERO + N_DYNAMIC_ROBOTWIN


def dynamic_libero_tasks():
    tasks = tuple(main_track_tasks())
    if len(tasks) != N_DYNAMIC_LIBERO:
        raise RuntimeError(f"Dynamic-LIBERO main track is {len(tasks)}, expected {N_DYNAMIC_LIBERO}")
    return tasks


def dynamic_robotwin_tasks():
    tasks = tuple(main_tasks())
    if len(tasks) != N_DYNAMIC_ROBOTWIN:
        raise RuntimeError(
            f"Dynamic-RoboTwin main track is {len(tasks)}, expected {N_DYNAMIC_ROBOTWIN}"
        )
    return tasks


def official_tasks() -> dict[str, tuple]:
    libero = dynamic_libero_tasks()
    robotwin = dynamic_robotwin_tasks()
    if len(libero) + len(robotwin) != N_TASKS:
        raise RuntimeError("official task count is not 95")
    return {"dynamic_libero": libero, "dynamic_robotwin": robotwin}
