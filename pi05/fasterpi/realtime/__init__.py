"""Real-Time Dynamic benchmark integration for FasterPI.

Heavy openpi imports are lazy so the LeRobot RoboCasa path can load without
having ``openpi`` installed in the same environment.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "OpenPIRealtimePolicy",
    "install_speedup",
    "load_libero_policy",
    "load_robotwin_policy",
    "warmup_policy",
    "libero_obs_to_openpi",
    "robotwin_obs_to_openpi",
    "measure_libero_latency",
    "measure_robotwin_latency",
    "LeRobotRoboCasaPolicy",
    "load_lerobot_robocasa_policy",
    "warmup_lerobot_policy",
]


def __getattr__(name: str) -> Any:
    if name in {
        "OpenPIRealtimePolicy",
        "install_speedup",
        "load_libero_policy",
        "load_robotwin_policy",
        "warmup_policy",
    }:
        from fasterpi.realtime import loader as _loader

        return getattr(_loader, name)
    if name in {"libero_obs_to_openpi", "robotwin_obs_to_openpi"}:
        from fasterpi.realtime import adapters as _adapters

        return getattr(_adapters, name)
    if name in {"measure_libero_latency", "measure_robotwin_latency"}:
        from fasterpi.realtime import measure as _measure

        return getattr(_measure, name)
    if name in {
        "LeRobotRoboCasaPolicy",
        "load_lerobot_robocasa_policy",
        "warmup_lerobot_policy",
    }:
        from fasterpi.realtime import lerobot_robocasa as _lr

        return getattr(_lr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
