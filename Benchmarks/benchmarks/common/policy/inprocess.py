"""In-process PolicyClient adapters wrapping existing FasterWAM bridges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _reset_ctx_accel(ctx: Any) -> None:
    model = getattr(ctx, "model", None)
    if model is None:
        return
    from fasterwam.accel import reset_inference_state

    reset_inference_state(model)


@dataclass
class LiberoInProcessPolicy:
    ctx: Any
    benchmark: str = "libero"

    def reset(self) -> None:
        _reset_ctx_accel(self.ctx)

    def predict(self, obs: Any, instruction: str, **kwargs: Any) -> np.ndarray:
        from benchmarks.dynamic_libero.env import libero_bridge as bridge

        chunk, _, _ = bridge.predict_action_chunk(obs, instruction, self.ctx)
        return np.asarray(chunk)

    def close(self) -> None:
        return


@dataclass
class RoboTwinInProcessPolicy:
    ctx: Any
    benchmark: str = "robotwin"

    def reset(self) -> None:
        _reset_ctx_accel(self.ctx)

    def predict(self, obs: Any, instruction: str, **kwargs: Any) -> np.ndarray:
        from benchmarks.dynamic_robotwin.env import robotwin_bridge as bridge

        chunk = bridge.predict_action_chunk(obs, instruction, self.ctx)
        return np.asarray(chunk)

    def close(self) -> None:
        return


@dataclass
class RoboCasaInProcessPolicy:
    ctx: Any
    benchmark: str = "robocasa"

    def reset(self) -> None:
        _reset_ctx_accel(self.ctx)

    def predict(self, obs: Any, instruction: str, **kwargs: Any) -> np.ndarray:
        from benchmarks.realtime_robocasa.env import robocasa_bridge as bridge

        chunk = bridge.predict_action_chunk(obs, instruction, self.ctx)
        return np.asarray(chunk)

    def close(self) -> None:
        return
