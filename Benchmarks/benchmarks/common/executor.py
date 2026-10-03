"""Replan / execute loop with freeze or async stale-chunk backends."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .protocol import RealtimeBackend
from .thinking import apply_thinking_async, apply_thinking_freeze


@dataclass
class RealtimeExecutor:
    """Manages chunk consumption, replan cadence, and thinking-phase coupling."""

    replan_steps: int
    backend: RealtimeBackend = RealtimeBackend.FREEZE
    chunk: Any | None = None
    idx: int = 0
    steps_since_replan: int = 0
    n_replans: int = 0
    stale_executed_total: int = 0
    stale_hold_ticks_total: float = 0.0
    applied_freeze_total: float = 0.0
    leftover_len_sum: int = 0
    chunk_len_sum: int = 0
    n_len_samples: int = 0
    _last_thinking: dict[str, float | int] = field(default_factory=dict)

    def needs_replan(self) -> bool:
        if self.chunk is None:
            return True
        if self.idx >= len(self.chunk):
            return True
        return self.steps_since_replan >= int(self.replan_steps)

    def on_replan(
        self,
        new_chunk: Any,
        driver: Any,
        n_freeze: float,
        *,
        async_step_fn: Callable[[Any], None] | None = None,
        max_ticks: float | None = None,
    ) -> None:
        """Run thinking phase then swap in ``new_chunk``."""
        self.n_replans += 1
        stale_tail = []
        if self.chunk is not None and self.idx < len(self.chunk):
            stale_tail = list(self.chunk[self.idx :])
        new_len = 0 if new_chunk is None else len(new_chunk)
        self.leftover_len_sum += len(stale_tail)
        self.chunk_len_sum += int(new_len)
        self.n_len_samples += 1
        if self.n_replans <= 3:
            print(
                f"[queue] replan={self.n_replans} chunk_len={new_len} "
                f"leftover={len(stale_tail)} replan_steps={self.replan_steps} "
                f"backend={self.backend.value}",
                flush=True,
            )

        if self.backend == RealtimeBackend.ASYNC:
            if async_step_fn is None:
                raise ValueError("async backend requires async_step_fn")
            stats = apply_thinking_async(
                driver, n_freeze, stale_tail, async_step_fn, max_ticks=max_ticks
            )
        else:
            applied = apply_thinking_freeze(driver, n_freeze, max_ticks=max_ticks)
            stats = {
                "applied_freeze": applied,
                "stale_executed": 0,
                "stale_hold_ticks": applied if not stale_tail else 0.0,
            }

        self._last_thinking = stats
        self.applied_freeze_total += float(stats.get("applied_freeze", 0.0))
        self.stale_executed_total += int(stats.get("stale_executed", 0))
        self.stale_hold_ticks_total += float(stats.get("stale_hold_ticks", 0.0))

        self.chunk = new_chunk
        self.idx = 0
        self.steps_since_replan = 0

    def next_action(self) -> Any:
        if self.chunk is None:
            raise RuntimeError("next_action called before any replan")
        action = self.chunk[self.idx]
        self.idx += 1
        self.steps_since_replan += 1
        return action

    def open_loop_break(self, n_freeze: float) -> bool:
        return not math.isfinite(float(n_freeze))

    def metrics(self, n_freeze: float) -> dict[str, Any]:
        n = int(self.n_len_samples)
        return {
            "n_replans": int(self.n_replans),
            "applied_freeze": float(self.applied_freeze_total),
            "stale_executed": int(self.stale_executed_total),
            "stale_hold_ticks": float(self.stale_hold_ticks_total),
            "backend": self.backend.value,
            "replan_steps": int(self.replan_steps),
            "last_thinking": dict(self._last_thinking),
            "n_freeze": float(n_freeze) if math.isfinite(float(n_freeze)) else None,
            "chunk_len_mean": (self.chunk_len_sum / n) if n else None,
            "leftover_len_mean": (self.leftover_len_sum / n) if n else None,
        }
