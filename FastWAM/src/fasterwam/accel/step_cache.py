"""Denoising-step cache for the action expert.

The action flow-matching loop integrates

    x_{i+1} = x_i + v_i * delta_i

and accounts for ~80% of per-replan latency, so skipping steps is the highest-
ceiling algorithmic lever. A skipped step must still supply a velocity for the
un-computed `v_i`; this controller holds the last computed one (zero-order).

Why the gate reads the *input* latent rather than the velocity: the scheduler
advances `x` on every step, including held ones, so the input drift keeps
refreshing during a hold and the gate reacts as soon as the trajectory moves. A
velocity-based gate can only refresh when a step is actually computed, so it
runs stale exactly during the hold it is supposed to guard — measured to be
10-28% worse in action MSE at a matched skip rate despite correlating better
with per-step hold error.

Holding the most recent velocity likewise beats filling with an average of
recent velocities: the gate only skips steps where the latent has settled, and
there the newest velocity is the best available estimate.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .base import Controller
from .config import StepCacheConfig


def relative_l1_tensor(a: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Device-resident relative L1; compare on GPU and sync only at branch points."""
    denom = reference.abs().mean().clamp(min=1e-6)
    return (a - reference).abs().mean() / denom


def relative_l1(a: torch.Tensor, reference: torch.Tensor) -> float:
    """Host scalar relative L1 — prefer ``relative_l1_tensor`` on hot paths."""
    return float(relative_l1_tensor(a, reference).item())


class StepCacheController(Controller):
    name = "step_cache"
    REQUIRED_ATTRS = ("infer_action", "_predict_action_noise_with_cache")

    def __init__(self, model: Any, config: StepCacheConfig | None = None):
        super().__init__(model)
        self.config = config or StepCacheConfig()
        self._computed = 0
        self._skipped = 0
        self._force_refresh = 0
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self._last_input: torch.Tensor | None = None
        self._last_velocity: torch.Tensor | None = None
        self._drift: torch.Tensor | None = None
        self._drift_threshold_tensor: torch.Tensor | None = None
        self._consecutive_skips = 0
        self._step = 0

    def _drift_threshold(self, latents_action: torch.Tensor) -> torch.Tensor:
        t = self._drift_threshold_tensor
        if t is None or t.device != latents_action.device:
            t = torch.tensor(
                self.config.threshold,
                device=latents_action.device,
                dtype=torch.float32,
            )
            self._drift_threshold_tensor = t
        return t

    def _install(self) -> None:
        # Patch the denoise loop entry first so every replan starts from clean
        # state; a stale velocity must never leak across replans.
        self._patch(self.model, "infer_action", self._wrap_infer_action)
        self._patch(self.model, "_predict_action_noise_with_cache", self._wrap_predict)

    def _wrap_infer_action(self, original: Callable) -> Callable:
        def infer_action(*args, **kwargs):
            self._reset_episode_state()
            return original(*args, **kwargs)

        return infer_action

    def _wrap_predict(self, original: Callable) -> Callable:
        def predict(*args, latents_action: torch.Tensor | None = None, **kwargs):
            if latents_action is None:
                if not args:
                    raise TypeError("step_cache: `latents_action` is required")
                latents_action, args = args[0], args[1:]

            step = self._step
            self._step += 1

            if self._last_input is not None:
                step_drift = relative_l1_tensor(latents_action, self._last_input)
                self._drift = (
                    step_drift
                    if self._drift is None
                    else self._drift + step_drift
                )

            cfg = self.config
            can_skip = (
                step >= cfg.cold_steps
                and self._last_velocity is not None
                and self._consecutive_skips < cfg.max_consecutive_skips
            )
            if can_skip and self._drift is not None:
                can_skip = bool((self._drift < self._drift_threshold(latents_action)).item())
            if can_skip:
                self._consecutive_skips += 1
                self._skipped += 1
                self._last_input = latents_action.clone()
                return self._last_velocity

            if (
                step >= cfg.cold_steps
                and self._last_velocity is not None
                and self._consecutive_skips >= cfg.max_consecutive_skips
            ):
                self._force_refresh += 1

            velocity = original(*args, latents_action=latents_action, **kwargs)
            self._computed += 1
            self._last_input = latents_action.clone()
            self._last_velocity = velocity
            self._drift = None
            self._consecutive_skips = 0
            return velocity

        return predict

    def reset(self) -> None:
        self._reset_episode_state()

    def _on_uninstall(self) -> None:
        self._reset_episode_state()

    def stats(self) -> dict[str, Any]:
        total = self._computed + self._skipped
        return {
            "computed": self._computed,
            "skipped": self._skipped,
            "force_refresh": self._force_refresh,
            "skip_rate": (self._skipped / total) if total else 0.0,
        }
