"""Shared cross-replan state for video-token drift tracking.

``video_token_cache`` and ``chunk_residual_cache`` both gate on relative L1 of
``video_tokens`` vs the previous replan. When both are installed, their prefill
wrappers nest (chunk_residual outer, video_token inner). This holder computes
drift once per prefill (outer ``enter_prefill``) and stores a single read-only
clone of the last tokens (outer ``exit_prefill``).
"""

from __future__ import annotations

import torch

from .step_cache import relative_l1_tensor

VIDEO_TOKEN_STATE_ATTR = "_fasterwam_video_token_state"


class VideoTokenState:
    """Controller-owned, read-only snapshot of the last replan's video tokens."""

    def __init__(self) -> None:
        self._last_tokens: torch.Tensor | None = None
        self.drift: float = float("inf")
        self._prefill_depth = 0

    @property
    def last_tokens(self) -> torch.Tensor | None:
        return self._last_tokens

    @property
    def owns_prefill_scope(self) -> bool:
        return self._prefill_depth == 0

    def enter_prefill(self, current: torch.Tensor) -> float:
        """Begin a prefill call; compute drift only on the outermost entry."""
        if self._prefill_depth == 0:
            self.drift = self._compute_drift(current)
        self._prefill_depth += 1
        return self.drift

    def exit_prefill(self, current: torch.Tensor) -> None:
        """End a prefill call; commit the token clone only on the outermost exit."""
        self._prefill_depth -= 1
        if self._prefill_depth == 0:
            self._last_tokens = current.detach().clone()

    def _compute_drift(self, current: torch.Tensor) -> float:
        ref = self._last_tokens
        if ref is None:
            return float("inf")
        current = current.detach()
        if ref.shape != current.shape:
            return float("inf")
        if ref.device != current.device or ref.dtype != current.dtype:
            return float("inf")
        return float(relative_l1_tensor(current, ref).item())

    def reset(self) -> None:
        self._last_tokens = None
        self.drift = float("inf")
        self._prefill_depth = 0


def get_video_token_state(model: object) -> VideoTokenState | None:
    return getattr(model, VIDEO_TOKEN_STATE_ATTR, None)


def ensure_video_token_state(model: object) -> VideoTokenState:
    state = get_video_token_state(model)
    if state is None:
        state = VideoTokenState()
        setattr(model, VIDEO_TOKEN_STATE_ATTR, state)
    return state


def clear_video_token_state(model: object) -> None:
    if hasattr(model, VIDEO_TOKEN_STATE_ATTR):
        try:
            delattr(model, VIDEO_TOKEN_STATE_ATTR)
        except AttributeError:
            pass
