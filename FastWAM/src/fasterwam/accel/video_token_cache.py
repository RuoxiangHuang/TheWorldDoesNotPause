"""Cross-replan video KV cache with static-token refresh.

Within a replan, FastWAM already prefills video K/V once. Across replans the
observation usually changes only locally, so recomputing the full 30-layer
video stack every time is wasteful. Full-cache reuse was measured and rejected
as too lossy when applied unconditionally; this controller keeps the reuse but
gates it:

  - global relative-L1 of `video_tokens` below `full_reuse_threshold`
    -> return the previous K/V (skip prefill entirely)
  - else a token-wise mask selects dynamic tokens; if the refresh fraction is
    at most `max_refresh_ratio`, run the sparse prefill path that recomputes
    only those tokens and copies static K/V from the prior cache
  - otherwise fall through to a full prefill

Cold replans and a consecutive-reuse cap bound how stale the cache can get.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .base import Controller
from .config import VideoTokenCacheConfig
from .world_state import VideoTokenState, ensure_video_token_state


def _clone_cache(cache: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{"k": entry["k"].clone(), "v": entry["v"].clone()} for entry in cache]


def token_relative_l1(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-token relative L1 over the feature dim. Returns shape `[S]`."""
    # [B, S, D] -> reduce batch then feature.
    diff = (current - reference).abs().mean(dim=(0, 2))
    denom = reference.abs().mean(dim=(0, 2)).clamp(min=1e-6)
    return diff / denom


class VideoTokenCacheController(Controller):
    name = "video_token_cache"
    REQUIRED_ATTRS = ("infer_action",)

    @classmethod
    def missing_requirements(cls, model: Any, config: Any = None) -> list[str]:
        missing = super().missing_requirements(model, config)
        if not hasattr(getattr(model, "mot", None), "prefill_video_cache"):
            missing.append("mot.prefill_video_cache")
        return missing

    def __init__(
        self,
        model: Any,
        config: VideoTokenCacheConfig | None = None,
        *,
        video_state: VideoTokenState | None = None,
    ):
        super().__init__(model)
        self.config = config or VideoTokenCacheConfig()
        self._video_state = video_state or ensure_video_token_state(model)
        self._full_reuse = 0
        self._sparse_refresh = 0
        self._full_prefill = 0
        self._tokens_refreshed = 0
        self._tokens_total = 0
        self._reset_state()

    def _reset_state(self) -> None:
        self._last_cache: list[dict[str, torch.Tensor]] | None = None
        self._replan = 0
        self._consecutive_reuse = 0

    def reset(self) -> None:
        """Drop the cross-replan cache (call at episode boundaries)."""
        self._reset_state()
        self._video_state.reset()

    def _install(self) -> None:
        self._patch(self.model.mot, "prefill_video_cache", self._wrap_prefill)

    def _wrap_prefill(self, original: Callable) -> Callable:
        def prefill_video_cache(
            video_tokens: torch.Tensor,
            video_freqs: torch.Tensor,
            video_t_mod: torch.Tensor,
            video_context_payload=None,
            video_attention_mask: torch.Tensor | None = None,
            refresh_mask: torch.Tensor | None = None,
            previous_cache=None,
            **kwargs,
        ):
            owns_prefill_scope = self._video_state.owns_prefill_scope
            if owns_prefill_scope:
                self._video_state.enter_prefill(video_tokens)
            try:
                # Honour an explicit sparse request from a caller; otherwise decide.
                if refresh_mask is not None:
                    cache = original(
                        video_tokens,
                        video_freqs,
                        video_t_mod,
                        video_context_payload,
                        video_attention_mask,
                        refresh_mask=refresh_mask,
                        previous_cache=previous_cache,
                        **kwargs,
                    )
                    self._remember(video_tokens, cache)
                    self._full_prefill += 1
                    return cache

                cfg = self.config
                replan = self._replan
                self._replan += 1
                seq_len = int(video_tokens.shape[1])
                self._tokens_total += seq_len
                last_tokens = self._video_state.last_tokens

                must_full = (
                    last_tokens is None
                    or self._last_cache is None
                    or replan < cfg.cold_replans
                    or self._consecutive_reuse >= cfg.max_consecutive_reuses
                    or (cfg.force_refresh_every > 0 and replan > 0 and replan % cfg.force_refresh_every == 0)
                    or last_tokens.shape != video_tokens.shape
                    or last_tokens.device != video_tokens.device
                    or last_tokens.dtype != video_tokens.dtype
                )
                if must_full:
                    cache = original(
                        video_tokens,
                        video_freqs,
                        video_t_mod,
                        video_context_payload,
                        video_attention_mask,
                        **kwargs,
                    )
                    self._remember(video_tokens, cache)
                    self._consecutive_reuse = 0
                    self._full_prefill += 1
                    self._tokens_refreshed += seq_len
                    return cache

                global_drift = self._video_state.drift
                if global_drift < cfg.full_reuse_threshold:
                    self._full_reuse += 1
                    self._consecutive_reuse += 1
                    # Denoise only reads K/V; return the controller-owned cache
                    # without cloning every layer (see mot.forward_action_with_video_cache).
                    return self._last_cache

                per_token = token_relative_l1(video_tokens, last_tokens)
                refresh = per_token >= cfg.token_threshold
                num_tokens = refresh.numel()
                refresh_count = int(refresh.sum().item())
                max_refresh = int(cfg.max_refresh_ratio * num_tokens)
                if refresh_count > 0 and refresh_count <= max_refresh:
                    cache = original(
                        video_tokens,
                        video_freqs,
                        video_t_mod,
                        video_context_payload,
                        video_attention_mask,
                        refresh_mask=refresh,
                        previous_cache=self._last_cache,
                        **kwargs,
                    )
                    self._remember(video_tokens, cache)
                    self._consecutive_reuse = 0
                    self._sparse_refresh += 1
                    self._tokens_refreshed += refresh_count
                    return cache

                cache = original(
                    video_tokens,
                    video_freqs,
                    video_t_mod,
                    video_context_payload,
                    video_attention_mask,
                    **kwargs,
                )
                self._remember(video_tokens, cache)
                self._consecutive_reuse = 0
                self._full_prefill += 1
                self._tokens_refreshed += seq_len
                return cache
            finally:
                if owns_prefill_scope:
                    self._video_state.exit_prefill(video_tokens)

        return prefill_video_cache

    def _remember(self, video_tokens: torch.Tensor, cache: list[dict[str, torch.Tensor]]) -> None:
        self._last_cache = _clone_cache(cache)

    def _on_uninstall(self) -> None:
        self._reset_state()

    def stats(self) -> dict[str, Any]:
        decisions = self._full_reuse + self._sparse_refresh + self._full_prefill
        return {
            "full_reuse": self._full_reuse,
            "sparse_refresh": self._sparse_refresh,
            "full_prefill": self._full_prefill,
            "reuse_rate": (self._full_reuse / decisions) if decisions else 0.0,
            "sparse_rate": (self._sparse_refresh / decisions) if decisions else 0.0,
            "token_refresh_rate": (
                self._tokens_refreshed / self._tokens_total if self._tokens_total else 0.0
            ),
        }
