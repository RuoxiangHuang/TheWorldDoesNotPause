"""Cross-chunk residual reuse for the FastWAM action DiT.

Across consecutive replans the same denoising index k tends to see a highly
correlated block residual R_k = h^L - h^0 *when the action latent at k is also
similar*. Caching R_k and applying

    h^L ≈ h^0_new + R̃_k

lets a replan absorb fresh proprio / action-latent conditioning through
`pre_dit` while skipping the expensive MoT stack on a hit.

Install order places this controller *inside* `step_cache` and *outside*
`compile`:

- residual hits short-circuit in Python and never enter the compiled predict
  (avoids CUDA-graph thrash from intermittent reuse);
- steps that `step_cache` skips never reach this wrapper, so a held velocity
  cannot be paired with a stale residual (no double-staleness);
- residual misses call the (possibly compiled) original predict; R is captured
  by wrapping `mot.forward_action_with_video_cache` so compile still accelerates
  refresh steps.

Gates (all must pass):

- cold_chunks / force_refresh_every (paper τ) / max_consecutive_reuses /
  warm_steps / max_cache_step (paper skips the tail; -1 = no cap) /
  min_full_tail (always recompute the last T steps of this NFE)
- max_consecutive_reuses=0 means unlimited (paper τ=0 never refreshes)
- condition-side relative L1 on video tokens from prefill
- action-latent relative L1 vs the latent that produced R_k (blocks independent
  noise seeds from consuming a stale residual; paper schedule sets this huge)
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .base import Controller
from .config import ChunkResidualCacheConfig
from .step_cache import relative_l1_tensor
from .world_state import VideoTokenState, ensure_video_token_state


class ChunkResidualCacheController(Controller):
    name = "chunk_residual_cache"
    REQUIRED_ATTRS = ("infer_action", "_predict_action_noise_with_cache", "action_expert")

    @classmethod
    def missing_requirements(cls, model: Any, config: Any = None) -> list[str]:
        missing = super().missing_requirements(model, config)
        expert = getattr(model, "action_expert", None)
        if expert is not None:
            for attr in ("pre_dit", "post_dit"):
                if not hasattr(expert, attr):
                    missing.append(f"action_expert.{attr}")
        mot = getattr(model, "mot", None)
        if mot is None or not hasattr(mot, "forward_action_with_video_cache"):
            missing.append("mot.forward_action_with_video_cache")
        return missing

    def __init__(
        self,
        model: Any,
        config: ChunkResidualCacheConfig | None = None,
        *,
        video_state: VideoTokenState | None = None,
    ):
        super().__init__(model)
        self.config = config or ChunkResidualCacheConfig()
        self._video_state = video_state or ensure_video_token_state(model)
        self._full_compute = 0
        self._residual_hits = 0
        self._gate_blocked = 0
        self._latent_blocked = 0
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        """Drop cross-chunk tables (episode boundary / uninstall)."""
        self._residuals: dict[int, torch.Tensor] = {}
        self._residual_latents: dict[int, torch.Tensor] = {}
        self._consecutive_reuses: dict[int, int] = {}
        self._chunk = 0
        self._step = 0
        self._nfe: int | None = None
        self._reuse_allowed_this_chunk = False
        self._capture_residual = False
        self._pending_residual: torch.Tensor | None = None
        self._latent_threshold_tensor: torch.Tensor | None = None

    def _latent_threshold(self, latents_action: torch.Tensor) -> torch.Tensor:
        t = self._latent_threshold_tensor
        if t is None or t.device != latents_action.device:
            t = torch.tensor(
                self.config.action_latent_threshold,
                device=latents_action.device,
                dtype=torch.float32,
            )
            self._latent_threshold_tensor = t
        return t

    def reset(self) -> None:
        """Public episode-boundary reset."""
        self._reset_episode_state()
        self._video_state.reset()

    def _install(self) -> None:
        self._patch(self.model.mot, "forward_action_with_video_cache", self._wrap_forward)
        if hasattr(self.model.mot, "prefill_video_cache"):
            self._patch(self.model.mot, "prefill_video_cache", self._wrap_prefill)
        self._patch(self.model, "infer_action", self._wrap_infer_action)
        self._patch(self.model, "_predict_action_noise_with_cache", self._wrap_predict)

    def _wrap_forward(self, original: Callable) -> Callable:
        def forward_action_with_video_cache(action_tokens: torch.Tensor, *args, **kwargs):
            out = original(action_tokens, *args, **kwargs)
            if self._capture_residual:
                # R = h^L - h^0 for the MoT stack; clone so a later write cannot
                # mutate the table entry (CUDA-graph static buffers).
                self._pending_residual = (out - action_tokens).detach().clone()
            return out

        return forward_action_with_video_cache

    def _wrap_infer_action(self, original: Callable) -> Callable:
        def infer_action(*args, **kwargs):
            nfe = kwargs.get("num_inference_steps")
            if nfe is not None:
                nfe = int(nfe)
                if self._nfe is not None and nfe != self._nfe:
                    self._residuals.clear()
                    self._residual_latents.clear()
                    self._consecutive_reuses.clear()
                self._nfe = nfe

            self._step = 0
            chunk = self._chunk
            self._chunk += 1

            cfg = self.config
            force = (
                cfg.force_refresh_every > 0
                and chunk > 0
                and chunk % cfg.force_refresh_every == 0
            )
            # Condition drift is finalized in prefill when that hook runs.
            self._reuse_allowed_this_chunk = (
                chunk >= cfg.cold_chunks and not force and bool(self._residuals)
            )
            return original(*args, **kwargs)

        return infer_action

    def _wrap_prefill(self, original: Callable) -> Callable:
        def prefill_video_cache(video_tokens: torch.Tensor, *args, **kwargs):
            owns_prefill_scope = self._video_state.owns_prefill_scope
            if owns_prefill_scope:
                self._video_state.enter_prefill(video_tokens)
            try:
                condition_drift = self._video_state.drift

                cfg = self.config
                chunk_idx = max(self._chunk - 1, 0)
                force = (
                    cfg.force_refresh_every > 0
                    and chunk_idx > 0
                    and chunk_idx % cfg.force_refresh_every == 0
                )
                allowed = (
                    chunk_idx >= cfg.cold_chunks
                    and not force
                    and bool(self._residuals)
                    and condition_drift < cfg.condition_threshold
                )
                if (
                    not allowed
                    and chunk_idx >= cfg.cold_chunks
                    and not force
                    and condition_drift >= cfg.condition_threshold
                ):
                    self._gate_blocked += 1
                self._reuse_allowed_this_chunk = allowed

                return original(video_tokens, *args, **kwargs)
            finally:
                if owns_prefill_scope:
                    self._video_state.exit_prefill(video_tokens)

        return prefill_video_cache

    def _wrap_predict(self, original: Callable) -> Callable:
        def predict(
            *args,
            latents_action: torch.Tensor | None = None,
            timestep_action=None,
            context=None,
            context_mask=None,
            video_kv_cache=None,
            attention_mask=None,
            video_seq_len=None,
            **kwargs,
        ):
            if latents_action is None:
                if not args:
                    raise TypeError("chunk_residual_cache: `latents_action` is required")
                latents_action, args = args[0], args[1:]
            if timestep_action is None and args:
                timestep_action, args = args[0], args[1:]
            if context is None and args:
                context, args = args[0], args[1:]
            if context_mask is None and args:
                context_mask, args = args[0], args[1:]
            if video_kv_cache is None and args:
                video_kv_cache, args = args[0], args[1:]
            if attention_mask is None and args:
                attention_mask, args = args[0], args[1:]
            if video_seq_len is None and args:
                video_seq_len, args = args[0], args[1:]

            k = self._step
            self._step += 1

            can_reuse = self._can_reuse(k, latents_action)
            if can_reuse:
                self._residual_hits += 1
                self._consecutive_reuses[k] = self._consecutive_reuses.get(k, 0) + 1
                return self._thin_predict(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    step_index=k,
                )

            self._pending_residual = None
            self._capture_residual = True
            try:
                velocity = original(
                    *args,
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                    **kwargs,
                )
            finally:
                self._capture_residual = False
            if self._pending_residual is not None:
                self._residuals[k] = self._pending_residual
                self._residual_latents[k] = latents_action.detach().clone()
            self._pending_residual = None
            self._consecutive_reuses[k] = 0
            self._full_compute += 1
            return velocity

        return predict

    def _can_reuse(self, step_index: int, latents_action: torch.Tensor) -> bool:
        cfg = self.config
        if not self._reuse_allowed_this_chunk:
            return False
        if step_index not in self._residuals:
            return False
        if step_index < cfg.warm_steps:
            return False
        if cfg.max_cache_step >= 0 and step_index > cfg.max_cache_step:
            return False
        # Short schedules (RoboTwin NFE=4) sit entirely inside k<=6. Keep the
        # last T steps on the real MoT so the current frame still conditions
        # the large late flow updates.
        nfe = self._nfe
        if cfg.min_full_tail > 0 and nfe is not None and step_index >= nfe - cfg.min_full_tail:
            return False
        # 0 = unlimited, matching paper τ=0 (refresh only via force_refresh_every).
        if (
            cfg.max_consecutive_reuses > 0
            and self._consecutive_reuses.get(step_index, 0) >= cfg.max_consecutive_reuses
        ):
            return False
        ref = self._residual_latents.get(step_index)
        if ref is None:
            return False
        # Action-latent gate: independent noise seeds make R non-transferable.
        if latents_action.shape != ref.shape:
            return False
        if ref.device != latents_action.device or ref.dtype != latents_action.dtype:
            ref = ref.to(device=latents_action.device, dtype=latents_action.dtype)
        if bool(
            (relative_l1_tensor(latents_action.detach(), ref) >= self._latent_threshold(latents_action)).item()
        ):
            self._latent_blocked += 1
            return False
        return True

    def _thin_predict(
        self,
        latents_action: torch.Tensor,
        timestep_action,
        context,
        context_mask,
        step_index: int,
    ) -> torch.Tensor:
        expert = self.model.action_expert
        # Thin path needs the real pre_dit / post_dit; unit-test stubs may omit
        # context — fall through is not expected on a reuse decision.
        action_pre = expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        residual = self._residuals[step_index]
        h0 = action_pre["tokens"]
        hL = h0 + residual.to(device=h0.device, dtype=h0.dtype)
        return expert.post_dit(hL, action_pre)

    def _on_uninstall(self) -> None:
        self._reset_episode_state()

    def stats(self) -> dict[str, Any]:
        total = self._full_compute + self._residual_hits
        return {
            "full_compute": self._full_compute,
            "residual_hits": self._residual_hits,
            "hit_rate": (self._residual_hits / total) if total else 0.0,
            "gate_blocked": self._gate_blocked,
            "latent_blocked": self._latent_blocked,
            "cached_steps": len(self._residuals),
            "chunks": self._chunk,
        }
