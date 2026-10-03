"""FasterPI: FastWAM-style training-free accelerate stack for openpi pi0.5 (PI0Pytorch).

Controllers (install order: restore_eager → graph → sdpa → compile_prefix → compile → vision → d0 → prefix_kv/split_prefix → fuse_loop → chunk_residual_cache → step_cache → adaptive_nfe).

Default FasterPI spec: compile+chunk_residual_cache+step_cache+compile_prefix (alias ``fasterpi``).
Adaptive NFE and the visual-encoder cache (``vision``) stay available as dirs
and are not in the default.
Previous defaults: compile+chunk_residual_cache+vision+adaptive_nfe+step_cache (``fasterpi_v2``),
compile+chunk_residual_cache+vision (``fasterpi_v1``).
Older spec tokens ``c3`` and ``d1`` are accepted as aliases of
``chunk_residual_cache`` and ``step_cache``.

  compile      — torch.compile(denoise_step) when chunk_residual_cache is off; with it,
                 compile the expert body inside ChunkResidual instead.
  graph        — inductor CUDA graphs + cudagraph_mark_step_begin on compiled
                 denoise (and compiled SigLIP if compile_prefix is on). Same
                 images / full NFE; no cache approximation. Needs a single
                 infer thread (serial HTTP). Default compile evals keep graphs off.
  vision       — cache SigLIP image embeddings when RGB barely moved (vae_cache analog).
  d0           — cache language-token embeddings (usually noop on pi0.5).
  prefix_kv    — sparse refresh of prefix past_key_values across replans (default off
                 unless requested; conservative gate).
  chunk_residual_cache — cross-replan residual R=h^L-h^0 at the same denoise index.
  step_cache   — hold last velocity while x_t drift stays small.
  compile_prefix — torch.compile(SigLIP embed_image); batch images when vision is off.

obs_pipeline is a runner-side helper (not installed on the model).
"""

from __future__ import annotations

import concurrent.futures
import math
import os
from typing import Any, Iterable

import torch

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

from fasterpi.world_state import get_world_state


def rel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.abs().mean().clamp(min=1e-6)
    return float((a - b).abs().mean() / denom)


def cosine_delta_vec(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    an = torch.linalg.vector_norm(a).clamp(min=1e-8)
    bn = torch.linalg.vector_norm(b).clamp(min=1e-8)
    sim = float(((a / an) * (b / bn)).sum().clamp(-1.0, 1.0).item())
    return 1.0 - sim


def _world_gate_on(model) -> bool:
    return bool(getattr(model, "_fasterpi_world_gate", False))


def _attn_impl(model) -> str:
    return str(getattr(model, "_fasterpi_attn_impl", "eager") or "eager")


def _world_allows(model, kind: str) -> bool:
    """Extra permission for caches. Off → existing RGB/state gates only."""
    if not _world_gate_on(model):
        return True
    ws = get_world_state()
    if ws is None:
        return False
    return bool(ws.cache_ok(kind))


def restore_eager(model):
    """Undo openpi's built-in torch.compile(sample_actions) and any FasterPI patches."""
    cls = type(model)
    model.sample_actions = cls.sample_actions.__get__(model, cls)
    model.denoise_step = cls.denoise_step.__get__(model, cls)
    if hasattr(cls, "embed_prefix"):
        model.embed_prefix = cls.embed_prefix.__get__(model, cls)
    pg = getattr(model, "paligemma_with_expert", None)
    if pg is not None:
        pg_cls = type(pg)
        if hasattr(pg_cls, "forward"):
            pg.forward = pg_cls.forward.__get__(pg, pg_cls)
        if hasattr(pg_cls, "embed_language_tokens"):
            pg.embed_language_tokens = pg_cls.embed_language_tokens.__get__(pg, pg_cls)
        if hasattr(pg_cls, "embed_image"):
            pg.embed_image = pg_cls.embed_image.__get__(pg, pg_cls)
        lm = getattr(getattr(pg, "paligemma", None), "language_model", None)
        if lm is not None and hasattr(type(lm), "forward"):
            lm.forward = type(lm).forward.__get__(lm, type(lm))
    model._fasterpi_world_gate = False
    model._fasterpi_adaptive_replan = False
    model._fasterpi_adaptive_nfe = False
    model._fasterpi_host = False
    model._fasterpi_attn_impl = "eager"
    model._fasterpi_cudagraphs = False
    return model


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------


class CompileController:
    """Compile per-step denoise. With ``graph``, also capture inductor CUDA graphs."""

    def __init__(self, model, mode: str = "reduce-overhead"):
        if mode not in ("reduce-overhead", "default", "max-autotune"):
            raise ValueError(mode)
        self.model = model
        self.mode = mode
        self._orig = None
        try:
            import torch._dynamo as dynamo

            dynamo.config.suppress_errors = True
        except Exception:  # noqa: BLE001
            pass

    def install(self):
        if self._orig is not None:
            return self
        self._orig = self.model.denoise_step
        compiled = torch.compile(self._orig, mode=self.mode)
        use_graph = bool(getattr(self.model, "_fasterpi_cudagraphs", False))

        def denoise_compiled(*args, **kw):
            if use_graph:
                torch.compiler.cudagraph_mark_step_begin()
            return compiled(*args, **kw).clone()

        self.model.denoise_step = denoise_compiled
        print(
            f"[compile] denoise_step compiled (mode={self.mode} cudagraphs={use_graph})",
            flush=True,
        )
        return self

    def uninstall(self):
        if self._orig is not None:
            self.model.denoise_step = self._orig
            self._orig = None


# ---------------------------------------------------------------------------
# step_cache — within-chunk velocity hold
# ---------------------------------------------------------------------------


class StepCacheController:
    """Hold the last velocity while accumulated x_t drift stays under threshold."""

    def __init__(
        self,
        model,
        threshold: float = 0.20,
        max_consecutive_skips: int = 2,
        cold_steps: int = 2,
    ):
        self.model = model
        self.threshold = float(threshold)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.cold_steps = int(cold_steps)
        self._orig_denoise = None
        self._orig_sample = None
        self._nfe = 10
        self.reset_state()
        self.stats = {"computed": 0, "skipped": 0, "force_refresh": 0}

    def reset_state(self):
        self._last_input = None
        self._last_pred = None
        self._acc = 0.0
        self._consec = 0
        self._step_idx = 0

    def install(self):
        if self._orig_denoise is not None:
            return self
        self._orig_denoise = self.model.denoise_step
        self._orig_sample = self.model.sample_actions
        ctrl = self

        def denoise_cached(*args, **kw):
            x = args[3]
            timestep = args[4] if len(args) > 4 else kw.get("timestep")
            # Prefer timestep-derived index so skips stay aligned with diffusion k.
            if timestep is not None and ctrl._nfe > 0:
                t = float(timestep.detach().float().reshape(-1)[0].item())
                step = int(round((1.0 - t) * ctrl._nfe))
                step = max(0, min(ctrl._nfe - 1, step))
            else:
                step = ctrl._step_idx
            ctrl._step_idx = step + 1

            if ctrl._last_input is not None:
                ctrl._acc += rel_l1(x, ctrl._last_input)
            signal_ok = ctrl._acc < ctrl.threshold

            can_skip = (
                step >= ctrl.cold_steps
                and ctrl._last_pred is not None
                and ctrl._consec < ctrl.max_consecutive_skips
                and signal_ok
                and _world_allows(ctrl.model, "step_cache")
            )
            if can_skip:
                ctrl._consec += 1
                ctrl.stats["skipped"] += 1
                ctrl._last_input = x.clone()
                return ctrl._last_pred

            if (
                step >= ctrl.cold_steps
                and ctrl._last_pred is not None
                and ctrl._consec >= ctrl.max_consecutive_skips
                and signal_ok
                and _world_allows(ctrl.model, "step_cache")
            ):
                ctrl.stats["force_refresh"] += 1

            pred = ctrl._orig_denoise(*args, **kw)
            ctrl.stats["computed"] += 1
            ctrl._last_input = x.clone()
            ctrl._last_pred = pred
            ctrl._acc = 0.0
            ctrl._consec = 0
            return pred

        def sample_with_reset(*a, **k):
            ctrl.reset_state()
            if "num_steps" in k:
                ctrl._nfe = int(k["num_steps"])
            elif len(a) >= 4 and a[3] is not None:
                ctrl._nfe = int(a[3])
            return ctrl._orig_sample(*a, **k)

        self.model.denoise_step = denoise_cached
        self.model.sample_actions = sample_with_reset
        print(
            f"[step_cache] installed (threshold={self.threshold}, "
            f"max_skip={self.max_consecutive_skips}, cold={self.cold_steps})"
        )
        return self

    def uninstall(self):
        if self._orig_denoise is not None:
            self.model.denoise_step = self._orig_denoise
            self.model.sample_actions = self._orig_sample
            self._orig_denoise = None
            self._orig_sample = None


# ---------------------------------------------------------------------------
# d0 — text cache
# ---------------------------------------------------------------------------


class TextCacheController:
    """Exact cache of language-token embeddings across replans."""

    def __init__(self, model, max_entries: int = 64):
        self.model = model
        self.max_entries = max_entries
        self._cache: dict[tuple, torch.Tensor] = {}
        self._orig = None
        self.stats = {"hits": 0, "misses": 0}

    def install(self):
        pg = self.model.paligemma_with_expert
        if self._orig is not None:
            return self
        self._orig = pg.embed_language_tokens
        ctrl = self

        @torch.no_grad()
        def embed_cached(lang_tokens):
            key = tuple(lang_tokens.flatten().tolist())
            hit = ctrl._cache.get(key)
            if hit is not None:
                ctrl.stats["hits"] += 1
                return hit.clone()
            ctrl.stats["misses"] += 1
            emb = ctrl._orig(lang_tokens)
            if len(ctrl._cache) >= ctrl.max_entries:
                ctrl._cache.pop(next(iter(ctrl._cache)))
            ctrl._cache[key] = emb.detach()
            return emb

        pg.embed_language_tokens = embed_cached
        print("[d0] language-embedding cache installed")
        return self

    def uninstall(self):
        if self._orig is not None:
            self.model.paligemma_with_expert.embed_language_tokens = self._orig
            self._orig = None


# ---------------------------------------------------------------------------
# vision — SigLIP embed cache (vae_cache analog)
# ---------------------------------------------------------------------------


class VisionCacheController:
    """Reuse vision-tower embeddings when RGB relative L1 stays under threshold."""

    def __init__(self, model, threshold: float = 0.02, max_entries: int = 8):
        self.model = model
        self.threshold = float(threshold)
        self.max_entries = int(max_entries)
        self._orig = None
        self._slots: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.stats = {"hits": 0, "misses": 0}

    def install(self):
        pg = self.model.paligemma_with_expert
        if self._orig is not None:
            return self
        self._orig = pg.embed_image
        ctrl = self

        @torch.no_grad()
        def embed_cached(img):
            if not _world_allows(ctrl.model, "vision"):
                ctrl.stats["misses"] += 1
                emb = ctrl._orig(img)
                ctrl._slots.append((img.detach().clone(), emb.detach().clone()))
                if len(ctrl._slots) > ctrl.max_entries:
                    ctrl._slots.pop(0)
                return emb
            for prev_img, prev_emb in ctrl._slots:
                if prev_img.shape == img.shape and rel_l1(img, prev_img) < ctrl.threshold:
                    ctrl.stats["hits"] += 1
                    return prev_emb.clone()
            ctrl.stats["misses"] += 1
            emb = ctrl._orig(img)
            ctrl._slots.append((img.detach().clone(), emb.detach().clone()))
            if len(ctrl._slots) > ctrl.max_entries:
                ctrl._slots.pop(0)
            return emb

        pg.embed_image = embed_cached
        print(f"[vision] image-embedding cache installed (thr={self.threshold})")
        return self

    def reset(self):
        self._slots.clear()
        self.stats = {"hits": 0, "misses": 0}

    def uninstall(self):
        if self._orig is not None:
            self.model.paligemma_with_expert.embed_image = self._orig
            self._orig = None
            self.reset()


# ---------------------------------------------------------------------------
# prefix_kv — sparse prefix KV refresh (video_token_cache analog, conservative)
# ---------------------------------------------------------------------------


class PrefixKVCacheController:
    """Reuse image+language prefix KV. State always comes from the current step.

    ``gate=rgb_l1`` compares raw images to the previous frame (legacy).
    ``gate=siglip_anchor`` compares SigLIP mean-pool cosine to the cache source
    (CLIRA condition-refresh). Age is ``max_consecutive_reuses``.
    """

    def __init__(
        self,
        model,
        image_threshold: float = 0.01,
        state_threshold: float = 0.02,
        cold_chunks: int = 1,
        max_consecutive_reuses: int = 2,
        gate: str = "rgb_l1",
    ):
        self.model = model
        self.image_threshold = float(image_threshold)
        self.state_threshold = float(state_threshold)
        self.cold_chunks = int(cold_chunks)
        self.max_consecutive_reuses = int(max_consecutive_reuses)
        self.gate = str(gate or "rgb_l1")
        self._orig_sample = None
        self._chunk = 0
        self._consec = 0
        self._cached_kv = None
        self._cached_prefix_pad = None
        self._last_images: list[torch.Tensor] | None = None
        self._last_state: torch.Tensor | None = None
        self._anchor_phis: list[torch.Tensor] | None = None
        self.stats = {"hits": 0, "misses": 0, "last_delta": None}

    def reset(self):
        self._chunk = 0
        self._consec = 0
        self._cached_kv = None
        self._cached_prefix_pad = None
        self._last_images = None
        self._last_state = None
        self._anchor_phis = None
        self.stats["last_delta"] = None

    def _image_drift(self, images) -> float:
        if self._last_images is None:
            return float("inf")
        drifts = []
        for a, b in zip(images, self._last_images, strict=False):
            if a.shape != b.shape:
                return float("inf")
            drifts.append(rel_l1(a, b))
        return max(drifts) if drifts else float("inf")

    def _siglip_phis(self, images, img_masks) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pg = self.model.paligemma_with_expert
        embs, phis = [], []
        for img, mask in zip(images, img_masks, strict=True):
            emb = pg.embed_image(img)
            embs.append(emb)
            if mask.ndim == 1 and not bool(mask.reshape(-1)[0].item()):
                continue
            phis.append(emb.float().mean(dim=1).reshape(-1).detach())
        return embs, phis

    def _delta_anchor(self, phis: list[torch.Tensor]) -> float:
        if not phis or self._anchor_phis is None or len(self._anchor_phis) != len(phis):
            return float("inf")
        return max(cosine_delta_vec(a, b) for a, b in zip(phis, self._anchor_phis))

    def _prefix_from_embs(self, img_embs, img_masks, lang_tokens, lang_masks):
        model = self.model
        pg = model.paligemma_with_expert
        packed, pad_masks, att_list = [], [], []
        for img_emb, img_mask in zip(img_embs, img_masks, strict=True):
            bsize, ntok = img_emb.shape[:2]
            packed.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, ntok))
            att_list += [0] * ntok
        lang_emb = pg.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(float(lang_emb.shape[-1]))
        packed.append(lang_emb)
        pad_masks.append(lang_masks)
        att_list += [0] * lang_emb.shape[1]
        prefix_embs = torch.cat(packed, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_list, dtype=torch.bool, device=prefix_pad_masks.device)
        att_masks = att_masks[None, :].expand(prefix_pad_masks.shape[0], -1)
        att_2d = make_att_2d_masks(prefix_pad_masks, att_masks)
        pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = model._prepare_attention_masks_4d(att_2d)
        pg.paligemma.language_model.config._attn_implementation = _attn_impl(model)
        _, past = pg.forward(
            attention_mask=att_4d,
            position_ids=pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return past, prefix_pad_masks

    def install(self):
        if self._orig_sample is not None:
            return self
        self._orig_sample = self.model.sample_actions
        ctrl = self
        model = self.model
        siglip_gate = ctrl.gate == "siglip_anchor"

        @torch.no_grad()
        def sample_with_prefix_kv(device, observation, noise=None, num_steps=10):
            bsize = observation.state.shape[0]
            if noise is None:
                actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
                noise = model.sample_noise(actions_shape, device)

            images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
                observation, train=False
            )

            img_embs = None
            phis = None
            if siglip_gate:
                img_embs, phis = ctrl._siglip_phis(images, img_masks)
                delta = ctrl._delta_anchor(phis)
                ctrl.stats["last_delta"] = delta
                can_reuse = (
                    ctrl._chunk >= ctrl.cold_chunks
                    and ctrl._cached_kv is not None
                    and ctrl._consec < ctrl.max_consecutive_reuses
                    and delta <= ctrl.image_threshold
                    and _world_allows(model, "prefix_kv")
                )
            else:
                img_drift = ctrl._image_drift(images)
                state_drift = (
                    rel_l1(state, ctrl._last_state) if ctrl._last_state is not None else float("inf")
                )
                ctrl.stats["last_delta"] = img_drift
                can_reuse = (
                    ctrl._chunk >= ctrl.cold_chunks
                    and ctrl._cached_kv is not None
                    and ctrl._consec < ctrl.max_consecutive_reuses
                    and img_drift < ctrl.image_threshold
                    and state_drift < ctrl.state_threshold
                    and _world_allows(model, "prefix_kv")
                )

            if can_reuse:
                ctrl.stats["hits"] += 1
                ctrl._consec += 1
                past_key_values = ctrl._cached_kv
                prefix_pad_masks = ctrl._cached_prefix_pad
            else:
                ctrl.stats["misses"] += 1
                ctrl._consec = 0
                if siglip_gate:
                    past_key_values, prefix_pad_masks = ctrl._prefix_from_embs(
                        img_embs, img_masks, lang_tokens, lang_masks
                    )
                    ctrl._anchor_phis = [p.detach().clone() for p in phis]
                else:
                    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                        images, img_masks, lang_tokens, lang_masks
                    )
                    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
                    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
                        _attn_impl(model)  # noqa: SLF001
                    )
                    _, past_key_values = model.paligemma_with_expert.forward(
                        attention_mask=prefix_att_2d_masks_4d,
                        position_ids=prefix_position_ids,
                        past_key_values=None,
                        inputs_embeds=[prefix_embs, None],
                        use_cache=True,
                    )
                ctrl._cached_kv = past_key_values
                ctrl._cached_prefix_pad = prefix_pad_masks

            ctrl._last_images = [im.detach().clone() for im in images]
            ctrl._last_state = state.detach().clone()
            ctrl._chunk += 1

            dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                v_t = model.denoise_step(
                    state, prefix_pad_masks, past_key_values, x_t, expanded_time
                )
                x_t = x_t + dt * v_t
                time = time + dt
            return x_t

        self.model.sample_actions = sample_with_prefix_kv
        print(
            f"[prefix_kv] installed (gate={self.gate}, img_thr={self.image_threshold}, "
            f"state_thr={self.state_threshold}, cold={self.cold_chunks}, "
            f"Amax={self.max_consecutive_reuses})"
        )
        return self

    def uninstall(self):
        if self._orig_sample is not None:
            self.model.sample_actions = self._orig_sample
            self._orig_sample = None
            self.reset()


# ---------------------------------------------------------------------------
# chunk_residual_cache — cross-replan residual
# ---------------------------------------------------------------------------


class ChunkResidualCacheController:
    """Residual reuse: h^L ≈ h^0_new + R̃_k across replans at denoise index k."""

    def __init__(
        self,
        model,
        *,
        condition_threshold: float = 0.05,
        latent_threshold: float = 0.15,
        cold_chunks: int = 1,
        max_consecutive_reuses: int = 4,
        compile_mode: str | None = "reduce-overhead",
    ):
        self.model = model
        self.condition_threshold = float(condition_threshold)
        self.latent_threshold = float(latent_threshold)
        self.cold_chunks = int(cold_chunks)
        self.max_consecutive_reuses = int(max_consecutive_reuses)
        self.compile_mode = compile_mode
        self._orig_denoise = None
        self._orig_sample = None
        self._expert_fn = None
        self._nfe = 10
        self._reset_tables()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "gate_blocked": 0,
            "latent_blocked": 0,
            "force_refresh": 0,
        }

    def _reset_tables(self):
        self._residuals: dict[int, torch.Tensor] = {}
        self._residual_latents: dict[int, torch.Tensor] = {}
        self._consecutive_reuses: dict[int, int] = {}
        self._chunk = 0
        self._reuse_allowed = False
        self._last_state: torch.Tensor | None = None

    def reset(self):
        self._reset_tables()

    def _step_index(self, timestep) -> int:
        t = float(timestep.detach().float().reshape(-1)[0].item())
        k = int(round((1.0 - t) * self._nfe))
        return max(0, min(self._nfe - 1, k))

    def _build_expert_fn(self):
        model = self.model

        def expert_body(
            suffix_embs,
            full_att_2d_masks_4d,
            position_ids,
            past_key_values,
            adarms_cond,
        ):
            model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = _attn_impl(
                model
            )  # noqa: SLF001
            outputs_embeds, _ = model.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return outputs_embeds[1]

        if self.compile_mode:
            try:
                import torch._dynamo as dynamo

                dynamo.config.suppress_errors = True
            except Exception:  # noqa: BLE001
                pass
            expert_body = torch.compile(expert_body, mode=self.compile_mode)
            print(f"[chunk_residual_cache] expert body compiled (mode={self.compile_mode})")
        return expert_body

    def install(self):
        if self._orig_denoise is not None:
            return self
        self._orig_denoise = self.model.denoise_step
        self._orig_sample = self.model.sample_actions
        self._expert_fn = self._build_expert_fn()
        ctrl = self
        model = self.model
        horizon = model.config.action_horizon

        def denoise_residual(state, prefix_pad_masks, past_key_values, x_t, timestep):
            k = ctrl._step_index(timestep)

            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(
                state, x_t, timestep
            )
            if (
                model.paligemma_with_expert.paligemma.language_model.layers[0]
                .self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

            can_hit = (
                ctrl._reuse_allowed
                and k in ctrl._residuals
                and ctrl._consecutive_reuses.get(k, 0) < ctrl.max_consecutive_reuses
            )
            if can_hit:
                prev_x = ctrl._residual_latents.get(k)
                if prev_x is not None and rel_l1(x_t, prev_x) >= ctrl.latent_threshold:
                    ctrl.stats["latent_blocked"] += 1
                    can_hit = False

            forced_by_cap = (
                ctrl._reuse_allowed
                and k in ctrl._residuals
                and ctrl._consecutive_reuses.get(k, 0) >= ctrl.max_consecutive_reuses
            )
            if forced_by_cap:
                ctrl.stats["force_refresh"] += 1

            if can_hit:
                ctrl.stats["hits"] += 1
                ctrl._consecutive_reuses[k] = ctrl._consecutive_reuses.get(k, 0) + 1
                hl = suffix_embs + ctrl._residuals[k].to(dtype=suffix_embs.dtype)
                suffix_out = hl[:, -horizon:].to(dtype=torch.float32)
                return model.action_out_proj(suffix_out)

            # full expert path
            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = model._prepare_attention_masks_4d(full_att_2d_masks)

            hl = ctrl._expert_fn(
                suffix_embs, full_att_2d_masks_4d, position_ids, past_key_values, adarms_cond
            )
            # clone to detach from CUDA-graph static buffers
            R = (hl - suffix_embs).detach().clone()
            ctrl._residuals[k] = R
            ctrl._residual_latents[k] = x_t.detach().clone()
            ctrl._consecutive_reuses[k] = 0
            ctrl.stats["misses"] += 1

            suffix_out = hl[:, -horizon:].to(dtype=torch.float32)
            return model.action_out_proj(suffix_out)

        def sample_with_gate(*args, **kwargs):
            if "num_steps" in kwargs:
                ctrl._nfe = int(kwargs["num_steps"])
            elif len(args) >= 4 and args[3] is not None:
                ctrl._nfe = int(args[3])
            # peek state from observation when available for condition gate
            observation = args[1] if len(args) > 1 else kwargs.get("observation")
            state = None
            if observation is not None and hasattr(observation, "state"):
                state = observation.state
            if state is not None and ctrl._last_state is not None:
                drift = rel_l1(state, ctrl._last_state)
                ctrl._reuse_allowed = (
                    ctrl._chunk >= ctrl.cold_chunks
                    and drift < ctrl.condition_threshold
                    and _world_allows(model, "chunk_residual_cache")
                )
                if not ctrl._reuse_allowed and ctrl._chunk >= ctrl.cold_chunks:
                    ctrl.stats["gate_blocked"] += 1
            else:
                ctrl._reuse_allowed = ctrl._chunk >= ctrl.cold_chunks and _world_allows(
                    model, "chunk_residual_cache"
                )
            if state is not None:
                ctrl._last_state = state.detach().clone()
            ctrl._chunk += 1
            return ctrl._orig_sample(*args, **kwargs)

        self.model.denoise_step = denoise_residual
        self.model.sample_actions = sample_with_gate
        print(
            f"[chunk_residual_cache] installed (cond_thr={self.condition_threshold}, "
            f"latent_thr={self.latent_threshold}, cold={self.cold_chunks})"
        )
        return self

    def uninstall(self):
        if self._orig_denoise is not None:
            self.model.denoise_step = self._orig_denoise
            self.model.sample_actions = self._orig_sample
            self._orig_denoise = None
            self._orig_sample = None
            self._expert_fn = None
            self._reset_tables()


# ---------------------------------------------------------------------------
# split_prefix — per-stream image embed + language freeze + optional KV skip
# ---------------------------------------------------------------------------


class SplitPrefixController:
    """Dual-rate prefix: language frozen; agentview vs wrist refreshed independently.

    When every stream is reused, skip the PaliGemma prefix forward (KV reuse).
    World-gate (if on): agentview follows object motion; wrist follows eef proximity.
    """

    def __init__(
        self,
        model,
        image_threshold: float = 0.02,
        cold_chunks: int = 1,
        max_consecutive_kv: int = 4,
    ):
        self.model = model
        self.image_threshold = float(image_threshold)
        self.cold_chunks = int(cold_chunks)
        self.max_consecutive_kv = int(max_consecutive_kv)
        self._orig_sample = None
        self.reset()
        self.stats = {
            "kv_hits": 0,
            "kv_misses": 0,
            "img_hits": 0,
            "img_misses": 0,
            "lang_hits": 0,
            "lang_misses": 0,
        }

    def reset(self):
        self._chunk = 0
        self._kv_consec = 0
        self._cached_kv = None
        self._cached_prefix_pad = None
        self._img_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        self._lang_key = None
        self._lang_emb = None

    def _img_kind(self, idx: int, n_img: int) -> str:
        if n_img <= 1 or idx == 0:
            return "agentview"
        return "wrist"

    def _may_reuse_image(self, idx: int, img: torch.Tensor, n_img: int) -> bool:
        if not _world_allows(self.model, self._img_kind(idx, n_img)):
            return False
        if idx >= len(self._img_cache) or self._img_cache[idx] is None:
            return False
        prev, _ = self._img_cache[idx]
        return prev.shape == img.shape and rel_l1(img, prev) < self.image_threshold

    def install(self):
        if self._orig_sample is not None:
            return self
        self._orig_sample = self.model.sample_actions
        ctrl = self
        model = self.model

        @torch.no_grad()
        def sample_with_split_prefix(device, observation, noise=None, num_steps=10):
            bsize = observation.state.shape[0]
            if noise is None:
                actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
                noise = model.sample_noise(actions_shape, device)

            images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
                observation, train=False
            )
            n_img = len(images)
            while len(ctrl._img_cache) < n_img:
                ctrl._img_cache.append(None)

            embs = []
            pad_masks = []
            att_masks: list[int] = []
            all_img_reused = True
            for i, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):
                if ctrl._may_reuse_image(i, img, n_img):
                    img_emb = ctrl._img_cache[i][1].clone()
                    ctrl.stats["img_hits"] += 1
                else:
                    img_emb = model.paligemma_with_expert.embed_image(img)
                    ctrl._img_cache[i] = (img.detach().clone(), img_emb.detach().clone())
                    ctrl.stats["img_misses"] += 1
                    all_img_reused = False
                bsize_i, num_img_embs = img_emb.shape[:2]
                embs.append(img_emb)
                pad_masks.append(img_mask[:, None].expand(bsize_i, num_img_embs))
                att_masks += [0] * num_img_embs

            lang_key = tuple(lang_tokens.detach().flatten().tolist())
            if ctrl._lang_emb is not None and ctrl._lang_key == lang_key:
                lang_emb = ctrl._lang_emb.clone()
                ctrl.stats["lang_hits"] += 1
            else:
                lang_emb = model.paligemma_with_expert.embed_language_tokens(lang_tokens)
                lang_emb = lang_emb * (lang_emb.shape[-1] ** 0.5)
                ctrl._lang_key = lang_key
                ctrl._lang_emb = lang_emb.detach().clone()
                ctrl.stats["lang_misses"] += 1
            embs.append(lang_emb)
            pad_masks.append(lang_masks)
            att_masks += [0] * lang_emb.shape[1]

            prefix_embs = torch.cat(embs, dim=1)
            prefix_pad_masks = torch.cat(pad_masks, dim=1)
            att = torch.tensor(att_masks, dtype=torch.bool, device=prefix_pad_masks.device)
            prefix_att_masks = att[None, :].expand(prefix_pad_masks.shape[0], att.shape[0])

            can_kv = (
                ctrl._chunk >= ctrl.cold_chunks
                and ctrl._cached_kv is not None
                and ctrl._kv_consec < ctrl.max_consecutive_kv
                and all_img_reused
                and _world_allows(model, "prefix_kv")
            )
            if can_kv:
                ctrl.stats["kv_hits"] += 1
                ctrl._kv_consec += 1
                past_key_values = ctrl._cached_kv
                prefix_pad_masks = ctrl._cached_prefix_pad
            else:
                ctrl.stats["kv_misses"] += 1
                ctrl._kv_consec = 0
                prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
                model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
                    _attn_impl(model)  # noqa: SLF001
                )
                _, past_key_values = model.paligemma_with_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                )
                ctrl._cached_kv = past_key_values
                ctrl._cached_prefix_pad = prefix_pad_masks

            ctrl._chunk += 1
            dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while time >= -dt / 2:
                expanded_time = time.expand(bsize)
                v_t = model.denoise_step(
                    state, prefix_pad_masks, past_key_values, x_t, expanded_time
                )
                x_t = x_t + dt * v_t
                time = time + dt
            return x_t

        self.model.sample_actions = sample_with_split_prefix
        print(
            f"[split_prefix] installed (img_thr={self.image_threshold}, cold={self.cold_chunks})"
        )
        return self

    def uninstall(self):
        if self._orig_sample is not None:
            self.model.sample_actions = self._orig_sample
            self._orig_sample = None
            self.reset()


# ---------------------------------------------------------------------------
# adaptive_nfe — world-state diffusion step count
# ---------------------------------------------------------------------------


class AdaptiveNFEController:
    """Lower denoise steps when the world is slow / object is held."""

    def __init__(self, model, default_steps: int = 10):
        self.model = model
        self.default_steps = int(default_steps)
        self._orig_sample = None
        self.stats = {"calls": 0, "nfe_sum": 0}

    def install(self):
        if self._orig_sample is not None:
            return self
        self._orig_sample = self.model.sample_actions
        ctrl = self

        def sample_adaptive(device, observation, noise=None, num_steps=10, **kw):
            ws = get_world_state()
            nfe = int(num_steps) if num_steps is not None else ctrl.default_steps
            if ws is not None:
                nfe = int(ws.recommended_nfe(nfe))
            ctrl.stats["calls"] += 1
            ctrl.stats["nfe_sum"] += nfe
            return ctrl._orig_sample(device, observation, noise=noise, num_steps=nfe, **kw)

        self.model.sample_actions = sample_adaptive
        print(f"[adaptive_nfe] installed (default={self.default_steps})")
        return self

    def reset(self):
        self.stats = {"calls": 0, "nfe_sum": 0}

    def uninstall(self):
        if self._orig_sample is not None:
            self.model.sample_actions = self._orig_sample
            self._orig_sample = None
            self.reset()


# ---------------------------------------------------------------------------
# obs_pipeline — runner helper
# ---------------------------------------------------------------------------


class ObsPipeline:
    """Async observation prefetch for eval runners (lossless, model-agnostic)."""

    def __init__(self, max_workers: int = 1):
        self._ex = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._fut: concurrent.futures.Future | None = None
        self.stats = {"submitted": 0, "taken": 0}

    def submit(self, fn, *args, **kwargs):
        self._fut = self._ex.submit(fn, *args, **kwargs)
        self.stats["submitted"] += 1
        return self._fut

    def take(self, timeout: float | None = None):
        if self._fut is None:
            raise RuntimeError("ObsPipeline.take() called before submit()")
        out = self._fut.result(timeout=timeout)
        self.stats["taken"] += 1
        self._fut = None
        return out

    def shutdown(self):
        self._ex.shutdown(wait=False)


# ---------------------------------------------------------------------------
# sdpa — override openpi's forced eager attention
# ---------------------------------------------------------------------------


class SdpaController:
    """Force SDPA (or FlashAttention-2) on every PaliGemma/expert forward.

    openpi writes ``_attn_implementation = "eager"`` immediately before forward;
    wrapping ``forward`` reapplies SDPA after that assignment.

    Do not stack with chunk_residual_cache ``reduce-overhead`` CUDA graphs: attention masks change
    shape across denoise steps and recapture dozens of graphs (measured ~30×
    slower than eager attention on LIBERO t00.pick).
    """

    def __init__(self, model, impl: str | None = None):
        self.model = model
        self.impl = impl or "sdpa"
        self._orig = None
        self.stats = {"forwards": 0, "impl": self.impl}

    def install(self):
        pg = self.model.paligemma_with_expert
        if self._orig is not None:
            return self
        self._orig = pg.forward
        impl = self.impl
        self.model._fasterpi_attn_impl = impl
        orig = self._orig
        ctrl = self

        def forward_sdpa(*args, **kwargs):
            pg.paligemma.language_model.config._attn_implementation = impl  # noqa: SLF001
            pg.gemma_expert.model.config._attn_implementation = impl  # noqa: SLF001
            vision = getattr(pg.paligemma, "vision_tower", None)
            if vision is not None and hasattr(vision, "config"):
                vision.config._attn_implementation = impl  # noqa: SLF001
            qdtype = pg.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            mask = kwargs.get("attention_mask")
            if mask is None and args:
                mask = args[0] if len(args) > 0 else None
            if mask is not None and torch.is_tensor(mask) and mask.dtype != qdtype:
                mask = mask.to(dtype=qdtype)
                if "attention_mask" in kwargs:
                    kwargs["attention_mask"] = mask
                elif args:
                    args = (mask,) + args[1:]
            ctrl.stats["forwards"] += 1
            return orig(*args, **kwargs)

        pg.forward = forward_sdpa
        print(f"[sdpa] paligemma/expert attention -> {impl}", flush=True)
        return self

    def reset(self):
        self.stats = {"forwards": 0, "impl": self.impl}

    def uninstall(self):
        if self._orig is not None:
            self.model.paligemma_with_expert.forward = self._orig
            self._orig = None
            self.model._fasterpi_attn_impl = "eager"


# ---------------------------------------------------------------------------
# compile_prefix — batched SigLIP + compiled image/prefix towers
# ---------------------------------------------------------------------------


class CompilePrefixController:
    """Batch all prefix images into one SigLIP call and compile embed_image."""

    def __init__(self, model, mode: str = "reduce-overhead"):
        self.model = model
        self.mode = mode
        self._orig_embed_image = None
        self._orig_embed_prefix = None
        self.stats = {"batched": 0, "fallback": 0}

    def install(self):
        if self._orig_embed_prefix is not None:
            return self
        model = self.model
        pg = model.paligemma_with_expert
        self._orig_embed_image = pg.embed_image
        self._orig_embed_prefix = model.embed_prefix
        try:
            import torch._dynamo as dynamo

            dynamo.config.suppress_errors = True
        except Exception:  # noqa: BLE001
            pass
        compiled_embed = torch.compile(self._orig_embed_image, mode=self.mode)
        ctrl = self
        use_graph = bool(getattr(model, "_fasterpi_cudagraphs", False))

        def embed_image_compiled(img):
            if use_graph:
                torch.compiler.cudagraph_mark_step_begin()
            return compiled_embed(img).clone()

        pg.embed_image = embed_image_compiled

        def embed_prefix_batched(images, img_masks, lang_tokens, lang_masks):
            embs = []
            pad_masks = []
            att_masks: list[int] = []
            # vision (installed after us) wraps embed_image for B=1 cache hits.
            # Batching would bypass that cache (cat has a different batch dim).
            vision_wrapped = pg.embed_image is not embed_image_compiled
            can_batch = (
                (not vision_wrapped)
                and bool(images)
                and len(images) > 1
                and all(im.shape == images[0].shape for im in images)
            )
            if can_batch:
                ctrl.stats["batched"] += 1
                cat = torch.cat(images, dim=0)
                all_emb = embed_image_compiled(cat)
                b0 = int(images[0].shape[0])
                chunks = all_emb.split(b0, dim=0)
                for img_emb, img_mask in zip(chunks, img_masks, strict=True):
                    bsize, num_img_embs = img_emb.shape[:2]
                    embs.append(img_emb)
                    pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                    att_masks += [0] * num_img_embs
            else:
                ctrl.stats["fallback"] += 1
                embed_one = pg.embed_image
                for img, img_mask in zip(images, img_masks, strict=True):
                    img_emb = embed_one(img)
                    bsize, num_img_embs = img_emb.shape[:2]
                    embs.append(img_emb)
                    pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                    att_masks += [0] * num_img_embs

            lang_emb = pg.embed_language_tokens(lang_tokens)
            lang_emb = lang_emb * (lang_emb.shape[-1] ** 0.5)
            embs.append(lang_emb)
            pad_masks.append(lang_masks)
            att_masks += [0] * lang_emb.shape[1]
            prefix_embs = torch.cat(embs, dim=1)
            prefix_pad_masks = torch.cat(pad_masks, dim=1)
            att = torch.tensor(att_masks, dtype=torch.bool, device=prefix_pad_masks.device)
            prefix_att_masks = att[None, :].expand(prefix_pad_masks.shape[0], att.shape[0])
            return prefix_embs, prefix_pad_masks, prefix_att_masks

        model.embed_prefix = embed_prefix_batched
        print(
            f"[compile_prefix] batched SigLIP + compiled embed_image "
            f"(mode={self.mode} cudagraphs={use_graph})",
            flush=True,
        )
        return self

    def reset(self):
        self.stats = {"batched": 0, "fallback": 0}

    def uninstall(self):
        if self._orig_embed_image is not None:
            self.model.paligemma_with_expert.embed_image = self._orig_embed_image
            self._orig_embed_image = None
        if self._orig_embed_prefix is not None:
            self.model.embed_prefix = self._orig_embed_prefix
            self._orig_embed_prefix = None


# ---------------------------------------------------------------------------
# fuse_loop — prefix once, then an eager Euler denoise loop
# ---------------------------------------------------------------------------


class FuseLoopController:
    """Prefix once, then a tight Python Euler loop (same structure as openpi).

    Compiling this loop with ``torch.compile`` is unsafe on the FasterPI stack:
    dynamo traces into step_cache / chunk_residual_cache, recaptures per-timestep graphs, and
    overwrites CUDA-graph tensor outputs stored for residual reuse. Keep the
    loop eager so those caches still own the compiled expert.
    """

    def __init__(self, model, mode: str = "reduce-overhead"):
        self.model = model
        self.mode = mode
        self._orig_sample = None
        self.stats = {"calls": 0, "compiled": 0}

    def _euler(self, nfe: int, state, prefix_pad_masks, past_key_values, x_t):
        nfe = int(nfe)
        model = self.model
        dt = torch.tensor(-1.0 / nfe, dtype=torch.float32, device=x_t.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=x_t.device)
        bsize = state.shape[0]
        for _ in range(nfe):
            v_t = model.denoise_step(
                state, prefix_pad_masks, past_key_values, x_t, time.expand(bsize)
            )
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t

    def install(self):
        if self._orig_sample is not None:
            return self
        self._orig_sample = self.model.sample_actions
        ctrl = self
        model = self.model

        @torch.no_grad()
        def sample_fused(device, observation, noise=None, num_steps=10, **kw):
            bsize = observation.state.shape[0]
            if noise is None:
                actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
                noise = model.sample_noise(actions_shape, device)
            images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
                observation, train=False
            )
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
            model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
                _attn_impl(model)  # noqa: SLF001
            )
            _, past_key_values = model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            nfe = int(num_steps)
            ctrl.stats["calls"] += 1
            return ctrl._euler(nfe, state, prefix_pad_masks, past_key_values, noise)

        self.model.sample_actions = sample_fused
        print(
            "[fuse_loop] sample_actions -> prefix + eager Euler "
            "(compile off: chunk_residual_cache/step_cache)",
            flush=True,
        )
        return self

    def reset(self):
        self.stats["calls"] = 0

    def uninstall(self):
        if self._orig_sample is not None:
            self.model.sample_actions = self._orig_sample
            self._orig_sample = None


# ---------------------------------------------------------------------------
# graph — inductor CUDA graphs (no approximation; same images / full NFE)
# ---------------------------------------------------------------------------


class KernelGraphController:
    """Turn on inductor CUDA graphs for compiled denoise / SigLIP.

    ``torch.compile(mode=reduce-overhead)`` already *asks* for graphs, but the
    RoboTwin HTTP workers historically set ``TORCHINDUCTOR_CUDAGRAPHS=0`` because
    ThreadingMixIn ran ``/predict`` off the capture thread. Serial HTTP plus
    ``cudagraph_mark_step_begin`` makes graphs safe again. This controller does
    not skip NFEs or reuse vision/action caches.
    """

    def __init__(self, model):
        self.model = model

    def install(self):
        self.model._fasterpi_cudagraphs = True
        os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "1"
        try:
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraphs = True
        except Exception:  # noqa: BLE001
            pass
        print(
            "[graph] inductor CUDA graphs ON "
            "(mark_step_begin on compiled denoise / SigLIP; serial infer thread)",
            flush=True,
        )
        return self

    def reset(self):
        return None

    def uninstall(self):
        self.model._fasterpi_cudagraphs = False


# ---------------------------------------------------------------------------
# stack assembly
# ---------------------------------------------------------------------------

VALID_DIRS = (
    "compile",
    "graph",
    "vision",
    "d0",
    "prefix_kv",
    "split_prefix",
    "chunk_residual_cache",
    "step_cache",
    "world_gate",
    "adaptive_replan",
    "adaptive_nfe",
    "sdpa",
    "compile_prefix",
    "fuse_loop",
    "host",
)

# Current FasterPI default (LIBERO in-process t00.pick vs v2 nfe+step_cache:
# ~12% lower L static/move; compile_prefix compiled SigLIP on vision misses).
DEFAULT_FASTERPI_SPEC = "compile+chunk_residual_cache+step_cache+compile_prefix"
# Paper name for the default stack. `ours` and `fasterpi` are the same spec.
# Leave-one-out of the optional modules. Compile and residual reuse stay.
OURS_NO_ADAPTIVE_NFE_SPEC = "compile+chunk_residual_cache+vision+step_cache+compile_prefix"
OURS_NO_VISION_SPEC = "compile+chunk_residual_cache+adaptive_nfe+step_cache+compile_prefix"
OURS_NO_STEP_CACHE_SPEC = "compile+chunk_residual_cache+vision+adaptive_nfe+compile_prefix"
OURS_NO_D1_SPEC = OURS_NO_STEP_CACHE_SPEC
FASTERPI_V2_SPEC = "compile+chunk_residual_cache+vision+adaptive_nfe+step_cache"
FASTERPI_V1_SPEC = "compile+chunk_residual_cache+vision"
# World-blind paper Cache: same modules as v1, looser similarity so reuse
# continues while objects move (no W_c / adaptive_nfe).
FASTERPI_CACHE_SPEC = "compile+chunk_residual_cache+vision"
# Ablations of paper Cache at the same world-blind reuse gates (not default tight).
FASTERPI_CACHE_VISION_SPEC = "compile+vision"
FASTERPI_CACHE_RESIDUAL_SPEC = "compile+chunk_residual_cache"
FASTERPI_CACHE_C3_SPEC = FASTERPI_CACHE_RESIDUAL_SPEC
# Kernel path: compile + fused/compiled SigLIP + CUDA graphs. No vision or residual.
FASTERPI_KERNEL_SPEC = "compile+compile_prefix+graph"
FASTERPI_COMPILE_GRAPH_SPEC = "compile+graph"
# A-only CLIRA: compile + residual, SigLIP probe, skip prefix. No step_cache / adaptive_nfe.
FASTERPI_CLIRA_SPEC = "compile+chunk_residual_cache+compile_prefix+prefix_kv"
WORLD_BLIND_CACHE_KWARGS = {
    "vision_threshold": 0.12,
    "chunk_residual_condition_threshold": 0.25,
    "chunk_residual_latent_threshold": 0.50,
    "chunk_residual_cold_chunks": 0,
    "chunk_residual_max_consecutive_reuses": 8,
}
WORLD_BLIND_CACHE_ALIASES = frozenset(
    {
        "cache",
        "worldblind",
        "fasterpi_cache",
        "cache_vision",
        "cache_c3",
        "cache_residual",
    }
)
CLIRA_PREFIX_KWARGS = {
    "prefix_gate": "siglip_anchor",
    "prefix_image_threshold": 0.005,
    "prefix_state_threshold": 1e9,
    "prefix_cold_chunks": 1,
    "prefix_max_consecutive_reuses": 4,
}
SPEEDUP_ALIASES = {
    "fasterpi": DEFAULT_FASTERPI_SPEC,
    "fasterpi_v2": FASTERPI_V2_SPEC,
    "fasterpi_v1": FASTERPI_V1_SPEC,
    "pace": DEFAULT_FASTERPI_SPEC,
    "ours": DEFAULT_FASTERPI_SPEC,
    "ours_no_adaptive_nfe": OURS_NO_ADAPTIVE_NFE_SPEC,
    "ours_no_vision": OURS_NO_VISION_SPEC,
    "ours_no_step_cache": OURS_NO_STEP_CACHE_SPEC,
    "ours_no_d1": OURS_NO_STEP_CACHE_SPEC,
    "baseline": "eager",
    "cache": FASTERPI_CACHE_SPEC,
    "worldblind": FASTERPI_CACHE_SPEC,
    "fasterpi_cache": FASTERPI_CACHE_SPEC,
    "cache_vision": FASTERPI_CACHE_VISION_SPEC,
    "cache_residual": FASTERPI_CACHE_RESIDUAL_SPEC,
    "cache_c3": FASTERPI_CACHE_RESIDUAL_SPEC,
    "compile_graph": FASTERPI_COMPILE_GRAPH_SPEC,
    "kernel": FASTERPI_KERNEL_SPEC,
    "clira": FASTERPI_CLIRA_SPEC,
    "fasterpi_clira": FASTERPI_CLIRA_SPEC,
}


DIR_ALIASES = {
    "c3": "chunk_residual_cache",
    "d1": "step_cache",
}


def canonicalize_dirs(dirs: Iterable[str]) -> list[str]:
    """Map retired spec tokens onto the ``*_cache`` names."""
    out: list[str] = []
    for raw in dirs:
        name = DIR_ALIASES.get(str(raw).strip(), str(raw).strip())
        if name and name not in out:
            out.append(name)
    return out


def resolve_speedup_spec(spec: str) -> str:
    """Expand aliases (``fasterpi`` → default dirs). Eager/builtin pass through."""
    key = (spec or "eager").strip()
    resolved = SPEEDUP_ALIASES.get(key, key)
    if resolved in ("", "none", "eager", "builtin"):
        return resolved
    return "+".join(canonicalize_dirs(resolved.replace("+", ",").split(",")))


def install_stack(
    model,
    dirs: Iterable[str],
    *,
    policy=None,
    compile_mode: str = "reduce-overhead",
    step_cache_threshold: float = 0.20,
    step_cache_max_skip: int = 2,
    step_cache_cold: int = 2,
    vision_threshold: float = 0.02,
    chunk_residual_condition_threshold: float = 0.05,
    chunk_residual_latent_threshold: float = 0.15,
    chunk_residual_cold_chunks: int = 1,
    chunk_residual_max_consecutive_reuses: int = 4,
    prefix_image_threshold: float = 0.01,
    prefix_state_threshold: float = 0.02,
    prefix_cold_chunks: int = 1,
    prefix_max_consecutive_reuses: int = 2,
    prefix_gate: str = "rgb_l1",
):
    """Install requested accelerate directions.

    dirs: subset of VALID_DIRS.
    Always restores eager first (removes openpi whole-loop max-autotune).
    When both compile and chunk_residual_cache are requested, compile is absorbed
    into the residual expert body so residual hits never enter a CUDA graph.

    ``world_gate`` / ``adaptive_replan`` are flags (no extra GPU graph).
    ``split_prefix`` replaces ``prefix_kv`` if both are requested.
    """
    dirs = canonicalize_dirs(dirs)
    unknown = [d for d in dirs if d not in VALID_DIRS]
    if unknown:
        raise ValueError(f"unknown accelerate dirs {unknown}; valid={VALID_DIRS}")

    restore_eager(model)
    model._fasterpi_world_gate = "world_gate" in dirs
    model._fasterpi_adaptive_replan = "adaptive_replan" in dirs
    model._fasterpi_adaptive_nfe = "adaptive_nfe" in dirs
    model._fasterpi_host = "host" in dirs
    model._fasterpi_attn_impl = "sdpa" if "sdpa" in dirs else "eager"

    installed: list[Any] = []
    has_residual = "chunk_residual_cache" in dirs
    has_split = "split_prefix" in dirs

    if "graph" in dirs:
        installed.append(KernelGraphController(model).install())

    if "sdpa" in dirs:
        installed.append(SdpaController(model).install())

    if "compile_prefix" in dirs:
        installed.append(CompilePrefixController(model, mode=compile_mode).install())

    # compile only when residual cache is absent (it owns / compiles the expert path)
    if "compile" in dirs and not has_residual:
        installed.append(CompileController(model, mode=compile_mode).install())

    if "vision" in dirs:
        installed.append(VisionCacheController(model, threshold=vision_threshold).install())

    if "d0" in dirs:
        installed.append(TextCacheController(model).install())

    if has_split:
        installed.append(SplitPrefixController(model).install())
    elif "prefix_kv" in dirs:
        installed.append(
            PrefixKVCacheController(
                model,
                image_threshold=prefix_image_threshold,
                state_threshold=prefix_state_threshold,
                cold_chunks=prefix_cold_chunks,
                max_consecutive_reuses=prefix_max_consecutive_reuses,
                gate=prefix_gate,
            ).install()
        )

    if "fuse_loop" in dirs:
        installed.append(FuseLoopController(model, mode=compile_mode).install())

    if has_residual:
        installed.append(
            ChunkResidualCacheController(
                model,
                condition_threshold=chunk_residual_condition_threshold,
                latent_threshold=chunk_residual_latent_threshold,
                cold_chunks=chunk_residual_cold_chunks,
                max_consecutive_reuses=chunk_residual_max_consecutive_reuses,
                compile_mode=compile_mode if "compile" in dirs else None,
            ).install()
        )

    if "step_cache" in dirs:
        installed.append(
            StepCacheController(
                model,
                threshold=step_cache_threshold,
                max_consecutive_skips=step_cache_max_skip,
                cold_steps=step_cache_cold,
            ).install()
        )

    if "adaptive_nfe" in dirs:
        installed.append(AdaptiveNFEController(model).install())

    if policy is not None:
        policy._sample_actions = model.sample_actions

    def _reset_caches() -> None:
        for c in installed:
            fn = getattr(c, "reset", None)
            if callable(fn):
                fn()

    model._fasterpi_reset_caches = _reset_caches
    model._fasterpi_controllers = list(installed)

    print(f"[fasterpi] stack active: {dirs}", flush=True)
    extra = []
    if vision_threshold != 0.02:
        extra.append(f"vision_thr={vision_threshold}")
    if chunk_residual_condition_threshold != 0.05 or chunk_residual_latent_threshold != 0.15:
        extra.append(
            f"residual_cond={chunk_residual_condition_threshold} "
            f"residual_lat={chunk_residual_latent_threshold} "
            f"residual_cold={chunk_residual_cold_chunks} "
            f"residual_reuse={chunk_residual_max_consecutive_reuses}"
        )
    if extra:
        print(f"[fasterpi] cache gates: {' '.join(extra)}", flush=True)
    return installed
