"""FastWAM perception probe: queued/desync/lookahead actions + cached-z history."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from benchmarks.dynamic_libero.eval.coupling_math import (
    camera_phis_horizontal,
    cosine_delta,
    mean_abs,
    mean_pool,
    relative_l1,
)
from benchmarks.dynamic_libero.eval.coupling_probe2_common import clone_kv, f32
from benchmarks.dynamic_libero.eval.perception_probe_common import DELAYS, empty_variant_fields


class FastWAMPerceptionProbe:
    def __init__(self, ctx: Any, *, nfe: int = 10, delays: tuple[int, ...] = DELAYS):
        self.ctx = ctx
        self.model = ctx.model
        self.nfe = int(nfe)
        self.delays = tuple(int(d) for d in delays)
        self._prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._E = None

    def reset(self) -> None:
        return None

    def _eval_mod(self):
        if self._E is None:
            from benchmarks.dynamic_libero.env import libero_bridge as bridge

            self._E = bridge._import_eval_libero_single()
        return self._E

    def _prompt_txt(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

        prompt = DEFAULT_PROMPT.format(task=instruction)
        hit = self._prompt_cache.get(prompt)
        if hit is not None:
            return hit[0].clone(), hit[1].clone()
        ctx, mask = self.model.encode_prompt(prompt)
        self._prompt_cache[prompt] = (ctx.detach(), mask.detach())
        return ctx.clone(), mask.clone()

    def _obs_tensors(self, obs: dict):
        E = self._eval_mod()
        model = self.model
        cfg = self.ctx.cfg
        image, proprio, _ = E._obs_to_model_input(
            obs,
            cfg=cfg,
            processor=self.ctx.processor,
            width=self.ctx.vw,
            height=self.ctx.vh,
            device=self.ctx.device,
            dtype=model.torch_dtype,
        )
        image = image.to(device=model.device, dtype=model.torch_dtype)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        proprio = proprio.to(device=model.device, dtype=model.torch_dtype)
        return image, proprio

    def _prefill(self, image, context, context_mask, tiled: bool, latents=None):
        model = self.model
        if latents is None:
            latents = model._encode_input_image_latents_tensor(input_image=image, tiled=tiled)
        fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
        timestep_video = torch.zeros(
            (latents.shape[0],), dtype=latents.dtype, device=model.device
        )
        video_pre = model.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        tokens = video_pre["tokens"]
        patch = model.video_expert.patch_size
        ph, pw = int(patch[1]), int(patch[2])
        if latents.ndim == 5:
            lh, lw = int(latents.shape[3]), int(latents.shape[4])
        else:
            lh, lw = int(latents.shape[2]), int(latents.shape[3])
        phis_cam = camera_phis_horizontal(
            tokens, latent_h=lh, latent_w=lw, patch_h=ph, patch_w=pw, n_cam=2
        )
        phi = mean_pool(tokens)
        video_seq_len = int(tokens.shape[1])
        horizon = int(self.ctx.action_horizon)
        attention_mask = model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=horizon,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=tokens.device,
        )
        ctx_mask = video_pre.get("context_mask", video_pre.get("mask"))
        kv = model.mot.prefill_video_cache(
            video_tokens=tokens,
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": ctx_mask,
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        return kv, phi, phis_cam, attention_mask, video_seq_len, video_pre, latents

    def _prefill_tokens(self, tokens, video_pre, attention_mask, video_seq_len):
        model = self.model
        ctx_mask = video_pre.get("context_mask", video_pre.get("mask"))
        return model.mot.prefill_video_cache(
            video_tokens=tokens,
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": ctx_mask,
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

    def _integrate(self, noise, kv, attn, vlen, context, context_mask):
        model = self.model
        cfg = self.ctx.cfg
        sigma_shift = cfg.EVALUATION.get("sigma_shift")
        sigma_shift = None if sigma_shift is None else float(sigma_shift)
        x = noise.clone()
        steps, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(self.nfe),
            device=model.device,
            dtype=x.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_d in zip(steps, deltas):
            t = step_t.unsqueeze(0).to(dtype=x.dtype, device=model.device)
            v = model._predict_action_noise_with_cache(
                latents_action=x,
                timestep_action=t,
                context=context,
                context_mask=context_mask,
                video_kv_cache=kv,
                attention_mask=attn,
                video_seq_len=vlen,
            )
            x = model.infer_action_scheduler.step(v, step_d, x)
        return x[0].detach()

    def _encode_frame(self, obs: dict, instruction: str) -> dict[str, Any] | None:
        model = self.model
        cfg = self.ctx.cfg
        tiled = bool(cfg.EVALUATION.get("tiled", False))
        image, proprio = self._obs_tensors(obs)
        ctx_txt, mask_txt = self._prompt_txt(instruction)
        ctx_txt = ctx_txt.to(device=model.device, dtype=model.torch_dtype)
        mask_txt = mask_txt.to(device=model.device, dtype=torch.bool)
        ctx_prop, mask_prop = model._append_proprio_to_context(
            context=ctx_txt, context_mask=mask_txt, proprio=proprio
        )
        kv_prop, phi, phis_cam, attn, vlen, video_pre, latents = self._prefill(
            image, ctx_prop, mask_prop, tiled
        )
        kv_vis, _, _, attn_vis, vlen_vis, _, _ = self._prefill(
            image, ctx_txt, mask_txt, tiled, latents=latents
        )
        if vlen_vis != vlen:
            kv_vis = None
            attn_vis = attn
        return {
            "kv_prop": clone_kv(kv_prop),
            "kv_vis": clone_kv(kv_vis) if kv_vis is not None else None,
            "phi": phi.detach(),
            "phis_cam": [p.detach() for p in phis_cam],
            "attn": attn,
            "vlen": int(vlen),
            "ctx_prop": ctx_prop,
            "mask_prop": mask_prop,
            "tokens": tokens_detached(video_pre["tokens"]),
            "video_pre": {
                "freqs": video_pre["freqs"],
                "t_mod": video_pre["t_mod"],
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
        }

    @torch.no_grad()
    def measure_episode(self, frames: list[dict]) -> tuple[list[dict], list[np.ndarray]]:
        packs: list[dict[str, Any] | None] = []
        for fr in frames:
            packs.append(self._encode_frame(fr["obs"], fr["instruction"]))

        horizon = int(self.ctx.action_horizon)
        dim = int(self.model.action_expert.action_dim)
        rows: list[dict] = []
        phis: list[np.ndarray] = []
        prev_a = None
        T = len(frames)
        for t, pack in enumerate(packs):
            rec = empty_variant_fields(self.delays)
            rec["nfe"] = self.nfe
            rec["capture_lag_ms"] = float(frames[t].get("capture_lag_ms") or 0.0)
            rec["infer_ms"] = frames[t].get("infer_ms")
            if pack is None:
                rows.append(rec)
                continue
            noise = torch.randn(
                (1, horizon, dim),
                device=self.model.device,
                dtype=self.model.torch_dtype,
            )
            a_sync = self._integrate(
                noise, pack["kv_prop"], pack["attn"], pack["vlen"], pack["ctx_prop"], pack["mask_prop"]
            )
            if prev_a is not None:
                rec["err_full_consec_l1"] = relative_l1(a_sync, prev_a)
                rec["err_full_consec_abs"] = mean_abs(a_sync, prev_a)
            if t > 0 and packs[t - 1] is not None:
                rec["delta_cos_adj"] = cosine_delta(pack["phi"], packs[t - 1]["phi"])
                rec["delta_cam_adj"] = _cam_delta(pack["phis_cam"], packs[t - 1]["phis_cam"])
            for d in self.delays:
                if t >= d and packs[t - d] is not None:
                    old = packs[t - d]
                    rec[f"delta_cos_d{d}"] = cosine_delta(pack["phi"], old["phi"])
                    rec[f"delta_cam_d{d}"] = _cam_delta(pack["phis_cam"], old["phis_cam"])
                    rec[f"motion_rate_d{d}"] = rec[f"delta_cos_d{d}"] / float(d)
                    if old["vlen"] == pack["vlen"]:
                        a_delay = self._integrate(
                            noise, old["kv_prop"], old["attn"], old["vlen"], old["ctx_prop"], old["mask_prop"]
                        )
                        rec[f"err_delay_d{d}_l1"] = relative_l1(a_delay, a_sync)
                        rec[f"err_delay_d{d}_abs"] = mean_abs(a_delay, a_sync)
                        if old["kv_vis"] is not None:
                            a_img = self._integrate(
                                noise, old["kv_vis"], old["attn"], old["vlen"], pack["ctx_prop"], pack["mask_prop"]
                            )
                            rec[f"err_img_old_d{d}_l1"] = relative_l1(a_img, a_sync)
                            rec[f"err_img_old_d{d}_abs"] = mean_abs(a_img, a_sync)
                        if pack["kv_vis"] is not None:
                            a_prop = self._integrate(
                                noise, pack["kv_vis"], pack["attn"], pack["vlen"], old["ctx_prop"], old["mask_prop"]
                            )
                            rec[f"err_prop_old_d{d}_l1"] = relative_l1(a_prop, a_sync)
                            rec[f"err_prop_old_d{d}_abs"] = mean_abs(a_prop, a_sync)
                if t + d < T and packs[t + d] is not None:
                    fut = packs[t + d]
                    if fut["vlen"] == pack["vlen"]:
                        a_ahead = self._integrate(
                            noise, fut["kv_prop"], fut["attn"], fut["vlen"], fut["ctx_prop"], fut["mask_prop"]
                        )
                        rec[f"err_ahead_d{d}_l1"] = relative_l1(a_ahead, a_sync)
                        rec[f"err_ahead_d{d}_abs"] = mean_abs(a_ahead, a_sync)
            if t >= 1 and packs[t - 1] is not None:
                prev = packs[t - 1]
                try:
                    if prev["tokens"].shape == pack["tokens"].shape:
                        mixed = 0.5 * pack["tokens"] + 0.5 * prev["tokens"]
                        kv_mix = self._prefill_tokens(
                            mixed, pack["video_pre"], pack["attn"], pack["vlen"]
                        )
                        a_mix = self._integrate(
                            noise, kv_mix, pack["attn"], pack["vlen"], pack["ctx_prop"], pack["mask_prop"]
                        )
                        rec["err_mix_k2_l1"] = relative_l1(a_mix, a_sync)
                        rec["err_mix_k2_abs"] = mean_abs(a_mix, a_sync)
                except Exception:
                    pass
            phi_np = pack["phi"].detach().float().cpu().numpy().astype(np.float16, copy=False)
            rec["phi_dim"] = int(phi_np.size)
            rows.append({k: f32(v) if isinstance(v, float) else v for k, v in rec.items()})
            phis.append(np.asarray(phi_np).reshape(-1))
            prev_a = a_sync
        packs.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rows, phis


def tokens_detached(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.detach()


def _cam_delta(a, b) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    return max(cosine_delta(x, y) for x, y in zip(a, b))
