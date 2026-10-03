"""FastWAM shadow probe: current-W reduced NFE vs stale KV, same noise."""

from __future__ import annotations

from typing import Any

import torch

from benchmarks.dynamic_libero.eval.coupling_math import (
    camera_phis_horizontal,
    cosine_delta,
    mean_pool,
    mse,
    relative_l1,
)


def _clone_kv(cache: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{"k": e["k"].detach().clone(), "v": e["v"].detach().clone()} for e in cache]


class FastWAMCouplingProbe:
    def __init__(self, ctx: Any, *, nfe_full: int = 10, nfe_mid: int = 8, nfe_cheap: int = 6):
        self.ctx = ctx
        self.model = ctx.model
        self.nfe_full = int(nfe_full)
        self.nfe_mid = int(nfe_mid)
        self.nfe_cheap = int(nfe_cheap)
        self._prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._prev_phi: torch.Tensor | None = None
        self._prev_phis_cam: list[torch.Tensor] | None = None
        self._prev_kv: list[dict[str, torch.Tensor]] | None = None
        self._prev_video_seq: int | None = None
        self._E = None

    def reset(self) -> None:
        self._prev_phi = None
        self._prev_phis_cam = None
        self._prev_kv = None
        self._prev_video_seq = None

    def _eval_mod(self):
        if self._E is None:
            from benchmarks.dynamic_libero.env import libero_bridge as bridge

            self._E = bridge._import_eval_libero_single()
        return self._E

    def _prompt_ctx(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

        prompt = DEFAULT_PROMPT.format(task=instruction)
        hit = self._prompt_cache.get(prompt)
        if hit is not None:
            return hit[0].clone(), hit[1].clone()
        ctx, mask = self.model.encode_prompt(prompt)
        self._prompt_cache[prompt] = (ctx.detach(), mask.detach())
        return ctx.clone(), mask.clone()

    @torch.no_grad()
    def measure(self, obs: dict, instruction: str) -> dict[str, Any]:
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

        tiled = bool(cfg.EVALUATION.get("tiled", False))
        sigma_shift = cfg.EVALUATION.get("sigma_shift")
        sigma_shift = None if sigma_shift is None else float(sigma_shift)

        context, context_mask = self._prompt_ctx(instruction)
        context = context.to(device=model.device, dtype=model.torch_dtype)
        context_mask = context_mask.to(device=model.device, dtype=torch.bool)
        context, context_mask = model._append_proprio_to_context(
            context=context, context_mask=context_mask, proprio=proprio
        )

        first_frame_latents = model._encode_input_image_latents_tensor(
            input_image=image, tiled=tiled
        )
        fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=model.device,
        )
        video_pre = model.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        tokens = video_pre["tokens"]
        patch = model.video_expert.patch_size
        ph, pw = int(patch[1]), int(patch[2])
        # VAE latents: [B,C,F,H,W] or [B,C,H,W]
        lat = first_frame_latents
        if lat.ndim == 5:
            lh, lw = int(lat.shape[3]), int(lat.shape[4])
        else:
            lh, lw = int(lat.shape[2]), int(lat.shape[3])
        phis_cam = camera_phis_horizontal(
            tokens, latent_h=lh, latent_w=lw, patch_h=ph, patch_w=pw, n_cam=2
        )
        phi = mean_pool(tokens)

        delta = None
        delta_cam = None
        if self._prev_phi is not None:
            delta = cosine_delta(phi, self._prev_phi)
            if self._prev_phis_cam is not None and len(self._prev_phis_cam) == len(phis_cam):
                delta_cam = max(
                    cosine_delta(a, b) for a, b in zip(phis_cam, self._prev_phis_cam)
                )
            else:
                delta_cam = delta

        video_seq_len = int(tokens.shape[1])
        horizon = int(self.ctx.action_horizon)
        attention_mask = model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=horizon,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=tokens.device,
        )
        kv = model.mot.prefill_video_cache(
            video_tokens=tokens,
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        noise = torch.randn(
            (1, horizon, model.action_expert.action_dim),
            device=model.device,
            dtype=model.torch_dtype,
        )

        def _integrate(nfe: int, kv_use, attn, vlen: int) -> torch.Tensor:
            x = noise.clone()
            steps, deltas = model.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=int(nfe),
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
                    video_kv_cache=kv_use,
                    attention_mask=attn,
                    video_seq_len=vlen,
                )
                x = model.infer_action_scheduler.step(v, step_d, x)
            return x[0].detach()

        a_full = _integrate(self.nfe_full, kv, attention_mask, video_seq_len)
        a_mid = _integrate(self.nfe_mid, kv, attention_mask, video_seq_len)
        a_cheap = _integrate(self.nfe_cheap, kv, attention_mask, video_seq_len)

        rec: dict[str, Any] = {
            "delta_cos": delta,
            "delta_cam_max": delta_cam,
            "err_nfe_mid_l1": relative_l1(a_mid, a_full),
            "err_nfe_cheap_l1": relative_l1(a_cheap, a_full),
            "err_nfe_mid_mse": mse(a_mid, a_full),
            "err_nfe_cheap_mse": mse(a_cheap, a_full),
            "err_stale_l1": None,
            "err_stale_mse": None,
            "nfe_full": self.nfe_full,
            "nfe_mid": self.nfe_mid,
            "nfe_cheap": self.nfe_cheap,
        }
        if self._prev_kv is not None and self._prev_video_seq == video_seq_len:
            a_stale = _integrate(
                self.nfe_full, self._prev_kv, attention_mask, video_seq_len
            )
            rec["err_stale_l1"] = relative_l1(a_stale, a_full)
            rec["err_stale_mse"] = mse(a_stale, a_full)

        self._prev_phi = phi.detach().clone()
        self._prev_phis_cam = [p.detach().clone() for p in phis_cam]
        self._prev_kv = _clone_kv(kv)
        self._prev_video_seq = video_seq_len
        return rec
