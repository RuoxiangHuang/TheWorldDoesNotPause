"""FastWAM probe-2: cost split, cache age, visual-only KV vs proprio-baked KV."""

from __future__ import annotations

from collections import deque
from typing import Any

import torch

from benchmarks.dynamic_libero.eval.coupling_math import (
    camera_phis_horizontal,
    cosine_delta,
    mean_abs,
    mean_pool,
    relative_l1,
)
from benchmarks.dynamic_libero.eval.coupling_probe2_common import (
    AGES,
    CudaTimer,
    clone_kv,
    f32,
)


class FastWAMProbe2:
    def __init__(self, ctx: Any, *, nfe: int = 10, ages: tuple[int, ...] = AGES):
        self.ctx = ctx
        self.model = ctx.model
        self.nfe = int(nfe)
        self.ages = tuple(int(a) for a in ages)
        self._prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._hist: deque[dict[str, Any]] = deque(maxlen=max(self.ages) + 1)
        self._prev_a_full: torch.Tensor | None = None
        self._E = None
        self._n_measure = 0

    def reset(self) -> None:
        self._hist.clear()
        self._prev_a_full = None

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

    def _prefill(
        self,
        image,
        context,
        context_mask,
        tiled: bool,
        timer: CudaTimer | None,
        latents=None,
    ):
        model = self.model
        tm = timer or CudaTimer(False)
        if latents is None:
            with tm.span("vae"):
                latents = model._encode_input_image_latents_tensor(input_image=image, tiled=tiled)
        fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
        timestep_video = torch.zeros(
            (latents.shape[0],), dtype=latents.dtype, device=model.device
        )
        with tm.span("predit"):
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
        with tm.span("prefill"):
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
        return kv, phi, phis_cam, attention_mask, video_seq_len, tm.ms, latents

    def _integrate(self, noise, nfe, kv, attn, vlen, context, context_mask, timer=None):
        model = self.model
        cfg = self.ctx.cfg
        sigma_shift = cfg.EVALUATION.get("sigma_shift")
        sigma_shift = None if sigma_shift is None else float(sigma_shift)
        x = noise.clone()
        steps, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(nfe),
            device=model.device,
            dtype=x.dtype,
            shift_override=sigma_shift,
        )
        tm = timer or CudaTimer(False)
        with tm.span("action"):
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
        return x[0].detach(), tm.ms.get("action")

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

        ctx_txt, mask_txt = self._prompt_txt(instruction)
        ctx_txt = ctx_txt.to(device=model.device, dtype=model.torch_dtype)
        mask_txt = mask_txt.to(device=model.device, dtype=torch.bool)
        ctx_prop, mask_prop = model._append_proprio_to_context(
            context=ctx_txt, context_mask=mask_txt, proprio=proprio
        )

        timer = CudaTimer(True)
        kv_prop, phi, phis_cam, attn, vlen, ms_prop, latents = self._prefill(
            image, ctx_prop, mask_prop, tiled, timer
        )
        kv_vis, _, _, attn_vis, vlen_vis, ms_vis, _ = self._prefill(
            image, ctx_txt, mask_txt, tiled, CudaTimer(False), latents=latents
        )
        if vlen_vis != vlen:
            kv_vis = None

        horizon = int(self.ctx.action_horizon)
        noise = torch.randn(
            (1, horizon, model.action_expert.action_dim),
            device=model.device,
            dtype=model.torch_dtype,
        )
        act_timer = CudaTimer(True)
        a_full, ms_act = self._integrate(
            noise, self.nfe, kv_prop, attn, vlen, ctx_prop, mask_prop, act_timer
        )
        a_full_vis = None
        if kv_vis is not None and len(self._hist) < 3:
            a_full_vis, _ = self._integrate(
                noise, self.nfe, kv_vis, attn_vis, vlen, ctx_prop, mask_prop, CudaTimer(False)
            )

        rec: dict[str, Any] = {
            "nfe": self.nfe,
            "ms_vae": ms_prop.get("vae"),
            "ms_predit": ms_prop.get("predit"),
            "ms_prefill": ms_prop.get("prefill"),
            "ms_action": ms_act,
            "ms_predit_vis": ms_vis.get("predit"),
            "ms_prefill_vis": ms_vis.get("prefill"),
            "err_vis_vs_prop_l1": None,
            "err_vis_vs_prop_abs": None,
            "err_full_consec_l1": None,
            "err_full_consec_abs": None,
            "delta_cam_adj": None,
            "delta_cos_adj": None,
        }
        if a_full_vis is not None:
            rec["err_vis_vs_prop_l1"] = relative_l1(a_full_vis, a_full)
            rec["err_vis_vs_prop_abs"] = mean_abs(a_full_vis, a_full)
        if self._prev_a_full is not None:
            rec["err_full_consec_l1"] = relative_l1(a_full, self._prev_a_full)
            rec["err_full_consec_abs"] = mean_abs(a_full, self._prev_a_full)
        if self._hist:
            prev = self._hist[-1]
            rec["delta_cos_adj"] = cosine_delta(phi, prev["phi"])
            rec["delta_cam_adj"] = max(
                cosine_delta(a, b) for a, b in zip(phis_cam, prev["phis_cam"])
            ) if prev.get("phis_cam") and len(prev["phis_cam"]) == len(phis_cam) else rec["delta_cos_adj"]

        for age in self.ages:
            rec[f"delta_cam_age{age}"] = None
            rec[f"delta_cos_age{age}"] = None
            rec[f"err_stale_kv_l1_age{age}"] = None
            rec[f"err_stale_kv_abs_age{age}"] = None
            rec[f"err_stale_vis_l1_age{age}"] = None
            rec[f"err_stale_vis_abs_age{age}"] = None
            rec[f"err_exec_kv_vs_vis_l1_age{age}"] = None
            if len(self._hist) < age:
                continue
            snap = self._hist[-age]
            if snap["vlen"] != vlen:
                continue
            dcos = cosine_delta(phi, snap["phi"])
            dcam = dcos
            if snap.get("phis_cam") and len(snap["phis_cam"]) == len(phis_cam):
                dcam = max(cosine_delta(a, b) for a, b in zip(phis_cam, snap["phis_cam"]))
            rec[f"delta_cam_age{age}"] = dcam
            rec[f"delta_cos_age{age}"] = dcos
            a_stale, _ = self._integrate(
                noise, self.nfe, snap["kv_prop"], attn, vlen, ctx_prop, mask_prop
            )
            rec[f"err_stale_kv_l1_age{age}"] = relative_l1(a_stale, a_full)
            rec[f"err_stale_kv_abs_age{age}"] = mean_abs(a_stale, a_full)
            if age == 1 and snap.get("kv_vis") is not None:
                a_vis, _ = self._integrate(
                    noise, self.nfe, snap["kv_vis"], attn, vlen, ctx_prop, mask_prop
                )
                rec[f"err_stale_vis_l1_age{age}"] = relative_l1(a_vis, a_full)
                rec[f"err_stale_vis_abs_age{age}"] = mean_abs(a_vis, a_full)
                rec[f"err_exec_kv_vs_vis_l1_age{age}"] = relative_l1(a_stale, a_vis)

        self._hist.append(
            {
                "phi": phi.detach().clone(),
                "phis_cam": [p.detach().clone() for p in phis_cam],
                "kv_prop": clone_kv(kv_prop),
                "kv_vis": clone_kv(kv_vis) if kv_vis is not None else None,
                "vlen": vlen,
            }
        )
        self._prev_a_full = a_full.detach().clone()
        self._n_measure += 1
        rec["delta_cam_max"] = rec.get("delta_cam_age1")
        rec["delta_cos"] = rec.get("delta_cos_age1")
        rec["err_stale_l1"] = rec.get("err_stale_kv_l1_age1")
        return {k: f32(v) if isinstance(v, float) else v for k, v in rec.items()}
