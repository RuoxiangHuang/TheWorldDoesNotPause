"""π₀.₅ perception probe: queued/desync/lookahead actions + cached SigLIP history."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from benchmarks.dynamic_libero.eval.coupling_math import cosine_delta, mean_abs, relative_l1
from benchmarks.dynamic_libero.eval.coupling_probe2_common import clone_past, f32
from benchmarks.dynamic_libero.eval.perception_probe_common import DELAYS, empty_variant_fields
from fasterpi.realtime.adapters import libero_obs_to_openpi
from fasterpi.realtime.loader import _batch_to_torch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


class PiPerceptionProbe:
    def __init__(self, rt_policy: Any, *, nfe: int = 10, delays: tuple[int, ...] = DELAYS):
        self.rt = rt_policy
        self.policy = rt_policy.policy
        self.model = rt_policy.policy._model
        self.nfe = int(nfe)
        self.delays = tuple(int(d) for d in delays)

    def reset(self) -> None:
        return None

    def _element(self, obs: dict, instruction: str) -> dict:
        return libero_obs_to_openpi(obs, instruction, resize=int(self.rt.resize))

    def _observation(self, element: dict):
        from openpi.models import model as _model_mod

        inputs = self.policy._input_transform(element)
        device = self.policy._pytorch_device
        batched = _batch_to_torch(inputs, device)
        return _model_mod.Observation.from_dict(batched)

    def _attn_impl(self) -> str:
        return str(getattr(self.model, "_fasterpi_attn_impl", "eager") or "eager")

    def _embed_images_phis(self, images, img_masks):
        model = self.model
        pg = model.paligemma_with_expert
        embs = []
        phis = []
        for img, mask in zip(images, img_masks, strict=True):
            emb = pg.embed_image(img)
            embs.append(emb)
            m = mask
            if m.ndim == 1:
                valid = bool(m.reshape(-1)[0].item())
                if valid:
                    phis.append(emb.float().mean(dim=1).reshape(-1).detach())
            else:
                phis.append(emb.float().mean(dim=1).reshape(-1).detach())
        return embs, phis

    def _prefix_from_embs(self, img_embs, img_masks, lang_tokens, lang_masks):
        model = self.model
        pg = model.paligemma_with_expert
        pad_masks = []
        att_masks_list = []
        packed = []
        for img_emb, img_mask in zip(img_embs, img_masks, strict=True):
            bsize, num_img_embs = img_emb.shape[:2]
            packed.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks_list += [0] * num_img_embs
        lang_emb = pg.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * (float(lang_emb.shape[-1]) ** 0.5)
        packed.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks_list += [0] * lang_emb.shape[1]
        prefix_embs = torch.cat(packed, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks_list, dtype=torch.bool, device=prefix_pad_masks.device)
        att_masks = att_masks[None, :].expand(prefix_pad_masks.shape[0], -1)
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = model._prepare_attention_masks_4d(prefix_att_2d)
        pg.paligemma.language_model.config._attn_implementation = self._attn_impl()
        _, past = pg.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return past, prefix_pad_masks

    def _denoise(self, state, prefix_pad_masks, past, noise):
        model = self.model
        device = noise.device
        bsize = state.shape[0]
        dt = torch.tensor(-1.0 / int(self.nfe), dtype=torch.float32, device=device)
        x_t = noise.clone()
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = model.denoise_step(
                state, prefix_pad_masks, past, x_t, time.expand(bsize)
            )
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t[0].detach()

    def _encode_frame(self, obs: dict, instruction: str) -> dict[str, Any]:
        element = self._element(obs, instruction)
        observation = self._observation(element)
        model = self.model
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
            observation, train=False
        )
        img_embs, phis = self._embed_images_phis(images, img_masks)
        past, prefix_pad = self._prefix_from_embs(
            img_embs, img_masks, lang_tokens, lang_masks
        )
        phi = torch.stack(phis, dim=0).mean(dim=0) if phis else None
        return {
            "past": clone_past(past),
            "prefix_pad": prefix_pad.detach(),
            "state": state,
            "phis": [p.detach() for p in phis],
            "phi": phi.detach() if phi is not None else None,
            "img_embs": [e.detach() for e in img_embs],
            "img_masks": img_masks,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
        }

    def _deltas(self, phis, prev_phis) -> tuple[float | None, float | None]:
        if not phis or prev_phis is None or len(prev_phis) != len(phis):
            return None, None
        ds = [cosine_delta(a, b) for a, b in zip(phis, prev_phis)]
        phi_now = torch.stack(phis, dim=0).mean(dim=0)
        phi_prev = torch.stack(prev_phis, dim=0).mean(dim=0)
        return max(ds), cosine_delta(phi_now, phi_prev)

    @torch.no_grad()
    def measure_episode(self, frames: list[dict]) -> tuple[list[dict], list[np.ndarray]]:
        packs: list[dict[str, Any] | None] = []
        for fr in frames:
            packs.append(self._encode_frame(fr["obs"], fr["instruction"]))

        device = self.policy._pytorch_device
        model = self.model
        rows: list[dict] = []
        phis_out: list[np.ndarray] = []
        prev_a = None
        T = len(frames)
        for t, pack in enumerate(packs):
            rec = empty_variant_fields(self.delays)
            rec["nfe"] = self.nfe
            rec["capture_lag_ms"] = float(frames[t].get("capture_lag_ms") or 0.0)
            rec["infer_ms"] = frames[t].get("infer_ms")
            if pack is None or pack.get("phi") is None:
                rows.append(rec)
                continue
            bsize = pack["state"].shape[0]
            noise = model.sample_noise(
                (bsize, model.config.action_horizon, model.config.action_dim), device
            )
            a_sync = self._denoise(pack["state"], pack["prefix_pad"], pack["past"], noise)
            if prev_a is not None:
                rec["err_full_consec_l1"] = relative_l1(a_sync, prev_a)
                rec["err_full_consec_abs"] = mean_abs(a_sync, prev_a)
            if t > 0 and packs[t - 1] is not None:
                rec["delta_cam_adj"], rec["delta_cos_adj"] = self._deltas(
                    pack["phis"], packs[t - 1]["phis"]
                )
            for d in self.delays:
                if t >= d and packs[t - d] is not None:
                    old = packs[t - d]
                    rec[f"delta_cam_d{d}"], rec[f"delta_cos_d{d}"] = self._deltas(
                        pack["phis"], old["phis"]
                    )
                    if rec[f"delta_cos_d{d}"] is not None:
                        rec[f"motion_rate_d{d}"] = rec[f"delta_cos_d{d}"] / float(d)
                    a_delay = self._denoise(old["state"], old["prefix_pad"], old["past"], noise)
                    rec[f"err_delay_d{d}_l1"] = relative_l1(a_delay, a_sync)
                    rec[f"err_delay_d{d}_abs"] = mean_abs(a_delay, a_sync)
                    a_img = self._denoise(pack["state"], old["prefix_pad"], old["past"], noise)
                    rec[f"err_img_old_d{d}_l1"] = relative_l1(a_img, a_sync)
                    rec[f"err_img_old_d{d}_abs"] = mean_abs(a_img, a_sync)
                    a_prop = self._denoise(old["state"], pack["prefix_pad"], pack["past"], noise)
                    rec[f"err_prop_old_d{d}_l1"] = relative_l1(a_prop, a_sync)
                    rec[f"err_prop_old_d{d}_abs"] = mean_abs(a_prop, a_sync)
                if t + d < T and packs[t + d] is not None:
                    fut = packs[t + d]
                    a_ahead = self._denoise(fut["state"], fut["prefix_pad"], fut["past"], noise)
                    rec[f"err_ahead_d{d}_l1"] = relative_l1(a_ahead, a_sync)
                    rec[f"err_ahead_d{d}_abs"] = mean_abs(a_ahead, a_sync)
            if t >= 1 and packs[t - 1] is not None:
                prev = packs[t - 1]
                if len(prev["img_embs"]) == len(pack["img_embs"]) and all(
                    a.shape == b.shape for a, b in zip(prev["img_embs"], pack["img_embs"])
                ):
                    embs_mix = [
                        0.5 * a + 0.5 * b for a, b in zip(pack["img_embs"], prev["img_embs"])
                    ]
                    past_mix, pad_mix = self._prefix_from_embs(
                        embs_mix, pack["img_masks"], pack["lang_tokens"], pack["lang_masks"]
                    )
                    a_mix = self._denoise(pack["state"], pad_mix, past_mix, noise)
                    rec["err_mix_k2_l1"] = relative_l1(a_mix, a_sync)
                    rec["err_mix_k2_abs"] = mean_abs(a_mix, a_sync)
            phi_np = pack["phi"].detach().float().cpu().numpy().astype(np.float16, copy=False)
            rec["phi_dim"] = int(phi_np.size)
            rows.append({k: f32(v) if isinstance(v, float) else v for k, v in rec.items()})
            phis_out.append(np.asarray(phi_np).reshape(-1))
            prev_a = a_sync
        packs.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rows, phis_out
