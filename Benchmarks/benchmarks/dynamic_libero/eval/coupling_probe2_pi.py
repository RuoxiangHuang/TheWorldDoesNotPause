"""π₀.₅ probe-2: cost split, cache age, stale images vs cached prefix KV."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from benchmarks.dynamic_libero.eval.coupling_math import cosine_delta, mean_abs, relative_l1
from benchmarks.dynamic_libero.eval.coupling_probe2_common import (
    AGES,
    CudaTimer,
    clone_past,
    f32,
)
from fasterpi.realtime.adapters import libero_obs_to_openpi
from fasterpi.realtime.loader import _batch_to_torch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


class PiProbe2:
    def __init__(self, rt_policy: Any, *, nfe: int = 10, ages: tuple[int, ...] = AGES):
        self.rt = rt_policy
        self.policy = rt_policy.policy
        self.model = rt_policy.policy._model
        self.nfe = int(nfe)
        self.ages = tuple(int(a) for a in ages)
        self._hist: deque[dict[str, Any]] = deque(maxlen=max(self.ages) + 1)
        self._prev_a_full: torch.Tensor | None = None

    def reset(self) -> None:
        self._hist.clear()
        self._prev_a_full = None

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

    def _embed_images_phis(self, images, img_masks, timer: CudaTimer):
        model = self.model
        pg = model.paligemma_with_expert
        embs = []
        phis = []
        with timer.span("siglip"):
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

    def _prefix_from_embs(self, img_embs, img_masks, lang_tokens, lang_masks, timer: CudaTimer):
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
        with timer.span("prefix"):
            _, past = pg.forward(
                attention_mask=prefix_att_4d,
                position_ids=prefix_pos,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        return past, prefix_pad_masks

    def _denoise(self, state, prefix_pad_masks, past, noise, nfe: int, timer: CudaTimer | None):
        model = self.model
        device = noise.device
        bsize = state.shape[0]
        dt = torch.tensor(-1.0 / int(nfe), dtype=torch.float32, device=device)
        x_t = noise.clone()
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        tm = timer or CudaTimer(False)
        with tm.span("action"):
            while time >= -dt / 2:
                v_t = model.denoise_step(
                    state, prefix_pad_masks, past, x_t, time.expand(bsize)
                )
                x_t = x_t + dt * v_t
                time = time + dt
        return x_t[0].detach(), tm.ms.get("action")

    def _deltas(self, phis, prev_phis) -> tuple[float | None, float | None]:
        if not phis or prev_phis is None or len(prev_phis) != len(phis):
            return None, None
        ds = [cosine_delta(a, b) for a, b in zip(phis, prev_phis)]
        phi_now = torch.stack(phis, dim=0).mean(dim=0)
        phi_prev = torch.stack(prev_phis, dim=0).mean(dim=0)
        return max(ds), cosine_delta(phi_now, phi_prev)

    @torch.no_grad()
    def measure(self, obs: dict, instruction: str) -> dict[str, Any]:
        element = self._element(obs, instruction)
        observation = self._observation(element)
        model = self.model
        device = self.policy._pytorch_device
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
            observation, train=False
        )
        timer = CudaTimer(True)
        img_embs, phis = self._embed_images_phis(images, img_masks, timer)
        past, prefix_pad = self._prefix_from_embs(
            img_embs, img_masks, lang_tokens, lang_masks, timer
        )
        bsize = observation.state.shape[0]
        noise = model.sample_noise(
            (bsize, model.config.action_horizon, model.config.action_dim), device
        )
        a_full, ms_act = self._denoise(state, prefix_pad, past, noise, self.nfe, timer)

        rec: dict[str, Any] = {
            "nfe": self.nfe,
            "ms_siglip": timer.ms.get("siglip"),
            "ms_prefix": timer.ms.get("prefix"),
            "ms_action": ms_act,
            "err_full_consec_l1": None,
            "err_full_consec_abs": None,
            "delta_cam_adj": None,
            "delta_cos_adj": None,
        }
        if self._prev_a_full is not None:
            rec["err_full_consec_l1"] = relative_l1(a_full, self._prev_a_full)
            rec["err_full_consec_abs"] = mean_abs(a_full, self._prev_a_full)
        if self._hist:
            rec["delta_cam_adj"], rec["delta_cos_adj"] = self._deltas(phis, self._hist[-1]["phis"])

        for age in self.ages:
            rec[f"delta_cam_age{age}"] = None
            rec[f"delta_cos_age{age}"] = None
            rec[f"err_stale_img_l1_age{age}"] = None
            rec[f"err_stale_img_abs_age{age}"] = None
            rec[f"err_stale_prefix_l1_age{age}"] = None
            rec[f"err_stale_prefix_abs_age{age}"] = None
            rec[f"err_exec_img_vs_prefix_l1_age{age}"] = None
            if len(self._hist) < age:
                continue
            snap = self._hist[-age]
            dcam, dcos = self._deltas(phis, snap["phis"])
            rec[f"delta_cam_age{age}"] = dcam
            rec[f"delta_cos_age{age}"] = dcos
            a_kv, _ = self._denoise(
                state, snap["prefix_pad"], snap["past"], noise, self.nfe, CudaTimer(False)
            )
            rec[f"err_stale_prefix_l1_age{age}"] = relative_l1(a_kv, a_full)
            rec[f"err_stale_prefix_abs_age{age}"] = mean_abs(a_kv, a_full)

            if age != 1:
                continue
            stale_el = dict(element)
            stale_el["observation/image"] = snap["image"]
            stale_el["observation/wrist_image"] = snap["wrist"]
            stale_obs = self._observation(stale_el)
            simg, smask, slang, slang_m, _sstate = model._preprocess_observation(
                stale_obs, train=False
            )
            t2 = CudaTimer(False)
            sembs, _ = self._embed_images_phis(simg, smask, t2)
            spast, spad = self._prefix_from_embs(sembs, smask, slang, slang_m, t2)
            a_img, _ = self._denoise(state, spad, spast, noise, self.nfe, t2)
            rec[f"err_stale_img_l1_age{age}"] = relative_l1(a_img, a_full)
            rec[f"err_stale_img_abs_age{age}"] = mean_abs(a_img, a_full)
            rec[f"err_exec_img_vs_prefix_l1_age{age}"] = relative_l1(a_kv, a_img)

        self._hist.append(
            {
                "phis": [p.detach().clone() for p in phis],
                "past": clone_past(past),
                "prefix_pad": prefix_pad.detach().clone(),
                "image": np.array(element["observation/image"], copy=True),
                "wrist": np.array(element["observation/wrist_image"], copy=True),
            }
        )
        self._prev_a_full = a_full.detach().clone()
        rec["delta_cam_max"] = rec.get("delta_cam_age1")
        rec["delta_cos"] = rec.get("delta_cos_age1")
        rec["err_stale_l1"] = rec.get("err_stale_prefix_l1_age1")
        return {k: f32(v) if isinstance(v, float) else v for k, v in rec.items()}
