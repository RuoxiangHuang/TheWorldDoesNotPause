"""π₀.₅ shadow probe: current-W reduced NFE vs stale images, same noise."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from benchmarks.dynamic_libero.eval.coupling_math import cosine_delta, mse, relative_l1
from fasterpi.realtime.adapters import libero_obs_to_openpi
from fasterpi.realtime.loader import _batch_to_torch


class PiCouplingProbe:
    def __init__(self, rt_policy: Any, *, nfe_full: int = 10, nfe_mid: int = 8, nfe_cheap: int = 6):
        self.rt = rt_policy
        self.policy = rt_policy.policy
        self.model = rt_policy.policy._model
        self.nfe_full = int(nfe_full)
        self.nfe_mid = int(nfe_mid)
        self.nfe_cheap = int(nfe_cheap)
        self._prev_phis: list[torch.Tensor] | None = None
        self._prev_images: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self) -> None:
        self._prev_phis = None
        self._prev_images = None

    def _element(self, obs: dict, instruction: str) -> dict:
        return libero_obs_to_openpi(obs, instruction, resize=int(self.rt.resize))

    def _observation(self, element: dict):
        from openpi.models import model as _model_mod

        inputs = self.policy._input_transform(element)
        device = self.policy._pytorch_device
        batched = _batch_to_torch(inputs, device)
        return _model_mod.Observation.from_dict(batched), inputs

    def _capture_image_phis(self, observation) -> list[torch.Tensor]:
        model = self.model
        images, img_masks, lang_tokens, lang_masks, _state = model._preprocess_observation(
            observation, train=False
        )
        phis: list[torch.Tensor] = []
        pg = model.paligemma_with_expert
        for img, mask in zip(images, img_masks, strict=True):
            emb = pg.embed_image(img)
            m = mask
            if m.ndim == 1:
                valid = bool(m.reshape(-1)[0].item())
                if not valid:
                    continue
                phis.append(emb.float().mean(dim=1).reshape(-1).detach())
            else:
                # mask is per-batch image validity, tokens all used
                phis.append(emb.float().mean(dim=1).reshape(-1).detach())
        del lang_tokens, lang_masks
        return phis

    @torch.no_grad()
    def measure(self, obs: dict, instruction: str) -> dict[str, Any]:
        element = self._element(obs, instruction)
        observation, _inputs = self._observation(element)
        device = self.policy._pytorch_device
        phis = self._capture_image_phis(observation)

        delta = None
        delta_cam = None
        if self._prev_phis is not None and len(self._prev_phis) == len(phis) and phis:
            ds = [cosine_delta(a, b) for a, b in zip(phis, self._prev_phis)]
            delta_cam = max(ds)
            # global: mean of per-camera phis then cosine (not concat-pool of prefix)
            phi_now = torch.stack(phis, dim=0).mean(dim=0)
            phi_prev = torch.stack(self._prev_phis, dim=0).mean(dim=0)
            delta = cosine_delta(phi_now, phi_prev)

        bsize = observation.state.shape[0]
        noise = self.model.sample_noise(
            (bsize, self.model.config.action_horizon, self.model.config.action_dim),
            device,
        )

        def _sample(nfe: int, obs_use) -> torch.Tensor:
            return self.model.sample_actions(
                device, obs_use, noise=noise.clone(), num_steps=int(nfe)
            )[0].detach()

        a_full = _sample(self.nfe_full, observation)
        a_mid = _sample(self.nfe_mid, observation)
        a_cheap = _sample(self.nfe_cheap, observation)

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
        if self._prev_images is not None:
            stale_el = dict(element)
            stale_el["observation/image"] = self._prev_images[0]
            stale_el["observation/wrist_image"] = self._prev_images[1]
            stale_obs, _ = self._observation(stale_el)
            a_stale = _sample(self.nfe_full, stale_obs)
            rec["err_stale_l1"] = relative_l1(a_stale, a_full)
            rec["err_stale_mse"] = mse(a_stale, a_full)

        self._prev_phis = [p.detach().clone() for p in phis]
        self._prev_images = (
            np.array(element["observation/image"], copy=True),
            np.array(element["observation/wrist_image"], copy=True),
        )
        return rec
