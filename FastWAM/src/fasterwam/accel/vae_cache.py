"""VAE observation cache across consecutive replans.

When the RGB observation barely moves, re-encoding through the Wan VAE (~2% of
a replan, but 100% redundant) is pure waste. This controller memoizes the last
encoded latent and returns it when the input image relative-L1 stays below a
threshold. Exact under a hit; cold / forced refreshes keep it from going stale.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .base import Controller
from .config import VaeCacheConfig
from .step_cache import relative_l1_tensor


class VaeCacheController(Controller):
    name = "vae_cache"
    REQUIRED_ATTRS = ("_encode_input_image_latents_tensor",)

    def __init__(self, model: Any, config: VaeCacheConfig | None = None):
        super().__init__(model)
        self.config = config or VaeCacheConfig()
        self._hits = 0
        self._misses = 0
        self._reset_state()

    def _reset_state(self) -> None:
        self._last_image: torch.Tensor | None = None
        self._last_latent: torch.Tensor | None = None
        self._call = 0
        self._consecutive = 0

    def reset(self) -> None:
        self._reset_state()

    def _install(self) -> None:
        self._patch(self.model, "_encode_input_image_latents_tensor", self._wrap_encode)

    def _wrap_encode(self, original: Callable) -> Callable:
        def encode(input_image: torch.Tensor, tiled: bool = False, **kwargs):
            cfg = self.config
            call = self._call
            self._call += 1

            img = input_image
            if img.ndim == 3:
                img = img.unsqueeze(0)

            can_hit = (
                self._last_image is not None
                and self._last_latent is not None
                and call >= cfg.cold_calls
                and self._consecutive < cfg.max_consecutive_hits
                and (cfg.force_refresh_every == 0 or call % cfg.force_refresh_every != 0)
                and self._last_image.shape == img.shape
                and self._last_image.device == img.device
                and self._last_image.dtype == img.dtype
            )
            if can_hit and float(relative_l1_tensor(img, self._last_image).item()) < cfg.threshold:
                self._hits += 1
                self._consecutive += 1
                return self._last_latent.clone()

            latent = original(input_image=input_image, tiled=tiled, **kwargs)
            self._misses += 1
            self._consecutive = 0
            self._last_image = img.detach().clone()
            self._last_latent = latent.detach().clone()
            return latent

        return encode

    def _on_uninstall(self) -> None:
        self._reset_state()

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
        }
