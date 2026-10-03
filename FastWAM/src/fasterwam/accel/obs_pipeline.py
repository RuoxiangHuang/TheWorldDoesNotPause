"""Observation-encode / action-denoise pipeline overlap (P1).

While the robot executes the previous action chunk, the next observation can
already be VAE-encoded on the GPU. This controller does not change numerics of
a single `infer_action` call; it exposes an async prefetch API the policy
runner can drive:

    pipe = stack.controllers['obs_pipeline']
    pipe.prefetch(next_image)
    # ... execute actions on the robot / sim ...
    out = model.infer_action(input_image=next_image, ...)  # consumes prefetch

If no prefetch matches, inference is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from .base import Controller
from .config import ObsPipelineConfig


@dataclass
class _PrefetchHandle:
    image: torch.Tensor
    latent: torch.Tensor
    event: Optional[torch.cuda.Event]


class ObsPipelineController(Controller):
    """Overlap next-frame VAE encode with current chunk execution."""

    name = "obs_pipeline"
    REQUIRED_ATTRS = ("_encode_input_image_latents_tensor", "infer_action")

    def __init__(self, model: Any, config: ObsPipelineConfig | None = None):
        super().__init__(model)
        self.config = config or ObsPipelineConfig()
        self._prefetch: _PrefetchHandle | None = None
        self._original_encode: Callable | None = None
        self._hits = 0
        self._misses = 0
        self._submitted = 0

    def _install(self) -> None:
        self._patch(self.model, "_encode_input_image_latents_tensor", self._wrap_encode)

    def prefetch(self, input_image: torch.Tensor, tiled: bool = False) -> None:
        """Kick off a VAE encode for the next observation (non-blocking on CUDA)."""
        if not self.config.enabled or self._original_encode is None:
            return
        img = input_image
        if img.ndim == 3:
            img = img.unsqueeze(0)
        latent = self._original_encode(input_image=img, tiled=tiled)
        event = None
        if torch.cuda.is_available() and img.is_cuda:
            event = torch.cuda.Event()
            event.record()
        self._prefetch = _PrefetchHandle(image=img.detach().clone(), latent=latent, event=event)
        self._submitted += 1

    def _wrap_encode(self, original: Callable) -> Callable:
        self._original_encode = original

        def encode(input_image: torch.Tensor, tiled: bool = False, **kwargs):
            img = input_image
            if img.ndim == 3:
                img = img.unsqueeze(0)
            handle = self._prefetch
            if (
                handle is not None
                and handle.image.shape == img.shape
                and handle.image.device == img.device
                and handle.image.dtype == img.dtype
                and torch.equal(handle.image, img)
            ):
                if handle.event is not None:
                    handle.event.synchronize()
                self._hits += 1
                latent = handle.latent
                self._prefetch = None
                return latent.clone()
            self._misses += 1
            self._prefetch = None
            return original(input_image=input_image, tiled=tiled, **kwargs)

        return encode

    def reset(self) -> None:
        self._prefetch = None

    def _on_uninstall(self) -> None:
        self._prefetch = None
        self._original_encode = None

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "submitted": self._submitted,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
        }
