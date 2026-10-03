"""Text-embedding cache: memoize `encode_prompt` across replans.

The task prompt is fixed for the duration of an episode, but the policy
re-encodes it through T5 on every replan (~36 ms, ~6% of a replan). Caching by
prompt string returns bit-identical embeddings, so this controller is exact.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Union

import torch

from .base import Controller
from .config import TextCacheConfig


class TextCacheController(Controller):
    name = "text_cache"
    REQUIRED_ATTRS = ("encode_prompt",)

    def __init__(self, model: Any, config: TextCacheConfig | None = None):
        super().__init__(model)
        self.config = config or TextCacheConfig()
        self._cache: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}
        self._hits = 0
        self._misses = 0

    def _install(self) -> None:
        self._patch(self.model, "encode_prompt", self._wrap)

    def _wrap(self, original: Callable) -> Callable:
        @torch.no_grad()
        def encode_prompt(prompt: Union[str, Sequence[str]]):
            key = prompt if isinstance(prompt, str) else tuple(prompt)
            hit = self._cache.get(key)
            if hit is not None:
                self._hits += 1
                emb, mask = hit
            else:
                self._misses += 1
                emb, mask = original(prompt)
                emb, mask = emb.detach(), mask.detach()
                if len(self._cache) >= self.config.max_entries:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = (emb, mask)
            # Hand out copies: callers may append proprio tokens in place.
            return emb.clone(), mask.clone()

        return encode_prompt

    def _on_uninstall(self) -> None:
        self._cache.clear()

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
            "entries": len(self._cache),
        }
