"""torch.compile the action denoise step and the video prefill.

The action step is launch-bound, not compute-bound: pushing ~32 action tokens
through the 30-layer action expert takes roughly 69x the weight-bandwidth
floor, spent on hundreds of tiny kernel launches and Python dispatch over the
layer loop. Compiling with `reduce-overhead` (CUDA graphs) removes that
overhead and is lossless.

This composes multiplicatively with the step cache: the step cache cuts how
many steps run, this cuts what each step costs.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .base import Controller
from .config import CompileConfig


class CompileController(Controller):
    name = "compile"

    @classmethod
    def missing_requirements(cls, model: Any, config: Any = None) -> list[str]:
        config = config or CompileConfig()
        missing = []
        if config.predict and not hasattr(model, "_predict_action_noise_with_cache"):
            missing.append("_predict_action_noise_with_cache")
        if config.prefill and not hasattr(getattr(model, "mot", None), "prefill_video_cache"):
            missing.append("mot.prefill_video_cache")
        return missing

    def __init__(self, model: Any, config: CompileConfig | None = None):
        super().__init__(model)
        self.config = config or CompileConfig()
        self._targets: list[str] = []

    def _install(self) -> None:
        if self.config.suppress_dynamo_errors:
            # `pre_dit` slices a buffer by a data-dependent length, which breaks
            # the graph. Falling back to eager for that op keeps the rest
            # compiled instead of failing the whole inference.
            try:
                import torch._dynamo as dynamo

                dynamo.config.suppress_errors = True
            except Exception:
                pass

        if self.config.predict:
            self._patch(self.model, "_predict_action_noise_with_cache", self._wrap_predict)
            self._targets.append("predict")
        if self.config.prefill:
            self._patch(self.model.mot, "prefill_video_cache", self._wrap_prefill)
            self._targets.append("prefill")

    def _wrap_predict(self, original: Callable) -> Callable:
        compiled = torch.compile(original, mode=self.config.mode)

        def predict(*args, **kwargs):
            # `reduce-overhead` may return a static CUDA-graph buffer that the
            # next call overwrites. The step cache holds this velocity across
            # skipped steps, so it has to own its copy.
            return compiled(*args, **kwargs).clone()

        return predict

    def _wrap_prefill(self, original: Callable) -> Callable:
        prefill_mode = self.config.prefill_mode or self.config.mode
        compiled = torch.compile(original, mode=prefill_mode)
        # Sparse refresh has a data-dependent token index; keep a second compiled
        # entrypoint that still accepts refresh_mask (Inductor falls back where
        # needed) so intermittent sparse calls are not pure eager Python.
        compiled_sparse = torch.compile(original, mode="default")

        def prefill_video_cache(*args, **kwargs):
            use_sparse = kwargs.get("refresh_mask") is not None or (
                len(args) >= 6 and args[5] is not None
            )
            cache = compiled_sparse(*args, **kwargs) if use_sparse else compiled(*args, **kwargs)
            # Same static-buffer hazard, and the K/V is read by every denoise
            # step of the replan: without the copy the predict graph's memory
            # pool can overwrite the video memory mid-replan.
            return [{"k": entry["k"].clone(), "v": entry["v"].clone()} for entry in cache]

        return prefill_video_cache

    def _on_uninstall(self) -> None:
        self._targets.clear()

    def stats(self) -> dict[str, Any]:
        return {"mode": self.config.mode, "targets": list(self._targets)}
