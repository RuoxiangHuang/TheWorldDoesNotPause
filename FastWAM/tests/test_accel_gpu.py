"""Real-`torch.compile` tests. Skipped without CUDA.

The rest of the suite stubs out `torch.compile`, which cannot catch the one
hazard that only appears under `mode="reduce-overhead"`: its CUDA graphs hand
back static output buffers that later calls overwrite, while the step cache
holds a velocity across steps and the video K/V is read all replan long. These
tests exercise that interaction for real.

They pay a one-time compile cost (tens of seconds), so the stand-in model is
kept small.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fasterwam.accel import AccelConfig, AccelStack

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = "cuda"
DT = torch.bfloat16
SA, SV, HID, LAYERS = 32, 24, 256, 4


class _MoT(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(HID, 2 * HID)

    def prefill_video_cache(self, video_tokens):
        k, v = self.enc(video_tokens).chunk(2, dim=-1)
        return [{"k": k, "v": v} for _ in range(LAYERS)]


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mot = _MoT()
        self.inp = nn.Linear(7, HID)
        self.layers = nn.ModuleList(nn.Linear(HID, HID) for _ in range(LAYERS))
        self.head = nn.Linear(HID, 7)

    def encode_prompt(self, prompt):
        return (
            torch.ones(1, 4, HID, device=DEV, dtype=DT),
            torch.ones(1, 4, dtype=torch.bool, device=DEV),
        )

    def _predict_action_noise_with_cache(self, latents_action, timestep_action=None, video_kv_cache=None):
        x = self.inp(latents_action)
        for layer, cache in zip(self.layers, video_kv_cache):
            x = torch.tanh(layer(x)) + cache["k"].mean() + cache["v"].mean()
        return self.head(x)

    @torch.no_grad()
    def infer_action(self, num_inference_steps: int = 8, prompt: str = "task"):
        self.encode_prompt(prompt)
        torch.manual_seed(0)
        video = torch.randn(1, SV, HID, device=DEV, dtype=DT)
        cache = self.mot.prefill_video_cache(video)
        latents = torch.randn(1, SA, 7, device=DEV, dtype=DT)
        for step in range(num_inference_steps):
            velocity = self._predict_action_noise_with_cache(
                latents_action=latents, timestep_action=step, video_kv_cache=cache
            )
            latents = latents + velocity * 0.1
        return latents


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return _Model().to(DEV, DT).eval()


@pytest.fixture(scope="module")
def reference(model):
    return model.infer_action().float()


def test_full_stack_runs_and_stays_close_to_the_reference(model, reference):
    stack = AccelStack(model, AccelConfig.from_any(None, apply_env=False)).install()
    try:
        assert sorted(stack.controllers) == ["compile", "step_cache", "text_cache"]
        out = model.infer_action().float()
        assert torch.isfinite(out).all()
        deviation = ((out - reference).abs().mean() / reference.abs().mean()).item()
        assert deviation < 0.2, f"accelerated output diverged: relL1={deviation:.3e}"
        assert stack.stats()["step_cache"]["skipped"] > 0
    finally:
        stack.uninstall()


def test_uninstall_restores_bit_identical_output(model, reference):
    stack = AccelStack(model, AccelConfig.from_any(None, apply_env=False)).install()
    model.infer_action()
    stack.uninstall()
    assert torch.equal(model.infer_action().float(), reference)


def test_compile_only_is_numerically_faithful(model, reference):
    cfg = AccelConfig.from_any(
        {"step_cache": {"enabled": False}, "text_cache": {"enabled": False}}, apply_env=False
    )
    stack = AccelStack(model, cfg).install()
    try:
        out = model.infer_action().float()
        deviation = ((out - reference).abs().mean() / reference.abs().mean()).item()
        # Compilation only reassociates kernels; deviation must stay at noise level.
        assert deviation < 0.05, f"compile changed the result: relL1={deviation:.3e}"
    finally:
        stack.uninstall()


def test_held_velocity_survives_the_next_compiled_call(model):
    """Without the `.clone()` in the compile wrapper, `reduce-overhead` would
    overwrite the velocity the step cache is holding."""
    cfg = AccelConfig.from_any({"step_cache": {"enabled": False}}, apply_env=False)
    stack = AccelStack(model, cfg).install()
    try:
        video = torch.randn(1, SV, HID, device=DEV, dtype=DT)
        cache = model.mot.prefill_video_cache(video)
        latents = torch.randn(1, SA, 7, device=DEV, dtype=DT)

        with torch.no_grad():
            first = model._predict_action_noise_with_cache(
                latents_action=latents, timestep_action=0, video_kv_cache=cache
            )
            held = first.clone()
            for step in range(1, 4):
                model._predict_action_noise_with_cache(
                    latents_action=latents + step, timestep_action=step, video_kv_cache=cache
                )
        assert torch.equal(first, held), "a later compiled call mutated an earlier output"
    finally:
        stack.uninstall()


def test_prefilled_video_cache_is_stable_across_denoise_steps(model):
    cfg = AccelConfig.from_any({"step_cache": {"enabled": False}}, apply_env=False)
    stack = AccelStack(model, cfg).install()
    try:
        video = torch.randn(1, SV, HID, device=DEV, dtype=DT)
        with torch.no_grad():
            cache = model.mot.prefill_video_cache(video)
            snapshot = [{"k": e["k"].clone(), "v": e["v"].clone()} for e in cache]
            latents = torch.randn(1, SA, 7, device=DEV, dtype=DT)
            for step in range(4):
                model._predict_action_noise_with_cache(
                    latents_action=latents, timestep_action=step, video_kv_cache=cache
                )
        for entry, expected in zip(cache, snapshot):
            assert torch.equal(entry["k"], expected["k"]), "video K was overwritten mid-replan"
            assert torch.equal(entry["v"], expected["v"]), "video V was overwritten mid-replan"
    finally:
        stack.uninstall()
