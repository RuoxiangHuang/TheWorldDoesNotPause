"""Integration tests for `prepare_for_inference`.

The deployment contract is that going through this function is what turns
acceleration on, and that it does so in an order that cannot fail: eval mode
before install, since the stack refuses to attach to a training model.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fasterwam.accel import AccelConfig
from fasterwam.accel.stack import get_stack
from fasterwam.runtime import ACCEL_CONFIG_ATTR, _attach_accel_config, prepare_for_inference


class TinyMoT(nn.Module):
    def prefill_video_cache(self, **kwargs):
        return [{"k": torch.ones(1, 8, 4), "v": torch.ones(1, 8, 4)}]


class TinyModel(nn.Module):
    """A stand-in exposing the surface `prepare_for_inference` touches."""

    def __init__(self):
        super().__init__()
        self.mot = TinyMoT()
        self.weight = nn.Parameter(torch.ones(1))
        self.loaded_from: str | None = None

    def load_checkpoint(self, path):
        self.loaded_from = str(path)

    def encode_prompt(self, prompt):
        return torch.ones(1, 4, 8), torch.ones(1, 4, dtype=torch.bool)

    def _predict_action_noise_with_cache(self, latents_action, **kwargs):
        return torch.full_like(latents_action, 1e-3)

    def infer_action(self, num_inference_steps: int = 8, prompt: str = "task"):
        self.encode_prompt(prompt)
        self.mot.prefill_video_cache()
        latents = torch.ones(1, 4, 7)
        for step in range(num_inference_steps):
            latents = latents + self._predict_action_noise_with_cache(latents_action=latents) * 0.1
        return latents


@pytest.fixture(autouse=True)
def _stub_compile(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)


def test_prepare_for_inference_evals_then_accelerates():
    model = TinyModel()
    assert model.training  # nn.Module default; the stack would refuse this
    model, stack = prepare_for_inference(model)
    assert not model.training
    assert stack.active
    assert get_stack(model) is stack
    model.infer_action()


def test_attached_config_is_used_when_none_is_passed():
    model = TinyModel()
    _attach_accel_config(model, {"step_cache": {"threshold": 0.42}})
    assert isinstance(getattr(model, ACCEL_CONFIG_ATTR), AccelConfig)
    _model, stack = prepare_for_inference(model)
    assert stack.controllers["step_cache"].config.threshold == pytest.approx(0.42)


def test_explicit_accel_argument_overrides_the_attached_config():
    model = TinyModel()
    _attach_accel_config(model, None)
    _model, stack = prepare_for_inference(model, accel={"enabled": False})
    assert not stack.active


def test_env_overlay_does_not_mutate_the_attached_config(monkeypatch):
    model = TinyModel()
    _attach_accel_config(model, {"step_cache": {"threshold": 0.2}})
    monkeypatch.setenv("FASTERWAM_ACCEL_STEP_CACHE_THRESHOLD", "0.4")
    _model, stack = prepare_for_inference(model)
    assert stack.controllers["step_cache"].config.threshold == pytest.approx(0.4)
    # The config recorded at construction must be left as authored.
    assert getattr(model, ACCEL_CONFIG_ATTR).step_cache.threshold == pytest.approx(0.2)


def test_missing_checkpoint_raises_by_default(tmp_path):
    model = TinyModel()
    with pytest.raises(FileNotFoundError):
        prepare_for_inference(model, checkpoint_path=tmp_path / "absent.pt")


def test_missing_checkpoint_can_be_tolerated(tmp_path):
    model = TinyModel()
    _model, stack = prepare_for_inference(
        model, checkpoint_path=tmp_path / "absent.pt", strict_checkpoint=False
    )
    assert stack.active
    assert model.loaded_from is None


def test_existing_checkpoint_is_loaded_before_install(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"")
    model = TinyModel()
    _model, stack = prepare_for_inference(model, checkpoint_path=ckpt)
    assert model.loaded_from == str(ckpt)
    assert stack.active


def test_accelerated_and_reference_paths_agree_closely():
    """The stack is near-lossless: with a settled latent the skipped steps must
    not move the result far from the reference integration."""
    reference = TinyModel().eval().infer_action(num_inference_steps=10)

    model = TinyModel()
    _model, stack = prepare_for_inference(model)
    fast = model.infer_action(num_inference_steps=10)
    assert stack.stats()["step_cache"]["skipped"] > 0

    deviation = (fast - reference).abs().mean() / reference.abs().mean()
    assert deviation < 0.01
