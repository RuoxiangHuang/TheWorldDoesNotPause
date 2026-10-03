"""Unit tests for the acceleration stack.

These use a stand-in model rather than a real checkpoint, so they run on CPU in
under a second and cover the parts that are easy to get subtly wrong: the
install order, the skip gate, and whether uninstall really restores the
reference path. `torch.compile` is stubbed out — what is tested here is the
wrapper contract around it, not the compiler.
"""

from __future__ import annotations

import torch

from fasterwam.accel import AccelConfig, AccelStack, accelerate
from fasterwam.accel.stack import get_stack


class FakeMoT:
    def __init__(self):
        self.prefill_calls = 0
        self.dit_calls = 0
        self.last_refresh_mask = None

    def prefill_video_cache(
        self,
        video_tokens=None,
        video_freqs=None,
        video_t_mod=None,
        video_context_payload=None,
        video_attention_mask=None,
        refresh_mask=None,
        previous_cache=None,
        **kwargs,
    ):
        self.prefill_calls += 1
        self.last_refresh_mask = None if refresh_mask is None else refresh_mask.clone()
        tokens = video_tokens if video_tokens is not None else torch.ones(1, 8, 4)
        seq = tokens.shape[1]
        if refresh_mask is not None and previous_cache is not None and not refresh_mask.all():
            cache = [{"k": e["k"].clone(), "v": e["v"].clone()} for e in previous_cache]
            dyn = refresh_mask.nonzero(as_tuple=False).reshape(-1)
            for entry in cache:
                entry["k"].index_copy_(1, dyn, tokens.index_select(1, dyn))
                entry["v"].index_copy_(1, dyn, tokens.index_select(1, dyn))
            return cache
        return [{"k": tokens.clone(), "v": tokens.clone()}]

    def forward_action_with_video_cache(self, action_tokens, **kwargs):
        self.dit_calls += 1
        # Constant residual R = 1 so thin-path reuse is easy to assert.
        return action_tokens + 1.0


class FakeActionExpert:
    def pre_dit(self, action_tokens, timestep=None, context=None, context_mask=None):
        return {
            "tokens": action_tokens.clone(),
            "freqs": torch.zeros(action_tokens.shape[1], 1, 4),
            "t_mod": torch.zeros(1, 6, 4),
            "context": context if context is not None else torch.ones(1, 4, 8),
            "context_mask": context_mask,
        }

    def post_dit(self, tokens, pre_state):
        # Velocity proportional to hidden state so residual reuse changes output.
        return tokens * 1e-3


class FakeModel:
    """Exposes the same hooks the controllers patch on a real FastWAM."""

    def __init__(self, velocity_scale: float = 1e-3):
        self.training = False
        self.mot = FakeMoT()
        self.action_expert = FakeActionExpert()
        self.velocity_scale = velocity_scale
        self.predict_calls = 0
        self.encode_calls = 0
        self._token_phase = 0.0

    def encode_prompt(self, prompt):
        self.encode_calls += 1
        return torch.ones(1, 4, 8), torch.ones(1, 4, dtype=torch.bool)

    def _predict_action_noise_with_cache(
        self,
        latents_action,
        timestep_action=None,
        context=None,
        context_mask=None,
        video_kv_cache=None,
        attention_mask=None,
        video_seq_len=None,
        **kwargs,
    ):
        self.predict_calls += 1
        if context is None:
            context = torch.ones(1, 4, 8)
        if video_kv_cache is None:
            video_kv_cache = [{"k": torch.ones(1, 8, 4), "v": torch.ones(1, 8, 4)}]
        if attention_mask is None:
            attention_mask = torch.ones(12, 12, dtype=torch.bool)
        if video_seq_len is None:
            video_seq_len = 8
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(tokens, action_pre)

    def _encode_input_image_latents_tensor(self, input_image, tiled=False, **kwargs):
        self.encode_image_calls = getattr(self, "encode_image_calls", 0) + 1
        return torch.ones(1, 48, 1, 4, 4)

    def infer_action(
        self,
        num_inference_steps: int = 10,
        prompt: str = "task",
        token_drift: float = 0.0,
    ):
        context, context_mask = self.encode_prompt(prompt)
        base = torch.ones(1, 8, 4)
        tokens = base + token_drift
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=tokens,
            video_freqs=torch.zeros(8, 1, 4),
            video_t_mod=torch.zeros(1, 8, 6, 4),
            video_context_payload=None,
            video_attention_mask=torch.ones(8, 8, dtype=torch.bool),
        )
        latents = torch.ones(1, 4, 7)
        attention_mask = torch.ones(12, 12, dtype=torch.bool)
        for step in range(num_inference_steps):
            velocity = self._predict_action_noise_with_cache(
                latents_action=latents,
                timestep_action=torch.tensor([float(step)]),
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=8,
            )
            latents = latents + velocity * 0.1
        return latents


class VideoOnlyModel:
    """Like Wan22Core: has a prompt encoder but no action-denoising path."""

    def __init__(self):
        self.training = False

    def encode_prompt(self, prompt):
        return torch.ones(1, 4, 8), torch.ones(1, 4, dtype=torch.bool)


def _identity_compile(monkeypatch):
    """Make torch.compile a no-op so the wrapper contract is what gets tested."""
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)


# --- config -----------------------------------------------------------------


def test_defaults_enable_core_controllers():
    cfg = AccelConfig()
    assert cfg.active
    assert cfg.text_cache.active and cfg.step_cache.active and cfg.compile.active
    # video_token_cache is a first-class section but off in AccelConfig defaults /
    # accel=default; enable via accel=p0 / p1 / full.
    assert not cfg.video_token_cache.active
    assert "video_token_cache" in AccelConfig._SECTIONS
    assert not cfg.chunk_residual_cache.active
    assert "chunk_residual_cache" in AccelConfig._SECTIONS


def test_env_overrides_win_over_config(monkeypatch):
    monkeypatch.setenv("FASTERWAM_ACCEL_STEP_CACHE_THRESHOLD", "0.35")
    monkeypatch.setenv("FASTERWAM_ACCEL_COMPILE", "0")
    cfg = AccelConfig.from_any({"step_cache": {"threshold": 0.2}})
    assert cfg.step_cache.threshold == 0.35
    assert not cfg.compile.active


def test_global_env_kill_switch(monkeypatch):
    monkeypatch.setenv("FASTERWAM_ACCEL", "0")
    assert not AccelConfig.from_any(None).active


def test_unknown_config_key_is_rejected():
    try:
        AccelConfig.from_any({"step_cahce": {}})
    except ValueError as exc:
        assert "step_cahce" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a misspelled key")


# --- install / uninstall ----------------------------------------------------


def test_uninstall_restores_the_reference_path(monkeypatch):
    _identity_compile(monkeypatch)
    model = FakeModel()
    reference = model.infer_action(num_inference_steps=10)
    baseline_calls = model.predict_calls

    stack = accelerate(model)
    assert stack.active
    model.infer_action(num_inference_steps=10)
    assert model.predict_calls < baseline_calls * 2  # some steps were skipped

    stack.uninstall()
    assert not stack.active
    assert get_stack(model) is None
    # No leftover instance attributes shadowing the class methods.
    assert "infer_action" not in vars(model)
    assert "_predict_action_noise_with_cache" not in vars(model)
    assert "encode_prompt" not in vars(model)

    model.predict_calls = 0
    restored = model.infer_action(num_inference_steps=10)
    assert model.predict_calls == baseline_calls
    assert torch.equal(restored, reference)


def test_stack_works_as_a_context_manager(monkeypatch):
    _identity_compile(monkeypatch)
    model = FakeModel()
    with AccelStack(model, AccelConfig()) as stack:
        assert stack.active
        model.infer_action(num_inference_steps=6)
    assert not stack.active
    assert "encode_prompt" not in vars(model)


def test_double_install_is_refused(monkeypatch):
    _identity_compile(monkeypatch)
    model = FakeModel()
    accelerate(model)
    try:
        accelerate(model)
    except RuntimeError as exc:
        assert "already has an acceleration stack" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError on double install")


def test_training_model_is_refused():
    model = FakeModel()
    model.training = True
    try:
        accelerate(model)
    except RuntimeError as exc:
        assert "training mode" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError for a training model")


def test_disabled_config_installs_nothing():
    model = FakeModel()
    model.training = True  # allowed, because nothing gets installed
    stack = accelerate(model, {"enabled": False})
    assert not stack.active
    assert "encode_prompt" not in vars(model)


def test_controllers_without_hooks_are_skipped(monkeypatch):
    _identity_compile(monkeypatch)
    model = VideoOnlyModel()
    stack = accelerate(model)
    # Only the prompt cache applies; the action-path controllers are absent.
    assert sorted(stack.controllers) == ["text_cache"]
    stack.uninstall()


def test_compile_installs_before_step_cache(monkeypatch):
    """The order is a correctness constraint, so assert it directly."""
    compiled: list[str] = []

    def tracking_compile(fn, **kwargs):
        compiled.append(getattr(fn, "__name__", repr(fn)))
        return fn

    monkeypatch.setattr(torch, "compile", tracking_compile)
    model = FakeModel()
    accelerate(model)
    # compile must have wrapped the model's own forward, not the step cache's
    # Python gate (which is named `predict`).
    assert "_predict_action_noise_with_cache" in compiled
    assert "predict" not in compiled


# --- step cache behaviour ---------------------------------------------------


def test_cold_steps_always_compute():
    model = FakeModel()
    stack = accelerate(
        model,
        {"compile": {"enabled": False}, "step_cache": {"cold_steps": 3, "threshold": 1e9}},
    )
    model.infer_action(num_inference_steps=3)
    assert model.predict_calls == 3
    assert stack.stats()["step_cache"]["skipped"] == 0


def test_consecutive_skips_are_bounded():
    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            # A huge threshold would otherwise skip everything after warmup.
            "step_cache": {"cold_steps": 1, "threshold": 1e9, "max_consecutive_skips": 2},
        },
    )
    model.infer_action(num_inference_steps=10)
    stats = stack.stats()["step_cache"]
    assert stats["computed"] + stats["skipped"] == 10
    # With at most 2 held steps per computed one, a third of steps must compute.
    assert stats["computed"] >= 10 // 3


def test_a_tight_threshold_disables_skipping():
    model = FakeModel()
    stack = accelerate(
        model,
        {"compile": {"enabled": False}, "step_cache": {"threshold": 0.0}},
    )
    model.infer_action(num_inference_steps=8)
    assert stack.stats()["step_cache"]["skipped"] == 0
    assert model.predict_calls == 8


def test_state_does_not_leak_between_replans():
    model = FakeModel()
    stack = accelerate(
        model,
        {"compile": {"enabled": False}, "step_cache": {"cold_steps": 2, "threshold": 1e9}},
    )
    model.infer_action(num_inference_steps=4)
    first = stack.controllers["step_cache"].stats()["computed"]
    model.infer_action(num_inference_steps=4)
    second = stack.controllers["step_cache"].stats()["computed"] - first
    # Each replan pays the same cold-start cost; a held velocity must not carry
    # over from the previous replan.
    assert first == second


def test_skipped_steps_reuse_the_last_velocity():
    model = FakeModel()
    accelerate(
        model,
        {"compile": {"enabled": False}, "step_cache": {"cold_steps": 1, "threshold": 1e9}},
    )
    latents = torch.ones(1, 4, 7)
    first = model._predict_action_noise_with_cache(latents_action=latents)
    second = model._predict_action_noise_with_cache(latents_action=latents + 1e-6)
    assert model.predict_calls == 1
    assert torch.equal(first, second)


# --- text cache -------------------------------------------------------------


def test_text_cache_is_exact_and_isolated():
    model = FakeModel()
    stack = accelerate(model, {"compile": {"enabled": False}, "step_cache": {"enabled": False}})
    first, mask = model.encode_prompt("pick up the bowl")
    second, _ = model.encode_prompt("pick up the bowl")
    assert torch.equal(first, second)
    assert model.encode_calls == 1

    # A caller mutating the returned embedding must not poison the cache.
    first.add_(5.0)
    third, _ = model.encode_prompt("pick up the bowl")
    assert torch.equal(third, second)

    model.encode_prompt("a different task")
    stats = stack.stats()["text_cache"]
    assert stats["hits"] == 2 and stats["misses"] == 2


def test_text_cache_evicts_at_capacity():
    model = FakeModel()
    stack = accelerate(
        model,
        {"compile": {"enabled": False}, "step_cache": {"enabled": False},
         "text_cache": {"max_entries": 2}},
    )
    for prompt in ("a", "b", "c"):
        model.encode_prompt(prompt)
    assert stack.stats()["text_cache"]["entries"] == 2


# --- compile wrappers -------------------------------------------------------


def test_compiled_outputs_are_cloned(monkeypatch):
    """reduce-overhead can hand back a reused static buffer; the step cache
    holds those tensors across steps, so the wrappers must copy them."""
    static_out = torch.zeros(1, 4, 7)
    static_cache = [{"k": torch.zeros(1, 8, 4), "v": torch.zeros(1, 8, 4)}]

    class StaticBufferModel(FakeModel):
        def _predict_action_noise_with_cache(self, latents_action, **kwargs):
            self.predict_calls += 1
            return static_out

    model = StaticBufferModel()
    model.mot.prefill_video_cache = lambda **kw: static_cache
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    accelerate(model, {"step_cache": {"enabled": False}})

    velocity = model._predict_action_noise_with_cache(latents_action=torch.ones(1, 4, 7))
    cache = model.mot.prefill_video_cache()
    assert velocity is not static_out
    assert cache[0]["k"] is not static_cache[0]["k"]

    static_out.fill_(9.0)
    static_cache[0]["k"].fill_(9.0)
    assert velocity.abs().sum() == 0
    assert cache[0]["k"].abs().sum() == 0


# --- video token cache ------------------------------------------------------


def test_video_token_cache_reuses_identical_tokens(monkeypatch):
    _identity_compile(monkeypatch)
    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            "step_cache": {"enabled": False},
            "video_token_cache": {
                "enabled": True,
                "cold_replans": 1,
                "full_reuse_threshold": 0.02,
                "max_consecutive_reuses": 8,
                "force_refresh_every": 0,
            },
        },
    )
    model.infer_action(token_drift=0.0)
    model.infer_action(token_drift=0.0)
    model.infer_action(token_drift=0.0)
    stats = stack.stats()["video_token_cache"]
    assert stats["full_prefill"] == 1
    assert stats["full_reuse"] == 2
    assert model.mot.prefill_calls == 1


def test_video_token_cache_sparse_refresh_on_local_drift(monkeypatch):
    _identity_compile(monkeypatch)
    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            "step_cache": {"enabled": False},
            "video_token_cache": {
                "enabled": True,
                "cold_replans": 1,
                "full_reuse_threshold": 0.001,
                "token_threshold": 0.05,
                "max_refresh_ratio": 0.5,
                "force_refresh_every": 0,
                "max_consecutive_reuses": 8,
            },
        },
    )
    model.infer_action(token_drift=0.0)
    tokens = torch.ones(1, 8, 4)
    tokens[:, :2] += 1.0
    model.mot.prefill_video_cache(
        video_tokens=tokens,
        video_freqs=torch.zeros(8, 1, 4),
        video_t_mod=torch.zeros(1, 8, 6, 4),
        video_context_payload=None,
        video_attention_mask=torch.ones(8, 8, dtype=torch.bool),
    )
    stats = stack.stats()["video_token_cache"]
    assert stats["sparse_refresh"] == 1
    assert model.mot.last_refresh_mask is not None
    assert int(model.mot.last_refresh_mask.sum().item()) == 2


def test_video_token_cache_force_refresh_bounds_staleness(monkeypatch):
    _identity_compile(monkeypatch)
    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            "step_cache": {"enabled": False},
            "video_token_cache": {
                "enabled": True,
                "cold_replans": 1,
                "full_reuse_threshold": 1e9,
                "max_consecutive_reuses": 100,
                "force_refresh_every": 3,
            },
        },
    )
    for _ in range(6):
        model.infer_action(token_drift=0.0)
    stats = stack.stats()["video_token_cache"]
    assert stats["full_prefill"] >= 2


def test_vae_cache_hits_on_identical_images(monkeypatch):
    _identity_compile(monkeypatch)
    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            "step_cache": {"enabled": False},
            "vae_cache": {"enabled": True, "cold_calls": 1, "threshold": 0.01, "force_refresh_every": 0},
        },
    )
    img = torch.ones(1, 3, 8, 8)
    model._encode_input_image_latents_tensor(img)
    model._encode_input_image_latents_tensor(img)
    model._encode_input_image_latents_tensor(img)
    stats = stack.stats()["vae_cache"]
    assert stats["misses"] == 1
    assert stats["hits"] == 2


def test_obs_pipeline_prefetch_is_consumed(monkeypatch):
    _identity_compile(monkeypatch)
    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            "step_cache": {"enabled": False},
            "obs_pipeline": {"enabled": True},
        },
    )
    pipe = stack.controllers["obs_pipeline"]
    img = torch.ones(1, 3, 8, 8)
    pipe.prefetch(img)
    out = model._encode_input_image_latents_tensor(img)
    assert out.shape[1] == 48
    assert stack.stats()["obs_pipeline"]["hits"] == 1


# --- chunk residual cache ---------------------------------------------------


def test_chunk_residual_cache_reuses_across_chunks(monkeypatch):
    _identity_compile(monkeypatch)
    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            "step_cache": {"enabled": False},
            "chunk_residual_cache": {
                "enabled": True,
                "cold_chunks": 1,
                "condition_threshold": 1e9,
                "action_latent_threshold": 1e9,
                "max_consecutive_reuses": 8,
                "force_refresh_every": 0,
            },
        },
    )
    model.infer_action(num_inference_steps=4, token_drift=0.0)
    dit_after_cold = model.mot.dit_calls
    assert dit_after_cold == 4
    model.infer_action(num_inference_steps=4, token_drift=0.0)
    # Second chunk should hit residual and skip MoT for all 4 steps.
    assert model.mot.dit_calls == dit_after_cold
    stats = stack.stats()["chunk_residual_cache"]
    assert stats["full_compute"] == 4
    assert stats["residual_hits"] == 4
    assert stats["hit_rate"] == 0.5


def test_chunk_residual_blocks_divergent_action_latents(monkeypatch):
    _identity_compile(monkeypatch)
    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            "step_cache": {"enabled": False},
            "chunk_residual_cache": {
                "enabled": True,
                "cold_chunks": 1,
                "condition_threshold": 1e9,
                # drift >= 0 blocks every reuse, including exact matches.
                "action_latent_threshold": 0.0,
                "max_consecutive_reuses": 8,
            },
        },
    )
    model.infer_action(num_inference_steps=2, token_drift=0.0)
    model.infer_action(num_inference_steps=2, token_drift=0.0)
    stats = stack.stats()["chunk_residual_cache"]
    assert stats["residual_hits"] == 0
    assert stats["latent_blocked"] >= 1


def test_chunk_residual_force_refresh(monkeypatch):
    _identity_compile(monkeypatch)
    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            "step_cache": {"enabled": False},
            "chunk_residual_cache": {
                "enabled": True,
                "cold_chunks": 1,
                "condition_threshold": 1e9,
                "action_latent_threshold": 1e9,
                "max_consecutive_reuses": 100,
                "force_refresh_every": 2,
            },
        },
    )
    for _ in range(4):
        model.infer_action(num_inference_steps=2, token_drift=0.0)
    stats = stack.stats()["chunk_residual_cache"]
    # chunk0 cold full, chunk1 reuse, chunk2 force full, chunk3 reuse
    assert stats["full_compute"] == 4  # 2 steps * 2 full chunks
    assert stats["residual_hits"] == 4


def test_chunk_residual_skips_when_step_cache_holds(monkeypatch):
    """step_cache outermost: held steps must not read/write residual tables."""
    _identity_compile(monkeypatch)
    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            "step_cache": {
                "enabled": True,
                "cold_steps": 1,
                "threshold": 1e9,
                "max_consecutive_skips": 10,
            },
            "chunk_residual_cache": {
                "enabled": True,
                "cold_chunks": 1,
                "condition_threshold": 1e9,
                "action_latent_threshold": 1e9,
                "max_consecutive_reuses": 8,
            },
        },
    )
    assert AccelStack.INSTALL_ORDER.index("chunk_residual_cache") < AccelStack.INSTALL_ORDER.index(
        "step_cache"
    )
    model.infer_action(num_inference_steps=4, token_drift=0.0)
    residual = stack.controllers["chunk_residual_cache"]
    # Only computed (non-held) steps enter residual; with cold=1 and unbounded
    # consecutive skips, steps 1..3 are held → only step 0 writes R.
    assert len(residual._residuals) == 1
    assert 0 in residual._residuals


def test_chunk_residual_install_order_outside_compile(monkeypatch):
    _identity_compile(monkeypatch)
    assert AccelStack.INSTALL_ORDER.index("compile") < AccelStack.INSTALL_ORDER.index(
        "chunk_residual_cache"
    )
    assert AccelStack.INSTALL_ORDER.index("chunk_residual_cache") < AccelStack.INSTALL_ORDER.index(
        "step_cache"
    )


def test_chunk_residual_in_summary():
    cfg = AccelConfig.from_any(
        {"chunk_residual_cache": {"enabled": True, "condition_threshold": 0.05}},
        apply_env=False,
    )
    assert "chunk_residual_cache" in cfg.summary()


def test_shared_video_token_state_unifies_drift_and_clone(monkeypatch):
    """Both controllers share one VideoTokenState; drift computed once per prefill."""
    _identity_compile(monkeypatch)
    from fasterwam.accel.world_state import VideoTokenState

    drift_calls = {"n": 0}
    token_clones = {"n": 0}
    original_compute = VideoTokenState._compute_drift
    original_exit = VideoTokenState.exit_prefill

    def counting_compute(self, current):
        drift_calls["n"] += 1
        return original_compute(self, current)

    def counting_exit(self, current):
        token_clones["n"] += 1
        return original_exit(self, current)

    monkeypatch.setattr(VideoTokenState, "_compute_drift", counting_compute)
    monkeypatch.setattr(VideoTokenState, "exit_prefill", counting_exit)

    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            "step_cache": {"enabled": False},
            "video_token_cache": {
                "enabled": True,
                "cold_replans": 1,
                "full_reuse_threshold": 0.02,
                "max_consecutive_reuses": 8,
                "force_refresh_every": 0,
            },
            "chunk_residual_cache": {
                "enabled": True,
                "cold_chunks": 1,
                "condition_threshold": 0.02,
                "action_latent_threshold": 1e9,
                "max_consecutive_reuses": 8,
                "force_refresh_every": 0,
            },
        },
    )
    vtc = stack.controllers["video_token_cache"]
    crc = stack.controllers["chunk_residual_cache"]
    assert vtc._video_state is crc._video_state

    model.infer_action(num_inference_steps=2, token_drift=0.0)
    model.infer_action(num_inference_steps=2, token_drift=0.0)
    model.infer_action(num_inference_steps=2, token_drift=0.0)

    assert drift_calls["n"] == 3
    assert token_clones["n"] == 3
    assert stack.stats()["video_token_cache"]["full_reuse"] == 2
    assert stack.stats()["chunk_residual_cache"]["residual_hits"] == 4


def test_shared_video_token_state_video_only_still_works(monkeypatch):
    """video_token_cache alone still owns enter/exit when chunk_residual is off."""
    _identity_compile(monkeypatch)
    from fasterwam.accel.world_state import VideoTokenState

    drift_calls = {"n": 0}
    original_compute = VideoTokenState._compute_drift

    def counting_compute(self, current):
        drift_calls["n"] += 1
        return original_compute(self, current)

    monkeypatch.setattr(VideoTokenState, "_compute_drift", counting_compute)

    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            "step_cache": {"enabled": False},
            "video_token_cache": {
                "enabled": True,
                "cold_replans": 1,
                "full_reuse_threshold": 0.02,
                "max_consecutive_reuses": 8,
            },
        },
    )
    model.infer_action(token_drift=0.0)
    model.infer_action(token_drift=0.0)
    assert drift_calls["n"] == 2
    assert stack.stats()["video_token_cache"]["full_reuse"] == 1


def test_shared_video_token_state_chunk_only_still_works(monkeypatch):
    """chunk_residual_cache alone still tracks drift via shared state."""
    _identity_compile(monkeypatch)
    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            "step_cache": {"enabled": False},
            "chunk_residual_cache": {
                "enabled": True,
                "cold_chunks": 1,
                "condition_threshold": 1e9,
                "action_latent_threshold": 1e9,
                "max_consecutive_reuses": 8,
            },
        },
    )
    model.infer_action(num_inference_steps=2, token_drift=0.0)
    model.infer_action(num_inference_steps=2, token_drift=0.0)
    stats = stack.stats()["chunk_residual_cache"]
    assert stats["residual_hits"] == 2


def test_stack_reset_drops_cross_episode_video_and_residual(monkeypatch):
    """Episode-boundary reset must force a cold prefill / residual miss."""
    _identity_compile(monkeypatch)
    model = FakeModel()
    stack = accelerate(
        model,
        {
            "compile": {"enabled": False},
            "step_cache": {"enabled": False},
            "video_token_cache": {
                "enabled": True,
                "cold_replans": 1,
                "full_reuse_threshold": 0.02,
                "max_consecutive_reuses": 8,
                "force_refresh_every": 0,
            },
            "chunk_residual_cache": {
                "enabled": True,
                "cold_chunks": 1,
                "condition_threshold": 0.02,
                "action_latent_threshold": 1e9,
                "max_consecutive_reuses": 8,
                "force_refresh_every": 0,
            },
        },
    )
    model.infer_action(num_inference_steps=2, token_drift=0.0)
    model.infer_action(num_inference_steps=2, token_drift=0.0)
    warmed = stack.stats()
    assert warmed["video_token_cache"]["full_reuse"] >= 1
    assert warmed["chunk_residual_cache"]["residual_hits"] >= 1

    stack.reset()
    after = stack.stats()
    # Counters are cumulative; the next call must take the cold path.
    model.infer_action(num_inference_steps=2, token_drift=0.0)
    cold = stack.stats()
    assert cold["video_token_cache"]["full_reuse"] == warmed["video_token_cache"]["full_reuse"]
    assert cold["video_token_cache"]["full_prefill"] == warmed["video_token_cache"]["full_prefill"] + 1
    assert cold["chunk_residual_cache"]["residual_hits"] == warmed["chunk_residual_cache"]["residual_hits"]
    vtc = stack.controllers["video_token_cache"]
    crc = stack.controllers["chunk_residual_cache"]
    assert vtc._last_cache is not None
    assert crc._chunk == 1
    assert vtc._video_state.last_tokens is not None


