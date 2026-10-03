"""Checks that the hydra `accel` config group is wired to the model correctly.

The model configs reach the acceleration stack through an `accel: ${accel}`
interpolation, which is easy to break silently: a broken interpolation would
only surface at model construction, i.e. after a checkpoint load.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from fasterwam.accel import AccelConfig
from fasterwam.utils.config_resolvers import register_default_resolvers

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
TASK = "task=libero_uncond_2cam224_1e-4"


@pytest.fixture(scope="module", autouse=True)
def _resolvers():
    register_default_resolvers()


def _accel(*overrides: str) -> AccelConfig:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="train", overrides=[TASK, *overrides])
    # apply_env=False so a developer's shell cannot flip the assertions.
    return AccelConfig.from_any(cfg.model.accel, apply_env=False)


def test_default_group_enables_the_full_stack():
    accel = _accel()
    assert accel.text_cache.active
    assert accel.step_cache.active and accel.step_cache.threshold == pytest.approx(0.20)
    assert accel.compile.active and accel.compile.predict and accel.compile.prefill
    assert not accel.video_token_cache.active


def test_p0_group_enables_video_token_cache():
    accel = _accel("accel=p0")
    assert accel.video_token_cache.active
    assert accel.step_cache.active and accel.compile.active
    assert not accel.vae_cache.active


def test_p1_group_enables_vae_cache():
    accel = _accel("accel=p1")
    assert accel.video_token_cache.active
    assert accel.vae_cache.active
    assert not accel.obs_pipeline.active


def test_full_group_enables_every_controller():
    accel = _accel("accel=full")
    assert accel.text_cache.active
    assert accel.step_cache.active
    assert accel.compile.active and accel.compile.prefill_mode == "default"
    assert accel.video_token_cache.active
    assert accel.vae_cache.active
    assert accel.obs_pipeline.active
    assert accel.chunk_residual_cache.active


def test_action0_group_enables_chunk_residual():
    accel = _accel("accel=action0")
    assert accel.chunk_residual_cache.active
    assert accel.step_cache.active and accel.compile.active
    assert not accel.video_token_cache.active
    assert accel.chunk_residual_cache.condition_threshold == pytest.approx(0.05)


def test_default_disables_chunk_residual():
    accel = _accel()
    assert not accel.chunk_residual_cache.active


def test_baseline_group_is_the_reference_path():
    assert not _accel("accel=baseline").active
    assert not _accel("accel=off").active


def test_pace_is_the_paper_name_for_ours():
    pace = _accel("accel=pace")
    ours = _accel("accel=ours")
    default = _accel()
    for cfg in (pace, ours):
        assert cfg.text_cache.active and cfg.step_cache.active and cfg.compile.active
        assert not cfg.chunk_residual_cache.active
        assert not cfg.video_token_cache.active
        assert not cfg.vae_cache.active
        assert cfg.step_cache.threshold == pytest.approx(0.20)
        assert cfg.step_cache.max_consecutive_skips == 2
        assert cfg.step_cache.cold_steps == 2
        assert cfg.text_cache.max_entries == 64
    assert pace.step_cache.threshold == pytest.approx(default.step_cache.threshold)
    assert pace.step_cache.threshold == pytest.approx(ours.step_cache.threshold)


def test_off_group_disables_everything():
    assert not _accel("accel=off").active


def test_lossless_group_keeps_every_denoise_step():
    accel = _accel("accel=lossless")
    assert accel.active
    assert not accel.step_cache.active
    assert accel.text_cache.active and accel.compile.active


def test_individual_fields_are_overridable_from_the_cli():
    accel = _accel("accel.step_cache.threshold=0.35", "accel.compile.prefill=false")
    assert accel.step_cache.threshold == pytest.approx(0.35)
    assert not accel.compile.prefill
    assert accel.compile.active  # predict is still on


@pytest.mark.parametrize(
    "model_group, target",
    [
        ("fastwam", "fasterwam.runtime.create_fastwam"),
        ("fastwam_joint", "fasterwam.runtime.create_fastwam_joint"),
        ("fastwam_idm", "fasterwam.runtime.create_fastwam_idm"),
    ],
)
def test_every_model_config_targets_this_package_and_carries_accel(model_group, target):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="train", overrides=[TASK, f"model={model_group}"])
    assert cfg.model._target_ == target
    assert AccelConfig.from_any(cfg.model.accel, apply_env=False).active


def test_dataset_configs_target_this_package():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="train", overrides=[TASK])
    assert cfg.data.train._target_.startswith("fasterwam.")
    assert cfg.data.train.processor._target_.startswith("fasterwam.")
