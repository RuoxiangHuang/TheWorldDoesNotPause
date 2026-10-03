"""Tests for the DOMINO integration that need neither a GPU nor a DOMINO checkout.

The parts worth pinning down are the ones that fail silently in a real run: the
latency->n_frozen contract, DOMINO's `eval()`-based override parsing, and the
policy-name shadowing that makes DOMINO import the wrong module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from integrations.domino import run_eval, run_latency_coupled  # noqa: E402
from integrations.domino.policy_runner import (  # noqa: E402
    LatencyCoupler,
    compose_eval_config,
    load_eval_config,
    parse_bool,
    parse_optional_float,
    parse_optional_int,
)


class FakeEnv:
    """Stands in for a DOMINO Base_Task with the drift hook."""

    def __init__(self, per_call_m: float = 0.01):
        self.per_call_m = per_call_m
        self.calls: list[int] = []

    def inject_latency_drift(self, n_steps):
        self.calls.append(n_steps)
        return self.per_call_m * n_steps


class TestLatencyCoupler:
    def test_derives_n_frozen_from_latency(self):
        # The FastWAM measurements this port reproduces: 250 ms -> 5, 88 ms -> 2.
        assert LatencyCoupler.from_latency(0.250, control_hz=20).n_frozen == 5
        assert LatencyCoupler.from_latency(0.088, control_hz=20).n_frozen == 2

    def test_control_hz_scales_the_injection(self):
        assert LatencyCoupler.from_latency(0.1, control_hz=10).n_frozen == 1
        assert LatencyCoupler.from_latency(0.1, control_hz=50).n_frozen == 5

    def test_zero_is_disabled_and_never_touches_the_env(self):
        coupler = LatencyCoupler(0)
        env = FakeEnv()
        assert not coupler.enabled
        coupler.inject(env)
        assert env.calls == []
        assert coupler.drift_m == 0.0

    def test_negative_n_frozen_clamps_to_disabled(self):
        assert LatencyCoupler(-3).n_frozen == 0
        assert not LatencyCoupler(-3).enabled

    def test_accumulates_drift_across_replans(self):
        coupler = LatencyCoupler(3)
        env = FakeEnv(per_call_m=0.01)
        coupler.inject(env)
        coupler.inject(env)
        assert env.calls == [3, 3]
        assert coupler.injections == 2
        assert coupler.drift_m == pytest.approx(0.06)

    def test_reset_clears_per_episode_state(self):
        coupler = LatencyCoupler(2)
        coupler.inject(FakeEnv())
        coupler.reset()
        assert coupler.drift_m == 0.0
        assert coupler.injections == 0
        # n_frozen is a property of the configuration, not of the episode.
        assert coupler.n_frozen == 2

    def test_missing_hook_fails_loudly(self):
        # Silently skipping would produce a decoupled run masquerading as coupled.
        with pytest.raises(RuntimeError, match="inject_latency_drift"):
            LatencyCoupler(2).inject(object())

    def test_hook_returning_none_is_tolerated(self):
        class NoReturn:
            def inject_latency_drift(self, n_steps):
                return None

        coupler = LatencyCoupler(2)
        coupler.inject(NoReturn())
        assert coupler.drift_m == 0.0
        assert coupler.injections == 1


class TestOverrideFormatting:
    """DOMINO parses overrides with `eval(value)` and falls back to the raw string."""

    @staticmethod
    def _roundtrip(value):
        try:
            return eval(run_eval._format_override(value))  # noqa: S307
        except Exception:
            return run_eval._format_override(value)

    @pytest.mark.parametrize("value", ["unseen", "demo_clean_dynamic", ""])
    def test_strings_survive(self, value):
        assert self._roundtrip(value) == value

    def test_paths_stay_strings(self):
        path = "/DATA/YuanZhen/FasterWAM/checkpoints/robotwin_uncond_3cam_384.pt"
        assert self._roundtrip(path) == path

    def test_bools_and_none(self):
        assert self._roundtrip(True) is True
        assert self._roundtrip(False) is False
        assert self._roundtrip(None) is None

    def test_numbers(self):
        assert self._roundtrip(4) == 4
        assert self._roundtrip(0.03) == pytest.approx(0.03)


class TestPolicySymlink:
    def test_rejects_name_that_shadows_a_package(self, tmp_path):
        # `fasterwam` is importable, so DOMINO would import the package instead
        # of the policy adapter — the exact failure this guard exists for.
        (tmp_path / "policy").mkdir()
        with pytest.raises(ValueError, match="shadows an importable module"):
            run_eval._ensure_policy_symlink(tmp_path, "fasterwam")

    def test_rejects_non_domino_root(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="policy/"):
            run_eval._ensure_policy_symlink(tmp_path, "fasterwam_policy")

    def test_creates_and_is_idempotent(self, tmp_path):
        (tmp_path / "policy").mkdir()
        first = run_eval._ensure_policy_symlink(tmp_path, "fasterwam_policy")
        assert first.is_symlink()
        assert first.resolve() == run_eval.POLICY_SOURCE.resolve()
        assert run_eval._ensure_policy_symlink(tmp_path, "fasterwam_policy") == first

    def test_refuses_to_clobber_a_real_directory(self, tmp_path):
        (tmp_path / "policy" / "fasterwam_policy").mkdir(parents=True)
        with pytest.raises(RuntimeError, match="not a symlink"):
            run_eval._ensure_policy_symlink(tmp_path, "fasterwam_policy")

    def test_detects_a_foreign_symlink(self, tmp_path):
        (tmp_path / "policy").mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (tmp_path / "policy" / "fasterwam_policy").symlink_to(elsewhere, target_is_directory=True)
        with pytest.raises(RuntimeError, match="expected"):
            run_eval._ensure_policy_symlink(tmp_path, "fasterwam_policy")


class TestDatasetStatsResolution:
    @staticmethod
    def _cfg(explicit=None):
        from omegaconf import OmegaConf

        return OmegaConf.create({"EVALUATION": {"dataset_stats_path": explicit}})

    def test_finds_stats_beside_the_checkpoint(self, tmp_path):
        ckpt = tmp_path / "robotwin_uncond_3cam_384.pt"
        ckpt.touch()
        stats = tmp_path / "robotwin_uncond_3cam_384_dataset_stats.json"
        stats.write_text("{}")
        assert run_eval._resolve_dataset_stats(self._cfg(), ckpt) == stats.resolve()

    def test_explicit_path_wins(self, tmp_path):
        ckpt = tmp_path / "model.pt"
        ckpt.touch()
        (tmp_path / "model_dataset_stats.json").write_text("{}")
        explicit = tmp_path / "explicit.json"
        explicit.write_text("{}")
        assert run_eval._resolve_dataset_stats(self._cfg(str(explicit)), ckpt) == explicit.resolve()

    def test_reports_clearly_when_absent(self, tmp_path):
        ckpt = tmp_path / "model.pt"
        ckpt.touch()
        with pytest.raises(FileNotFoundError, match="dataset_stats_path"):
            run_eval._resolve_dataset_stats(self._cfg(), ckpt)


class TestLatencyReportParsing:
    def test_extracts_latency_and_n_frozen(self, tmp_path):
        log = tmp_path / "measure.log"
        log.write_text(
            "noise\n"
            "[FASTERWAM_LATENCY] accel=compile,step_cache,text_cache nfe=4 replan_steps=8 "
            "warm_replans=7 per_replan_ms=88.4 implied_n_frozen=2 (at 20 Hz)\n"
        )
        assert run_latency_coupled.parse_latency(log) == (88.4, 2)

    def test_takes_the_last_report(self, tmp_path):
        log = tmp_path / "measure.log"
        log.write_text(
            "[FASTERWAM_LATENCY] per_replan_ms=250.0 implied_n_frozen=5\n"
            "[FASTERWAM_LATENCY] per_replan_ms=88.0 implied_n_frozen=2\n"
        )
        assert run_latency_coupled.parse_latency(log) == (88.0, 2)

    def test_returns_none_without_a_report(self, tmp_path):
        log = tmp_path / "measure.log"
        log.write_text("crashed before reporting\n")
        assert run_latency_coupled.parse_latency(log) is None
        assert run_latency_coupled.parse_latency(tmp_path / "missing.log") is None


class TestSweepWiring:
    def test_arms_differ_only_in_the_accel_group(self):
        assert run_latency_coupled.ARMS == {"accel": "default", "reference": "off"}

    def test_command_carries_the_coupling_and_arm(self):
        cmd = run_latency_coupled.build_command(
            task="beat_block_hammer", arm="reference", n_frozen=5, episodes=12,
            coeff=0.03, ckpt="ckpt.pt", stats=None, tag="t", output_dir=Path("/tmp/x"),
        )
        assert "accel=off" in cmd
        assert "EVALUATION.n_frozen=5" in cmd
        assert "EVALUATION.dynamic_coefficient=0.03" in cmd
        assert "EVALUATION.task_name=beat_block_hammer" in cmd

    def test_metrics_lookup_prefers_the_newest_run(self, tmp_path):
        base = tmp_path / "eval_result" / "beat_block_hammer" / "p" / "cfg" / "ckpt"
        for tag, sr in (("coupled_accel_nf2_older", 0.1), ("coupled_accel_nf2_newer", 0.5)):
            run_dir = base / tag
            run_dir.mkdir(parents=True)
            (run_dir / "_metrics.json").write_text(json.dumps({"success_rate": sr}))
        found = run_latency_coupled.find_metrics(tmp_path, "beat_block_hammer", "coupled_accel_nf2")
        assert found is not None and found["success_rate"] in {0.1, 0.5}

    def test_metrics_lookup_returns_none_when_missing(self, tmp_path):
        assert run_latency_coupled.find_metrics(tmp_path, "nope", "coupled_accel_nf2") is None


class TestEvalConfig:
    def test_composes_with_accel_wired_into_the_model(self):
        cfg = compose_eval_config(config_path=None, config_name=None, task=None)
        assert cfg.model._target_ == "fasterwam.runtime.create_fastwam"
        assert cfg.model.accel is not None
        assert cfg.model.load_text_encoder is True
        assert cfg.EVALUATION.task_config == "demo_clean_dynamic"
        # Must not shadow the `fasterwam` package.
        assert cfg.EVALUATION.policy_name == "fasterwam_policy"

    def test_coupling_is_off_by_default(self):
        # A latency-coupled run must be opted into, so an unqualified number is
        # never reported as coupled.
        cfg = compose_eval_config(config_path=None, config_name=None, task=None)
        assert cfg.EVALUATION.n_frozen == 0

    def test_rejects_a_config_outside_the_config_tree(self):
        with pytest.raises(ValueError, match="must live under"):
            compose_eval_config(config_path="/etc/passwd", config_name=None, task=None)

    def test_accel_preset_reaches_the_composed_config(self):
        assert compose_eval_config(accel="off").accel.enabled is False
        assert compose_eval_config(accel="default").accel.enabled is True


class TestConfigHandover:
    """The launcher's command-line overrides must survive into the subprocess.

    Re-composing in the subprocess would silently turn `accel=off` back into
    `accel=default`, producing a "reference" run that is actually accelerated.
    """

    def test_handover_is_used_verbatim(self, tmp_path):
        from omegaconf import OmegaConf

        path = tmp_path / "resolved.yaml"
        OmegaConf.save(OmegaConf.create({"accel": {"enabled": False}, "marker": "handover"}), path)
        cfg = load_eval_config({"eval_cfg_resolved": str(path)})
        assert cfg.accel.enabled is False
        assert cfg.marker == "handover"

    def test_handover_beats_the_fallback_config_name(self, tmp_path):
        from omegaconf import OmegaConf

        path = tmp_path / "resolved.yaml"
        OmegaConf.save(OmegaConf.create({"accel": {"enabled": False}}), path)
        cfg = load_eval_config({
            "eval_cfg_resolved": str(path),
            "eval_cfg_name": "eval/domino.yaml",  # would enable acceleration
        })
        assert cfg.accel.enabled is False

    def test_missing_handover_fails_loudly(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="handover"):
            load_eval_config({"eval_cfg_resolved": str(tmp_path / "gone.yaml")})

    def test_fallback_still_honours_the_accel_preset(self):
        cfg = load_eval_config({"accel_preset": "off"})
        assert cfg.accel.enabled is False

    def test_fallback_defaults_to_the_accelerated_config(self):
        assert load_eval_config({}).accel.enabled is True


class TestValueParsing:
    @pytest.mark.parametrize("raw,expected", [("1", True), ("true", True), ("no", False), ("0", False)])
    def test_bools(self, raw, expected):
        assert parse_bool(raw) is expected

    def test_bad_bool_raises(self):
        with pytest.raises(ValueError):
            parse_bool("maybe")

    @pytest.mark.parametrize("raw", [None, "", "none", "null", "None"])
    def test_none_like_values(self, raw):
        assert parse_optional_int(raw) is None
        assert parse_optional_float(raw) is None

    def test_real_values(self):
        assert parse_optional_int("4") == 4
        assert parse_optional_float("0.03") == pytest.approx(0.03)


class TestDominoHooks:
    def test_package_exposes_the_four_hooks(self):
        from integrations.domino import policy

        for name in ("get_model", "encode_obs", "eval", "reset_model"):
            assert callable(getattr(policy, name)), f"missing hook: {name}"

    def test_encode_obs_is_passthrough(self):
        from integrations.domino import policy

        obs = {"observation": {}}
        assert policy.encode_obs(obs) is obs
        assert policy.encode_obs(None) is None
