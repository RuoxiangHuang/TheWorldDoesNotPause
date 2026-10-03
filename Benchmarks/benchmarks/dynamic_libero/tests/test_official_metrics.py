"""Unit tests for official contract + curve metrics (no GPU)."""

from __future__ import annotations

import math

import pytest

from benchmarks.common.protocol import DEFAULT_BACKEND, PROTOCOL_VERSION
from benchmarks.dynamic_libero.env.official import (
    OFFICIAL,
    OFFICIAL_MOVERS,
    OFFICIAL_TARGET_REGISTRY,
    normalize_speed,
)
from benchmarks.dynamic_libero.env.movers import parse_mover_language, parse_movers_flag, resolve_rails_spec
from benchmarks.dynamic_libero.metrics.aggregate import (
    auc_sr,
    bootstrap_ci,
    curve_metrics,
    enrich_table_with_curve_metrics,
    normalize_auc_sr,
    v50,
)


def test_official_protocol_matches_common():
    assert OFFICIAL.protocol == PROTOCOL_VERSION == "real_time_v2"
    assert OFFICIAL.backend == DEFAULT_BACKEND.value == "async"
    assert OFFICIAL.replan_steps == 10
    assert OFFICIAL.episodes_per_task == 5
    assert OFFICIAL_MOVERS is False
    assert OFFICIAL_TARGET_REGISTRY is True
    assert OFFICIAL.mover_language == "keep"
    assert "linear" in OFFICIAL.trajectories and "sine" in OFFICIAL.trajectories
    assert "smooth_turn" not in OFFICIAL.trajectories
    assert 0.0005 in OFFICIAL.speeds
    assert OFFICIAL.speeds[0] == 0.0
    assert OFFICIAL.speeds[-1] == OFFICIAL.v_max
    assert 0.0006 not in OFFICIAL.speeds  # paper working point, not official 60 grid


def test_parse_speed_list_paper_alias_is_async_curve():
    from benchmarks.dynamic_libero.env.official import (
        PAPER_SLICE_SPEEDS,
        parse_speed_list,
    )

    assert parse_speed_list("paper") == list(PAPER_SLICE_SPEEDS)
    assert parse_speed_list("slice") == list(PAPER_SLICE_SPEEDS)
    assert 0.0 in PAPER_SLICE_SPEEDS
    assert 0.0006 in PAPER_SLICE_SPEEDS
    assert parse_speed_list("0.0,0.0006") == [0.0, 0.0006]
    assert parse_speed_list("official")[0] == 0.0
    assert parse_speed_list("official") == list(OFFICIAL.speeds)


def test_normalize_speed():
    assert normalize_speed(0.0) == 0.0
    assert normalize_speed(0.010) == 1.0
    assert abs(normalize_speed(0.005) - 0.5) < 1e-9


def test_parse_movers_defaults_off():
    assert parse_movers_flag(None) is False
    assert parse_movers_flag("off") is False
    assert parse_movers_flag("on") is True
    assert parse_mover_language(None) == "keep"


def test_resolve_rails_spec_registry_without_assets():
    spec = resolve_rails_spec("libero_object", 0, movers_enabled=False, target_registry=True)
    assert spec is not None
    assert spec.target  # non-empty rails target name
    assert resolve_rails_spec("libero_object", 0, movers_enabled=False, target_registry=False) is None
    on = resolve_rails_spec("libero_object", 0, movers_enabled=True, target_registry=False)
    assert on is not None


def test_auc_and_v50():
    speeds = [0.0, 0.002, 0.004, 0.006]
    srs = [1.0, 0.8, 0.4, 0.0]
    assert auc_sr(speeds, srs) > 0
    nau = normalize_auc_sr(speeds, srs, v_min=0.0, v_max=0.006)
    assert 0.0 < nau < 1.0
    v = v50(speeds, srs, threshold=0.5)
    assert v is not None
    assert 0.002 < v < 0.004
    assert v50([0.0, 0.01], [0.9, 0.8], threshold=0.5) is None


def test_bootstrap_ci_bernoulli():
    ci = bootstrap_ci([1, 1, 0, 1, 0], n_boot=500, seed=1)
    assert ci["n"] == 5
    assert 0.0 <= ci["lo"] <= ci["mean"] <= ci["hi"] <= 1.0


def test_enrich_table_curves():
    table = {
        "sr": {
            "off": {"linear": {"0.0": 0.8, "0.001": 0.4, "0.003": 0.1}},
            "full": {"linear": {"0.0": 0.8, "0.001": 0.6, "0.003": 0.3}},
        },
        "latency": {},
        "delta_full_minus_off": {},
    }
    enriched = enrich_table_with_curve_metrics(table, v_min=0.0, v_max=0.003)
    assert "curves" in enriched
    assert math.isfinite(enriched["curves"]["full"]["linear"]["norm_auc_sr"])
    assert "linear" in enriched["delta_norm_auc_full_minus_off"]
    m = curve_metrics({"0": 1.0, "1": 0.0})
    assert m["v50"] == pytest.approx(0.5)
