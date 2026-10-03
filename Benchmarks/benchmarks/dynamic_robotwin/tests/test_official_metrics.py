"""Unit tests for Dynamic-RoboTwin official contract + metrics (no GPU)."""

from __future__ import annotations

import math

import pytest

from benchmarks.common.protocol import DEFAULT_BACKEND, PROTOCOL_VERSION
from benchmarks.dynamic_robotwin.env.official import (
    OFFICIAL,
    OFFICIAL_RAILS_COLLIDE,
    normalize_speed,
    parse_task_names,
)
from benchmarks.dynamic_robotwin.env.task_registry import V1_TASKS
from benchmarks.dynamic_robotwin.metrics.aggregate import (
    auc_sr,
    bootstrap_ci,
    enrich_table_with_curve_metrics,
    v50,
)


def test_official_protocol_matches_common():
    assert OFFICIAL.protocol == PROTOCOL_VERSION == "real_time_v2"
    assert OFFICIAL.backend == DEFAULT_BACKEND.value == "async"
    assert OFFICIAL.replan_steps == 8
    assert OFFICIAL.policy_hz == 20.0
    assert OFFICIAL.episodes_per_task == 5
    assert OFFICIAL_RAILS_COLLIDE is False
    assert OFFICIAL.contact_switch == "medium"
    assert "contact_trigger" in OFFICIAL.dynamic_modes
    assert "occlusion" in OFFICIAL.dynamic_modes
    assert "contact_chain" in OFFICIAL.dynamic_modes
    assert "linear" in OFFICIAL.trajectories and "sine" in OFFICIAL.trajectories
    assert OFFICIAL.speeds[0] == 0.0
    assert OFFICIAL.speeds[-1] == OFFICIAL.v_max
    assert set(OFFICIAL.tasks) == set(V1_TASKS)
    assert OFFICIAL.n_episodes_macro() == len(V1_TASKS) * 5


def test_normalize_speed():
    assert normalize_speed(0.0) == 0.0
    assert normalize_speed(0.010) == 1.0
    assert abs(normalize_speed(0.005) - 0.5) < 1e-9


def test_parse_task_names():
    assert parse_task_names("all") == list(V1_TASKS)
    assert parse_task_names("place_empty_cup,handover_block") == [
        "place_empty_cup",
        "handover_block",
    ]
    with pytest.raises(KeyError):
        parse_task_names("not_a_task")


def test_parse_task_names_seed_catalog():
    names = parse_task_names("recon")
    assert parse_task_names("seed") == names
    assert parse_task_names("wave2") == names
    assert len(names) == 15 * 5
    assert names[0] == "place_empty_cup.pick"
    assert "place_empty_cup.place.irregular" in names
    assert "move_can_pot.both" in names
    assert "place_mouse_pad.place.irregular" in names
    assert "place_fan.pick" in names
    assert parse_task_names("place_empty_cup.both") == ["place_empty_cup.both"]
    # Official `all` does not silently include bumper-car reconstructions.
    assert "place_empty_cup.pick" not in parse_task_names("all")
    assert "move_can_pot" not in V1_TASKS
    assert "place_mouse_pad" not in V1_TASKS
    assert "place_fan" not in V1_TASKS
    assert "place_phone_stand.both" in names
    assert "place_a2b_right.place.irregular" in names
    assert "place_phone_stand.both" not in parse_task_names("all")
    assert parse_task_names("place_container_plate.pick") == ["place_container_plate.pick"]
    demo = parse_task_names("recon-demo")
    assert demo == parse_task_names("wave2-demo")
    assert len(demo) == 15 * 3


def test_parse_task_names_extra_catalog():
    names = parse_task_names("extra")
    assert names == parse_task_names("appendix")
    assert "catch_shuttlecock" in names
    assert "catch_bunny_toy" in names
    assert "stop_rolling_orange" in names
    assert "pursue_toycar" in names
    assert "collide_pool_balls" in names
    assert parse_task_names("catch_shuttlecock") == ["catch_shuttlecock"]
    assert "catch_shuttlecock" not in parse_task_names("all")
    assert "catch_shuttlecock" not in parse_task_names("recon")


def test_metrics_reexport():
    speeds = [0.0, 0.002, 0.004]
    srs = [1.0, 0.5, 0.0]
    assert auc_sr(speeds, srs) > 0
    assert v50(speeds, srs, threshold=0.5) == pytest.approx(0.002)
    ci = bootstrap_ci([1, 0, 1, 1], n_boot=200, seed=0)
    assert 0.0 <= ci["lo"] <= ci["mean"] <= ci["hi"] <= 1.0
    table = {
        "sr": {"off": {"linear": {"0.0": 0.8, "0.001": 0.3}}, "full": {"linear": {"0.0": 0.8, "0.001": 0.5}}},
        "latency": {},
        "delta_full_minus_off": {},
    }
    enriched = enrich_table_with_curve_metrics(table, v_min=0.0, v_max=0.001)
    assert math.isfinite(enriched["curves"]["full"]["linear"]["norm_auc_sr"])


def test_dynamics_cli_dict():
    d = OFFICIAL.dynamics_cli()
    assert d["contact_trigger"] == "medium"
    assert d["occlusion"] == "on"
    assert d["secondary_rails"] == "auto"
