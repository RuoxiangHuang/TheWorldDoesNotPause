"""CPU tests for hybrid grasp protocol helpers."""

from __future__ import annotations

import pytest

from benchmarks.common.grasp import (
    DEFAULT_ENGAGE_ALPHA,
    RailsPhase,
    engage_radius,
    parse_grasp_mode,
    parse_place_freeze,
    should_inject_release_velocity,
    soft_kp,
)


def test_parse_place_freeze():
    assert parse_place_freeze(None) == "both"
    assert parse_place_freeze(True) == "both"
    assert parse_place_freeze(False) == "off"
    assert parse_place_freeze("all") == "all"
    with pytest.raises(ValueError):
        parse_place_freeze("maybe")


def test_parse_grasp_mode():
    assert parse_grasp_mode(None) == "ghost"
    assert parse_grasp_mode("hybrid_a") == "hybrid_a"
    assert parse_grasp_mode("hybrid_b") == "hybrid_b"
    with pytest.raises(ValueError):
        parse_grasp_mode("invalid")


def test_engage_radius():
    assert engage_radius(0.06) == pytest.approx(0.06 * DEFAULT_ENGAGE_ALPHA)
    assert engage_radius(0.12, alpha=2.0) == pytest.approx(0.24)


def test_soft_kp_decay():
    r = engage_radius(0.06)
    assert soft_kp(r, r) == pytest.approx(4.0)
    assert soft_kp(0.0, r) == pytest.approx(0.0)
    assert soft_kp(r / 2, r) == pytest.approx(2.0)


def test_should_inject_release_velocity():
    assert should_inject_release_velocity("ghost", phase=RailsPhase.PURSUIT) is True
    assert should_inject_release_velocity("ghost", phase=RailsPhase.ENGAGE) is True
    assert should_inject_release_velocity("hybrid_a", phase=RailsPhase.PURSUIT) is True
    assert should_inject_release_velocity("hybrid_a", phase=RailsPhase.ENGAGE) is False
    assert should_inject_release_velocity("hybrid_b", phase=RailsPhase.PURSUIT, had_contact=True) is False
