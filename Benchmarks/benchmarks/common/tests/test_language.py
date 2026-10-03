"""Unit tests for B-track motion language (no GPU)."""

from __future__ import annotations

import pytest

from benchmarks.common.language import (
    DEFAULT_MOTION_HINT_TEMPLATE,
    append_motion_hint,
    apply_language_mode,
    humanize_noun,
    motion_hint,
    parse_language_mode,
)
from benchmarks.dynamic_robotwin.env.task_registry import (
    default_instruction,
    instruction_noun,
)


def test_parse_language_mode():
    assert parse_language_mode(None) == "keep"
    assert parse_language_mode("motion") == "motion"
    assert parse_language_mode("scene") == "scene"
    with pytest.raises(ValueError):
        parse_language_mode("rewrite")


def test_spatial_scene_rewrites_stale_locative():
    from benchmarks.common.language import SPATIAL_CARRIER_SCENE_INSTRUCTION

    stale = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    out = apply_language_mode(stale, "scene")
    assert out == SPATIAL_CARRIER_SCENE_INSTRUCTION
    assert "place it on the plate" in out
    assert "ramekin" not in out
    drawer = "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"
    assert apply_language_mode(drawer, "scene") == SPATIAL_CARRIER_SCENE_INSTRUCTION
    soup = "pick up the alphabet soup and place it in the basket"
    assert apply_language_mode(soup, "scene") == soup
    assert apply_language_mode(stale, "keep") == stale


def test_libero_style_motion_suffix():
    base = "pick up the alphabet soup and place it in the basket"
    out = apply_language_mode(base, "motion", noun="alphabet soup")
    assert out.startswith(base)
    assert "The alphabet soup is moving; grasp it before it leaves reach." in out
    assert apply_language_mode(base, "keep", noun="alphabet soup") == base


def test_robotwin_style_motion_suffix():
    base = default_instruction("place_empty_cup")
    out = apply_language_mode(base, "motion", noun=instruction_noun("place_empty_cup"))
    assert out.startswith(base.rstrip("."))
    assert "The cup is moving; grasp it before it leaves reach." in out


def test_idempotent_append():
    base = "Pick up the cup and place it onto the coaster."
    once = append_motion_hint(base, noun="cup")
    twice = append_motion_hint(once, noun="cup")
    assert once == twice


def test_humanize_and_template():
    assert humanize_noun("alphabet_soup") == "alphabet soup"
    assert motion_hint(noun="toy car") == DEFAULT_MOTION_HINT_TEMPLATE.format(noun="toy car")


def test_instruction_nouns_cover_whitelist():
    from benchmarks.dynamic_robotwin.env.task_registry import V1_TASKS

    for t in V1_TASKS:
        n = instruction_noun(t)
        assert isinstance(n, str) and len(n) > 0
