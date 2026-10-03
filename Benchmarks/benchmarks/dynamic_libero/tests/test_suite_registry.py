"""CPU unit tests for LIBERO suite registry."""

from __future__ import annotations

import pytest

from benchmarks.dynamic_libero.env.suite_registry import (
    ALL_SUITES,
    TRAINED_SUITES,
    is_trained_suite,
    normalize_suite_name,
    parse_task_ids,
    resolve_suites,
    suite_n_tasks,
)


def test_trained_suites_cover_training_data():
    assert len(TRAINED_SUITES) == 4
    assert "libero_object" in TRAINED_SUITES
    assert "libero_90" not in TRAINED_SUITES
    assert len(ALL_SUITES) == 5


def test_aliases():
    assert normalize_suite_name("spatial") == "libero_spatial"
    assert normalize_suite_name("long") == "libero_10"


def test_resolve_suites_trained():
    assert resolve_suites("trained") == list(TRAINED_SUITES)
    assert resolve_suites("all-trained") == list(TRAINED_SUITES)


def test_resolve_suites_all():
    assert resolve_suites("all") == list(ALL_SUITES)


def test_parse_task_ids_all():
    assert parse_task_ids("all", "libero_object") == list(range(10))
    assert parse_task_ids("*", "libero_90") == list(range(90))


def test_parse_task_ids_explicit():
    assert parse_task_ids("0,2,4", "libero_spatial") == [0, 2, 4]


def test_parse_task_ids_out_of_range():
    with pytest.raises(ValueError):
        parse_task_ids("10", "libero_object")


def test_is_trained_suite():
    assert is_trained_suite("libero_goal")
    assert not is_trained_suite("libero_90")


def test_suite_n_tasks():
    assert suite_n_tasks("libero_10") == 10
    assert suite_n_tasks("libero_90") == 90
