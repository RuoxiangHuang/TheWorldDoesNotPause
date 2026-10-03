"""CPU tests for scene axes and outcome tri-states."""

from __future__ import annotations

import pytest

from benchmarks.common.outcomes import (
    GraspHoldTracker,
    outcome_fields,
    read_grasp_hold,
    read_rails_released,
    unknown_bool,
)
from benchmarks.common.scene import (
    DEFAULT_LAYOUT_SPEED,
    SceneCondition,
    resolve_scene_condition,
    scene_from_preset,
    trajectory_speed_for_condition,
)


def test_hold_and_drive_share_layout_speed():
    hold = scene_from_preset("apparatus_hold", speed=0.0006)
    drive = scene_from_preset("apparatus_drive", speed=0.0006)
    assert hold.scene == drive.scene == "modified"
    assert hold.apparatus == drive.apparatus == "visible"
    assert hold.layout_speed == drive.layout_speed == pytest.approx(0.0006)
    assert hold.constrains_rails() and drive.constrains_rails()
    assert not hold.advances_rails() and drive.advances_rails()
    assert trajectory_speed_for_condition(hold, 0.0) == pytest.approx(0.0006)
    assert trajectory_speed_for_condition(drive, 0.0006) == pytest.approx(0.0006)


def test_speed_zero_does_not_skip_catalog_install():
    cond = resolve_scene_condition(speed=0.0, catalog_job=True)
    assert cond.scene == "modified"
    assert cond.apparatus == "visible"
    assert cond.rails == "hold"
    assert cond.world_clock == "realtime"
    assert cond.uses_layout_actor()
    assert cond.apply_n_freeze(4.5) == pytest.approx(4.5)


def test_original_has_no_apparatus():
    cond = scene_from_preset("original", speed=0.0006)
    assert cond.scene == "original"
    assert cond.apparatus == "off"
    assert not cond.modifies_layout()
    assert not cond.constrains_rails()
    assert trajectory_speed_for_condition(cond, 0.0006) == pytest.approx(0.0)


def test_world_clock_pause_zeroes_n_freeze():
    cond = SceneCondition(world_clock="pause")
    assert cond.apply_n_freeze(3.0) == 0.0


def test_hidden_keeps_layout_without_mesh():
    cond = scene_from_preset("ghost_hold", speed=0.001)
    assert cond.apparatus == "hidden"
    assert cond.uses_layout_actor()
    assert not cond.installs_visual_actor()
    assert cond.layout_speed == pytest.approx(0.001)


def test_auto_non_catalog_speed_zero_is_original():
    cond = resolve_scene_condition(speed=0.0, catalog_job=False)
    assert cond.scene == "original" and cond.rails == "off"


def test_outcome_fields_do_not_alias_release_to_hold():
    row = outcome_fields(
        rails_released=True,
        grasp_hold=None,
        task_success=True,
    )
    assert row["rails_released"] is True
    assert row["grasp_hold"] is None
    assert row["grasp_success"] is None
    assert row["task_success"] is True


def test_historical_missing_grasp_hold_is_unknown():
    ep = {"success": True, "released": True, "grasp_success": False}
    assert read_grasp_hold(ep) is None
    assert read_rails_released(ep) is True
    assert unknown_bool(None) is None


def test_grasp_hold_requires_lift_and_follow():
    tr = GraspHoldTracker(hold_radius_m=0.06, lift_m=0.02, min_ticks=3)
    tr.set_seated_z(0.90)
    for _ in range(5):
        tr.update(dist_m=0.01, object_z=0.905)  # 5 mm, not enough lift
    assert tr.hold is False
    assert tr.snapshot()["grasp_hold"] is False
    for _ in range(3):
        tr.update(dist_m=0.01, object_z=0.93)
    assert tr.hold is True
    assert tr.snapshot()["grasp_hold_reason"] == "lift_and_follow"


def test_catalog_hold_layout_speed_matches_drive():
    hold = resolve_scene_condition(speed=0.0, catalog_job=True)
    drive = resolve_scene_condition(speed=0.0006, catalog_job=True)
    assert hold.rails == "hold" and drive.rails == "drive"
    assert hold.layout_speed == drive.layout_speed == pytest.approx(DEFAULT_LAYOUT_SPEED)
    assert hold.modifies_layout() and drive.modifies_layout()
    assert trajectory_speed_for_condition(hold, 0.0) == trajectory_speed_for_condition(
        drive, 0.0006
    )


def test_default_layout_speed_constant():
    assert DEFAULT_LAYOUT_SPEED == pytest.approx(0.0006)


def test_presentation_demo_shows_bumper_and_test_uses_ghost_rails():
    demo = resolve_scene_condition(
        speed=0.0006, catalog_job=True, presentation="demo"
    )
    test = resolve_scene_condition(
        speed=0.0006, catalog_job=True, presentation="ghost_rail"
    )
    assert demo.apparatus == "visible" and demo.installs_visual_actor()
    assert demo.rails == "drive"
    assert test.apparatus == "hidden" and not test.installs_visual_actor()
    assert test.uses_layout_actor() and test.rails == "drive"
    assert test.layout_speed == demo.layout_speed


def test_presentation_does_not_touch_original():
    cond = resolve_scene_condition(
        preset="original", speed=0.0006, presentation="demo"
    )
    assert cond.scene == "original" and cond.apparatus == "off"


def test_explicit_apparatus_overrides_presentation():
    cond = resolve_scene_condition(
        speed=0.0006,
        catalog_job=True,
        presentation="test",
        apparatus="visible",
        apparatus_explicit=True,
    )
    assert cond.apparatus == "visible"


def test_presentation_relabels_auto_catalog():
    from argparse import Namespace

    from benchmarks.common.scene import iter_scene_conditions

    demo_args = Namespace(
        scene_preset="auto",
        scene=None,
        apparatus=None,
        rails=None,
        world_clock=None,
        layout_speed=None,
        presentation="demo",
    )
    test_args = Namespace(**{**demo_args.__dict__, "presentation": "test"})
    demo_label, demo = iter_scene_conditions(
        demo_args, speed=0.0, catalog_job=True
    )[0]
    test_label, test = iter_scene_conditions(
        test_args, speed=0.0006, catalog_job=True
    )[0]
    assert demo_label == "apparatus_hold" and demo.rails == "hold"
    assert test_label == "ghost_drive" and test.apparatus == "hidden"
