"""Tests for the Dynamic-LIBERO 1→3 task catalog and rails variants."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.dynamic_libero.env.dynamic_tasks import (
    get_dynamic_task,
    parse_dynamic_tasks,
    parse_rails_variant,
    seed_tasks,
)
from benchmarks.dynamic_libero.env.pusher import make_pusher_scene


def test_seed_soup_is_three_tasks():
    seeds = seed_tasks()
    assert [t.id for t in seeds] == [
        "libero_object.t00.pick",
        "libero_object.t00.place",
        "libero_object.t00.both",
    ]
    langs = {t.language for t in seeds}
    assert langs == {"pick up the alphabet soup and place it in the basket"}
    assert [t.variant for t in seeds] == ["pick", "place", "both"]
    assert all(t.pick_target == "alphabet_soup_1" for t in seeds)
    assert all(t.place_target == "basket_1" for t in seeds)
    assert seeds[0].pick_motion and not seeds[0].place_motion
    assert seeds[1].place_motion and not seeds[1].pick_motion
    assert seeds[2].pick_motion and seeds[2].place_motion


def test_parse_dynamic_tasks_seed_alias():
    assert [t.id for t in parse_dynamic_tasks("seed")] == [
        t.id for t in seed_tasks()
    ]
    assert parse_dynamic_tasks("off") == []
    one = parse_dynamic_tasks("libero_object.t00.place")
    assert len(one) == 1 and one[0].variant == "place"


def test_object_suite_ready_is_thirty():
    ready = parse_dynamic_tasks("ready")
    object_ready = [t for t in ready if t.suite == "libero_object"]
    assert len(object_ready) == 30


def test_parse_rails_variant_aliases():
    assert parse_rails_variant("basket") == "place"
    assert parse_rails_variant("dual") == "both"
    assert parse_rails_variant("auto", default="pick") == "pick"


def test_make_pusher_scene_by_variant():
    pick, place = make_pusher_scene("pusher", "pick")
    assert pick is not None and place is None
    pick, place = make_pusher_scene("pusher", "place")
    assert pick is None and place is not None
    pick, place = make_pusher_scene("pusher", "both")
    assert pick is not None and place is not None
    pick, place = make_pusher_scene("carrier", "pick")
    assert pick is not None and place is None
    assert bool(getattr(pick, "is_carrier", False))
    pick, place = make_pusher_scene("carrier", "both")
    assert pick is not None and place is not None


def test_irregular_object_tasks_are_twenty_unique_smooth_curves():
    from benchmarks.dynamic_libero.env.dynamic_tasks import irregular_tasks
    from benchmarks.dynamic_libero.trajectories import build_trajectory

    tasks = irregular_tasks()
    assert len(tasks) == 20
    ids = [t.id for t in tasks]
    assert ids[0] == "libero_object.t00.pick.irregular"
    assert ids[1] == "libero_object.t00.place.irregular"
    assert len(set(ids)) == 20
    fingerprints = []
    names = []
    for t in tasks:
        assert t.status == "irregular"
        assert t.trajectory_kind == "curve"
        assert t.variant in ("pick", "place")
        kind, kw = t.traj_spec()
        assert kind == "curve"
        traj = build_trajectory(kind, speed=0.003, **kw)
        traj.reset(base_pos=[0.0, 0.0, 0.05])
        names.append(t.path_name)
        pts = np.stack([traj.xyz_at(k) for k in range(0, 80, 4)])
        fingerprints.append(tuple(np.round(pts[:, :2].ravel(), 4)))
        extent = float(np.max(np.abs(np.stack(traj.rel_waypoints)[:, :2])))
        assert extent <= 0.32
        assert float(traj._cum[-1]) >= 0.18
        prev_xy = None
        prev_h = None
        max_ddeg = 0.0
        for k in range(90):
            p = traj.xyz_at(float(k))
            if prev_xy is not None:
                d = p[:2] - prev_xy
                if float(np.linalg.norm(d)) < 1e-6:
                    break
                h = float(np.arctan2(d[1], d[0]))
                if prev_h is not None:
                    dh = (h - prev_h + np.pi) % (2.0 * np.pi) - np.pi
                    max_ddeg = max(max_ddeg, abs(float(np.degrees(dh))))
                prev_h = h
            prev_xy = p[:2].copy()
        assert max_ddeg < 12.0, f"{t.path_name} heading jump {max_ddeg:.1f} deg/tick"
    assert len(set(names)) == 20
    assert len(set(fingerprints)) == 20
    from benchmarks.dynamic_libero.env.irregular_paths import sample_curve

    clothoid = sample_curve("clothoid_right", n=160)
    ymax = float(np.max(np.abs(clothoid[:, 1])))
    end_dxy = clothoid[-1] - clothoid[-2]
    end_yaw = float(np.degrees(np.arctan2(end_dxy[1], end_dxy[0])))
    # BBQ pick used to yaw ~60° / 11 cm off-axis; that grasp-then-timeout
    # the basket. Keep a right-bending clothoid, but stay in the S-curve band.
    assert ymax < 0.05, ymax
    assert abs(end_yaw) < 30.0, end_yaw


def test_spatial_carrier_uses_scene_language():
    from benchmarks.common.language import SPATIAL_CARRIER_SCENE_INSTRUCTION
    from benchmarks.dynamic_libero.env.dynamic_tasks import spatial_tasks

    tasks = spatial_tasks()
    assert len(tasks) == 10
    assert [t.task_id for t in tasks] == list(range(10))
    assert all(t.motive == "carrier" for t in tasks)
    assert all(t.eval_split == "extension" for t in tasks)
    assert all(t.language_policy == "scene" for t in tasks)
    assert all(t.language == SPATIAL_CARRIER_SCENE_INSTRUCTION for t in tasks)
    assert tasks[4].source_language == (
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"
    )
    assert "drawer" not in tasks[4].language
    assert parse_dynamic_tasks("spatial")[0].id == "libero_spatial.t00.pick.carrier"
    assert parse_dynamic_tasks("extension") == tasks


def test_main_track_is_identity_object_tasks():
    from benchmarks.dynamic_libero.env.dynamic_tasks import main_track_tasks

    main = main_track_tasks()
    assert len(main) == 50
    assert all(t.suite == "libero_object" for t in main)
    assert all(t.eval_split == "main" for t in main)
    assert all(t.language_policy == "keep" for t in main)
    assert all("basket" in t.language for t in main)


def test_reconstructed_set_is_sixty():
    from benchmarks.dynamic_libero.env.dynamic_tasks import (
        RECONSTRUCTED_REPLACEMENTS,
        reconstructed_tasks,
    )

    tasks = reconstructed_tasks()
    assert len(tasks) == 60
    assert sum(1 for t in tasks if t.suite == "libero_object" and t.status != "irregular") == 30
    assert sum(1 for t in tasks if t.status == "irregular") == 20
    assert sum(1 for t in tasks if t.status == "spatial") == 10
    assert [t.id for t in parse_dynamic_tasks("reconstructed")] == [t.id for t in tasks]
    ids = {t.id for t in tasks}
    for tid in RECONSTRUCTED_REPLACEMENTS:
        assert tid in ids
    t03 = get_dynamic_task("libero_object.t03.pick.irregular")
    assert t03.path_name == "clothoid_right" and t03.trajectory_kind == "curve"
    for tid in (
        "libero_spatial.t00.pick.carrier",
        "libero_spatial.t04.pick.carrier",
        "libero_spatial.t05.pick.carrier",
        "libero_spatial.t07.pick.carrier",
        "libero_spatial.t09.pick.carrier",
    ):
        t = get_dynamic_task(tid)
        assert t.motive == "carrier" and t.place_target == "plate_1"


def test_reconstructed_linear_headings_are_diverse():
    from benchmarks.dynamic_libero.env.dynamic_tasks import reconstructed_tasks
    from benchmarks.dynamic_libero.trajectories import build_trajectory

    linear = [t for t in reconstructed_tasks() if t.trajectory_kind == "linear"]
    headings = [t.heading_deg for t in linear]
    assert all(h is not None for h in headings)
    assert len(set(round(h, 1) for h in headings)) >= 20
    spatial = [t for t in linear if t.status == "spatial"]
    assert all(5.0 <= float(t.heading_deg) <= 85.0 for t in spatial)
    # Object suite covers more than one cardinal direction (not all +x).
    obj = [t for t in linear if t.suite == "libero_object"]
    vecs = []
    for t in obj:
        kind, kw = t.traj_spec()
        assert kw.get("toward_center") is True
        traj = build_trajectory(kind, speed=0.003, **kw)
        traj.reset(np.zeros(3))
        vecs.append(tuple(np.round(traj._axis[:2], 3)))
    assert len(set(vecs)) >= 8


def test_deflect_linear_heading_misses_plate():
    from benchmarks.dynamic_libero.trajectories import build_trajectory
    from benchmarks.dynamic_libero.trajectories.linear import (
        corridor_clearance,
        deflect_linear_heading,
    )

    traj = build_trajectory("linear", speed=0.003, heading_deg=10.0)
    # Far enough left that a 0–90° candidate (spatial kitchen quadrant) can
    # miss the plate by 16 cm; the original 13 cm lateral gap could not.
    origin = np.array([-0.20, 0.18])
    plate = np.array([0.07, 0.19])
    assert corridor_clearance(origin, 10.0, plate) < 0.16
    got = deflect_linear_heading(traj, origin, plate, min_clearance=0.16)
    assert corridor_clearance(origin, got, plate) >= 0.16


def test_parse_dynamic_tasks_irregular():
    tasks = parse_dynamic_tasks("irregular")
    assert len(tasks) == 20
    assert all(t.status == "irregular" for t in tasks)


def test_react_track_aligns_with_main_pick():
    from benchmarks.dynamic_libero.env.dynamic_tasks import (
        main_object_pick_tasks,
        main_track_tasks,
        react_tasks,
        reconstructed_tasks,
        spatial_tasks,
    )
    from benchmarks.dynamic_libero.trajectories.smooth_turn import (
        DIAGNOSTIC_TURN_ANGLE_DEG,
        DIAGNOSTIC_TURN_DURATION_TICKS,
    )

    mains = main_object_pick_tasks()
    reacts = react_tasks()
    assert len(mains) == 10
    assert len(reacts) == 10
    assert [t.id for t in parse_dynamic_tasks("react")] == [t.id for t in reacts]
    for main, react in zip(mains, reacts):
        assert react.id == f"{main.id}.react"
        assert react.aligned_main_id == main.id
        assert react.language == main.language
        assert react.pick_target == main.pick_target
        assert react.place_target == main.place_target
        assert react.heading_deg == main.heading_deg
        assert react.variant == "pick"
        assert react.track == "react"
        assert react.event_type == "turn"
        assert react.transition_duration == DIAGNOSTIC_TURN_DURATION_TICKS
        assert react.event_magnitude == DIAGNOSTIC_TURN_ANGLE_DEG
        kind, kw = react.traj_spec()
        assert kind == "smooth_turn"
        assert kw["heading_deg"] == main.heading_deg
        assert kw["turn_duration_ticks"] == DIAGNOSTIC_TURN_DURATION_TICKS
        assert kw["turn_angle_deg"] == DIAGNOSTIC_TURN_ANGLE_DEG
    rec = reconstructed_tasks()
    assert len(rec) == 60
    assert all(t.track == "main" for t in rec)
    assert all(t.status != "react" for t in rec)
    assert sum(1 for t in rec if t.eval_split == "main") == 50
    assert sum(1 for t in rec if t.eval_split == "extension") == 10
    assert parse_dynamic_tasks("main") == main_track_tasks()
    assert parse_dynamic_tasks("extension") == spatial_tasks()
    assert all(t.eval_split == "react" for t in parse_dynamic_tasks("react"))


def test_latency_slice_is_five_aligned_picks():
    from benchmarks.dynamic_libero.env.dynamic_tasks import (
        LATENCY_SLICE_TASK_IDS,
        SLICE_EVENT_T_HI,
        SLICE_EVENT_T_LO,
        latency_slice_tasks,
        place_slice_tasks,
        react_slice_tasks,
    )

    mains = latency_slice_tasks()
    reacts = react_slice_tasks()
    places = place_slice_tasks()
    assert [t.task_id for t in mains] == list(LATENCY_SLICE_TASK_IDS)
    assert [t.task_id for t in reacts] == list(LATENCY_SLICE_TASK_IDS)
    assert [t.task_id for t in places] == list(LATENCY_SLICE_TASK_IDS)
    assert all(t.variant == "pick" and t.track == "main" for t in mains)
    assert all(t.track == "react" and t.variant == "pick" for t in reacts)
    assert all(t.variant == "place" and t.track == "main" and t.place_motion for t in places)
    assert all(not t.pick_motion for t in places)
    assert [t.id for t in parse_dynamic_tasks("latency")] == [t.id for t in mains]
    assert [t.id for t in parse_dynamic_tasks("place_slice")] == [t.id for t in places]
    assert [t.id for t in parse_dynamic_tasks("pi_place")] == [t.id for t in places]
    assert [t.id for t in parse_dynamic_tasks("react_slice")] == [t.id for t in reacts]
    for main, react in zip(mains, reacts):
        assert react.aligned_main_id == main.id
        assert react.language == main.language
        assert react.heading_deg == main.heading_deg
        assert react.event_t_lo == SLICE_EVENT_T_LO
        assert react.event_t_hi == SLICE_EVENT_T_HI
        _, kw = react.traj_spec()
        assert kw["event_t_lo"] == SLICE_EVENT_T_LO
        assert kw["event_t_hi"] == SLICE_EVENT_T_HI


def test_cli_defaults_do_not_override_slice_event_window():
    import argparse

    from benchmarks.dynamic_libero.eval.cli_common import traj_kwargs_from_args
    from benchmarks.dynamic_libero.env.dynamic_tasks import (
        SLICE_EVENT_T_HI,
        SLICE_EVENT_T_LO,
        get_dynamic_task,
    )
    from benchmarks.dynamic_libero.trajectories.smooth_turn import (
        default_smooth_turn_kwargs,
    )

    args = argparse.Namespace(
        trajectory="smooth_turn",
        toward_center=True,
        axis="x",
        direction=1.0,
        turn_event_t=None,
        turn_event_t_lo=None,
        turn_event_t_hi=None,
        turn_angle_deg=None,
        turn_duration_ticks=None,
        turn_sign=None,
    )
    cli_kw = traj_kwargs_from_args(args)
    _, spec_kw = get_dynamic_task("libero_object.t00.pick.react").traj_spec()
    merged = {**spec_kw, **cli_kw}
    for k, v in default_smooth_turn_kwargs().items():
        merged.setdefault(k, v)
    assert merged["event_t_lo"] == SLICE_EVENT_T_LO
    assert merged["event_t_hi"] == SLICE_EVENT_T_HI


def test_get_dynamic_task_unknown():
    try:
        get_dynamic_task("nope")
    except KeyError:
        return
    raise AssertionError("expected KeyError")
