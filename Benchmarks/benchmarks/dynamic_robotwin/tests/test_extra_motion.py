"""Native extra motions stay off the 95-task rails and keep moving at scale 1."""

from benchmarks.dynamic_robotwin.env.extra_motion import (
    BUNNY_START,
    ORANGE_START,
    TABLE_Z,
    bunny_pose,
    car_sample,
    orange_pose,
    pool_strike_velocity,
    shuttlecock_initial,
    step_shuttlecock,
)
from benchmarks.dynamic_robotwin.env.extra_tasks import extra_tasks
from benchmarks.dynamic_robotwin.env.official import parse_task_names
from benchmarks.official_tasks import N_TASKS, official_tasks


def test_five_extras_are_eval_ready_and_outside_the_95():
    extras = extra_tasks()
    assert [t.id for t in extras] == [
        "catch_shuttlecock",
        "catch_bunny_toy",
        "stop_rolling_orange",
        "pursue_toycar",
        "collide_pool_balls",
    ]
    assert all(t.eval_ready for t in extras)
    assert parse_task_names("extra") == [t.id for t in extras]
    paper = official_tasks()
    paper_ids = {t.id for t in paper["dynamic_libero"]} | {t.id for t in paper["dynamic_robotwin"]}
    assert len(paper_ids) == N_TASKS == 95
    assert paper_ids.isdisjoint({t.id for t in extras})


def test_shuttlecock_falls_and_seed_changes_the_lob():
    p, v = step_shuttlecock([0.2, 0.0, 0.84], [-0.55, 0.0, 3.05], 0.05)
    assert p[2] != 0.84
    assert v[2] < 3.05
    a = shuttlecock_initial(0, 1.0)
    b = shuttlecock_initial(1, 1.0)
    assert a[2] != b[2]


def test_bunny_hops_and_orange_rolls():
    xyz0, _ = bunny_pose(0.0)
    xyz_apex, _ = bunny_pose(0.18)
    assert abs(xyz0[0] - BUNNY_START[0]) < 1e-6
    assert xyz_apex[2] > TABLE_Z + 0.02
    start, _ = orange_pose(0.0)
    later, _ = orange_pose(1.2)
    assert abs(start[0] - ORANGE_START[0]) < 1e-6
    assert later[0] < start[0]


def test_toy_car_stays_parked_through_idle_then_drives():
    parked = car_sample(0.0)
    moving = car_sample(1.0)
    assert parked["speed"] == 0.0
    assert moving["speed"] > 0.0
    assert not (parked["xy"] == moving["xy"]).all()


def test_pool_cue_travels_toward_negative_y():
    linear, _ = pool_strike_velocity(1.0)
    assert linear[1] < 0.0
    held, _ = pool_strike_velocity(0.0)
    assert held[1] == 0.0
