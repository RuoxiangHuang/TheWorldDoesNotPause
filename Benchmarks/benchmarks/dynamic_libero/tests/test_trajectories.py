"""CPU unit tests for trajectory continuity (no MuJoCo / GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.dynamic_libero.trajectories import build_trajectory, clone_trajectory


KINDS = ["linear", "circle", "sine", "polyline", "stop_and_go", "random"]


@pytest.mark.parametrize("kind", KINDS)
def test_reset_and_step_continuity(kind: str):
    kwargs = {}
    if kind == "circle":
        kwargs = {"radius": 0.05, "omega": 0.02}
    elif kind == "sine":
        kwargs = {"amplitude": 0.02, "wavelength": 0.15}
    elif kind == "polyline":
        kwargs = {"waypoints": [[0, 0, 0], [0.1, 0, 0], [0.1, 0.05, 0]]}
    elif kind == "stop_and_go":
        kwargs = {"period": 20, "duty": 0.5}
    elif kind == "random":
        kwargs = {"n_waypoints": 5, "extent": 0.10, "seed": 0}

    traj = build_trajectory(kind, speed=0.003, **kwargs)
    base = np.array([0.1, -0.2, 0.05])
    traj.reset(base)
    p0 = traj.xyz_at(0.0)
    assert np.allclose(p0, base, atol=1e-9)

    prev = p0
    for _ in range(50):
        cur = traj.step(1.0)
        step = np.linalg.norm(cur - prev)
        # No teleport jumps larger than a few cm for these speeds.
        assert step < 0.05, f"{kind} jumped {step}"
        prev = cur


def test_linear_displacement():
    traj = build_trajectory("linear", speed=0.004, axis="x", direction=1.0)
    traj.reset(np.zeros(3))
    for _ in range(10):
        traj.step(1.0)
    assert np.allclose(traj.xyz_at(traj.t), [0.04, 0, 0], atol=1e-9)


def test_linear_heading_diagonal():
    traj = build_trajectory("linear", speed=0.010, heading_deg=45.0)
    traj.reset(np.zeros(3))
    traj.step(1.0)
    p = traj.xyz_at(traj.t)
    assert abs(p[0] - p[1]) < 1e-9
    assert p[0] > 0.0
    assert abs(p[2]) < 1e-12
    cloned = clone_trajectory(traj)
    assert cloned.heading_deg == pytest.approx(45.0)
    cloned.reset(np.zeros(3))
    cloned.step(1.0)
    assert np.allclose(cloned.xyz_at(cloned.t), p, atol=1e-9)


def test_linear_heading_toward_center_flips_outward():
    traj = build_trajectory("linear", speed=0.003, heading_deg=0.0, toward_center=True)
    traj.reset(np.array([0.20, 0.0, 0.05]))
    assert traj.direction < 0
    assert traj._axis[0] < 0


def test_deflect_uses_travel_heading_after_toward_center():
    from benchmarks.dynamic_libero.trajectories.linear import (
        corridor_clearance,
        deflect_linear_heading,
        effective_heading_deg,
        inward_heading_candidates,
    )

    pick = build_trajectory("linear", speed=0.003, heading_deg=0.0, toward_center=True)
    pick.reset(np.array([0.18, 0.0, 0.05]))
    place_xy = np.array([-0.18, 0.0])
    # Toward-center flipped +x into -x, which aims at the sibling.
    assert effective_heading_deg(pick) == pytest.approx(180.0)
    assert corridor_clearance(np.array([0.18, 0.0]), 180.0, place_xy) < 0.10
    got = deflect_linear_heading(
        pick,
        np.array([0.18, 0.0]),
        place_xy,
        min_clearance=0.14,
        candidates=inward_heading_candidates(np.array([0.18, 0.0])),
    )
    assert corridor_clearance(np.array([0.18, 0.0]), got, place_xy) >= 0.14
    assert pick.direction > 0


def test_speed_zero_static():
    traj = build_trajectory("linear", speed=0.0, axis="y")
    traj.reset(np.array([1.0, 2.0, 3.0]))
    for _ in range(20):
        traj.step(1.0)
    assert np.allclose(traj.xyz_at(traj.t), [1.0, 2.0, 3.0])


def test_stop_and_go_pauses():
    traj = build_trajectory("stop_and_go", speed=0.01, axis="x", period=10, duty=0.5)
    traj.reset(np.zeros(3))
    # First 5 ticks move, next 5 pause.
    for _ in range(5):
        traj.step(1.0)
    mid = traj.xyz_at(traj.t).copy()
    for _ in range(5):
        traj.step(1.0)
    assert np.allclose(traj.xyz_at(traj.t), mid)


def test_build_unknown():
    with pytest.raises(ValueError):
        build_trajectory("spiral", speed=0.001)


def test_random_polyline_seeded_and_varies():
    from benchmarks.dynamic_libero.trajectories import resolve_traj_complexity

    a = build_trajectory("random", speed=0.003, n_waypoints=6, extent=0.12, seed=7)
    b = build_trajectory("random", speed=0.003, n_waypoints=6, extent=0.12, seed=7)
    c = build_trajectory("random", speed=0.003, n_waypoints=6, extent=0.12, seed=8)
    base = np.array([0.05, -0.1, 0.02])
    a.reset(base)
    b.reset(base)
    c.reset(base)
    for _ in range(40):
        a.step(1.0)
        b.step(1.0)
        c.step(1.0)
    assert np.allclose(a.xyz_at(a.t), b.xyz_at(b.t))
    assert not np.allclose(a.xyz_at(a.t), c.xyz_at(c.t))

    kind, kw = resolve_traj_complexity("hard", trajectory="linear", seed=0)
    assert kind == "random"
    assert kw["n_waypoints"] == 5
    kind_m, _ = resolve_traj_complexity("medium", trajectory="linear", seed=0)
    assert kind_m == "sine"
    kind_n, kw_n = resolve_traj_complexity("none", trajectory="circle", overrides={"radius": 0.05})
    assert kind_n == "circle"
    assert kw_n["radius"] == 0.05


def test_clone_curve_keeps_id():
    from benchmarks.dynamic_libero.trajectories import clone_trajectory

    src = build_trajectory("curve", speed=0.003, curve_id="s_wave")
    src.reset(np.array([0.2, -0.1, 0.05]))
    dst = clone_trajectory(src)
    dst.reset(np.array([0.0, 0.0, 0.05]))
    assert dst.name == "curve"
    assert dst.curve_id == "s_wave"
    p0 = dst.xyz_at(0.0)
    p1 = dst.xyz_at(20.0)
    assert float(np.linalg.norm(p1 - p0)) > 0.04


def test_smooth_turn_world_clock_not_replan():
    from benchmarks.dynamic_libero.trajectories import clone_trajectory
    from benchmarks.dynamic_libero.trajectories.smooth_turn import mix_episode_seed

    base = np.array([0.12, -0.08, 0.04])
    a = build_trajectory(
        "smooth_turn",
        speed=0.001,
        event_seed=7,
        toward_center=False,
        axis="x",
        direction=1.0,
    )
    b = build_trajectory(
        "smooth_turn",
        speed=0.001,
        event_seed=7,
        toward_center=False,
        axis="x",
        direction=1.0,
    )
    a.reset(base)
    b.reset(base)
    assert a.event_t == pytest.approx(b.event_t)
    assert 20.0 <= float(a.event_t) <= 60.0
    snapped = 0
    for s in range(30):
        tr = build_trajectory(
            "smooth_turn",
            speed=0.001,
            event_seed=s,
            toward_center=False,
            axis="x",
            direction=1.0,
        )
        tr.reset(base)
        grid = float(tr.event_t) / 10.0
        if abs(grid - round(grid)) < 1e-9:
            snapped += 1
    assert snapped <= 4
    for t in (0.0, a.event_t, a.event_t + 6.0, 80.0):
        assert np.allclose(a.xyz_at(t), b.xyz_at(t), atol=1e-9)

    c = build_trajectory(
        "smooth_turn",
        speed=0.001,
        event_seed=8,
        toward_center=False,
        axis="x",
        direction=1.0,
    )
    c.reset(base)
    assert abs(float(c.event_t) - float(a.event_t)) > 1e-6

    # Same episode key → same mix; different init → different seed.
    assert mix_episode_seed(0, 0, 0) == mix_episode_seed(0, 0, 0)
    assert mix_episode_seed(0, 0, 0) != mix_episode_seed(0, 0, 1)

    cloned = clone_trajectory(a)
    cloned.reset(base)
    assert cloned.event_t == pytest.approx(a.event_t)
    assert cloned.turn_sign == a.turn_sign
    assert np.allclose(cloned.xyz_at(a.event_t + 12.0), a.xyz_at(a.event_t + 12.0), atol=1e-8)


def test_smooth_turn_continuity_and_heading_change():
    traj = build_trajectory(
        "smooth_turn",
        speed=0.001,
        event_t=25.3,
        turn_sign=1,
        turn_angle_deg=75.0,
        turn_duration_ticks=12.0,
        toward_center=False,
        axis="x",
        direction=1.0,
        event_seed=0,
    )
    traj.reset(np.zeros(3))
    prev = traj.xyz_at(0.0)
    for k in range(1, 80):
        cur = traj.xyz_at(float(k))
        assert float(np.linalg.norm(cur - prev)) < 0.01
        prev = cur
    # Inbound +x; after the turn heading should be ~+75°.
    v0 = traj.velocity_at(0.0, hz=20.0)
    v1 = traj.velocity_at(25.3 + 12.0, hz=20.0)
    assert v0[0] > 0 and abs(v0[1]) < 1e-9
    ang = np.degrees(np.arctan2(v1[1], v1[0]))
    assert ang == pytest.approx(75.0, abs=1.5)
    # Raised cosine: heading rate is ~0 at event entry.
    v_in = traj.velocity_at(25.3 + 0.05, hz=20.0)
    assert abs(np.arctan2(v_in[1], v_in[0])) < np.deg2rad(4.0)
    alias = build_trajectory("turn", speed=0.001, event_t=10.0, turn_sign=-1)
    assert alias.name == "smooth_turn"


def test_smooth_turn_event_spec_fields():
    traj = build_trajectory(
        "smooth_turn",
        speed=0.001,
        event_t=30.0,
        turn_sign=-1,
        turn_angle_deg=75.0,
        turn_duration_ticks=12.0,
        event_seed=11,
        toward_center=False,
        axis="x",
        direction=1.0,
    )
    traj.reset(np.zeros(3))
    spec = traj.event_spec()
    assert spec["event_type"] == "turn"
    assert spec["event_time"] == pytest.approx(30.0)
    assert spec["transition_duration"] == pytest.approx(12.0)
    assert spec["event_magnitude"] == pytest.approx(-75.0)
    assert spec["event_seed"] == 11
    assert spec["turn_event_t"] == pytest.approx(30.0)

