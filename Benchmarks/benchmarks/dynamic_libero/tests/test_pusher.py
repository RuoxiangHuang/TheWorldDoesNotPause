"""Unit tests for the pusher-car motive (car on rails, original pick mesh)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from benchmarks.dynamic_libero.env.driver import RealtimeDriver
from benchmarks.dynamic_libero.env.movers import MoverSpec
from benchmarks.dynamic_libero.env.pusher import (
    PUSHER_BODY,
    PusherCar,
    PusherConfig,
    car_xy_behind,
    inject_pusher_xml,
)
from benchmarks.dynamic_libero.trajectories import build_trajectory


class _FakeSimData:
    def __init__(self, qpos: np.ndarray):
        self._qpos = {0: qpos.copy()}
        self._qvel = {0: np.zeros(6)}

    def get_joint_qpos(self, _jname):
        return self._qpos[0].copy()

    def set_joint_qpos(self, _jname, q):
        self._qpos[0] = np.asarray(q, dtype=np.float64).copy()

    def get_joint_qvel(self, _jname):
        return self._qvel[0].copy()

    def set_joint_qvel(self, _jname, v):
        self._qvel[0] = np.asarray(v, dtype=np.float64).copy()


class _FakeSim:
    def __init__(self, qpos: np.ndarray):
        self.data = _FakeSimData(qpos)

    def forward(self):
        return None


class _FakeObj:
    def __init__(self):
        self.joints = ["free_joint"]
        self.horizontal_radius = 0.025
        self.bottom_offset = np.array([0.0, 0.0, -0.03])


class _FakeBase:
    def __init__(self, qpos: np.ndarray):
        self.sim = _FakeSim(qpos)
        self.objects_dict = {"alphabet_soup_1": _FakeObj()}
        self.obj_of_interest = ["alphabet_soup_1"]
        self._rtl_pre_patched = True

    def _pre_action(self, action, policy_step=False):
        return None


class _FakeEnv:
    def __init__(self, qpos: np.ndarray):
        self.env = _FakeBase(qpos)


def test_inject_pusher_keeps_fixed_body_no_joint():
    xml = """
    <mujoco>
      <worldbody>
        <body name="floor" pos="0 0 0"><geom type="plane" size="1 1 0.1"/></body>
      </worldbody>
    </mujoco>
    """
    out = inject_pusher_xml(xml, PusherConfig())
    root = ET.fromstring(out)
    bodies = [b.get("name") for b in root.find("worldbody").findall("body")]
    assert PUSHER_BODY in bodies
    geoms = [g.get("name") for g in root.iter("geom")]
    assert f"{PUSHER_BODY}_chassis" in geoms
    assert f"{PUSHER_BODY}_hood" in geoms
    assert f"{PUSHER_BODY}_bumper" in geoms
    assert f"{PUSHER_BODY}_wheel_0" in geoms
    assert "freejoint" not in out
    assert "<joint" not in out
    out2 = inject_pusher_xml(out, PusherConfig())
    n = sum(
        1
        for b in ET.fromstring(out2).find("worldbody").findall("body")
        if b.get("name") == PUSHER_BODY
    )
    assert n == 1


def test_car_sits_behind_object_along_forward():
    rails = np.array([0.10, -0.20])
    fwd = np.array([1.0, 0.0])
    cxy = car_xy_behind(
        rails_xy=rails, forward_xy=fwd, half_length=0.04, object_radius=0.025, gap=0.0
    )
    assert cxy[0] < rails[0]
    assert abs(cxy[1] - rails[1]) < 1e-9
    assert abs((rails[0] - cxy[0]) - 0.065) < 1e-9


def test_pusher_moves_object_without_yaw_snap():
    q0 = np.array([0.10, -0.20, 0.05, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env = _FakeEnv(q0.copy())
    traj = build_trajectory("linear", speed=0.001, axis="x", direction=1.0)
    mover = MoverSpec(
        target="alphabet_soup_1",
        category="alphabet_soup",
        noun="alphabet soup",
        motion="drive",
        height_offset=0.0,
    )
    car = PusherCar()
    driver = RealtimeDriver(env, traj, mover=mover, pusher=car, control_hz=20.0)
    driver.base = env.env
    env.env._rtl_driver = driver

    driver.on_episode_start()
    q_start = env.env.sim.data.get_joint_qpos("free_joint")
    assert np.allclose(q_start[:3], q0[:3]), "layout must not teleport the can"
    assert np.allclose(q_start[3:7], q0[3:7])
    assert car.cfg.pos[0] < q0[0], "car starts behind the can along +X"
    assert driver.motion == "pusher"
    assert driver.align_heading is False

    driver.begin_policy()
    q_t0 = env.env.sim.data.get_joint_qpos("free_joint")
    assert np.allclose(q_t0[:3], q0[:3], atol=1e-9)
    assert np.allclose(q_t0[3:7], q0[3:7]), "soup stays upright at engage"

    n = 20
    for _ in range(n):
        driver.on_control_step()
        driver._apply_rails_drive()
    q_n = env.env.sim.data.get_joint_qpos("free_joint")
    expected_x = q0[0] + n * 0.001
    assert abs(float(q_n[0]) - expected_x) < 1e-9
    assert abs(float(q_n[1]) - q0[1]) < 1e-9
    assert np.allclose(q_n[3:7], q0[3:7]), "pusher must not yaw-snap the pick mesh"
    assert float(car.cfg.pos[0]) < float(q_n[0])
    assert abs(float(driver.object_drift_m) - n * 0.001) < 1e-8


def test_begin_policy_does_not_slam_settled_can_into_floor():
    """Wait-period physics lifts a penetrating spawn; rails must not undo that."""
    q0 = np.array([0.10, -0.20, 0.015, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env = _FakeEnv(q0.copy())
    traj = build_trajectory("linear", speed=0.001, axis="x", direction=1.0)
    mover = MoverSpec(
        target="alphabet_soup_1",
        category="alphabet_soup",
        noun="alphabet soup",
        motion="drive",
        height_offset=0.0,
    )
    car = PusherCar()
    driver = RealtimeDriver(env, traj, mover=mover, pusher=car, control_hz=20.0)
    driver.base = env.env
    env.env._rtl_driver = driver
    driver.on_episode_start()
    q = env.env.sim.data.get_joint_qpos("free_joint").copy()
    q[2] = 0.038  # settled on the floor after wait
    env.env.sim.data.set_joint_qpos("free_joint", q)
    driver.begin_policy()
    q1 = env.env.sim.data.get_joint_qpos("free_joint")
    assert float(q1[2]) == 0.038
    assert abs(float(driver.trajectory.base_pos[2]) - 0.038) < 1e-9


def test_inject_carrier_has_bed_and_second_car_is_distinct():
    from benchmarks.dynamic_libero.env.pusher import (
        PLACE_PUSHER_BODY,
        carrier_config,
        place_bumper_config,
    )

    xml = "<mujoco><worldbody></worldbody></mujoco>"
    out = inject_pusher_xml(xml, carrier_config())
    geoms = [g.get("name") for g in ET.fromstring(out).iter("geom")]
    assert f"{PUSHER_BODY}_bed" in geoms
    assert f"{PUSHER_BODY}_tail" in geoms
    vis = next(g for g in ET.fromstring(out).iter("geom") if g.get("name") == f"{PUSHER_BODY}_chassis")
    assert vis.get("contype") == "0"
    assert not any((g.get("name") or "").endswith("_col") for g in ET.fromstring(out).iter("geom"))
    out2 = inject_pusher_xml(out, place_bumper_config())
    bodies = [b.get("name") for b in ET.fromstring(out2).find("worldbody").findall("body")]
    assert PUSHER_BODY in bodies
    assert PLACE_PUSHER_BODY in bodies
    assert "freejoint" not in out2


def test_carrier_lifts_object_onto_bed():
    from benchmarks.dynamic_libero.env.pusher import carrier_config

    q0 = np.array([0.10, -0.20, 0.05, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env = _FakeEnv(q0.copy())
    traj = build_trajectory("linear", speed=0.001, axis="x", direction=1.0)
    mover = MoverSpec(
        target="alphabet_soup_1",
        category="alphabet_soup",
        noun="alphabet soup",
        motion="drive",
    )
    car = PusherCar(carrier_config())
    driver = RealtimeDriver(env, traj, mover=mover, pusher=car, control_hz=20.0)
    driver.base = env.env
    env.env._rtl_driver = driver
    driver.on_episode_start()
    q1 = env.env.sim.data.get_joint_qpos("free_joint")
    assert np.allclose(q1[:3], q0[:3]), "carrier waits to settle before lifting"
    driver.begin_policy()
    q1 = env.env.sim.data.get_joint_qpos("free_joint")
    assert float(q1[2]) > float(q0[2]) + 0.01
    assert driver.motion == "carrier"
    assert np.allclose(q1[:2], q0[:2])
    driver.begin_policy()
    for _ in range(10):
        driver.on_control_step()
        driver._apply_rails_drive()
    q2 = env.env.sim.data.get_joint_qpos("free_joint")
    assert abs(float(q2[0]) - (q0[0] + 0.010)) < 1e-9
    assert float(q2[2]) > float(q0[2])


def test_oriented_box_zmin_axis_aligned():
    from benchmarks.dynamic_libero.env.pusher import oriented_box_zmin

    z = oriented_box_zmin(
        np.array([0.0, 0.0, 1.0]),
        np.eye(3),
        np.array([0.10, 0.20, 0.05]),
    )
    assert abs(z - 0.95) < 1e-12


def test_corridor_slides_blocker_off_path():
    from benchmarks.dynamic_libero.env.pusher import corridor_lateral_delta

    d = corridor_lateral_delta(
        np.array([0.12, 0.01]),
        start_xy=np.array([0.0, 0.0]),
        forward_xy=np.array([1.0, 0.0]),
        travel=0.40,
        rear=0.12,
        half_width=0.06,
    )
    assert d is not None
    assert abs(float(d[0])) < 1e-9
    assert float(d[1]) > 0.04
    miss = corridor_lateral_delta(
        np.array([0.12, 0.20]),
        start_xy=np.array([0.0, 0.0]),
        forward_xy=np.array([1.0, 0.0]),
        travel=0.40,
        rear=0.12,
        half_width=0.06,
    )
    assert miss is None


def test_polyline_corridor_clears_blocker_on_bent_leg():
    from benchmarks.dynamic_libero.env.pusher import corridor_delta_polyline
    import numpy as np

    verts = np.array([[0.0, 0.0], [0.12, 0.0], [0.12, 0.12]], dtype=np.float64)
    # On the second (vertical) leg, slightly to the left of x=0.12.
    d = corridor_delta_polyline(
        np.array([0.11, 0.08]),
        verts,
        half_width=0.05,
        rear=0.05,
    )
    assert d is not None
    assert abs(float(d[0])) > 0.02
    miss = corridor_delta_polyline(
        np.array([0.30, 0.08]),
        verts,
        half_width=0.05,
        rear=0.05,
    )
    assert miss is None


def test_place_variant_leaves_pick_target_still():
    q0 = np.array([0.10, -0.20, 0.05, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env = _FakeEnv(q0.copy())
    traj = build_trajectory("linear", speed=0.001, axis="x", direction=1.0)
    mover = MoverSpec(
        target="alphabet_soup_1",
        category="alphabet_soup",
        noun="alphabet soup",
        motion="drive",
        height_offset=0.0,
    )
    car = PusherCar()
    driver = RealtimeDriver(
        env,
        traj,
        mover=mover,
        pusher=None,
        place_pusher=car,
        rails_variant="place",
        control_hz=20.0,
    )
    driver.base = env.env
    env.env._rtl_driver = driver
    driver.on_episode_start()
    assert driver.rails_active is False
    assert driver.pick_motion is False
    assert driver.place_motion is True
    driver.begin_policy()
    for _ in range(20):
        driver.on_control_step()
        driver._apply_rails_drive()
    q_n = env.env.sim.data.get_joint_qpos("free_joint")
    assert np.allclose(q_n[:2], q0[:2], atol=1e-9)


def test_clear_path_protects_stationary_place_target():
    """Spatial pick-only: plate stays put even if it sits on the corridor."""
    from types import SimpleNamespace

    from benchmarks.dynamic_libero.env.pusher import (
        PusherCar,
        carrier_config,
        clear_path_obstacles,
    )

    class _Obj:
        def __init__(self, joint, radius=0.04):
            self.joints = [joint]
            self.horizontal_radius = radius
            self.bottom_offset = np.array([0.0, 0.0, -0.03])

    class _Data:
        def __init__(self):
            self.q = {
                "bowl_j": np.array([0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]),
                "plate_j": np.array([0.18, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]),
                "cookie_j": np.array([0.22, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]),
            }
            self.v = {k: np.zeros(6) for k in self.q}

        def get_joint_qpos(self, n):
            return self.q[n].copy()

        def set_joint_qpos(self, n, q):
            self.q[n] = np.asarray(q, dtype=np.float64).copy()

        def get_joint_qvel(self, n):
            return self.v[n].copy()

        def set_joint_qvel(self, n, v):
            self.v[n] = np.asarray(v, dtype=np.float64).copy()

    class _Sim:
        def __init__(self):
            self.data = _Data()

        def forward(self):
            return None

    class _Base:
        def __init__(self):
            self.sim = _Sim()
            self.objects_dict = {
                "akita_black_bowl_1": _Obj("bowl_j"),
                "plate_1": _Obj("plate_j", radius=0.08),
                "cookies_1": _Obj("cookie_j"),
            }

    traj = build_trajectory("linear", speed=0.001, axis="x", direction=1.0)
    traj.reset(base_pos=np.array([0.0, 0.0, 0.05]))
    car = PusherCar(carrier_config())
    car._traj = traj
    driver = SimpleNamespace(
        base=_Base(),
        pick_motion=True,
        place_motion=False,
        target="akita_black_bowl_1",
        place_target=None,
        place_target_name="plate_1",
        workspace_aabb=None,
    )
    n = clear_path_obstacles(driver, [car])
    plate_xy = driver.base.sim.data.get_joint_qpos("plate_j")[:2]
    cookie_xy = driver.base.sim.data.get_joint_qpos("cookie_j")[:2]
    assert np.allclose(plate_xy, [0.18, 0.0], atol=1e-9)
    assert abs(float(cookie_xy[1])) > 0.02
    assert n >= 1


def test_car_world_aabb_respects_yaw():
    from benchmarks.dynamic_libero.env.pusher import PusherCar, car_world_aabb, carrier_config

    car = PusherCar(carrier_config())
    car.cfg.pos[:] = [0.0, 0.0, 0.93]
    car.cfg.forward_xy[:] = [1.0, 0.0]
    lo, hi = car_world_aabb(car)
    assert hi[0] - lo[0] > hi[1] - lo[1]  # longer along +x
    car.cfg.forward_xy[:] = [0.0, 1.0]
    lo2, hi2 = car_world_aabb(car)
    assert hi2[1] - lo2[1] > hi2[0] - lo2[0]


def test_propose_tabletop_xy_leaves_cabinet():
    from benchmarks.dynamic_libero.env.pusher import (
        PusherCar,
        car_clips_fixtures,
        carrier_config,
        propose_tabletop_xy,
        _trial_car_pose,
    )

    car = PusherCar(carrier_config())
    fixtures = [
        (
            "wooden_cabinet_1_base",
            np.array([-0.23, -0.53, 0.76], dtype=np.float64),
            np.array([0.26, -0.09, 1.29], dtype=np.float64),
        )
    ]
    parked = propose_tabletop_xy(
        car,
        current_xy=np.array([0.085, -0.13], dtype=np.float64),
        fwd=np.array([1.0, 0.0], dtype=np.float64),
        table_z=0.90,
        fixtures=fixtures,
        plate_xy=np.array([0.07, 0.19], dtype=np.float64),
    )
    _trial_car_pose(car, parked, np.array([1.0, 0.0]), 0.90)
    assert not car_clips_fixtures(car, fixtures)
    assert float(parked[1]) > -0.09


def test_table_support_z_prefers_plate():
    from types import SimpleNamespace

    from benchmarks.dynamic_libero.env.pusher import table_support_z

    class _Obj:
        def __init__(self, joint, z, offset=-0.03):
            self.joints = [joint]
            self.bottom_offset = np.array([0.0, 0.0, offset])

    class _Data:
        def __init__(self):
            self.q = {
                "bowl_j": np.array([0.08, -0.13, 1.12, 1.0, 0.0, 0.0, 0.0]),
                "plate_j": np.array([0.07, 0.19, 0.93, 1.0, 0.0, 0.0, 0.0]),
            }

        def get_joint_qpos(self, n):
            return self.q[n].copy()

    class _Sim:
        def __init__(self):
            self.data = _Data()
            self.model = None

        def forward(self):
            return None

    driver = SimpleNamespace(
        base=SimpleNamespace(
            sim=_Sim(),
            objects_dict={
                "akita_black_bowl_1": _Obj("bowl_j", 1.12),
                "plate_1": _Obj("plate_j", 0.93),
            },
        ),
        place_target="plate_1",
        place_target_name="plate_1",
        target="akita_black_bowl_1",
    )
    z = table_support_z(driver, "akita_black_bowl_1")
    # plate joint 0.93 + bottom_offset -0.03
    assert abs(z - 0.90) < 1e-6


def test_standoff_includes_the_bumper_nose():
    from benchmarks.dynamic_libero.env.pusher import bumper_half_for, nose_extent

    car = PusherCar()
    nose = nose_extent(car.cfg.half_length)
    assert abs(nose - (car.cfg.half_length + 2.0 * bumper_half_for(car.cfg.half_length))) < 1e-9
    assert abs(car.standoff - (nose + car.obj_radius + car.gap)) < 1e-9
    assert car.standoff > car.cfg.half_length + car.obj_radius
    root = ET.fromstring(inject_pusher_xml("<mujoco><worldbody/></mujoco>", car.cfg))
    bumper = next(g for g in root.iter("geom") if g.get("name") == f"{PUSHER_BODY}_bumper")
    pos = [float(v) for v in bumper.get("pos").split()]
    half = [float(v) for v in bumper.get("size").split()]
    assert abs((pos[0] + half[0]) - nose) < 1e-6

