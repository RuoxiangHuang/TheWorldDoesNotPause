"""CPU unit tests (no SAPIEN / GPU required)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.dynamic_libero.trajectories import build_trajectory
from benchmarks.dynamic_robotwin.env.task_registry import (
    BIMANUAL_TASKS,
    HANDOFF_TASKS,
    TARGET_ATTR,
    V1_TASKS,
    default_instruction,
    get_task_meta,
    parse_dynamic_modes,
    resolve_target_attr,
)


def test_whitelist_complete():
    assert len(V1_TASKS) >= 8
    assert "handover_block" in V1_TASKS
    assert "handover_mic" in V1_TASKS
    for t in V1_TASKS:
        assert resolve_target_attr(t) == TARGET_ATTR[t]
        assert isinstance(default_instruction(t), str)
        assert len(default_instruction(t)) > 0


def test_place_attr_cup():
    from benchmarks.dynamic_robotwin.env.task_registry import PLACE_ATTR, resolve_place_attr

    assert PLACE_ATTR["place_empty_cup"] == "coaster"
    assert resolve_place_attr("place_empty_cup") == "coaster"
    assert resolve_place_attr("grab_roller") is None


def test_ensure_play_once_eval_attrs_sets_arm_tag():
    from benchmarks.dynamic_robotwin.env.robotwin_bridge import ensure_play_once_eval_attrs

    class _Pose:
        def __init__(self, x):
            self.p = np.array([x, 0.0, 0.74], dtype=np.float64)

    class _Actor:
        def __init__(self, x):
            self._x = x

        def get_pose(self):
            return _Pose(self._x)

    class _Env:
        pass

    left = _Env()
    left.object = _Actor(-0.12)
    ensure_play_once_eval_attrs(left)
    assert str(left.arm_tag) == "left"

    right = _Env()
    right.object = _Actor(0.18)
    ensure_play_once_eval_attrs(right)
    assert str(right.arm_tag) == "right"

    kept = _Env()
    kept.arm_tag = "left"
    kept.object = _Actor(0.18)
    ensure_play_once_eval_attrs(kept)
    assert kept.arm_tag == "left"


def test_cup_pusher_catalog():
    from benchmarks.dynamic_robotwin.env.dynamic_tasks import get_dynamic_task, recon_tasks

    ids = [t.id for t in recon_tasks() if t.task_name == "place_empty_cup"]
    assert ids == [
        "place_empty_cup.pick",
        "place_empty_cup.place",
        "place_empty_cup.both",
        "place_empty_cup.pick.irregular",
        "place_empty_cup.place.irregular",
    ]
    irr = get_dynamic_task("place_empty_cup.pick.irregular")
    kind, kw = irr.traj_spec()
    assert kind == "curve" and kw["curve_id"] == "s_wave"
    both = get_dynamic_task("place_empty_cup.both")
    assert both.variant == "both" and both.place_attr == "coaster"
    from benchmarks.dynamic_robotwin.env.dynamic_tasks import resolve_eval_spec

    ghost = resolve_eval_spec("place_empty_cup")
    assert ghost.spawn_pushers is False and ghost.rails_variant == "pick"
    car = resolve_eval_spec("place_empty_cup.pick")
    assert car.spawn_pushers is True and car.task_name == "place_empty_cup"
    assert car.heading_deg == pytest.approx(0.0)
    kind, kw = get_dynamic_task("place_empty_cup.pick").traj_spec()
    assert kind == "linear" and kw["heading_deg"] == pytest.approx(0.0)
    assert kw["toward_center"] is True


def test_parse_pi_speedups_off_cache():
    from benchmarks.dynamic_robotwin.eval.cli_common import parse_pi_speedups

    assert parse_pi_speedups("off=eager,cache=cache") == [
        ("off", "eager"),
        ("cache", "cache"),
    ]
    assert parse_pi_speedups("eager") == [("off", "eager")]
    assert parse_pi_speedups("cache") == [("cache", "cache")]


def test_latency_slice_is_five_identity_skills():
    from benchmarks.dynamic_robotwin.env.dynamic_tasks import (
        LATENCY_SLICE_SKILLS,
        latency_slice_tasks,
    )
    from benchmarks.dynamic_robotwin.env.official import parse_task_names

    picks = latency_slice_tasks(variant="pick")
    places = latency_slice_tasks(variant="place")
    assert [t.task_name for t in picks] == list(LATENCY_SLICE_SKILLS)
    assert [t.task_name for t in places] == list(LATENCY_SLICE_SKILLS)
    assert all(t.variant == "pick" and t.trajectory_kind == "linear" for t in picks)
    assert all(t.variant == "place" and t.trajectory_kind == "linear" for t in places)
    assert parse_task_names("latency") == [t.id for t in picks]
    assert parse_task_names("pick_slice") == [t.id for t in picks]
    assert parse_task_names("place_slice") == [t.id for t in places]


def test_main_track_is_linear_identity_not_curves():
    from benchmarks.dynamic_robotwin.env.dynamic_tasks import (
        extension_tasks,
        main_tasks,
        recon_tasks,
    )
    from benchmarks.dynamic_robotwin.env.official import parse_task_names

    main = main_tasks()
    ext = extension_tasks()
    rec = recon_tasks()
    assert len(main) + len(ext) == len(rec)
    assert all(t.trajectory_kind == "linear" for t in main)
    assert all(t.eval_split == "main" for t in main)
    assert all(t.language_policy == "keep" for t in main)
    assert all(t.trajectory_kind == "curve" for t in ext)
    assert all(t.eval_split == "extension" for t in ext)
    assert parse_task_names("main") == [t.id for t in main]
    assert parse_task_names("extension") == [t.id for t in ext]
    # Not filtered by accelerator SR; every linear reconstruction is eligible.
    assert len(main) == 15 * 3


def test_catalog_eval_detects_reconstruction_ids():
    from benchmarks.dynamic_robotwin.env.dynamic_tasks import is_catalog_eval

    assert is_catalog_eval("place_empty_cup.pick") is True
    assert is_catalog_eval("place_empty_cup") is False


def test_recon_catalog_is_one_batch_with_pick_place_both_and_irregular():
    from benchmarks.dynamic_robotwin.env.dynamic_tasks import (
        RECON_SKILLS,
        get_dynamic_task,
        recon_demo_tasks,
        recon_tasks,
        resolve_eval_spec,
        wave2_tasks,
    )
    from benchmarks.dynamic_robotwin.env.task_registry import (
        RECON_TARGET_ATTR,
        resolve_place_attr,
        resolve_target_attr,
    )

    tasks = recon_tasks()
    assert wave2_tasks() == tasks
    assert len(RECON_SKILLS) == 15
    assert len(tasks) == 15 * 5
    ids = {t.id for t in tasks}
    assert "place_phone_stand.both" in ids
    assert "place_container_plate.pick.irregular" in ids
    assert "place_a2b_right.place.irregular" in ids
    assert "place_empty_cup.place.irregular" in ids
    assert "move_can_pot.both" in ids
    assert "place_mouse_pad.pick.irregular" in ids
    assert "place_fan.place.irregular" in ids
    irr = get_dynamic_task("place_object_basket.place.irregular")
    kind, kw = irr.traj_spec()
    assert irr.variant == "place" and kind == "curve" and kw["curve_id"] == "s_wave"
    both = get_dynamic_task("move_stapler_pad.both")
    assert both.pick_attr == "stapler" and both.place_attr == "pad"
    can = get_dynamic_task("move_can_pot.place")
    assert can.pick_attr == "can" and can.place_attr == "pot"
    mouse = get_dynamic_task("place_mouse_pad.both")
    assert mouse.pick_attr == "mouse" and mouse.place_attr == "target"
    fan = get_dynamic_task("place_fan.pick")
    assert fan.pick_attr == "fan" and fan.place_attr == "pad"
    linear = [t for t in tasks if t.trajectory_kind == "linear"]
    heads = sorted({round(float(t.heading_deg), 1) for t in linear})
    assert heads == [i * 24.0 for i in range(15)]
    demo = recon_demo_tasks()
    assert len(demo) == 15 * 3
    assert resolve_eval_spec("place_bread_skillet.pick").spawn_pushers is True
    ghost = resolve_eval_spec("place_container_plate")
    assert ghost.spawn_pushers is False and ghost.rails_variant == "pick"
    assert resolve_target_attr("place_can_basket") == RECON_TARGET_ATTR["place_can_basket"]
    assert resolve_place_attr("place_object_stand") == "displaystand"
    assert resolve_place_attr("place_object_scale") == "scale"
    assert resolve_target_attr("move_can_pot") == "can"
    assert resolve_place_attr("place_mouse_pad") == "target"
    assert resolve_place_attr("place_fan") == "pad"


def test_dotted_catalog_id_maps_to_bare_env_name():
    from benchmarks.dynamic_robotwin.env.dynamic_tasks import resolve_eval_spec
    from benchmarks.dynamic_robotwin.env.robotwin_bridge import resolve_robotwin_env_name

    assert resolve_eval_spec("place_phone_stand.both").task_name == "place_phone_stand"
    assert resolve_robotwin_env_name("place_phone_stand.both") == "place_phone_stand"
    assert resolve_robotwin_env_name("place_empty_cup") == "place_empty_cup"


def test_extra_catalog_not_in_recon():
    from benchmarks.dynamic_robotwin.env.dynamic_tasks import recon_tasks, resolve_eval_spec
    from benchmarks.dynamic_robotwin.env.extra_tasks import extra_tasks
    from benchmarks.dynamic_robotwin.env.task_registry import resolve_target_attr

    extra_ids = {t.id for t in extra_tasks()}
    recon_ids = {t.id for t in recon_tasks()}
    assert extra_ids.isdisjoint(recon_ids)
    spec = resolve_eval_spec("catch_shuttlecock")
    assert spec.trajectory_kind == "native"
    assert spec.spawn_pushers is False
    assert spec.language.startswith("Catch the flying")
    assert resolve_target_attr("catch_shuttlecock") == "object"


def test_named_radius_caps_known_meshes_not_unknown_actors():
    from benchmarks.dynamic_robotwin.env.pusher import object_xy_radius, pusher_object_radius

    assert object_xy_radius("container") == pytest.approx(0.045)
    assert object_xy_radius("can") == pytest.approx(0.032)
    assert object_xy_radius("mouse") == pytest.approx(0.040)
    assert object_xy_radius("fan") == pytest.approx(0.050)
    assert object_xy_radius("pot") == pytest.approx(0.090)
    assert object_xy_radius("target") == pytest.approx(0.065)
    assert object_xy_radius("object", default=None) is None
    assert object_xy_radius("target_object", default=None) is None

    class _Wrap:
        config = {
            "extents": [0.9875920171285272, 1.0172059744235127, 0.981945382705398],
            "scale": [0.088, 0.088, 0.088],
        }

    aabb = pusher_object_radius(_Wrap(), "object")
    named_cup = pusher_object_radius(_Wrap(), "cup")
    # Placement uses the larger footprint. The named 3.7 cm disc sits inside
    # this AABB, and taking the min used to drive the bumper into the mesh.
    assert named_cup == pytest.approx(aabb)
    assert named_cup > 0.037
    assert object_xy_radius("cup") == pytest.approx(0.037)

    class _BoxPad:
        config = {"extents": [0.01, 0.01, 0.01], "scale": [1.0, 1.0, 1.0]}

    pad_r = pusher_object_radius(_BoxPad(), "pad")
    assert pad_r == pytest.approx(0.055, abs=1e-6)
    mouse_pad_r = pusher_object_radius(_BoxPad(), "target")
    assert mouse_pad_r == pytest.approx(0.065, abs=1e-6)
    target_r = pusher_object_radius(_BoxPad(), "target")
    assert target_r == pytest.approx(0.065, abs=1e-6)


def test_skill_fit_covers_robotwin_and_ranks_cup_siblings():
    from benchmarks.dynamic_robotwin.env.dynamic_tasks import list_skill_fits

    all_fits = list_skill_fits()
    by_name = {s.task_name: s for s in all_fits}
    assert by_name["place_empty_cup"].status == "ready"
    ready = {s.task_name for s in list_skill_fits(status="ready")}
    assert "place_phone_stand" in ready
    assert "place_container_plate" in ready
    assert "place_object_basket" in ready
    skip = {s.task_name for s in list_skill_fits(status="skip")}
    assert "handover_block" in skip
    assert "open_microwave" in skip
    assert len(all_fits) >= 48
    assert len(all_fits) == len(by_name)  # unique names


def test_handoff_and_bimanual_metadata():
    assert set(HANDOFF_TASKS) == {"handover_block", "handover_mic"}
    assert "grab_roller" in BIMANUAL_TASKS
    for t in HANDOFF_TASKS:
        meta = get_task_meta(t)
        assert meta["bimanual"] is True
        assert meta["stress"] == "handoff"
        assert "handoff" in meta["dynamic_modes"]
    hb = get_task_meta("handover_block")
    assert hb["secondary_attr"] == "target_box"
    assert resolve_target_attr("handover_block") == "box"
    assert resolve_target_attr("handover_mic") == "microphone"


def test_parse_dynamic_modes():
    assert parse_dynamic_modes("none") == []
    assert parse_dynamic_modes("off") == []
    assert "contact_trigger" in parse_dynamic_modes(None)
    assert "occlusion" in parse_dynamic_modes("contact_trigger,occlusion")
    assert "handoff" in parse_dynamic_modes("all")
    with pytest.raises(ValueError):
        parse_dynamic_modes("not_a_mode")


@pytest.mark.parametrize("kind", ["linear", "sine", "circle"])
def test_shared_trajectories(kind: str):
    kwargs = {}
    if kind == "circle":
        kwargs = {"radius": 0.05, "omega": 0.02}
    traj = build_trajectory(kind, speed=0.003, **kwargs)
    traj.reset(np.zeros(3))
    for _ in range(20):
        traj.step(1.0)
    assert traj.t == 20.0


def test_robotwin_fractional_freeze_import():
    from benchmarks.dynamic_robotwin.eval.measure_latency import (
        freeze_ticks_from_latency,
        measure_latency_varying,
    )

    assert freeze_ticks_from_latency(0.021, 20.0) == pytest.approx(0.42)
    with pytest.raises(ValueError, match="must be > warmup"):
        measure_latency_varying("place_empty_cup", None, k=3, warmup=3)


def test_task_traj_defaults_and_toward_center():
    from benchmarks.dynamic_robotwin.env.task_registry import (
        default_release_radius,
        resolve_traj_kwargs,
    )

    assert default_release_radius("place_empty_cup") == pytest.approx(0.12)
    kw = resolve_traj_kwargs("place_empty_cup", {})
    assert kw["axis"] == "y"
    assert kw["toward_center"] is True
    # Explicit CLI overrides win and disable toward_center when direction set.
    kw2 = resolve_traj_kwargs(
        "place_empty_cup", {"axis": "x", "direction": 1.0, "toward_center": False}
    )
    assert kw2["axis"] == "x"
    assert kw2["toward_center"] is False

    traj = build_trajectory("linear", speed=0.001, axis="y", toward_center=True)
    traj.reset(np.array([0.2, 0.1, 0.7]))
    assert traj.direction == -1.0  # y>0 → drift toward center
    traj.reset(np.array([0.2, -0.1, 0.7]))
    assert traj.direction == 1.0


def test_contact_trigger_trajectory_switch():
    from benchmarks.dynamic_robotwin.env.dynamics import (
        apply_trajectory_switch,
        build_dynamic_config,
        resolve_contact_intensity,
    )

    mild = resolve_contact_intensity("mild")
    assert mild["reverse"] is True
    strong = resolve_contact_intensity("strong")
    assert strong["redirect"] is True
    assert strong["speed_scale"] == pytest.approx(2.0)

    traj = build_trajectory("linear", speed=0.002, axis="y", direction=1.0)
    traj.reset(np.array([0.0, 0.0, 0.8]))
    traj.step(5.0)
    before_dir = traj.direction
    before_speed = traj.speed
    mode = apply_trajectory_switch(traj, "medium", switch_mode="auto")
    assert mode == "accelerate"
    assert traj.direction == -before_dir
    assert traj.speed == pytest.approx(before_speed * 1.5)
    assert traj.t == pytest.approx(0.0)  # rebased

    cfg = build_dynamic_config(
        ["contact_trigger", "occlusion"],
        contact_switch="mild",
        occlusion=True,
        contact_chain=False,
    )
    assert cfg.wants("contact_trigger")
    assert cfg.wants("occlusion")
    assert not cfg.wants("contact_chain")
    assert cfg.contact_trigger.level == "mild"
    assert cfg.occlusion.enabled is True

    cfg_off = build_dynamic_config(["contact_trigger"], contact_switch="off")
    assert not cfg_off.wants("contact_trigger")
    assert cfg_off.contact_trigger.level == "off"


def test_occlusion_duty_cycle():
    from benchmarks.dynamic_robotwin.env.dynamics import occlusion_is_active

    assert occlusion_is_active(0.0, period_ticks=40.0, duty=0.35) is True
    assert occlusion_is_active(20.0, period_ticks=40.0, duty=0.35) is False
    assert occlusion_is_active(0.0, period_ticks=40.0, duty=0.0) is False
    assert occlusion_is_active(10.0, period_ticks=40.0, duty=1.0) is True


def test_contact_chain_multi_hop_without_sapien():
    from benchmarks.dynamic_robotwin.env.dynamics import (
        ContactChainConfig,
        ContactChainManager,
    )

    class _Env:
        @property
        def scene(self):
            raise RuntimeError("no sapien in unit test")

    cfg = ContactChainConfig(
        enabled=True, n_aux=3, impulse=0.4, spacing=0.08, hop_delay_ticks=2.0
    )
    mgr = ContactChainManager(cfg)
    mgr.setup(_Env(), np.array([0.0, 0.0, 0.8]))
    assert len(mgr.positions) == 3
    mgr.on_primary_contact(t_ticks=0.0)
    assert mgr.primary_fired
    assert mgr.n_impulses == 1
    assert mgr.hop_events[0]["source"] == "primary"
    # Proximity and/or delay fills the remaining hops.
    mgr.tick(2.0)
    mgr.tick(4.0)
    assert mgr.n_impulses == 3
    assert int(mgr.stats()["contact_chain_hops"]) == 3
    sources = [e["source"] for e in mgr.hop_events]
    assert sources[0] == "primary"
    assert set(sources[1:]).issubset({"proximity", "delay"})
    assert len(sources[1:]) == 2


def test_resolve_secondary_rails_auto():
    from benchmarks.dynamic_robotwin.env.dynamics import resolve_secondary_rails

    assert resolve_secondary_rails("handover_block", "auto") is True
    assert resolve_secondary_rails("handover_block", None) is True
    assert resolve_secondary_rails("handover_block", "off") is False
    assert resolve_secondary_rails("handover_mic", "auto") is False  # no secondary_attr
    assert resolve_secondary_rails("place_empty_cup", "auto") is False
    assert resolve_secondary_rails("place_empty_cup", "on") is True


def test_yaml_dynamics_defaults_align_with_cli_helper():
    from benchmarks.dynamic_robotwin.env.eval_config import dynamics_defaults_from_yaml

    d = dynamics_defaults_from_yaml()
    assert "contact_trigger" in d["modes"]
    assert d["contact_switch"] == "medium"
    assert d["secondary_rails"] == "auto"
    assert d["contact_chain"] == "on"
    assert int(d["contact_chain_n_aux"]) == 3


def test_driver_dynamics_stats_shape_without_sapien():
    """Driver episode_dynamics_stats works on a lightweight stub env."""
    from benchmarks.dynamic_robotwin.env.driver import RealtimeRoboTwinDriver
    from benchmarks.dynamic_robotwin.env.dynamics import build_dynamic_config

    class _Pose:
        def __init__(self, p):
            self.p = np.asarray(p, dtype=np.float64)
            self.q = np.array([1.0, 0.0, 0.0, 0.0])

    class _Entity:
        def __init__(self):
            self._p = np.array([0.0, 0.0, 0.8])

        def get_pose(self):
            return _Pose(self._p)

        def set_pose(self, pose):
            self._p = np.asarray(pose.p, dtype=np.float64)

        def find_component_by_type(self, *_a, **_k):
            return None

    class _Robot:
        def get_left_tcp_pose(self):
            return np.array([10.0, 10.0, 10.0, 0, 0, 0, 1])

        def get_right_tcp_pose(self):
            return np.array([10.0, 10.0, 10.0, 0, 0, 0, 1])

        def is_left_gripper_close(self):
            return False

        def is_right_gripper_close(self):
            return False

        def get_left_arm_jointState(self):
            return np.zeros(7)

        def get_right_arm_jointState(self):
            return np.zeros(7)

        def set_arm_joints(self, *a, **k):
            return None

        def set_gripper(self, *a, **k):
            return None

    class _Scene:
        def step(self):
            return None

        def create_actor_builder(self):
            raise RuntimeError("no sapien in unit test")

    class _Env:
        def __init__(self):
            self.cup = _Entity()
            self.robot = _Robot()
            self.scene = _Scene()

        def take_action(self, action, action_type="qpos"):
            return None

        def _update_render(self):
            return None

    traj = build_trajectory("linear", speed=0.003, axis="y", direction=1.0)
    dyn = build_dynamic_config(
        ["contact_trigger"],
        contact_switch="medium",
        occlusion=False,
        contact_chain=False,
    )
    env = _Env()
    drv = RealtimeRoboTwinDriver(
        env, traj, task_name="place_empty_cup", dynamic=dyn, physics_per_tick=1
    )
    # Skip occlusion/chain/sapien pin paths in this CPU stub.
    drv.target_wrapper, drv.entity = env.cup, env.cup
    drv.base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    traj.reset(base_pos=np.array([0.0, 0.0, 0.8]))
    drv.rails_active = True
    drv.motion_enabled = True
    drv.rails_drive = "pose_fallback"
    drv.ghost_sliding = True
    drv._pin = lambda: None  # type: ignore[method-assign]
    drv._chain.on_primary_contact = lambda: None  # type: ignore[method-assign]
    # Simulate TCP approaching the cup.
    env.robot.get_left_tcp_pose = lambda: np.array([0.0, 0.05, 0.8, 0, 0, 0, 1])
    drv._maybe_contact_trigger()
    stats = drv.episode_dynamics_stats()
    assert len(stats["trajectory_switches"]) == 1
    assert len(stats["contact_events"]) == 1
    assert stats["contact_events"][0]["arm"] == "left"
    assert stats["contact_switch_level"] == "medium"
    assert stats["rails_drive"] == "pose_fallback"
    assert stats["ghost_sliding"] is True


def test_rails_drive_mode_selection_and_velocity_path():
    """Choose kinematic/velocity/pose_fallback and exercise velocity drive."""
    sapien = pytest.importorskip("sapien")
    from benchmarks.dynamic_robotwin.env.driver import RealtimeRoboTwinDriver
    from benchmarks.dynamic_robotwin.env.dynamics import build_dynamic_config

    class _Pose:
        def __init__(self, p, q=None):
            self.p = np.asarray(p, dtype=np.float64)
            self.q = np.asarray(
                q if q is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

    class _Shape:
        def __init__(self):
            self.groups = [1, 1, 0, 0]

        def get_collision_groups(self):
            return list(self.groups)

        def set_collision_groups(self, groups):
            self.groups = list(groups)

    class _Comp:
        def __init__(self, *, kinematic_api=True, shapes=None):
            self.kinematic = False
            self.lin_v = np.zeros(3)
            self.ang_v = np.zeros(3)
            self.kinematic_target = None
            self._shapes = shapes or []
            self._kinematic_api = kinematic_api

        def get_collision_shapes(self):
            return list(self._shapes)

        def set_linear_velocity(self, v):
            self.lin_v = np.asarray(v, dtype=np.float64)

        def set_angular_velocity(self, v):
            self.ang_v = np.asarray(v, dtype=np.float64)

        def get_kinematic(self):
            return self.kinematic

        def set_kinematic(self, flag):
            if not self._kinematic_api:
                raise RuntimeError("no kinematic")
            self.kinematic = bool(flag)

        def set_kinematic_target(self, pose):
            if not self._kinematic_api:
                raise RuntimeError("no kinematic")
            self.kinematic_target = pose

    class _Entity:
        def __init__(self, comp):
            self._p = np.array([0.0, 0.0, 0.8])
            self._comp = comp

        def get_pose(self):
            return _Pose(self._p)

        def set_pose(self, pose):
            self._p = np.asarray(pose.p, dtype=np.float64)

        def find_component_by_type(self, typ):
            name = getattr(typ, "__name__", str(typ))
            if "RigidDynamic" in name or "RigidBody" in name:
                return self._comp
            return None

    class _Robot:
        def get_left_tcp_pose(self):
            return np.array([10.0, 10.0, 10.0, 0, 0, 0, 1])

        def get_right_tcp_pose(self):
            return np.array([10.0, 10.0, 10.0, 0, 0, 0, 1])

        def is_left_gripper_close(self):
            return False

        def is_right_gripper_close(self):
            return False

        def get_left_arm_jointState(self):
            return np.zeros(7)

        def get_right_arm_jointState(self):
            return np.zeros(7)

        def set_arm_joints(self, *a, **k):
            return None

        def set_gripper(self, *a, **k):
            return None

    class _Scene:
        def step(self):
            return None

    class _Env:
        def __init__(self, cup):
            self.cup = cup
            self.robot = _Robot()
            self.scene = _Scene()

        def take_action(self, action, action_type="qpos"):
            return None

        def _update_render(self):
            return None

    dyn = build_dynamic_config(
        ["contact_trigger"],
        contact_switch="off",
        occlusion=False,
        contact_chain=False,
    )

    # pose_fallback when no rigid dynamic component
    ent0 = _Entity(None)
    # Monkeypatch find to always None
    ent0.find_component_by_type = lambda *_a, **_k: None  # type: ignore[method-assign]
    env0 = _Env(ent0)
    traj0 = build_trajectory("linear", speed=0.003, axis="y", direction=1.0)
    drv0 = RealtimeRoboTwinDriver(
        env0, traj0, task_name="place_empty_cup", dynamic=dyn, physics_per_tick=2
    )
    assert drv0._choose_rails_drive(ent0) == "pose_fallback"

    # kinematic preferred when API present
    shape = _Shape()
    kcomp = _Comp(kinematic_api=True, shapes=[shape])
    kent = _Entity(kcomp)
    assert drv0._choose_rails_drive(kent) == "kinematic"

    # velocity when kinematic API missing but linear velocity exists
    class _VelOnly:
        def __init__(self):
            self.lin_v = np.zeros(3)
            self.ang_v = np.zeros(3)

        def get_collision_shapes(self):
            return []

        def set_linear_velocity(self, v):
            self.lin_v = np.asarray(v, dtype=np.float64)

        def set_angular_velocity(self, v):
            self.ang_v = np.asarray(v, dtype=np.float64)

    vent = _Entity(_VelOnly())
    assert drv0._choose_rails_drive(vent) == "velocity"

    # Full velocity drive + ghost filter + release
    class _VelComp:
        def __init__(self, shapes):
            self.lin_v = np.zeros(3)
            self.ang_v = np.zeros(3)
            self._shapes = shapes

        def get_collision_shapes(self):
            return list(self._shapes)

        def set_linear_velocity(self, v):
            self.lin_v = np.asarray(v, dtype=np.float64)

        def set_angular_velocity(self, v):
            self.ang_v = np.asarray(v, dtype=np.float64)

    vshape = _Shape()
    vcomp = _VelComp([vshape])
    vent2 = _Entity(vcomp)
    assert drv0._choose_rails_drive(vent2) == "velocity"
    env = _Env(vent2)
    traj = build_trajectory("linear", speed=0.003, axis="y", direction=1.0)
    drv = RealtimeRoboTwinDriver(
        env, traj, task_name="place_empty_cup", dynamic=dyn, physics_per_tick=2, policy_hz=20.0
    )
    drv.target_wrapper, drv.entity = vent2, vent2
    drv.base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    traj.reset(base_pos=np.array([0.0, 0.0, 0.8]))
    drv.rails_active = True
    drv.rails_drive = "velocity"
    drv.ghost_sliding = True
    assert drv._enable_ghost_collision_filter(vent2) is True
    assert vshape.groups[0] == 0 and vshape.groups[1] == 0
    drv.ghost_collision_filter = True
    traj.step(1.0)
    drv._pin()
    # Desired y increased by +0.003; dt_phys = 1/(20*2)=0.025 → v_y ≈ 0.12
    assert vcomp.lin_v[1] == pytest.approx(0.003 / (1.0 / 40.0), rel=1e-6)
    drv._release_rails()
    assert drv.rails_active is False
    assert drv.released is True
    assert vshape.groups[0] == 1 and vshape.groups[1] == 1
    stats = drv.episode_dynamics_stats()
    assert stats["rails_drive"] == "velocity"
    assert stats["ghost_sliding"] is True
    assert stats["ghost_collision_filter"] is True

    # Kinematic target path
    kshape = _Shape()
    kcomp2 = _Comp(kinematic_api=True, shapes=[kshape])
    kent2 = _Entity(kcomp2)
    envk = _Env(kent2)
    trajk = build_trajectory("linear", speed=0.003, axis="y", direction=1.0)
    drvk = RealtimeRoboTwinDriver(
        envk, trajk, task_name="place_empty_cup", dynamic=dyn, physics_per_tick=1
    )
    drvk.target_wrapper, drvk.entity = kent2, kent2
    drvk.base_quat = np.array([1.0, 0.0, 0.0, 0.0])
    trajk.reset(base_pos=np.array([0.0, 0.0, 0.8]))
    drvk.rails_active = True
    drvk.rails_drive = "kinematic"
    drvk.ghost_sliding = True
    trajk.step(1.0)
    drvk._pin()
    assert kcomp2.kinematic is True
    assert kcomp2.kinematic_target is not None
    xyz_des, _ = trajk.pose_at()
    assert np.allclose(np.asarray(kcomp2.kinematic_target.p), xyz_des)
    drvk._release_rails()
    assert kcomp2.kinematic is False
    assert sapien.physx.PhysxRigidDynamicComponent is not None


def test_orient_curve_inward_flips_away_from_origin():
    from benchmarks.dynamic_libero.trajectories import build_trajectory
    from benchmarks.dynamic_robotwin.env.pusher import orient_curve_inward

    traj = build_trajectory("curve", speed=0.003, curve_id="s_wave")
    traj.reset(np.array([0.22, -0.05, 0.78]))
    end_before = np.asarray(traj._pts[-1, :2], dtype=np.float64).copy()
    assert float(end_before[0]) > 0.0  # authored +X
    orient_curve_inward(traj)
    end_after = np.asarray(traj._pts[-1, :2], dtype=np.float64)
    # Spawn is +X, so the wave should now travel toward -X (table center).
    assert float(end_after[0]) < 0.0
    xyz = traj.xyz_at(40.0)
    assert float(xyz[0]) < float(traj.base_pos[0])


def test_cup_place_rails_tick_without_pushers():
    """place-only rails move the coaster and leave the cup still (CPU stub)."""
    sapien = pytest.importorskip("sapien")
    from benchmarks.dynamic_robotwin.env.driver import RealtimeRoboTwinDriver
    from benchmarks.dynamic_robotwin.env.dynamics import build_dynamic_config

    class _Pose:
        def __init__(self, p, q=None):
            self.p = np.asarray(p, dtype=np.float64)
            self.q = np.asarray(
                q if q is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

    class _Entity:
        def __init__(self, p):
            self._p = np.asarray(p, dtype=np.float64)

        def get_pose(self):
            return _Pose(self._p)

        def set_pose(self, pose):
            self._p = np.asarray(pose.p, dtype=np.float64)

        def find_component_by_type(self, *_a, **_k):
            return None

    class _Robot:
        def get_left_tcp_pose(self):
            return np.array([10.0, 10.0, 10.0, 0, 0, 0, 1])

        def get_right_tcp_pose(self):
            return np.array([10.0, 10.0, 10.0, 0, 0, 0, 1])

        def is_left_gripper_close(self):
            return False

        def is_right_gripper_close(self):
            return False

        def get_left_arm_jointState(self):
            return np.zeros(7)

        def get_right_arm_jointState(self):
            return np.zeros(7)

        def set_arm_joints(self, *a, **k):
            return None

        def set_gripper(self, *a, **k):
            return None

    class _Scene:
        def step(self):
            return None

    class _Env:
        def __init__(self):
            self.cup = _Entity([0.22, -0.08, 0.78])
            self.coaster = _Entity([0.02, 0.04, 0.74])
            self.robot = _Robot()
            self.scene = _Scene()

        def take_action(self, action, action_type="qpos"):
            return None

        def _update_render(self):
            return None

    env = _Env()
    cup0 = env.cup._p.copy()
    coaster0 = env.coaster._p.copy()
    traj = build_trajectory("linear", speed=0.003, axis="y", toward_center=True)
    drv = RealtimeRoboTwinDriver(
        env,
        traj,
        task_name="place_empty_cup",
        dynamic=build_dynamic_config("none"),
        physics_per_tick=1,
        rails_variant="place",
        spawn_pushers=False,
    )
    drv.on_episode_start()
    assert drv.pick_rails_active is False
    assert drv.place_rails_active is True
    drv.begin_policy()
    for _ in range(10):
        drv.tick(1.0)
    assert np.allclose(env.cup._p, cup0)
    assert not np.allclose(env.coaster._p, coaster0)
    assert abs(env.coaster._p[1] - coaster0[1]) == pytest.approx(0.03, rel=1e-6)
    assert sapien.Pose is not None


def test_pusher_standoff_clears_cup_aabb():
    from benchmarks.dynamic_robotwin.env.pusher import (
        CONTACT_GAP,
        FRONT_EXTENT,
        HALF_LENGTH,
        RoboTwinPusherCar,
        measure_actor_xy_radius,
    )

    class _Wrap:
        config = {
            "extents": [0.9875920171285272, 1.0172059744235127, 0.981945382705398],
            "scale": [0.088, 0.088, 0.088],
        }

    radius = measure_actor_xy_radius(_Wrap(), fallback=0.032)
    # No live pose → largest local half-extent. Placement then takes the
    # larger of this AABB and any named disc.
    raw = 1.0172059744235127 * 0.088 / 2.0
    assert radius == pytest.approx(raw, rel=1e-5)
    car = RoboTwinPusherCar(entity=None, palette="red", target_name="cup")
    car.obj_radius = radius
    assert car.standoff == pytest.approx(FRONT_EXTENT + radius + CONTACT_GAP)
    assert FRONT_EXTENT > HALF_LENGTH
    assert car.standoff > HALF_LENGTH + 0.032 + 0.004


def test_bumper_car_tires_touch_the_table_and_clear_corners():
    from benchmarks.dynamic_robotwin.env.pusher import (
        CONTACT_GAP,
        FRONT_EXTENT,
        PlanarFootprint,
        RoboTwinPusherCar,
        car_origin_z,
        wheel_bottom_local_z,
    )

    table = 0.742
    assert car_origin_z(table) + wheel_bottom_local_z() == pytest.approx(table)
    # Chassis bottom is one wheel radius above the tire contact.
    assert car_origin_z(table) - 0.016 > table

    car = RoboTwinPusherCar(entity=None, target_name="cup")
    car.obj_radius = 0.04
    car.footprint = PlanarFootprint(
        center_xy=[0.0, 0.0],
        half_xy=[0.04, 0.04],
    )
    axis = car.clearance([1.0, 0.0])
    diagonal = car.clearance([1.0, 1.0])
    assert axis == pytest.approx(FRONT_EXTENT + 0.04 + CONTACT_GAP)
    assert diagonal > axis


def test_measure_radius_is_xy_face_not_box_diagonal():
    """Spawn quat [0.5]*4 maps asset Y→world Z; planar radius is AABB half-width."""
    from benchmarks.dynamic_robotwin.env.pusher import measure_actor_xy_radius
    import numpy as np

    class _Pose:
        q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

        def to_transformation_matrix(self):
            raise RuntimeError("use quaternion path")

    class _Ent:
        def get_pose(self):
            return _Pose()

    class _Wrap:
        config = {
            "center": [0.00011744983637756174, 0.5084882382486132, -8.444826235722103e-05],
            "extents": [0.9875920171285272, 1.0172059744235127, 0.981945382705398],
            "scale": [0.088, 0.088, 0.088],
        }

    radius = measure_actor_xy_radius(_Wrap(), fallback=0.032, entity=_Ent())
    half_x = 0.9875920171285272 * 0.088 / 2.0
    half_z = 0.981945382705398 * 0.088 / 2.0
    face = max(half_x, half_z)
    hypot = float(np.hypot(half_x, half_z))
    assert radius == pytest.approx(face, abs=2e-5)
    assert radius < hypot  # must not use the AABB corner
    from benchmarks.dynamic_robotwin.env.pusher import pusher_object_radius

    placed = pusher_object_radius(_Wrap(), "cup", entity=_Ent())
    assert placed == pytest.approx(radius)
    assert placed > 0.037

