"""Tests for Dynamic-LIBERO v2 toy-car mover swap."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.dynamic_libero.env.movers import (
    MOVERS,
    all_mover_specs,
    get_mover_spec,
    mover_assets,
    mover_xml_path,
    rewrite_language,
)
from benchmarks.dynamic_libero.env.suite_registry import TRAINED_SUITES, suite_n_tasks
from benchmarks.dynamic_libero.tools.gen_movers import MOVER_CATEGORIES


LIBERO_ROOT = Path(os.environ.get("LIBERO_ROOT", "/DATA/YuanZhen/LIBERO"))


def _bddl_path(suite: str, task_id: int) -> Path:
    from libero.libero import benchmark

    bm = benchmark.get_benchmark_dict()[suite]()
    return Path(bm.get_task_bddl_file_path(task_id))


def _parse_bddl_objects(bddl_path: Path) -> set[str]:
    text = bddl_path.read_text()
    instances: set[str] = set()
    in_objects = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("(:objects"):
            in_objects = True
            continue
        if in_objects:
            if stripped.startswith("(:obj_of_interest") or stripped.startswith("(:init") or stripped.startswith("(:goal"):
                break
            if " - " in stripped:
                left, _right = stripped.split(" - ", 1)
                for inst in left.split():
                    if inst:
                        instances.add(inst)
    return instances


@pytest.mark.parametrize("suite", TRAINED_SUITES)
def test_movers_cover_all_tasks(suite: str):
    n = suite_n_tasks(suite)
    assert set(MOVERS[suite].keys()) == set(range(n))


def test_forty_task_specs():
    assert len(all_mover_specs()) == 40


@pytest.mark.parametrize("suite,task_id,spec", all_mover_specs())
def test_mover_target_in_bddl(suite: str, task_id: int, spec):
    bddl = _bddl_path(suite, task_id)
    if not bddl.is_file():
        pytest.skip(f"BDDL missing: {bddl}")
    instances = _parse_bddl_objects(bddl)
    assert spec.target in instances, f"{spec.target} not in {bddl}"


@pytest.mark.parametrize("category", sorted(MOVER_CATEGORIES))
def test_mover_mjcf_exists_and_compiles(category: str):
    xml_path = mover_xml_path(category)
    assert xml_path.is_file()
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nbody >= 2


@pytest.mark.parametrize("category", sorted(MOVER_CATEGORIES))
def test_mover_mjcf_has_required_sites(category: str):
    text = mover_xml_path(category).read_text()
    for site in ("bottom_site", "top_site", "horizontal_radius_site"):
        assert f'name="{site}"' in text


def test_mover_assets_restores_objects_dict():
    pytest.importorskip("libero")
    from libero.libero.envs.base_object import OBJECTS_DICT

    category = "alphabet_soup"
    before = OBJECTS_DICT[category]
    with mover_assets("libero_object", 0, enabled=True):
        assert OBJECTS_DICT[category] is not before
    assert OBJECTS_DICT[category] is before


def test_rewrite_language_alphabet_soup():
    spec = get_mover_spec("libero_object", 0)
    out = rewrite_language("pick up the alphabet soup and place it in the basket", spec)
    assert "toy car" in out.lower()
    assert "alphabet soup" not in out.lower()


@pytest.mark.parametrize("suite,task_id,spec", all_mover_specs())
def test_env_nq_nv_unchanged_with_movers(suite: str, task_id: int, spec):
    """Mover swap must not change sim DoF layout (pruned_init compatibility)."""
    pytest.importorskip("libero")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from benchmarks.dynamic_libero.env import libero_bridge as bridge
    from libero.libero import benchmark

    ctx = bridge.load_env_context(task_suite_name=suite, task_id=task_id, seed=0)
    task = ctx.task_suite.get_task(task_id)
    res = bridge.get_env_resolution()

    env_off, _ = bridge.get_libero_env(task, res, 0)
    nq_off = int(env_off.env.sim.model.nq)
    nv_off = int(env_off.env.sim.model.nv)
    env_off.close()

    with mover_assets(suite, task_id, enabled=True):
        env_on, _ = bridge.get_libero_env(task, res, 0)
    nq_on = int(env_on.env.sim.model.nq)
    nv_on = int(env_on.env.sim.model.nv)
    env_on.close()

    assert nq_on == nq_off, f"{suite} t{task_id}: nq {nq_on} != {nq_off}"
    assert nv_on == nv_off, f"{suite} t{task_id}: nv {nv_on} != {nv_off}"


@pytest.mark.parametrize("suite,task_id,spec", all_mover_specs())
def test_pruned_init_loads_with_movers(suite: str, task_id: int, spec):
    pytest.importorskip("libero")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from benchmarks.dynamic_libero.env import libero_bridge as bridge
    from benchmarks.dynamic_libero.env.driver import RealtimeDriver, wrap_env_with_driver
    from benchmarks.dynamic_libero.trajectories import build_trajectory

    ctx = bridge.load_env_context(task_suite_name=suite, task_id=task_id, seed=0)
    task = ctx.task_suite.get_task(task_id)
    res = bridge.get_env_resolution()
    init = list(ctx.task_suite.get_task_init_states(task_id))[0]

    env_off, _ = bridge.get_libero_env(task, res, 0)
    traj_off = build_trajectory("linear", speed=0.0)
    driver_off = RealtimeDriver(env_off, traj_off)
    wrap_env_with_driver(env_off, driver_off)
    env_off.reset()
    env_off.set_init_state(init)
    obj_off = env_off.env.objects_dict[spec.target]
    sim_off = env_off.env.sim
    sim_off.forward()
    bid_off = sim_off.model.body_name2id(obj_off.root_body)
    z_off = float(sim_off.data.body_xpos[bid_off][2])
    env_off.close()

    with mover_assets(suite, task_id, enabled=True) as mover_spec:
        env_on, _ = bridge.get_libero_env(task, res, 0)
        traj_on = build_trajectory("linear", speed=0.0)
        driver_on = RealtimeDriver(env_on, traj_on, mover=mover_spec)
        wrap_env_with_driver(env_on, driver_on)
        try:
            env_on.reset()
            env_on.set_init_state(init)
            assert driver_on.target == spec.target
            obj_on = env_on.env.objects_dict[spec.target]
            sim_on = env_on.env.sim
            sim_on.forward()
            bid_on = sim_on.model.body_name2id(obj_on.root_body)
            z_on = float(sim_on.data.body_xpos[bid_on][2])
            assert abs(z_on - z_off) < 1e-4, f"{suite} t{task_id}: z drift {z_off}->{z_on}"
        finally:
            env_on.close()
