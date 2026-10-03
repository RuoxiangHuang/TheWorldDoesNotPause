"""Lightweight CPU unit tests for FasterPI helpers (no GPU / no checkpoint)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import fasterpi as A
import torch


def test_rel_l1():
    a = torch.ones(4)
    b = torch.ones(4)
    assert A.rel_l1(a, b) == 0.0
    assert A.rel_l1(a * 2, b) > 0.5


def test_obs_pipeline():
    pipe = A.ObsPipeline()
    pipe.submit(lambda x: x + 1, 41)
    assert pipe.take() == 42
    pipe.shutdown()


def test_valid_dirs():
    assert "chunk_residual_cache" in A.VALID_DIRS
    assert "step_cache" in A.VALID_DIRS
    assert "c3" not in A.VALID_DIRS
    assert "d1" not in A.VALID_DIRS
    assert "vision" in A.VALID_DIRS
    assert "prefix_kv" in A.VALID_DIRS
    assert "world_gate" in A.VALID_DIRS
    assert "split_prefix" in A.VALID_DIRS
    assert "adaptive_replan" in A.VALID_DIRS
    assert "adaptive_nfe" in A.VALID_DIRS
    assert "sdpa" in A.VALID_DIRS
    assert "compile_prefix" in A.VALID_DIRS
    assert "fuse_loop" in A.VALID_DIRS
    assert "host" in A.VALID_DIRS
    assert "graph" in A.VALID_DIRS
    assert A.resolve_speedup_spec("compile_graph") == "compile+graph"
    assert A.resolve_speedup_spec("kernel") == "compile+compile_prefix+graph"
    assert A.DEFAULT_FASTERPI_SPEC == "compile+chunk_residual_cache+step_cache+compile_prefix"
    assert "adaptive_nfe" not in A.DEFAULT_FASTERPI_SPEC.split("+")
    assert "vision" not in A.DEFAULT_FASTERPI_SPEC.split("+")
    assert A.FASTERPI_V2_SPEC == "compile+chunk_residual_cache+vision+adaptive_nfe+step_cache"
    assert A.resolve_speedup_spec("fasterpi") == A.DEFAULT_FASTERPI_SPEC
    assert A.resolve_speedup_spec("fasterpi_v2") == A.FASTERPI_V2_SPEC
    assert A.resolve_speedup_spec("fasterpi_v1") == A.FASTERPI_V1_SPEC
    assert A.resolve_speedup_spec("pace") == A.DEFAULT_FASTERPI_SPEC
    assert A.resolve_speedup_spec("ours") == A.DEFAULT_FASTERPI_SPEC
    assert A.resolve_speedup_spec("compile+c3+d1+compile_prefix") == A.DEFAULT_FASTERPI_SPEC
    assert A.resolve_speedup_spec("ours_no_adaptive_nfe") == A.OURS_NO_ADAPTIVE_NFE_SPEC
    assert A.resolve_speedup_spec("ours_no_vision") == A.OURS_NO_VISION_SPEC
    assert A.resolve_speedup_spec("ours_no_step_cache") == A.OURS_NO_STEP_CACHE_SPEC
    assert A.resolve_speedup_spec("ours_no_d1") == A.OURS_NO_STEP_CACHE_SPEC
    assert "adaptive_nfe" not in A.OURS_NO_ADAPTIVE_NFE_SPEC.split("+")
    assert "vision" not in A.OURS_NO_VISION_SPEC.split("+")
    assert "step_cache" not in A.OURS_NO_STEP_CACHE_SPEC.split("+")
    assert A.resolve_speedup_spec("baseline") == "eager"


def test_chunk_return_len():
    from fasterpi.realtime.loader import chunk_return_len

    assert chunk_return_len(32, 8, False) == 8
    assert chunk_return_len(32, 8, True) == 32
    assert chunk_return_len(8, 8, True) == 8
    assert chunk_return_len(5, 8, False) == 5


def test_world_state_gates():
    from fasterpi.world_state import WorldState, world_scope, get_world_state

    fast = WorldState(phase="pursuit", object_speed=0.006)
    assert not fast.cache_ok("vision")
    assert not fast.cache_ok("prefix_kv")
    assert not fast.cache_ok("step_cache")
    assert not fast.cache_ok("d1")
    assert fast.recommended_replan_steps() <= 5
    assert fast.recommended_nfe() == 10
    parked = WorldState(phase="free", object_speed=0.0, released=True)
    assert parked.cache_ok("vision")
    assert parked.cache_ok("step_cache")
    assert parked.cache_ok("chunk_residual_cache")
    assert parked.recommended_nfe() == 6
    assert parked.recommended_replan_steps() == 10
    with world_scope(fast):
        assert get_world_state() is fast
    assert get_world_state() is None


def test_from_driver_converts_mps_to_tick():
    """Trajectory.velocity_at is m/s; WorldState.object_speed is m/tick."""
    import numpy as np
    from fasterpi.world_state import WorldState

    class _Traj:
        t = 0.0
        speed = 0.0006

        def velocity_at(self, t, hz=20.0):
            return np.array([self.speed * float(hz), 0.0, 0.0])

        def xyz_at(self, t):
            return np.zeros(3)

    class _Drv:
        trajectory = _Traj()
        control_hz = 20.0
        rails_phase = "pursuit"
        release_radius = 0.06
        engage_alpha = 1.75
        released = False

    mid = WorldState.from_driver(_Drv())
    assert abs(mid.object_speed - 0.0006) < 1e-9
    assert mid.recommended_replan_steps() == 10
    assert not mid.cache_ok("vision")

    _Drv.trajectory.speed = 0.004
    fast = WorldState.from_driver(_Drv())
    assert abs(fast.object_speed - 0.004) < 1e-9
    assert fast.recommended_replan_steps() == 5


def test_install_stack_rejects_unknown():
    class Dummy:
        pass

    try:
        A.install_stack(Dummy(), ["not_a_dir"])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_package_imports():
    from fasterpi.compat import install_stack as _is  # noqa: F401
    import pi05_accel  # noqa: F401 — root shim


if __name__ == "__main__":
    test_rel_l1()
    test_obs_pipeline()
    test_valid_dirs()
    test_install_stack_rejects_unknown()
    test_package_imports()
    test_world_state_gates()
    test_from_driver_converts_mps_to_tick()
    print("[ok] unit tests passed")
