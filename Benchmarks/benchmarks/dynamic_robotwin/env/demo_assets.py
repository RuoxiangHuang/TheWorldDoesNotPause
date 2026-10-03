"""Demo-only visual swap: handover_block red box → RoboTwin ``057_toycar``.

Monkey-patches ``create_box`` so the moving ``name='box'`` actor is spawned
from the textured toycar GLB. ``target_box`` and other settings stay unchanged.

Also forces a wheels-down pose: the stock handover_block spawn quat is for a
tall box.  The source GLB's actual long axis is local Z and its wheel-side
plane is local Y=0, so an identity pose stands the car on end.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np

# Unscaled GLB bounds (057_toycar/visual/base0.glb): local X=[-0.581, 0.581],
# Y=[0, 1.026], Z=[-0.949, 0.950]. JSON scale is 0.05.  The asset needs a
# +90° X rotation: local +Y becomes world +Z (wheel-side Y=0 is down), and
# local Z becomes a horizontal vehicle axis.
_TOYCAR_SCALE = 0.05
# RoboTwin table top ≈ box-center 0.842 − half_z 0.10
_TABLE_Z = 0.742
_TOYCAR_WHEEL_Y_MIN = 0.0
_TOYCAR_TABLE_CLEARANCE = 0.003
_TOYCAR_MASS_KG = 0.5
# SAPIEN uses wxyz. +90° around X maps the GLB's wheel-side +Y plane down.
_TOYCAR_UPRIGHT_QUAT_WXYZ = [
    float(np.cos(np.pi / 4.0)),
    float(np.sin(np.pi / 4.0)),
    0.0,
    0.0,
]


def _set_toycar_mass(actor) -> None:
    """Give the GLB a die-cast toy mass so it can actually shove scene props.

    RoboTwin's loader lands this asset at ~10 g, 12x lighter than the 5 cm
    contact-chain cubes (125 g), so a dynamic (non-kinematic) car stalls on
    first contact. Inertia is scaled by the same factor to stay consistent.
    """
    import sapien

    ent = getattr(actor, "actor", actor)
    try:
        comp = ent.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    except Exception:
        return
    if comp is None:
        return
    try:
        cur = float(comp.mass)
    except Exception:
        return
    if cur <= 1e-9:
        return
    k = _TOYCAR_MASS_KG / cur
    try:
        inertia = np.asarray(comp.inertia, dtype=np.float64).reshape(3) * k
    except Exception:
        inertia = None
    try:
        comp.mass = float(_TOYCAR_MASS_KG)
        if inertia is not None:
            comp.inertia = inertia.tolist()
    except Exception:
        pass


def _seat_toycar_upright(actor, pose) -> None:
    """Lay the GLB onto its wheels, just above the RoboTwin table."""
    import sapien

    try:
        p = np.asarray(pose.p if hasattr(pose, "p") else pose[:3], dtype=np.float64).reshape(3)
    except Exception:
        p = np.zeros(3, dtype=np.float64)
    # After +90° around X, local Y is world Z. Keep the wheel-side plane just
    # clear of the table; the small clearance avoids collision-mesh clipping.
    z_bottom = _TOYCAR_WHEEL_Y_MIN * _TOYCAR_SCALE
    p[2] = float(_TABLE_Z - z_bottom + _TOYCAR_TABLE_CLEARANCE)
    upright = sapien.Pose(p.tolist(), _TOYCAR_UPRIGHT_QUAT_WXYZ)
    try:
        if hasattr(actor, "set_pose"):
            actor.set_pose(upright)
        else:
            getattr(actor, "actor").set_pose(upright)
    except Exception:
        pass


@contextmanager
def temporary_toycar_moving_box(modelname: str = "057_toycar", model_id: int = 0) -> Iterator[None]:
    """While active, ``create_box(..., name='box')`` loads a toy car instead."""
    import importlib

    # ``envs.utils`` star-imports ``create_actor`` the *function*, which can
    # shadow the submodule; load the module file explicitly.
    ca = importlib.import_module("envs.utils.create_actor")
    orig = ca.create_box

    def _patched_create_box(
        scene,
        pose,
        half_size,
        color=None,
        is_static=False,
        name="",
        texture_id=None,
        boxtype="default",
    ):
        # Only the moving handover block; keep the blue target pad as a box.
        if str(name) == "box" and not is_static:
            actor = ca.create_actor(
                scene=scene,
                pose=pose,
                modelname=modelname,
                convex=True,
                is_static=False,
                model_id=model_id,
            )
            if actor is None:
                return orig(
                    scene,
                    pose,
                    half_size,
                    color=color,
                    is_static=is_static,
                    name=name,
                    texture_id=texture_id,
                    boxtype=boxtype,
                )
            # Keep RoboTwin attr / driver name expectations.
            try:
                if hasattr(actor, "set_name"):
                    actor.set_name("box")
                else:
                    ent = getattr(actor, "actor", actor)
                    if hasattr(ent, "set_name"):
                        ent.set_name("box")
            except Exception:
                pass
            # Toycar JSON lacks functional_matrix; graft box-like points so
            # handover_block.check_success() does not crash during demos.
            try:
                cfg = dict(getattr(actor, "config", None) or {})
                if not cfg.get("functional_matrix"):
                    cfg["functional_matrix"] = [
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, -1.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    ]
                if "scale" not in cfg:
                    cfg["scale"] = [_TOYCAR_SCALE] * 3
                actor.config = cfg
            except Exception:
                pass
            # Discard tall-box spawn tip; seat wheels on the table.
            _seat_toycar_upright(actor, pose)
            _set_toycar_mass(actor)
            return actor
        return orig(
            scene,
            pose,
            half_size,
            color=color,
            is_static=is_static,
            name=name,
            texture_id=texture_id,
            boxtype=boxtype,
        )

    ca.create_box = _patched_create_box
    patched_modules = [ca]
    # Star-imports bind the old function; retarget common holders.
    for mod_name in ("envs.utils", "envs.handover_block"):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "create_box"):
                setattr(mod, "create_box", _patched_create_box)
                patched_modules.append(mod)
        except Exception:
            continue
    try:
        yield
    finally:
        ca.create_box = orig
        for mod in patched_modules[1:]:
            try:
                setattr(mod, "create_box", orig)
            except Exception:
                pass
