"""Unit tests for motion motives and conveyor XML injection."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from benchmarks.dynamic_libero.env.conveyor import (
    CONVEYOR_BODY,
    ConveyorBelt,
    ConveyorConfig,
    inject_conveyor_xml,
)
from benchmarks.dynamic_libero.env.motives import resolve_motive


def test_suite_default_motives():
    assert resolve_motive("libero_object") == "pusher"
    assert resolve_motive("libero_10") == "pusher"
    assert resolve_motive("libero_spatial") == "toycar"
    assert resolve_motive("libero_object", "toycar") == "toycar"
    assert resolve_motive("libero_object", "conveyor") == "conveyor"
    assert resolve_motive("libero_spatial", "auto") == "toycar"
    assert resolve_motive("libero_object", "ghost") == "ghost"
    assert resolve_motive("libero_object", "push") == "pusher"
    assert resolve_motive("libero_object", "carrier") == "carrier"
    assert resolve_motive("libero_object", "flatbed") == "carrier"


def test_inject_conveyor_keeps_fixed_body_and_stripes():
    xml = """
    <mujoco>
      <worldbody>
        <body name="floor" pos="0 0 0"><geom type="plane" size="1 1 0.1"/></body>
      </worldbody>
    </mujoco>
    """
    cfg = ConveyorConfig(pos=np.array([0.1, 0.0, 0.008]), half_length=0.2, half_width=0.05)
    out = inject_conveyor_xml(xml, cfg)
    root = ET.fromstring(out)
    bodies = [b.get("name") for b in root.find("worldbody").findall("body")]
    assert CONVEYOR_BODY in bodies
    geoms = [g.get("name") for g in root.iter("geom")]
    assert f"{CONVEYOR_BODY}_deck" in geoms
    assert f"{CONVEYOR_BODY}_stripe_0" in geoms
    out2 = inject_conveyor_xml(out, cfg)
    n = sum(
        1
        for b in ET.fromstring(out2).find("worldbody").findall("body")
        if b.get("name") == CONVEYOR_BODY
    )
    assert n == 1


def test_inject_does_not_add_free_joint():
    xml = "<mujoco><worldbody></worldbody></mujoco>"
    out = inject_conveyor_xml(xml, ConveyorConfig())
    assert "freejoint" not in out
    assert "<joint" not in out


def test_layout_places_stationary_corridor():
    """Belt center should sit ahead of the start pose along travel, not on top of it."""
    belt = ConveyorBelt()
    belt._env = object()
    applied = {"pose": 0, "stripes": 0}
    belt._apply_pose = lambda: applied.__setitem__("pose", applied["pose"] + 1)  # type: ignore
    belt._apply_stripes = lambda: applied.__setitem__("stripes", applied["stripes"] + 1)  # type: ignore

    start = np.array([0.0, 0.0, 0.04])
    lift = belt.layout(object_xyz=start, axis="x", direction=1.0, travel=0.40, width=0.11)
    assert lift > 0.0
    assert belt.cfg.pos[0] > start[0] + 0.05
    assert abs(belt.cfg.pos[1] - start[1]) < 1e-9
    assert applied["pose"] == 1
