"""Unit tests for Real-Time obs adapters (no GPU)."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fasterpi.realtime.adapters import libero_obs_to_openpi, robotwin_obs_to_openpi


def test_libero_adapter_shapes():
    obs = {
        "agentview_image": np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.ones(3),
        "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
        "robot0_gripper_qpos": np.array([0.0, 0.0]),
    }
    out = libero_obs_to_openpi(obs, "pick up the mug")
    assert out["observation/image"].shape == (224, 224, 3)
    assert out["observation/wrist_image"].shape == (224, 224, 3)
    assert out["observation/state"].shape == (8,)
    assert out["prompt"] == "pick up the mug"


def test_robotwin_adapter_shapes():
    obs = {
        "observation": {
            "head_camera": {"rgb": np.zeros((128, 128, 3), dtype=np.uint8)},
            "right_camera": {"rgb": np.ones((128, 128, 3), dtype=np.uint8) * 2},
            "left_camera": {"rgb": np.ones((128, 128, 3), dtype=np.uint8) * 3},
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
    }
    out = robotwin_obs_to_openpi(obs, "place the cup")
    assert set(out["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    assert out["images"]["cam_high"].shape == (3, 128, 128)
    assert int(out["images"]["cam_left_wrist"].max()) == 3
    assert int(out["images"]["cam_right_wrist"].max()) == 2
    assert out["state"].shape == (14,)
    assert out["prompt"] == "place the cup"


if __name__ == "__main__":
    test_libero_adapter_shapes()
    test_robotwin_adapter_shapes()
    from fasterpi.realtime.codec import (
        decode_obs,
        encode_obs,
        pack_predict_request,
        pack_predict_response,
        unpack_predict_request,
        unpack_predict_response,
    )

    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    back = decode_obs(encode_obs(arr))
    assert np.allclose(arr, back)

    obs = {
        "video.robot0_agentview_left": np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8),
        "state.base_position": np.zeros(3, dtype=np.float32),
    }
    req = unpack_predict_request(
        pack_predict_request(obs=obs, instruction="pick", benchmark="robocasa")
    )
    assert np.array_equal(req["obs"]["video.robot0_agentview_left"], obs["video.robot0_agentview_left"])
    chunk = np.arange(12, dtype=np.float32).reshape(1, 12)
    assert np.allclose(unpack_predict_response(pack_predict_response(chunk)), chunk)
    print("[ok] realtime adapter tests passed")
