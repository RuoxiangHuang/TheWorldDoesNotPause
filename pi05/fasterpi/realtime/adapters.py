"""Observation adapters: Real-Time benchmark raw obs → openpi Policy.infer dict."""

from __future__ import annotations

import math

import numpy as np

try:
    from openpi_client import image_tools
except ImportError:  # pragma: no cover
    image_tools = None


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = math.sqrt(max(1e-12, 1.0 - quat[3] * quat[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def libero_obs_to_openpi(
    obs: dict, instruction: str, *, resize: int = 224, skip_resize: bool = False
) -> dict:
    """Convert robosuite/LIBERO obs to openpi pi0.5 LIBERO layout."""
    if image_tools is None:
        raise ImportError("openpi_client is required for LIBERO obs adaptation")
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    if not skip_resize:
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize))
    elif img.dtype != np.uint8:
        img = image_tools.convert_to_uint8(img)
        wrist = image_tools.convert_to_uint8(wrist)
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ).astype(np.float32),
        "prompt": str(instruction),
    }


def robotwin_obs_to_openpi(obs: dict, instruction: str) -> dict:
    """Convert RoboTwin env.get_obs() to openpi Aloha layout.

    Camera mapping matches ``RoboTwin/policy/pi0_accel_client/deploy_policy.py``:
    head → cam_high, left wrist → cam_left_wrist, right wrist → cam_right_wrist.
    """
    head = obs["observation"]["head_camera"]["rgb"]
    right = obs["observation"]["right_camera"]["rgb"]
    left = obs["observation"]["left_camera"]["rgb"]
    state = np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
    imgs = {}
    for key, frame in (
        ("cam_high", head),
        ("cam_left_wrist", left),
        ("cam_right_wrist", right),
    ):
        chw = np.transpose(np.ascontiguousarray(frame), (2, 0, 1))
        imgs[key] = chw.astype(np.uint8, copy=False)
    return {"state": state, "images": imgs, "prompt": str(instruction)}
