"""Shared helpers for perception probes A (see timely) and B (short history).

Delay unit is **replan steps** on the eager dump stream (same cadence as
CLIRA's cached z_c), not 20 Hz control ticks. Capture-to-infer-start is
zero in this freeze+MuJoCo stack: the obs handed to predict is already
the latest sim frame. Probe A therefore tests queued/desynced/lookahead
obs, not a camera FIFO.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from benchmarks.dynamic_libero.eval.run_coupling import DumpingPolicy, _copy_obs, _driver_meta

DELAYS = (1, 2, 4)
IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
PROP_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def mix_libero_obs(images_from: dict, proprio_from: dict) -> dict:
    """Images from one timestep, proprio from another (no extra copies)."""
    out: dict[str, Any] = {}
    for k in IMAGE_KEYS:
        if k in images_from:
            out[k] = images_from[k]
    for k in PROP_KEYS:
        if k in proprio_from:
            out[k] = proprio_from[k]
    return out


class TimedDumpingPolicy(DumpingPolicy):
    """Dump obs plus wall-clock infer span. Capture == infer-start here."""

    def predict(self, obs: Any, instruction: str, **kwargs: Any) -> np.ndarray:
        t0 = time.perf_counter()
        rec = {
            "obs": _copy_obs(obs),
            "instruction": str(instruction),
            "t_capture": t0,
            "capture_lag_ms": 0.0,
            **_driver_meta(kwargs.get("driver"), obs),
        }
        chunk = np.asarray(self.inner.predict(obs, instruction, **kwargs))
        t1 = time.perf_counter()
        rec["t_infer_end"] = t1
        rec["infer_ms"] = (t1 - t0) * 1000.0
        self.frames.append(rec)
        return chunk


def empty_variant_fields(delays: tuple[int, ...] = DELAYS) -> dict[str, None]:
    rec: dict[str, None] = {
        "err_mix_k2_l1": None,
        "err_mix_k2_abs": None,
        "err_full_consec_l1": None,
        "err_full_consec_abs": None,
        "delta_cos_adj": None,
        "delta_cam_adj": None,
    }
    for d in delays:
        rec[f"err_delay_d{d}_l1"] = None
        rec[f"err_delay_d{d}_abs"] = None
        rec[f"err_img_old_d{d}_l1"] = None
        rec[f"err_img_old_d{d}_abs"] = None
        rec[f"err_prop_old_d{d}_l1"] = None
        rec[f"err_prop_old_d{d}_abs"] = None
        rec[f"err_ahead_d{d}_l1"] = None
        rec[f"err_ahead_d{d}_abs"] = None
        rec[f"delta_cos_d{d}"] = None
        rec[f"delta_cam_d{d}"] = None
        rec[f"motion_rate_d{d}"] = None
    return rec
