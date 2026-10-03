#!/usr/bin/env python3
"""Convert extra oracle episode folders into a LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

ROOT = setup_sys_path()


FEATURES = {
    "observation.images.cam_high": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.cam_left_wrist": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.cam_right_wrist": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),
        "names": ["motors"],
    },
    "action": {
        "dtype": "float32",
        "shape": (14,),
        "names": ["motors"],
    },
}


def _iter_episodes(raw: Path):
    eps = sorted(raw.glob("rank*/ep*"))
    for ep_dir in eps:
        meta_p = ep_dir / "meta.json"
        if not meta_p.is_file():
            continue
        yield ep_dir, json.loads(meta_p.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-dir",
        default="evaluate_results/extra_oracle/catch_shuttlecock/raw",
    )
    p.add_argument(
        "--out-dir",
        default="data/extra_oracle/catch_shuttlecock",
    )
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--repo-id", default="extra_catch_shuttlecock")
    args = p.parse_args()

    raw = Path(args.raw_dir)
    if not raw.is_absolute():
        raw = ROOT / raw
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(f"{out} already exists; pick a new --out-dir")

    sys.path.insert(0, str(ROOT / "FastWAM" / "src"))
    from fasterwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset.create(
        repo_id=str(args.repo_id),
        fps=int(args.fps),
        features=FEATURES,
        root=out,
        robot_type="aloha",
        use_videos=True,
        video_codec="h264",
        is_compute_episode_stats_image=False,
    )
    n = 0
    for ep_dir, meta in _iter_episodes(raw):
        head = np.load(ep_dir / "head.npy")
        left = np.load(ep_dir / "left.npy")
        right = np.load(ep_dir / "right.npy")
        qpos = np.load(ep_dir / "qpos.npy")
        action = np.load(ep_dir / "action.npy")
        task = str(meta.get("instruction", "Catch the flying shuttlecock."))
        task4 = [task, task, task, task]
        t = int(qpos.shape[0])
        ep_idx = int(ds.meta.total_episodes)
        for i in range(t):
            ds.add_frame(
                {
                    "observation.images.cam_high": head[i],
                    "observation.images.cam_left_wrist": left[i],
                    "observation.images.cam_right_wrist": right[i],
                    "observation.state": qpos[i].astype(np.float32),
                    "action": action[i].astype(np.float32),
                },
                task=task4,
            )
            for key, arr in (
                ("observation.images.cam_high", head[i]),
                ("observation.images.cam_left_wrist", left[i]),
                ("observation.images.cam_right_wrist", right[i]),
            ):
                img_path = ds._get_image_file_path(ep_idx, key, i)
                img_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.ascontiguousarray(arr)).save(img_path, format="JPEG", quality=90)
        ds.save_episode()
        n += 1
        print(f"[to_lerobot] episode {n} frames={t} from {ep_dir}", flush=True)
    print(json.dumps({"episodes": n, "root": str(out)}, indent=2), flush=True)
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
