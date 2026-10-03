import os
from pathlib import Path


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resolve_checkpoint_path(path: str) -> Path:
    """Resolve a checkpoint path against the monorepo ``checkpoints/`` root.

    Configs often use ``checkpoints/foo.pt``. ``DIFFSYNTH_MODEL_BASE_PATH`` points
    at the checkpoints directory itself (``/path/to/FasterWAM/checkpoints``).
    """
    p = Path(path)
    if p.is_absolute():
        return p
    env_base = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
    if env_base:
        base = Path(env_base)
        rel = p.as_posix()
        if rel.startswith("checkpoints/"):
            return base / rel.split("/", 1)[1]
        return base / p
    # .../FasterWAM/FastWAM/src/fasterwam/utils/fs.py -> parents[4] is monorepo root
    repo = Path(__file__).resolve().parents[4]
    return repo / p
