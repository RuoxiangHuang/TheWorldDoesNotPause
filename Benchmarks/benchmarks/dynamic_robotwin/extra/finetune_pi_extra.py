#!/usr/bin/env python3
"""Fine-tune π₀.₅ on extra oracle NPZ demos (8-GPU DDP).

Loads ``pi05_robotwin2``, freezes PaliGemma, trains the action expert.
Saves a new safetensors dir that FasterPI can load with the same norm stats.

Uses the same input transform stack as serving (Aloha adapt_to_pi + z-score +
Paligemma tokenize) so eval can load the extra dir as a drop-in ckpt.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# π₀.₅ weights are PyTorch and must live on CUDA. OpenPI still `import jax`
# from gemma/transforms; park JAX on CPU so it cannot steal H20s from DDP.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _iter_eps(raw: Path):
    return sorted(p for p in raw.glob("rank*/ep*") if (p / "meta.json").is_file())


class ExtraWindows(Dataset):
    def __init__(self, raw_dir: Path, horizon: int = 32):
        self.horizon = int(horizon)
        self.items: list[tuple[Path, int]] = []
        self.prompt = "Catch the flying shuttlecock."
        for ep in _iter_eps(raw_dir):
            q = np.load(ep / "qpos.npy")
            t = int(q.shape[0])
            if t < 8:
                continue
            last = max(t - 1, 0)
            for i in range(0, last, 2):
                self.items.append((ep, i))
        if not self.items:
            raise RuntimeError(f"No extra windows under {raw_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        ep, t = self.items[idx]
        head = np.load(ep / "head.npy", mmap_mode="r")
        left = np.load(ep / "left.npy", mmap_mode="r")
        right = np.load(ep / "right.npy", mmap_mode="r")
        qpos = np.load(ep / "qpos.npy")
        T = int(qpos.shape[0])
        h = self.horizon
        acts = np.zeros((h, 14), dtype=np.float32)
        for k in range(h):
            j = min(t + 1 + k, T - 1)
            acts[k] = qpos[j]
        state = qpos[min(t, T - 1)].astype(np.float32)

        def _chw(img):
            im = Image.fromarray(np.asarray(img)).resize((224, 224), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
            return np.transpose(arr, (2, 0, 1)).copy()

        return {
            "state": state,
            "actions": acts,
            "cam_high": _chw(head[min(t, T - 1)]),
            "cam_left_wrist": _chw(left[min(t, T - 1)]),
            "cam_right_wrist": _chw(right[min(t, T - 1)]),
            "prompt": self.prompt,
        }


def setup_ddp():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    return world, rank, local, torch.device(f"cuda:{local}")


class _Bundle:
    def __init__(self, model, input_transform):
        self._model = model
        self._input_transform = input_transform


class _Obs:
    """Minimal Observation stand-in so we never import openpi.models.model (JAX)."""

    def __init__(self, images, image_masks, state, tokenized_prompt, tokenized_prompt_mask):
        self.images = images
        self.image_masks = image_masks
        self.state = state
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.token_ar_mask = None
        self.token_loss_mask = None


def _obs_from_dict(data: dict):
    images = data["image"]
    for key, img in list(images.items()):
        if torch.is_tensor(img) and img.dtype == torch.uint8:
            images[key] = img.to(torch.float32) / 255.0 * 2.0 - 1.0
            if images[key].ndim == 4 and images[key].shape[-1] == 3:
                images[key] = images[key].permute(0, 3, 1, 2)
        elif hasattr(img, "dtype") and np.issubdtype(img.dtype, np.uint8):
            images[key] = img.astype(np.float32) / 255.0 * 2.0 - 1.0
    return _Obs(
        images=images,
        image_masks=data["image_mask"],
        state=data["state"],
        tokenized_prompt=data.get("tokenized_prompt"),
        tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
    )


def _build_policy(ckpt: str, device: str):
    """Load PI0Pytorch weights onto CUDA first; OpenPI JAX imports happen after."""
    print(f"[pi-ft] pinning {device} before any openpi/jax import", flush=True)
    torch.zeros(1, device=device)
    print(
        f"[pi-ft] cuda reserved {torch.cuda.memory_reserved(device) / 1e9:.2f}GB",
        flush=True,
    )

    from types import SimpleNamespace

    import safetensors.torch as st
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    print("[pi-ft] constructing empty PI0Pytorch on CPU", flush=True)
    cfg = SimpleNamespace(
        pi05=True,
        dtype="bfloat16",
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=32,
        action_horizon=32,
        max_token_len=200,
        discrete_state_input=True,
        pytorch_compile_mode=None,
    )
    model = PI0Pytorch(config=cfg)
    weight_path = str(Path(ckpt) / "model.safetensors")
    print(f"[pi-ft] loading safetensors → {device}: {weight_path}", flush=True)
    st.load_model(model, weight_path, strict=False)
    to_bf16 = getattr(model.paligemma_with_expert, "to_bfloat16_for_selected_params", None)
    if callable(to_bf16):
        to_bf16("bfloat16")
    model.to(device)
    print(
        f"[pi-ft] weights on {device} allocated={torch.cuda.memory_allocated(device) / 1e9:.2f}GB",
        flush=True,
    )

    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.aloha_policy import AlohaInputs
    from openpi.shared import download as pi_download
    from openpi.shared.normalize import load as load_norm_stats
    from openpi.transforms import Normalize, PadStatesAndActions, TokenizePrompt, compose

    # Offline: PaligemmaTokenizer otherwise tries gs://big_vision/...
    _tok = Path("/DATA/YuanZhen/FasterPI/ckpt/paligemma_tokenizer/tokenizer.model")
    if _tok.is_file():
        _orig_dl = pi_download.maybe_download

        def _maybe_download(url, *args, **kwargs):
            if "paligemma_tokenizer.model" in str(url):
                return _tok
            return _orig_dl(url, *args, **kwargs)

        pi_download.maybe_download = _maybe_download
        print(f"[pi-ft] using local paligemma tokenizer {_tok}", flush=True)

    norm_id = "pi0.5_clean_randomize_joint_training"
    assets = Path(ckpt) / "assets" / norm_id
    if not (assets / "norm_stats.json").is_file():
        assets = Path(ckpt) / norm_id
    norm_stats = load_norm_stats(str(assets))
    input_transform = compose(
        [
            AlohaInputs(adapt_to_pi=True),
            Normalize(norm_stats, use_quantiles=True),
            TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=True),
            PadStatesAndActions(32),
        ]
    )
    print("[pi-ft] input transforms ready", flush=True)
    return _Bundle(model, input_transform)


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _stack_obs(items: list[dict], device) -> dict:
    skip = {"prompt", "actions"}
    out = {}
    keys = [k for k in items[0].keys() if k not in skip]
    for k in keys:
        v0 = items[0][k]
        if isinstance(v0, dict):
            out[k] = {
                sk: torch.as_tensor(np.stack([_to_numpy(it[k][sk]) for it in items]), device=device)
                for sk in v0
            }
        elif isinstance(v0, str):
            out[k] = [it[k] for it in items]
        else:
            arr = np.stack([_to_numpy(it[k]) for it in items], axis=0)
            out[k] = torch.as_tensor(arr, device=device)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-dir",
        default="/DATA/YuanZhen/FasterWAM/evaluate_results/extra_oracle/catch_shuttlecock/raw",
    )
    p.add_argument(
        "--ckpt",
        default="/DATA/YuanZhen/FasterPI/ckpt/pi05_robotwin2",
    )
    p.add_argument(
        "--out",
        default="/DATA/YuanZhen/FasterWAM/evaluate_results/extra_ckpts/pi05_extra_shuttlecock",
    )
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--horizon", type=int, default=32)
    args = p.parse_args()

    print(f"[pi-ft] rank start local={os.environ.get('LOCAL_RANK','0')}", flush=True)
    world, rank, local, device = setup_ddp()
    is_main = rank == 0
    print(f"[pi-ft] ddp ready world={world} rank={rank} device={device}", flush=True)

    sys.path.insert(0, "/DATA/YuanZhen/PI/openpi/src")
    sys.path.insert(0, "/DATA/YuanZhen/FasterPI")
    os.environ["ROBOTWIN_PI05_CKPT"] = str(args.ckpt)

    print(f"[pi-ft] loading policy from {args.ckpt}", flush=True)
    policy = _build_policy(str(args.ckpt), str(device))
    print("[pi-ft] policy loaded", flush=True)
    model = policy._model
    model.train()
    n_frozen = 0
    n_train = 0
    for name, param in model.named_parameters():
        freeze = "paligemma_with_expert.paligemma" in name and "gemma_expert" not in name
        if freeze:
            param.requires_grad_(False)
            n_frozen += param.numel()
        else:
            n_train += param.numel()
    if is_main:
        print(f"[pi-ft] frozen={n_frozen / 1e6:.1f}M train={n_train / 1e6:.1f}M", flush=True)

    ds = ExtraWindows(Path(args.raw_dir), horizon=int(args.horizon))
    sampler = None
    if world > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=2,
        drop_last=True,
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=1e-4,
    )
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], find_unused_parameters=True
        )

    step = 0
    model_mod = model.module if world > 1 else model
    it = iter(loader)
    while step < int(args.steps):
        if sampler is not None:
            sampler.set_epoch(step)
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        states = batch["state"].numpy()
        actions = batch["actions"].numpy()
        obs_list = []
        act_list = []
        for i in range(states.shape[0]):
            sample = {
                "state": states[i],
                "images": {
                    "cam_high": batch["cam_high"][i].numpy(),
                    "cam_left_wrist": batch["cam_left_wrist"][i].numpy(),
                    "cam_right_wrist": batch["cam_right_wrist"][i].numpy(),
                },
                "prompt": str(batch["prompt"][i]),
                "actions": actions[i],
            }
            tr = policy._input_transform(sample)
            obs_list.append(tr)
            act_list.append(_to_numpy(tr.get("actions", actions[i])))
        obs = _stack_obs(obs_list, device)
        acts = torch.as_tensor(np.stack(act_list, axis=0), device=device, dtype=torch.float32)
        observation = _obs_from_dict(obs)
        loss = model_mod.forward(observation, acts).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        step += 1
        if is_main and (step % 10 == 0 or step == 1):
            print(f"[pi-ft] step={step}/{args.steps} loss={float(loss.detach()):.4f}", flush=True)

    if is_main:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        import safetensors.torch as st

        st.save_model(model_mod, str(out / "model.safetensors"))
        src = Path(args.ckpt)
        for name in ("config.json", "metadata.pt"):
            src_f = src / name
            if src_f.is_file():
                shutil.copy2(src_f, out / name)
        src_assets = src / "assets"
        if src_assets.is_dir():
            dst = out / "assets"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src_assets, dst)
        readme = {"base": args.ckpt, "steps": args.steps, "raw": str(args.raw_dir)}
        (out / "extra_ft.json").write_text(json.dumps(readme, indent=2), encoding="utf-8")
        print(f"[pi-ft] saved {out}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
