#!/usr/bin/env python3
"""Micro-benchmark: default accel vs chunk_residual_cache (action0).

Compares per-replan wall time and action relative-L1 under a temporally
correlated observation stream. Residual reuse is most useful when consecutive
chunks see similar video tokens (small drift).

Example:
  PYTHONPATH=src /DATA/YuanZhen/conda_envs/fastwam/bin/python scripts/bench_accel_action.py \\
      --ckpt checkpoints/fastwam_release/libero_uncond_2cam224.pt \\
      --steps 4 --warmup 3 --iters 8 --device cuda:0 \\
      --out evaluate_results/accel_chunk_residual/libero_nfe4.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from fasterwam.accel import AccelConfig, AccelStack, get_stack
from fasterwam.runtime import prepare_for_inference
from fasterwam.utils.config_resolvers import register_default_resolvers


def _compose_model_cfg(task: str, accel: str):
    register_default_resolvers()
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[f"task={task}", f"accel={accel}", "model.load_text_encoder=true"],
        )


def _make_image_stream(
    n: int, h: int, w: int, seed: int, drift: float, device, dtype
) -> list[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.rand(1, 3, h, w, generator=g)
    images = []
    frame = base.clone()
    for i in range(n):
        if i > 0 and drift:
            frame = (frame + torch.randn(1, 3, h, w, generator=g) * drift).clamp(0, 1)
        images.append((frame * 2.0 - 1.0).to(device=device, dtype=dtype).clone())
    return images


def _relative_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.abs().mean().clamp(min=1e-6)
    return float((a - b).abs().mean() / denom)


def _build_accel_configs(names: list[str], task: str) -> dict[str, dict]:
    register_default_resolvers()
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")

    def _from_preset(preset: str) -> dict:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(
                config_name="train",
                overrides=[f"task={task}", f"accel={preset}", "model.load_text_encoder=true"],
            )
        return AccelConfig.from_any(cfg.model.accel, apply_env=False).to_dict()

    mapping = {
        "off": {"enabled": False},
        "default": _from_preset("default"),
        "action0": _from_preset("action0"),
        "full": _from_preset("full"),
        "action0_only": AccelConfig.from_any(
            {
                "enabled": True,
                "compile": {"enabled": False},
                "step_cache": {"enabled": False},
                "text_cache": {"enabled": True},
                "chunk_residual_cache": {"enabled": True, "condition_threshold": 0.05},
            },
            apply_env=False,
        ).to_dict(),
        "default_plus_residual": AccelConfig.from_any(
            {
                **_from_preset("default"),
                "chunk_residual_cache": {
                    "enabled": True,
                    "condition_threshold": 0.05,
                    "cold_chunks": 1,
                    "max_consecutive_reuses": 4,
                    "force_refresh_every": 0,
                    "warm_steps": 0,
                },
            },
            apply_env=False,
        ).to_dict(),
    }
    # action0 is default + residual; alias for clarity in reports
    mapping["default_plus_residual"] = mapping["action0"]
    unknown = set(names) - set(mapping)
    if unknown:
        raise ValueError(f"unknown configs: {sorted(unknown)}; choose from {sorted(mapping)}")
    return {name: mapping[name] for name in names}


def main() -> None:
    print("[bench] start", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="checkpoints/fastwam_release/libero_uncond_2cam224.pt",
    )
    parser.add_argument("--task", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--drift", type=float, default=0.01, help="cumulative RGB noise std")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--configs",
        default="off,default,action0",
        help="comma-separated: off,default,action0,full,action0_only",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--seed-mode",
        choices=("varying", "fixed"),
        default="fixed",
        help="fixed: same action noise seed across timed iters (residual-friendly); "
        "varying: seed=base+i (independent noise; latent gate should suppress hits)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    accel_cfgs = _build_accel_configs(config_names, args.task)

    print(f"[bench] compose task={args.task} device={device}", flush=True)
    cfg = _compose_model_cfg(args.task, "off")
    proprio_dim = int(cfg.model.proprio_dim)
    print(f"[bench] instantiate model proprio_dim={proprio_dim}", flush=True)
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    print(f"[bench] load checkpoint {args.ckpt}", flush=True)
    model, _ = prepare_for_inference(model, checkpoint_path=args.ckpt, accel={"enabled": False})
    print("[bench] model ready", flush=True)

    prompt = "The robot arm is performing the task: pick up the bowl"
    proprio = torch.zeros(1, proprio_dim, device=device, dtype=dtype)
    images = _make_image_stream(
        args.warmup + args.iters,
        args.height,
        args.width,
        seed=1000,
        drift=args.drift,
        device=device,
        dtype=dtype,
    )

    def timed_run(accel_cfg):
        stack = get_stack(model)
        if stack is not None:
            stack.uninstall()
        stack = AccelStack(model, AccelConfig.from_any(accel_cfg, apply_env=False)).install()

        for image in images[: args.warmup]:
            model.infer_action(
                prompt=prompt,
                input_image=image,
                action_horizon=args.action_horizon,
                proprio=proprio,
                num_inference_steps=args.steps,
                seed=0,
                rand_device="cpu",
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        actions = []
        for i, image in enumerate(images[args.warmup :]):
            seed = args.seed if args.seed_mode == "fixed" else args.seed + i
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.infer_action(
                prompt=prompt,
                input_image=image,
                action_horizon=args.action_horizon,
                proprio=proprio,
                num_inference_steps=args.steps,
                seed=seed,
                rand_device="cpu",
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            actions.append(out["action"])
        stats = stack.stats()
        stack.uninstall()
        return actions, times, stats

    ordered = list(accel_cfgs)
    if "off" in accel_cfgs and ordered[0] != "off":
        ordered = ["off"] + [n for n in ordered if n != "off"]

    results = {
        "meta": {
            "tag": args.tag,
            "ckpt": args.ckpt,
            "task": args.task,
            "device": device,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "warmup": args.warmup,
            "iters": args.iters,
            "drift": args.drift,
            "action_horizon": args.action_horizon,
            "seed_mode": args.seed_mode,
            "seed": args.seed,
            "configs": ordered,
        }
    }

    ref_actions = None
    off_ms = None
    off_steady = None
    default_ms = None
    default_steady = None
    for name in ordered:
        print(f"[bench] running config={name}", flush=True)
        actions, times, stats = timed_run(accel_cfgs[name])
        mean_ms = 1000.0 * sum(times) / len(times)
        steady = times[1:] if len(times) > 1 else times
        steady_ms = 1000.0 * sum(steady) / len(steady)
        print(f"[bench] done config={name} mean_ms={mean_ms:.1f} steady_ms={steady_ms:.1f}", flush=True)
        entry = {
            "mean_ms": mean_ms,
            "steady_ms": steady_ms,
            "stats": stats,
            "times_ms": [1000.0 * t for t in times],
        }
        if name == "off":
            ref_actions = actions
            off_ms = mean_ms
            off_steady = steady_ms
            entry["action_rel_l1_mean"] = 0.0
            entry["action_rel_l1_max"] = 0.0
        elif ref_actions is not None:
            rel = [_relative_l1(a, r) for a, r in zip(actions, ref_actions)]
            entry["action_rel_l1_mean"] = sum(rel) / len(rel)
            entry["action_rel_l1_max"] = max(rel)
        if off_ms is not None and name != "off":
            entry["speedup_vs_off"] = off_ms / mean_ms
            entry["steady_speedup_vs_off"] = off_steady / steady_ms
        if name == "default":
            default_ms = mean_ms
            default_steady = steady_ms
        results[name] = entry

    if default_ms is not None:
        for name in ("action0", "full", "action0_only"):
            if name in results:
                results[name]["speedup_vs_default"] = default_ms / results[name]["mean_ms"]
                results[name]["steady_speedup_vs_default"] = (
                    default_steady / results[name]["steady_ms"]
                )

    print(json.dumps(results, indent=2, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
