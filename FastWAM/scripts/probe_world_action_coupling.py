#!/usr/bin/env python3
"""Shadow probe: δ_world vs residual / step-hold error (world→action coupling).

Runs with accel disabled; hooks record per-(chunk,k) metrics without changing
the denoise trajectory. See FastWAM/doc/world_action_coupling.md.

Example:
  cd /DATA/YuanZhen/FasterWAM/FastWAM
  PYTHONPATH=src DIFFSYNTH_MODEL_BASE_PATH=/DATA/YuanZhen/FasterWAM/checkpoints \\
    /DATA/YuanZhen/conda_envs/fastwam/bin/python scripts/probe_world_action_coupling.py \\
    --ckpt ../checkpoints/fastwam_release/libero_uncond_2cam224.pt \\
    --task libero_uncond_2cam224_1e-4 --height 224 --width 448 \\
    --drifts 0,0.002,0.005,0.01,0.02,0.03 --seed-mode fixed --steps 4 \\
    --out ../evaluate_results/coupling_probe/libero_nfe4_fixed.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from fasterwam.runtime import prepare_for_inference
from fasterwam.utils.config_resolvers import register_default_resolvers


def _relative_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.abs().mean().clamp(min=1e-6)
    return float((a - b).abs().mean() / denom)


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


def _compose_model_cfg(task: str):
    register_default_resolvers()
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[f"task={task}", "accel=off", "model.load_text_encoder=true"],
        )


class CouplingProbe:
    """Temporary hooks on a live FastWAM model (accel must stay off)."""

    def __init__(self, model: Any, stream_id: str, drift_level: float):
        self.model = model
        self.stream_id = stream_id
        self.drift_level = drift_level
        self.records: list[dict[str, Any]] = []
        self.chunk = -1
        self.step = 0
        self._current_k = 0
        self.prev_video_tokens: torch.Tensor | None = None
        self.delta_world = float("inf")
        self.prev_latents_by_k: dict[int, torch.Tensor] = {}
        self.prev_chunk_latents_by_k: dict[int, torch.Tensor] = {}
        self.prev_residuals_by_k: dict[int, torch.Tensor] = {}
        self.pending_residuals: dict[int, torch.Tensor] = {}
        self.prev_v_star: torch.Tensor | None = None
        self._orig: dict[str, Any] = {}

    def install(self) -> None:
        probe = self
        self._orig["infer_action"] = self.model.infer_action
        self._orig["predict"] = self.model._predict_action_noise_with_cache
        self._orig["prefill"] = self.model.mot.prefill_video_cache
        self._orig["forward"] = self.model.mot.forward_action_with_video_cache

        def infer_action(*args, **kwargs):
            probe.chunk += 1
            probe.step = 0
            probe.prev_v_star = None
            probe.pending_residuals = {}
            out = probe._orig["infer_action"](*args, **kwargs)
            probe.prev_residuals_by_k = {k: v.clone() for k, v in probe.pending_residuals.items()}
            probe.prev_chunk_latents_by_k = {k: v.clone() for k, v in probe.prev_latents_by_k.items()}
            probe.pending_residuals = {}
            probe.prev_latents_by_k = {}
            return out

        def prefill(video_tokens, *args, **kwargs):
            tokens = video_tokens.detach()
            if probe.prev_video_tokens is not None and probe.prev_video_tokens.shape == tokens.shape:
                probe.delta_world = _relative_l1(tokens, probe.prev_video_tokens)
            else:
                probe.delta_world = float("inf") if probe.prev_video_tokens is not None else 0.0
            cache = probe._orig["prefill"](video_tokens, *args, **kwargs)
            probe.prev_video_tokens = tokens.clone()
            return cache

        def forward(action_tokens, *args, **kwargs):
            out = probe._orig["forward"](action_tokens, *args, **kwargs)
            k = probe._current_k
            probe.pending_residuals[k] = (out - action_tokens).detach().clone()
            return out

        def predict(
            latents_action,
            timestep_action,
            context,
            context_mask,
            video_kv_cache,
            attention_mask,
            video_seq_len,
        ):
            k = probe.step
            probe._current_k = k
            probe.step += 1

            latent_drift = float("inf")
            ref_lat = probe.prev_chunk_latents_by_k.get(k)
            if ref_lat is not None and ref_lat.shape == latents_action.shape:
                latent_drift = _relative_l1(latents_action.detach(), ref_lat)

            v_star = probe._orig["predict"](
                latents_action,
                timestep_action,
                context,
                context_mask,
                video_kv_cache,
                attention_mask,
                video_seq_len,
            )

            err_hold = None
            if probe.prev_v_star is not None:
                err_hold = _relative_l1(v_star.detach(), probe.prev_v_star)
            probe.prev_v_star = v_star.detach().clone()

            err_res = None
            v_res = probe._shadow_v_res(
                latents_action, timestep_action, context, context_mask, k
            )
            if v_res is not None:
                err_res = _relative_l1(v_res.detach(), v_star.detach())

            probe.records.append(
                {
                    "stream_id": probe.stream_id,
                    "drift_level": probe.drift_level,
                    "chunk": probe.chunk,
                    "k": k,
                    "timestep": float(timestep_action.reshape(-1)[0].item()),
                    "delta_world": probe.delta_world,
                    "latent_drift": latent_drift,
                    "err_res": err_res,
                    "err_hold": err_hold,
                }
            )
            probe.prev_latents_by_k[k] = latents_action.detach().clone()
            return v_star

        self.model.infer_action = infer_action
        self.model._predict_action_noise_with_cache = predict
        self.model.mot.prefill_video_cache = prefill
        self.model.mot.forward_action_with_video_cache = forward

    def uninstall(self) -> None:
        if "infer_action" in self._orig:
            self.model.infer_action = self._orig["infer_action"]
        if "predict" in self._orig:
            self.model._predict_action_noise_with_cache = self._orig["predict"]
        if "prefill" in self._orig:
            self.model.mot.prefill_video_cache = self._orig["prefill"]
        if "forward" in self._orig:
            self.model.mot.forward_action_with_video_cache = self._orig["forward"]

    def _shadow_v_res(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor | None:
        ref_r = self.prev_residuals_by_k.get(k)
        if ref_r is None:
            return None
        expert = self.model.action_expert
        action_pre = expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        h0 = action_pre["tokens"]
        hL = h0 + ref_r.to(device=h0.device, dtype=h0.dtype)
        return expert.post_dit(hL, action_pre)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--task", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--chunks", type=int, default=12, help="timed replans per drift level")
    parser.add_argument("--warmup-chunks", type=int, default=2)
    parser.add_argument("--drifts", default="0,0.002,0.005,0.01,0.02,0.03")
    parser.add_argument("--seed-mode", choices=("fixed", "varying"), default="fixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16
    drifts = [float(x) for x in args.drifts.split(",") if x.strip() != ""]

    print(f"[probe] load task={args.task} ckpt={args.ckpt}", flush=True)
    cfg = _compose_model_cfg(args.task)
    proprio_dim = int(cfg.model.proprio_dim)
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    model, _ = prepare_for_inference(model, checkpoint_path=args.ckpt, accel={"enabled": False})

    prompt = "The robot arm is performing the task: pick up the bowl"
    proprio = torch.zeros(1, proprio_dim, device=device, dtype=dtype)
    all_records: list[dict[str, Any]] = []

    for drift in drifts:
        stream_id = f"drift{drift:g}_{args.seed_mode}"
        n_frames = args.warmup_chunks + args.chunks
        images = _make_image_stream(
            n_frames, args.height, args.width, seed=1000 + int(drift * 1e6), drift=drift, device=device, dtype=dtype
        )
        probe = CouplingProbe(model, stream_id=stream_id, drift_level=drift)
        probe.install()
        try:
            for i, image in enumerate(images):
                seed = args.seed if args.seed_mode == "fixed" else args.seed + i
                model.infer_action(
                    prompt=prompt,
                    input_image=image,
                    action_horizon=args.action_horizon,
                    proprio=proprio,
                    num_inference_steps=args.steps,
                    seed=seed,
                    rand_device="cpu",
                )
            # drop warmup chunks from export
            all_records.extend(r for r in probe.records if r["chunk"] >= args.warmup_chunks)
            print(f"[probe] drift={drift} rows={len(probe.records)}", flush=True)
        finally:
            probe.uninstall()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in all_records:
            f.write(json.dumps(row) + "\n")
    print(f"[probe] wrote {len(all_records)} rows → {out}", flush=True)


if __name__ == "__main__":
    main()
