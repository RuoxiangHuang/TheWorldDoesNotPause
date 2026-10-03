#!/usr/bin/env bash
# Staggered 8-GPU P0 benches. Waits between pairs so CUDA init does not thrash.
set -euo pipefail
ROOT=/DATA/YuanZhen/FasterWAM
PY=/DATA/YuanZhen/conda_envs/fastwam/bin/python
OUT="$ROOT/evaluate_results/accel_p0_bench"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

launch() {
  local gpu="$1"; shift
  local tag="$1"; shift
  echo "[$(date +%H:%M:%S)] launch GPU${gpu} -> ${tag}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$PY" -u scripts/bench_accel_encoder.py \
    --device cuda --tag "$tag" --out "$OUT/${tag}.json" \
    --warmup 3 --iters 8 "$@" \
    >"$OUT/${tag}.log" 2>&1 &
  echo $! >"$OUT/${tag}.pid"
}

launch 0 libero_nfe4_d001 \
  --ckpt checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --task libero_uncond_2cam224_1e-4 --height 224 --width 448 \
  --steps 4 --drift 0.01 --configs off,default,p0
launch 1 libero_nfe10_d001 \
  --ckpt checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --task libero_uncond_2cam224_1e-4 --height 224 --width 448 \
  --steps 10 --drift 0.01 --configs off,default,p0

# Wait until first pair shows GPU memory
for _ in $(seq 1 60); do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  if [ "${mem:-0}" -gt 5000 ]; then break; fi
  sleep 5
done
echo "[$(date +%H:%M:%S)] GPU0 mem=${mem}MiB; launching next pair"

launch 2 libero_nfe4_d000 \
  --ckpt checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --task libero_uncond_2cam224_1e-4 --height 224 --width 448 \
  --steps 4 --drift 0.0 --configs off,default,p0
launch 3 libero_nfe4_d005 \
  --ckpt checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --task libero_uncond_2cam224_1e-4 --height 224 --width 448 \
  --steps 4 --drift 0.05 --configs off,default,p0
sleep 60

launch 4 robotwin_nfe4_d001 \
  --ckpt checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --task robotwin_uncond_3cam_384_1e-4 --height 384 --width 320 \
  --steps 4 --drift 0.01 --configs off,default,p0
launch 5 robotwin_nfe4_d000 \
  --ckpt checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --task robotwin_uncond_3cam_384_1e-4 --height 384 --width 320 \
  --steps 4 --drift 0.0 --configs off,default,p0
sleep 60

launch 6 libero_isolate_d001 \
  --ckpt checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --task libero_uncond_2cam224_1e-4 --height 224 --width 448 \
  --steps 4 --drift 0.01 --configs off,lossless,p0_only,p0
launch 7 robotwin_nfe4_d005 \
  --ckpt checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --task robotwin_uncond_3cam_384_1e-4 --height 384 --width 320 \
  --steps 4 --drift 0.05 --configs off,default,p0

echo "[$(date +%H:%M:%S)] all launched"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
