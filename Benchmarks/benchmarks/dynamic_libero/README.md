# Dynamic-LIBERO

论文中的 Dynamic-LIBERO：在 LIBERO 抓放任务上，推理期间世界继续演化。
用于评测冻结的 FastWAM / VLA，加速预设默认比较 `off` 与 `pace`。

Protocol details: [PROTOCOL.md](./PROTOCOL.md).

**Official contract** (`real_time_v2`): original LIBERO appearance (`movers=off`),
language `keep`, rails targets from `MoverSpec` registry, backend `async`,
N=5 inits, trajectories `{linear,sine}`, locked speed grid. See `env/official.py`.

## Layout

```
benchmarks/dynamic_libero/
  PROTOCOL.md              formal Real-Time definition + official contract
  env/official.py          frozen OfficialContract constants
  env/                     RealtimeDriver + LIBERO bridge + movers registry
  assets/movers/           optional toy-car MJCF (secondary track)
  tools/gen_movers.py      parametric asset generator
  eval/run_official.py     official suite entrypoint
  eval/run_fitness.py      task fitness scan (static SR + latency ΔSR)
  eval/run_static_gate.py  movers on vs off @ speed=0 (appearance SR cost)
  eval/run_suite.py        multi-suite paired eval
  eval/run_paired.py       single-suite paired off vs full
  eval/run_sr.py           speed × trajectory SR grid
  eval/run_smooth_turn.py  reaction track: aligned pick + one world-clock turn
  eval/analyze_smooth_turn.py  reaction-chain summary (event-experienced only)
  metrics/aggregate.py     SR tables + AUC-SR / v50 / bootstrap CI
  configs/                 YAML presets + schema (protocol: real_time_v2)
  tests/                   unit tests
```

## Quick start

```bash
cd /DATA/YuanZhen/FasterWAM
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
export PYTHONPATH=src:.
PY=/DATA/YuanZhen/conda_envs/fastwam/bin/python
CKPT=checkpoints/fastwam_release/libero_uncond_2cam224.pt

# Unit tests (CPU)
$PY -m pytest benchmarks/dynamic_libero/tests/test_trajectories.py \
  benchmarks/dynamic_libero/tests/test_official_metrics.py -q

# Official dry-run (prints frozen banner + argv)
$PY -m benchmarks.dynamic_libero.eval.run_official \
  --checkpoint $CKPT --mode sr --task-suite libero_object --task-ids 0 --dry-run

# Official SR grid (primary track)
$PY -m benchmarks.dynamic_libero.eval.run_official \
  --checkpoint $CKPT --mode sr --task-suite libero_object --task-ids all

# Seed reconstructed list: soup → pick / place / both
$PY -m benchmarks.dynamic_libero.eval.run_fitness \
  --checkpoint $CKPT --dynamic-tasks seed --init-ids 0,1

$PY -m benchmarks.common.viz.record_object_motion \
  --benchmark libero --dynamic-task libero_object.t00.place \
  --out-dir evaluate_results/demo_videos/dynamic_libero/place

# Spatial: bowl on a linear flatbed (original prompt)
$PY -m benchmarks.dynamic_libero.eval.run_fitness \
  --checkpoint $CKPT --dynamic-tasks spatial --init-ids 0,1

$PY -m benchmarks.common.viz.record_object_motion \
  --benchmark libero --dynamic-task libero_spatial.t02.pick.carrier \
  --out-dir evaluate_results/demo_videos/dynamic_libero/spatial_carrier

# Paired off vs full (defaults: movers=off, language=keep, backend=async)
$PY -m benchmarks.dynamic_libero.eval.run_paired \
  --checkpoint $CKPT \
  --task-suite libero_object --task-ids 0 --init-ids 0,1 \
  --trajectory linear --speed 0.001 --accels off,pace \
  --out-dir evaluate_results/dynamic_libero/paired

# Secondary track: toy-car appearance
$PY -m benchmarks.dynamic_libero.eval.run_paired \
  --checkpoint $CKPT --movers on --mover-language rewrite ...

# Static SR gate (appearance cost of toy-car swap)
$PY -m benchmarks.dynamic_libero.eval.run_static_gate \
  --checkpoint $CKPT --init-ids 0 \
  --out-dir evaluate_results/dynamic_libero/static_gate

# Full trained suites
$PY -m benchmarks.dynamic_libero.eval.run_suite \
  --checkpoint $CKPT \
  --suites trained --task-ids all --init-ids 0,1,2,3,4 \
  --trajectory linear --speed 0.001 --accels off,pace \
  --out-dir evaluate_results/dynamic_libero/suite_paired

# SR sweep (official trajectories default: linear,sine)
$PY -m benchmarks.dynamic_libero.eval.run_sr \
  --checkpoint $CKPT \
  --task-suite libero_object --task-ids 0 --init-ids 0,1,2 \
  --trajectories linear,sine --speeds 0.0,0.001,0.003,0.006 \
  --accels off,pace \
  --out-dir evaluate_results/dynamic_libero/sr_sweep

# Reaction extension: same cups as main object pick, plus one world-clock turn
$PY -m benchmarks.dynamic_libero.eval.run_smooth_turn \
  --checkpoint $CKPT --accels off,default --init-ids 0,1,2,3,4
```

## Aggregate (AUC-SR / v50)

```bash
python -m benchmarks.dynamic_libero.metrics.aggregate \
  --inputs 'evaluate_results/dynamic_libero/sr_sweep/**/sr_sweep.json' \
  --out evaluate_results/dynamic_libero/SUMMARY.md
```

## Notes

- **Primary track (default):** `--movers off` + `--mover-language keep` +
  `--language-mode keep` + target registry on. Absolute SR stays distributionally
  aligned with training. Motion is unannounced.
- **B-track (ablation):** `--language-mode motion` appends
  `The {noun} is moving; grasp it before it leaves reach.` without rewriting the
  goal clause (`run_official --language-track motion`).
- **Secondary track:** `--movers on` swaps rails targets to toy-car meshes;
  quantify appearance cost with `run_static_gate` before dynamic sweeps.
- Object speed is identical across accel configs; latency appears as longer
  thinking windows for slower stacks (`backend=async` by default).
- Latency uses **varying observations** (honest under video/vae/chunk caches).
- Success is LIBERO BDDL; the protocol does not invent a custom success function.
- `integrations/libero_viz` is a thin shim to `benchmarks.dynamic_libero.viz`.
