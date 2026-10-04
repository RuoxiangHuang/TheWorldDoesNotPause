# Dynamic-LIBERO

Package: `benchmarks.dynamic_libero`. Protocol: `real_time_v2`.

50 tasks. Default comparison: `off` vs `pace`.

Contract: `movers=off`, language `keep`, backend `async`, 5 inits, trajectories `linear,sine`. See `env/official.py` and [PROTOCOL.md](./PROTOCOL.md).

```bash
export PYTHONPATH="$PWD/FastWAM/src:$PWD/Benchmarks:$PWD/pi05"
export MUJOCO_GL=egl

python -m pytest Benchmarks/benchmarks/dynamic_libero/tests/test_trajectories.py -q

python -m benchmarks.dynamic_libero.eval.run_official \
  --checkpoint "$CKPT" --mode sr \
  --task-suite libero_object --task-ids 0 --dry-run

python -m benchmarks.dynamic_libero.eval.run_official \
  --checkpoint "$CKPT" --mode sr \
  --task-suite libero_object --task-ids all

python -m benchmarks.dynamic_libero.eval.run_paired \
  --checkpoint "$CKPT" \
  --task-suite libero_object --task-ids 0 --init-ids 0,1 \
  --trajectory linear --speed 0.001 --accels off,pace

python -m benchmarks.dynamic_libero.eval.run_sr \
  --checkpoint "$CKPT" \
  --task-suite libero_object --task-ids 0 --init-ids 0,1,2 \
  --trajectories linear,sine --speeds 0.0,0.001,0.003,0.006 \
  --accels off,pace

python -m benchmarks.dynamic_libero.metrics.aggregate \
  --inputs 'evaluate_results/dynamic_libero/sr_sweep/**/sr_sweep.json' \
  --out evaluate_results/dynamic_libero/SUMMARY.md
```

| Flag | Values |
|---|---|
| `--movers` | `off` (default), `on` |
| `--mover-language` | `keep` (default), `rewrite` |
| `--language-mode` | `keep` (default), `motion` |
| `--backend` | `async` (default), `freeze` |
