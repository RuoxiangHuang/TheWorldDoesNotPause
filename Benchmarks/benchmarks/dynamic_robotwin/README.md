# Dynamic-RoboTwin

Package: `benchmarks.dynamic_robotwin`. Protocol: `real_time_v2`.

45 tasks, plus 5 appendix tasks (`--task-names extra`) that are outside the 45. Default comparison: `off` vs `pace`.

Contract: `backend=async`, `rails_collide=false`, 5 seeds, trajectories `linear,sine`. See `env/official.py` and [PROTOCOL.md](./PROTOCOL.md).

```bash
export PYTHONPATH="$PWD/FastWAM/src:$PWD/Benchmarks:$PWD/pi05"
export MUJOCO_GL=egl
export ROBOTWIN_ROOT=/DATA/YuanZhen/FastWAM/third_party/RoboTwin

python -m pytest Benchmarks/benchmarks/dynamic_robotwin/tests -q

python -m benchmarks.dynamic_robotwin.eval.run_official \
  --checkpoint "$CKPT" --mode sr \
  --task-names place_empty_cup --dry-run

python -m benchmarks.dynamic_robotwin.eval.run_sr \
  --checkpoint "$CKPT" \
  --task-names place_empty_cup,handover_block \
  --seeds 0,1,2,3,4 \
  --trajectories linear,sine --speeds 0.0,0.001,0.003 \
  --accels off,pace

python -m benchmarks.dynamic_robotwin.eval.run_official \
  --checkpoint "$CKPT" --mode sr --task-names recon --dry-run

python -m benchmarks.dynamic_robotwin.eval.run_official \
  --checkpoint "$CKPT" --mode sr --task-names extra --dry-run

python -m benchmarks.dynamic_robotwin.eval.smoke_dynamics \
  --task-name place_empty_cup --steps 30

python -m benchmarks.dynamic_robotwin.metrics.aggregate \
  --inputs 'evaluate_results/dynamic_robotwin/sr_sweep/**/sr_sweep.json' \
  --out evaluate_results/dynamic_robotwin/SUMMARY.md
```

| Flag | Values |
|---|---|
| `--task-names` | whitelist, `all`, `recon`, `extra` |
| `--dynamic-modes` | `contact_trigger,occlusion,contact_chain` (default), `none`, `all` |
| `--contact-trigger` | `off`, `mild`, `medium`, `strong` |
| `--occlusion` | `on`, `off` |
| `--contact-chain` | `on`, `off` |
| `--secondary-rails` | `auto`, `on`, `off` |
| `--backend` | `async` (default), `freeze` |
| `--language-mode` | `keep` (default), `motion` |

Extra task ids: `catch_shuttlecock`, `catch_bunny_toy`, `stop_rolling_orange`, `pursue_toycar`, `collide_pool_balls`.
