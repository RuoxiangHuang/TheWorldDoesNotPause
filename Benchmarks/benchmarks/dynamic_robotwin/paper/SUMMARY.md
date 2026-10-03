# Paper assets — Dynamic-RoboTwin

| Asset | Command | Output |
|---|---|---|
| SR vs speed | `eval.run_sr --trajectories linear --speeds 0,...,0.010` | `evaluate_results/dynamic_robotwin/sr_sweep/` |
| Paired ΔSR | `eval.run_paired --accels off,pace` | `paired/*/summary.json` |
| Static sanity | `--speed 0.0` | off ≈ full |
| Wall-clock SBS | `viz.record_sbs` | `viz/rtl_tw_*_SIDEBYSIDE.mp4` |

Aggregate:

```bash
python -m benchmarks.dynamic_robotwin.metrics.aggregate \
  --inputs 'evaluate_results/dynamic_robotwin/sr_sweep/**/sr_sweep.json' \
  --out evaluate_results/dynamic_robotwin/SUMMARY.md
```
