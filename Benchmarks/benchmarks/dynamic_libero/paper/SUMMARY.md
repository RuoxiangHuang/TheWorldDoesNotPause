# Paper assets — Dynamic-LIBERO

This folder collects reproducible figures/tables for the FasterWAM paper narrative.

## Required assets

| Asset | Source command | Status |
|---|---|---|
| SR vs speed (linear) | `eval.run_sr --trajectories linear --speeds 0,0.002,...,0.010 --accels off,pace` | produce under `evaluate_results/dynamic_libero/sr_sweep/` |
| SR vs trajectory family | `eval.run_sr --trajectories linear,sine,circle,polyline --speeds 0.003` | same |
| Wall-clock SBS | `viz.record_sbs --trajectory linear --speed 0.003 --accels off,pace` | `evaluate_results/dynamic_libero/viz/` |
| Static sanity (`speed=0`) | same sweep; expect SR(off)≈SR(full) | table below after run |
| Ablation latency table | `off` / `default` / `action0` / `full` at working point `linear@0.003` | `paired/` |

## Causal claim (locked)

Paired experiments keep **policy weights and random seeds identical**; only the
accel stack (→ measured latency → `n_freeze`) differs. ΔSR is attributed to
inference delay, not to a differently trained policy.

## Working points (main text)

- `linear @ 0.003` and `linear @ 0.006`
- Accel pair: at least `off` vs `full`; optional `default`, `action0`

## How to refresh SUMMARY tables

```bash
python -m benchmarks.dynamic_libero.metrics.aggregate \
  --inputs 'evaluate_results/dynamic_libero/sr_sweep/**/sr_sweep.json' \
           'evaluate_results/dynamic_libero/paired/**/summary.json' \
  --out evaluate_results/dynamic_libero/SUMMARY.md
```

Copy the generated `SUMMARY.md` / `sr_curves.csv` next to figure scripts.

## Smoke verification (this repo)

Already produced under `evaluate_results/dynamic_libero/`:

- `SUMMARY.md` / `sr_curves.csv` — SR table from `run_sr` smoke
- `ABLATION_LATENCY.md` — off vs full latency
- `paired/linear_v0.0_*` — static sanity (both SUCCESS)
- `viz/rtl_*_SIDEBYSIDE.mp4` — wall-clock SBS (Real-Time driver, no `dynamic_libero_sr`)
