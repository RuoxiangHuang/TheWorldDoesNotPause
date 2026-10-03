# Config schema (Dynamic-LIBERO official)

| Key | Type | Meaning |
|---|---|---|
| `protocol` | str | Must be `real_time_v2` |
| `suite.name` | str | LIBERO suite (`libero_object`, `libero_spatial`, …) |
| `suite.task_ids` | list[int] \| `all` | Official: all tasks in suite |
| `suite.init_ids` | list[int] | Official: `[0,1,2,3,4]` (N=5) |
| `realtime.backend` | str | Official: `async`; ablation: `freeze` |
| `realtime.replan_steps` | int | Locked `10` for LIBERO |
| `realtime.control_freq` | int | Used for `n_freeze = L * f` (fractional ticks) |
| `realtime.release_radius` | float | Proximity release (metres) |
| `appearance.movers` | `on`\|`off` | Official primary: `off` (original meshes) |
| `appearance.mover_language` | `keep`\|`rewrite` | Official primary: `keep` |
| `appearance.target_registry` | bool | Official: `true` (MoverSpec rails targets) |
| `difficulty.speeds` | list[float] | Metres per control tick (`0` = static sanity) |
| `difficulty.trajectories[].type` | str | Official required: `linear`, `sine`. Diagnostic (not official): `smooth_turn` |
| `accel.presets` | list[str] | FasterWAM accel YAML names (`off`, `full`, …) |
| `latency.k` | int | Varying-obs latency samples; must be `> latency.warmup` |
| `latency.warmup` | int | Discarded warmup samples (default 3) |
| `metrics.report` | list[str] | Includes `auc_sr`, `v50`, `bootstrap_ci` |

CLI entrypoints (`run_official`, `run_paired`, `run_sr`, `run_fitness`) accept the
same knobs as flags; YAML is the canonical schema for paper sweeps.
