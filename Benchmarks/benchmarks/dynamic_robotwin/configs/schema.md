# Config schema (Dynamic-RoboTwin official)

| Key | Type | Meaning |
|---|---|---|
| `protocol` | str | Must be `real_time_v2` |
| `suite.task_names` | list[str] | Whitelist tasks (see `env/task_registry.py`) |
| `suite.seeds` | list[int] | Official: `[0,1,2,3,4]` (N=5) |
| `realtime.backend` | str | Official: `async`; ablation: `freeze` |
| `realtime.replan_steps` | int | Locked `8` for RoboTwin |
| `realtime.policy_hz` | float | Protocol tick rate (20) |
| `realtime.rails_collide` | bool | Official: `false` (ghost sliding); demo-only `true` |
| `dynamics.modes` | list[str] | Official: `contact_trigger,occlusion,contact_chain` |
| `dynamics.contact_switch` | str | Official: `medium` |
| `dynamics.secondary_rails` | `auto`\|`on`\|`off` | Auto for handoff+`secondary_attr` |
| `difficulty.speeds` | list[float] | Metres per policy tick (`0` = static sanity) |
| `difficulty.trajectories[].type` | str | Official required: `linear`, `sine` |
| `accel.presets` | list[str] | FasterWAM accel YAML names (`off`, `full`, …) |
| `latency.k` / `warmup` | int | Varying-obs latency samples; `k > warmup` |
| `metrics.report` | list[str] | Includes `auc_sr`, `v50`, `bootstrap_ci` |

CLI entrypoints (`run_official`, `run_paired`, `run_sr`, `run_fitness`) accept the
same knobs as flags; YAML is the canonical schema for paper sweeps.
