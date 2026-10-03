# Dynamic-LIBERO Protocol

## 1. Motivation

Static LIBERO pauses the simulator during policy inference. Wall-clock latency is
therefore invisible to success rate (SR): a lossless accelerator and its baseline
score identically. **Dynamic-LIBERO** puts the pick target on a
configurable trajectory and injects each configuration’s **measured per-replan
latency** into world evolution so that inference speed becomes a causal factor
for task success.

## 2. Real-Time latency coupling (`real_time_v2`)

Protocol version string: **`real_time_v2`** (shared with `benchmarks.common.protocol`).

Default backend is **`async`**: during thinking the robot executes the **stale
tail** of the previous action chunk; when the tail is empty it holds
(zero-order hold). Ablation backend **`freeze`** always holds the robot while
the object keeps moving.

1. At replan time \(t\), the policy observes \(\mathrm{obs}_t\) and runs inference
   with measured (or declared) latency \(L\) seconds.
2. During thinking, object motion continues for
   \(n_{\mathrm{freeze}}=L\cdot f_{\mathrm{ctrl}}\) **fractional** control ticks
   (no integer rounding). Sub-period latencies (e.g. 21 ms @ 20 Hz → 0.42 ticks)
   still couple into the world.
3. The policy then executes an action chunk of `replan_steps` ticks; the object
   advances **one trajectory step per control tick** (identical cm/frame on all
   accel configs).
4. When \(\|\mathrm{target}-\mathrm{eef}\| < r_{\mathrm{release}}\), rails
   **release** with **trajectory handoff velocity**. This is an *apparatus*
   event, not a grasp. Success uses the **unchanged LIBERO BDDL goal**.
   Grasp-hold is recorded separately (lift + sustained near-hand tracking).

```
obs -> infer (latency L) -> [object moves n_freeze ticks; robot: prior chunk | ZOH]
     -> execute chunk [object moves 1 tick per action]
     -> maybe rails-release -> BDDL success?
```

### Scene axes (independent of commanded speed)

Do **not** use `speed > 0` to decide whether to install cars, clear the
path, lift objects, or enable rails. Four flags:

| Axis | Values | Meaning |
|---|---|---|
| `scene` | `original` / `modified` | Official spawn vs path-clear / seating / support height |
| `apparatus` | `off` / `visible` / `hidden` | No actor, visible car/belt, or same layout without the mesh |
| `rails` | `off` / `hold` / `drive` | Free physics, constrain at \(t=0\), or advance the trajectory |
| `world_clock` | `pause` / `realtime` | Thinking does not advance the world vs inference wait is world time |

Hold and drive of the same modified scene **share** initial pose, corridor
clearing, support height, and release rules. Commanded `--speed` is the
drive / layout speed only. Stationary rails still constrain, so
original→hold includes the **full layout rewrite**, not appearance alone.

CLI: `--scene-preset original,apparatus_hold,apparatus_drive` (or
`ghost_hold` / `ghost_drive`). Catalog jobs with `--scene-preset auto` and
`speed=0` are **hold** (apparatus installed, world-clock still realtime),
not the original spawn. Original requires `--scene-preset original`.
`--world-clock pause` is the only switch that zeroes `n_freeze`.

### Episode outcomes (do not alias)

| Field | Meaning |
|---|---|
| `rails_released` | Distance / rule unpinned the rails constraint |
| `grasp_hold` | Object lifted ≥2 cm above seated z and stayed within `r_release` of the hand for ≥10 ticks (0.5 s @ 20 Hz). Gripper close is neither necessary nor sufficient. |
| `task_success` | Unchanged BDDL / `check_success` |
| `aux_events` | `place_freeze`, `release_kick`, `path_cleared`, `rails_release`, `trajectory_switch`, `script_weld`, … |

Missing historical fields stay JSON `null` (unknown). Never fill them with
zero, and never copy `released` into `grasp_hold`. Spot-check grasp-hold
with video; do not treat gripper closure as the definition.

### Main table vs extension

Main-track membership is **semantic**, not “did the accelerator win”:

- Identity tasks such as “pick up the soup and place it in the basket” stay
  on the main table (`--dynamic-tasks main`, 50 object linear+irregular).
- Spatial carriers that home a bowl onto a platform but still said “the
  bowl in the top drawer” are **extension**. Instruction is the public
  scene rule `pick up the black bowl on the moving platform and place it on
  the plate`; BDDL remains bowl-on-plate (`--dynamic-tasks extension`).
- `reconstructed` = 50 main + 10 spatial extension. React-track turns are
  diagnostic, not the official 60.

`eval/run_fitness.py` is a latency-sensitivity **diagnostic**. It does not
choose the main table.

### Locked replan horizon

| Benchmark | `replan_steps` |
|---|---:|
| LIBERO | 10 |
| RoboTwin | 8 |

Report **blind-window ratio** per episode:
\(\mathrm{BWR} = n_{\mathrm{freeze}} / (n_{\mathrm{freeze}} + \mathrm{replan\_steps})\).

### Policy server (process isolation)

Simulators may call an in-process FasterWAM bridge or a remote HTTP policy:

```bash
python -m benchmarks.common.policy.serve \
  --benchmark libero --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --port 8765
```

Eval with `--policy-url http://127.0.0.1:8765` (any VLA/WAM can implement the same JSON API).

### Hardware-decoupled SR(L) curve

In addition to measured-\(L\) paired eval, the suite supports **synthetic latency
injection** (`eval/run_sr_latency.py`):

| Anchor | \(L\) | Meaning |
|---|---|---|
| Oracle | \(0\) | \(n_{\mathrm{freeze}}=0\) |
| Open-loop | \(\infty\) | Thinking never finishes; robot never acts |
| Sweep | \(\{0, 25, 50, 100, 200, 400\}\) ms | Characteristic \(\mathrm{SR}(L)\) |

Methods report their measured \(L\) and read expected SR from the curve.

## 3. Official evaluation contract (frozen)

Use `env/official.py` / `eval/run_official.py` for comparable reports.

| Knob | Official value |
|---|---|
| `protocol` | `real_time_v2` |
| `backend` | `async` |
| `replan_steps` | 10 |
| `control_hz` | 20 |
| `release_radius` | 0.06 m |
| `max_steps` | 400 |
| `init_ids` | `{0,1,2,3,4}` → **N=5** / task |
| `task_ids` | **all** tasks in each trained suite |
| Suites | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| Trajectories (required) | `linear`, `sine` |
| Speeds (m/tick) | `{0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.006, 0.010}` |
| Appearance | **original LIBERO** (`movers=off`) |
| Language | `keep` |
| Rails target | always `MoverSpec` registry (`target_registry=on`) |
| Object-suite motive | **`pusher`**: toy car rides the rails and pushes the original mesh. Not a mesh swap. |
| Seed reconstructed list | soup→basket × `{pick, place, both}` (`--dynamic-tasks seed`) |

**Primary track** keeps training-aligned appearance/language so absolute SR is
interpretable. Toy-car **asset swap** (`movers=on`, `mover-language=rewrite`) is an
**optional secondary track**, not the official default. The object-suite
**pusher** car is a separate actor (original soup/cheese mesh stays); it is not
`movers=on`.

A static LIBERO pick-and-place can be reconstructed into **three** Dynamic-LIBERO
tasks without changing BDDL or the instruction: the grasp target moves, the
receptacle moves, or both move. Catalog: `env/dynamic_tasks.py`. Seed ids:

- `libero_object.t00.pick`
- `libero_object.t00.place`
- `libero_object.t00.both`

### Main track vs reaction extension

**Main track** (`linear` / `sine`, catalog `seed` / `ready` / `reconstructed`):
the target keeps moving and the robot has to keep following. A sine path
changes heading, but those changes are the structure of the whole trajectory,
not a labelled event. Official SR reports this track.

**Reaction extension** (`--dynamic-tasks react`): the same object, instruction,
and BDDL as the aligned main object-suite **pick** task
(`libero_object.tXX.pick` ↔ `libero_object.tXX.pick.react`). After inbound
motion on that same heading, one pre-sampled world-clock event fires. The
first cut is `pick` + one smooth turn. It is **not** part of the reconstructed
60 and is **not** added to `OFFICIAL_TRAJECTORIES`.

| Field | Meaning |
|---|---|
| `event_type` | `turn` (later: `decel` / `accel`) |
| `event_time` | world-clock start, sampled at reset |
| `transition_duration` | ticks over which the change completes |
| `event_magnitude` | turn angle (deg) or later speed change |
| `event_seed` | mix of `(eval seed, task_id, init_id)` so methods share \(p(t)\) |

If the object is grasped **before** the event, rails stay released (do not
drag the held object through the remaining path). The episode keeps full-task
BDDL success but is marked `event_experienced=false` and is excluded from
reaction-time stats.

Task selection for the **latency-sensitive pick slice** (`--dynamic-tasks latency` /
`react_slice`) is pre-registered: object-suite **pick only**, skills
`t00, t01, t04, t05, t06`, N=5 inits. Dropped because they are unsolvable or
out of distribution (not because Cache already won): BBQ (ungraspable), milk
(grasp-ok / place-fail), pudding (turn leaves the table). This slice is **not**
the official reconstructed 60.

The **place slice** (`--dynamic-tasks place_slice`) is the same five skills
with rails on the **basket**. Instruction and BDDL stay unchanged. Place-only
never parks the receptacle (`place_freeze=both` only applies to dual motion),
so the basket is driven for the whole episode. This is the π₀.₅ analog of the
FastWAM pick slice: π already tracks a moving cup; the remaining latency
bottleneck is dropping a static object into a basket that moved during think.
Not the official 60, and not a post-hoc ΔSR filter of reconstructed tasks.

## 4. Paired evaluation

For a fixed `(task_suite, task_id, init_idx, trajectory, speed, seed)`:

- Model weights and RNG seeds are identical across accel presets.
- **Only** the acceleration stack (hence \(L\) / \(n_{\mathrm{freeze}}\)) differs.
- SR gaps are attributed to inference latency, not to a different policy.

## 5. Latency measurement

- Warm up caches for `warmup` samples, then take the **median** of the
  remaining \(K-\mathrm{warmup}\) timed replans. **Requires \(K>\mathrm{warmup}\)**
  (otherwise measurement raises); never fold warmups into the median.
- Use **varying observations** between timed calls so video/VAE/chunk caches
  cannot collapse latency unrealistically.
- Report `latency_ms`, fractional `n_freeze`, and
  \(\mathrm{pursuit\_lag} \approx v\cdot n_{\mathrm{freeze}}\) (metres).

## 6. Difficulty axes

| Axis | Meaning | Official / exploratory values |
|---|---|---|
| `speed` | metres per control / render tick | Official grid above (`0` = static object, **backend still async**). Paper slice (`--speeds paper`): `{0, 0.0003, 0.0005, 0.0006, 0.0008, 0.001, 0.0015, 0.002}` with working cell **v=0.0006**. ~`0.001` is the older off/full separation band. |
| `trajectory` | path family | Official: `linear`, `sine`. Exploratory: `polyline`, `circle`, `stop_and_go`, `random`. Reaction extension (appendix §13, **not** the official grid): `smooth_turn` |
| `traj_complexity` | optional preset over path family | `none` (use `--trajectory`), `easy`→linear, `medium`→sine, `hard`/`chaotic`→random polyline |
| `task` | LIBERO pick targets with free joints | four **trained** suites below |

### Full-suite extension (`libero_uncond_2cam224.pt`)

The release checkpoint trains on four LIBERO suites (see `configs/data/libero_2cam.yaml`):

| Suite | Tasks | In release training |
|---|---:|---|
| `libero_spatial` | 10 | yes |
| `libero_object` | 10 | yes |
| `libero_goal` | 10 | yes |
| `libero_10` | 10 | yes |
| `libero_90` | 90 | **no** (optional OOD eval) |

Use `eval/run_suite.py --suites trained` or `eval/run_official.py` for macro
paired / SR grids. `--task-ids all` expands to every task in each suite.

## 7. Success, escape, timeout

- **Success**: LIBERO BDDL goal predicate (`done=True` from env).
- **Timeout**: exceed `max_steps` without success → failure.
- **Escape** (on by default): object leaves a per-suite / per-task workspace AABB
  before release → failure; reported as `escaped`. Disable with
  `--no-escape-check` or override via `--workspace-aabb`.
- Do **not** park the pusher/conveyor early so a slow policy can catch a stopped
  object (that would erase the latency effect). Default `max_disp` is large.

## 8. Required metrics

1. \(\mathrm{SR}(\mathrm{speed}, \mathrm{trajectory}, \mathrm{accel})\) with bootstrap CI when \(N\) is small
2. Paired \(\Delta\mathrm{SR}=\mathrm{SR}(\mathrm{full})-\mathrm{SR}(\mathrm{off})\)
3. **AUC-SR** / **normalized AUC-SR** over the official speed grid; paired ΔAUC
4. **v50**: interpolated speed where SR crosses 0.5
5. Latency / \(n_{\mathrm{freeze}}\) / pursuit lag / BWR
6. Catch-before-escape rate
7. Original-scene sanity: `--scene-preset original` (not `speed=0` on a
   catalog job, which is stationary apparatus)

Aggregate helpers: `metrics/aggregate.py` (`auc_sr`, `v50`, `bootstrap_ci`).

## 9. Scientific claim and limitations

**Claim.** Under this protocol, FasterWAM’s acceleration stack improves dynamic
grasp SR relative to an unaccelerated baseline by reducing blind-window object
drift.

**Limitations.** Absolute SR is protocol-specific. Report cross-config deltas and
causal isolation, not leaderboard absolutes against unrelated dynamic benchmarks.
Publish task fitness **diagnostics** so low-static-SR tasks are interpreted
honestly. Do **not** drop tasks from the main table because an accelerator
already won.

## 10. Rails targeting + optional movers (v2 mechanisms)

Rails targets are resolved from the per-task **`MoverSpec` registry**
(`env/movers.py`, 40 entries) — **not** inferred from BDDL — whenever
`target_registry=on` (official default). This stays true for the primary track
with original meshes.

| Mechanism | Detail |
|---|---|
| Target selection | Always `get_mover_spec(suite, task_id)` on official runs |
| Asset swap (optional) | Patch `OBJECTS_DICT[category]` → toy-car MJCF under `assets/movers/<category>/` |
| Motion | `drive`: yaw follows tangent after the first trajectory step; `roll`: optional spin |
| Height | `height_offset` default 0; if set, baked into trajectory origin at `begin_policy` |
| Language | Official: `keep`. Secondary track: `rewrite` → e.g. "toy car" |
| Primary vs secondary | `--movers off` (default) = original appearance; `--movers on` = toy-car track |
| Object / libero_10 motive | **`pusher`**: extra 0-DoF toy car rides the rails; original pick mesh is kinematically seated on the bumper (same cm/tick as ghost). Language and BDDL unchanged. `--motive ghost` restores invisible rails. `--motive conveyor` is the legacy belt demo. `--motive carrier` seats the pick mesh on a pickup flatbed. With `--rails-variant pick` (and `--dynamic-tasks spatial`) only the bowl rides; CLI `--motive carrier` without an explicit variant still also bumper-pushes the receptacle. Cars stay visual-only (`contype=0`); distractors on the travel corridor are nudged aside at episode start. Not the official default. |

**CLI flags** (all eval entrypoints): `--movers {on,off}` (default **`off`**),
`--mover-language {rewrite,keep}` (default **`keep`**),
`--motive {auto,pusher,carrier,conveyor,toycar,ghost}` (default **`auto`**),
`--no-target-registry` (legacy BDDL inference; not comparable).

**Known limitations (toy-car track).**

- Wheels are **static geometry** (no hinge joints) so MuJoCo `nq`/`nv` stay
  compatible with pruned inits; plausibility comes from car silhouette + heading.
- Collision is a single invisible box ≤ original object AABB; visual geoms are
  non-colliding (`contype=0`).
- Policies were trained on original LIBERO visuals; run `eval/run_static_gate.py`
  at `speed=0` to quantify appearance/language-only SR cost before dynamic sweeps.

**Regenerate assets** after LIBERO mesh updates:

```bash
python -m benchmarks.dynamic_libero.tools.gen_movers
```

## 11. Hybrid grasp modes (A/B control track)

**Official primary track unchanged:** `grasp_mode=ghost` (trajectory-first pin /
ghost sliding until distance release). Hybrid modes are **paired ablations** for
grasp-window physics — not comparable to official SR tables unless explicitly
reported.

| Mode | Pursuit | Engage (`dist < r_engage` or closing@engage) | Release |
|---|---|---|---|
| `ghost` | MuJoCo pin + zero velocity | — | `dist < r_release` |
| `hybrid_a` | pin (same as ghost) | stop pin; object collides | `dist < r_release` ∨ closing@engage |
| `hybrid_b` | pin | soft velocity PD toward trajectory (`kp` decays with distance) | same as A |

Defaults: `r_release=0.06` m, `r_engage = engage_alpha × r_release` (default
`engage_alpha=1.75` → ~0.105 m). Episode JSON fields: `grasp_mode`,
`rails_phase ∈ {pursuit, engage, free}`, `engage_t`, `release_t`,
`contact_before_release`, `release_velocity_injected`.

**CLI:** `--grasp-mode {ghost,hybrid_a,hybrid_b}` (default `ghost`),
`--engage-alpha`. Official `run_official` does **not** force hybrid.

**Demo:** policy-in-loop SBS with `--grasp-modes ghost,hybrid_b`:

```bash
python -m benchmarks.dynamic_libero.viz.record_sbs \
  --task-suite libero_object --task-id 0 --trajectory linear --speed 0.001 \
  --accels full --movers off --language-mode keep \
  --grasp-modes ghost,hybrid_b \
  --out-dir evaluate_results/demo_videos/dynamic_libero/grasp_hybrid
```

## 12. Reproducibility entrypoints

```bash
# Freeze banner + official SR grid (primary track)
python -m benchmarks.dynamic_libero.eval.run_official \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --mode sr --task-suite libero_object --task-ids all

# Task fitness scan (select publishable tasks)
python -m benchmarks.dynamic_libero.eval.run_fitness \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --suites trained --task-ids all

# Aggregate AUC / v50
python -m benchmarks.dynamic_libero.metrics.aggregate \
  --inputs 'evaluate_results/dynamic_libero/official/**/sr_sweep.json' \
  --out evaluate_results/dynamic_libero/SUMMARY.md
```

## 13. Appendix: reaction extension (not the official SR grid)

The official required trajectories remain **`linear`** and **`sine`** on the
locked speed grid. This appendix is the **reaction extension**: same cups,
instructions, and BDDL as the main object-suite pick catalog, plus one
controllable world-clock motion event.

Do **not** add `smooth_turn` to `OFFICIAL_TRAJECTORIES` or to `run_official`.
Do **not** fold `--dynamic-tasks react` into the reconstructed 60.

The first cut is **pick + one smooth turn**. A sine path can be used either as
continuous-motion SR (main) or as an event diagnostic if a turn is labelled;
the catalog currently labels only `smooth_turn`.

### Locked diagnostic knobs

| Knob | Value | Why |
|---|---|---|
| catalog | `--dynamic-tasks react` | 1:1 with `libero_object.tXX.pick` |
| `trajectory` | `smooth_turn` | inbound rail, then one world-clock turn |
| `speed` | `0.001` m/tick (2 cm/s @ 20 Hz) | same band as the main table, not a faster object |
| `replan_steps` | `10` | do not raise replan frequency |
| `control_hz` | `20` | plant rate unchanged |
| `backend` | `async` | same as the main protocol |
| `rails_variant` | `pick` | only the grasp target moves |
| `motive` | `ghost` | original mesh on rails; no extra car |
| inbound heading | copied from the aligned main pick | same corridor, then the event |
| compare | unaccelerated / Cache | isolate earlier takeover |

### World-clock event (required)

At episode `reset`, sample \(t_{\mathrm{event}}\sim\mathrm{Uniform}(20,60)\)
ticks **after** `begin_policy` from a stable mix of `(eval seed, task_id, init_id)`.
The same episode key must produce the same trajectory for every accel stack.
Heading then blends with a raised cosine over 12 ticks (0.6 s) by 75° (sign
sampled). `xyz_at(t)` is a function of trajectory time only.

**Allowed:** object turns at a pre-sampled simulation time.

**Forbidden:** turning when `predict` starts; turning at every replan boundary;
coupling the event to call count or latency. Those change the task across
methods and inject the change into the think window.

If rails have already released when \(t_{\mathrm{event}}\) arrives, do not pin
the held object back onto the path. Keep the BDDL episode result and set
`event_experienced=false`.

This slice tests **latest-position correction**, not motion prediction. A
frozen policy need not “know” a turn is in progress.

### Metrics (ground truth for eval only)

Record the chain

\[
\text{event} \to \text{first obs of the change} \to \text{new-plan takeover} \to \text{effective correction}
\]

plus close-time target–eef lateral error, `grasp_success` (rails released), and
full BDDL success. Effective correction: after a plan that *saw* the event
takes over, lateral error along the turn normal decreases for 3 consecutive
ticks and drops at least 2 mm. Reaction times are pooled only over
`event_experienced=true`.

Cache can shorten first-obs→takeover. It cannot shorten event→first obs
(that wait is the observation interval). If takeover moves earlier but
correction does not, the next issue is the action / handoff, not replan rate.

### Entrypoints

```bash
python -m benchmarks.dynamic_libero.eval.run_smooth_turn \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --accels off,default --init-ids 0,1,2,3,4

python -m benchmarks.dynamic_libero.eval.analyze_smooth_turn \
  --inputs evaluate_results/dynamic_libero/smooth_turn \
  --out evaluate_results/dynamic_libero/smooth_turn/SUMMARY.md
```

Equivalent via the generic sweep (still not official):

```bash
python -m benchmarks.dynamic_libero.eval.run_sr \
  --dynamic-tasks react --speeds 0.001 --replan-steps 10
```

