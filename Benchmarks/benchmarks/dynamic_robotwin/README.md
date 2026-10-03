# Dynamic-RoboTwin

论文中的 Dynamic-RoboTwin：在原生 RoboTwin（SAPIEN）上，推理期间世界继续演化。
除跟踪轨道外，还包含接触触发的轨迹切换、短时遮挡、移动物体交接和多跳接触链。
细节见 [PROTOCOL.md](./PROTOCOL.md)。加速预设默认比较 `off` 与 `pace`。

**Official contract** (`real_time_v2`): `backend=async`, `rails_collide=false`
(ghost sliding), paper dynamics mix, N=5 seeds, trajectories `{linear,sine}`,
locked speed grid. See `env/official.py`.

## vs LIBERO / DOMINO

| | Realtime LIBERO | Realtime RoboTwin | DOMINO integration |
|---|---|---|---|
| Backend | MuJoCo free-joint | SAPIEN rails (kinematic/velocity; ghost sliding) | DOMINO checkout |
| Time unit | control tick @ 20 Hz | **policy tick** @ 20 Hz | policy step + `inject_latency_drift` |
| Trajectories | local package | **reuses** `dynamic_libero.trajectories` | task-native dynamics |
| Primary stress | reaction / tracking | **bimanual timing** | task-native |
| Dependency | FastWAM eval helpers | native RoboTwin only | external DOMINO |

## Quick start

```bash
cd /DATA/YuanZhen/FasterWAM
export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints
export PYTHONPATH=src:.
export ROBOTWIN_ROOT=/DATA/YuanZhen/FastWAM/third_party/RoboTwin
PY=/DATA/YuanZhen/conda_envs/fastwam/bin/python
CKPT=checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt

$PY -m pytest Benchmarks/benchmarks/dynamic_robotwin/tests -q

# Official dry-run (prints frozen banner + argv)
$PY -m benchmarks.dynamic_robotwin.eval.run_official \
  --checkpoint $CKPT --mode sr --task-names place_empty_cup --dry-run

# Official SR grid (primary track)
$PY -m benchmarks.dynamic_robotwin.eval.run_official \
  --checkpoint $CKPT --mode sr --task-names place_empty_cup

# Task fitness (rails-only isolates latency; use primary for §9 stresses)
$PY -m benchmarks.dynamic_robotwin.eval.run_fitness \
  --checkpoint $CKPT --task-names all --seeds 0,1 \
  --dynamics-track rails_only

# Static sanity (speed=0; dynamic contact switches idle)
$PY -m benchmarks.dynamic_robotwin.eval.run_paired \
  --checkpoint $CKPT --task-name place_empty_cup --seeds 0 \
  --trajectory linear --speed 0.0 --accels off,pace \
  --dynamic-modes none \
  --out-dir evaluate_results/dynamic_robotwin/paired

# Paper-aligned combo (defaults: async backend, medium contact, …)
$PY -m benchmarks.dynamic_robotwin.eval.run_sr \
  --checkpoint $CKPT --task-names place_empty_cup,handover_block \
  --seeds 0,1,2,3,4 \
  --trajectories linear,sine --speeds 0.0,0.001,0.003 --accels off,pace \
  --out-dir evaluate_results/dynamic_robotwin/sr_sweep

# Handoff task: secondary-rails=auto enables receive-side target_box motion
$PY -m benchmarks.dynamic_robotwin.eval.run_paired \
  --checkpoint $CKPT --task-name handover_block --seeds 0,1 \
  --trajectory linear --speed 0.001 --accels off,pace \
  --secondary-rails auto \
  --out-dir evaluate_results/dynamic_robotwin/paired

# SAPIEN dynamics smoke (no policy weights)
$PY -m benchmarks.dynamic_robotwin.eval.smoke_dynamics \
  --task-name place_empty_cup --steps 30
```

## Aggregate (AUC-SR / v50)

```bash
python -m benchmarks.dynamic_robotwin.metrics.aggregate \
  --inputs 'evaluate_results/dynamic_robotwin/sr_sweep/**/sr_sweep.json' \
  --out evaluate_results/dynamic_robotwin/SUMMARY.md
```

## Task whitelist

See [`env/task_registry.py`](env/task_registry.py):

| Task | bimanual | stress |
|---|---|---|
| `place_empty_cup` … `place_a2b_left` | mostly no | tracking |
| `grab_roller` | yes | dual_grasp |
| `handover_block`, `handover_mic` | yes | **handoff** |

## Bumper-car reconstructions (`--task-names recon`)

One Dynamic-RoboTwin batch: **15 original skills × 5 catalog ids**
(`.pick` / `.place` / `.both` linear, `.pick.irregular` / `.place.irregular`
as `s_wave`). Official language and `check_success` stay; a bumper car
pushes the original mesh. Bare whitelist names stay ghost rails
(`spawn_pushers=False`). `seed` / `wave2` are aliases of `recon`.

Skills: `place_empty_cup`, `place_phone_stand`, `place_container_plate`,
`place_object_basket`, `place_can_basket`, `place_object_stand`,
`place_object_scale`, `place_bread_skillet`, `move_pillbottle_pad`,
`move_stapler_pad`, `place_a2b_left`, `place_a2b_right`, `move_can_pot`,
`place_mouse_pad`, `place_fan`.

```bash
# Record compact demo (both linear + irregular)
$PY -m benchmarks.dynamic_robotwin.viz.record_cup_pusher --suite recon-demo

# Eval the unified reconstructions (cars on)
$PY -m benchmarks.dynamic_robotwin.eval.run_official \
  --checkpoint $CKPT --mode sr --task-names recon --dry-run
```

Demos: `evaluate_results/demo_videos/dynamic_robotwin/<task>/`.
`--task-names all` stays ghost-rails. Bare whitelist names still do not spawn cars.
Which original skills should get the same treatment: `env/dynamic_tasks.py` (`SKILL_FIT`).

The **latency slice** (`--task-names latency` / `place_slice`) is a FastWAM analog
of the Dynamic-LIBERO pick/place slices: five reconstruction skills whose
*static* FastWAM-off SR was ≥ 0.6 (`place_empty_cup`, `place_can_basket`,
`move_can_pot`, `place_container_plate`, `place_object_stand`). Evaluate with
`--dynamic-modes none` (rails-only) and `backend=async`. This is **not** official
`all`, **not** the 30-id pick+place pool, and **not** a post-hoc Cache-ΔSR filter.

## Extra appendix (`--task-names extra`)

Independent intercept / chase catalog. **Not** in `all`, `recon`, or AUC.
See [`EXTRA.md`](./EXTRA.md) for the LoRA recipe that keeps weights identical
and only varies latency.

| id | kind | TTC | cine |
|---|---|---:|---|
| `catch_shuttlecock` | intercept | 0.36 s | `evaluate_results/demo_videos/catch_shuttlecock/` |
| `catch_bunny_toy` | intercept | 2.4 s | `evaluate_results/demo_videos/catch_bunny_toy/` |
| `stop_rolling_orange` | intercept | 2.4 s | `evaluate_results/demo_videos/stop_rolling_orange/` |
| `pursue_toycar` | chase | ~3 s | `evaluate_results/demo_videos/pursue_toycar/` |
| `collide_pool_balls` | physics demo | — | `evaluate_results/demo_videos/collide_pool_balls/` |

```bash
$PY -m benchmarks.dynamic_robotwin.viz.record_catch_shuttlecock --no-grasp
```

## Dynamic modes (CLI)

| Flag | Meaning |
|---|---|
| `--dynamic-modes` | from YAML (default `contact_trigger,occlusion,contact_chain`) / `none` / `all` |
| `--contact-trigger` | `off\|mild\|medium\|strong` |
| `--occlusion` | `on\|off` |
| `--contact-chain` | `on\|off` |
| `--secondary-rails` | `auto\|on\|off` |
| `--backend` | Official default **`async`**; `freeze` is ablation |
| `--language-mode` | `keep` (primary) / `motion` (B-track motion suffix) |

Episode JSON records `contact_events`, `trajectory_switches`,
`occlusion_active_ticks`, `contact_chain_hops` / `contact_chain_events`,
`rails_release_velocity_m_s`, plus `rails_drive` / `ghost_sliding`
(see PROTOCOL §9.1). B-track: `run_official --language-track motion`.
