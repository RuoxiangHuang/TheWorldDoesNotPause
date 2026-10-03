# Dynamic-RoboTwin Protocol

## 1. Motivation

Static RoboTwin pauses the simulator during policy inference, so wall-clock
latency is invisible to success rate (SR). **Dynamic-RoboTwin** places
a pick target on a configurable trajectory and injects each configuration’s
**measured per-replan latency** into world evolution so inference speed becomes
a causal factor for dual-arm task success.

This protocol is isomorphic to [Dynamic-LIBERO](../dynamic_libero/PROTOCOL.md).
Trajectory families are shared. The simulator backend is SAPIEN (RoboTwin), not
MuJoCo.

DOMINO ([`integrations/domino`](../../integrations/domino/)) is related work:
community dynamic scenes via an external checkout. This benchmark is
**self-contained on native RoboTwin** and does not require DOMINO.

## 2. Real-Time latency coupling (`real_time_v2`)

Protocol version string: **`real_time_v2`** (shared with `benchmarks.common.protocol`).

Default backend is **`async`**: during thinking the robot executes the **stale
tail** of the previous action chunk; when the tail is empty it holds
(zero-order hold). Ablation backend **`freeze`** always holds the robot while
the object keeps moving.

1. At replan \(t\), the policy observes \(\mathrm{obs}_t\) with measured latency
   \(L\) seconds (varying observations; CUDA synchronized).
2. During thinking the object advances for
   \(n_{\mathrm{freeze}}=L\cdot f_{\mathrm{policy}}\) **fractional policy
   ticks** (no integer rounding). Physics substeps scale as
   \(\mathrm{round}(n_{\mathrm{freeze}}\cdot\mathrm{physics\_per\_tick})\).
3. The policy executes `replan_steps` qpos actions via `take_action`. After each
   action the object advances **one policy tick** (hooked at the end of
   `take_action` to avoid double-counting with TOPP internals).
4. When \(\|\mathrm{target}-\mathrm{nearest\_TCP}\| < r_{\mathrm{release}}\)
   (gripper TCP, default \(0.12\,\mathrm{m}\); per-task overrides in
   `task_registry`), rails **release** with **`rails_release_velocity`**
   and PhysX resumes. This is an apparatus event, not a grasp. Success uses
   the task’s native `check_success()` / `eval_success`. Grasp-hold is
   lift ≥2 cm above seated z plus ≥10 ticks of near-hand tracking.

### Scene axes (independent of commanded speed)

Same four flags as Dynamic-LIBERO (`scene`, `apparatus`, `rails`,
`world_clock`). Hold and drive share initial pose, corridor, support
height, and release. Catalog `--task-names main` with `speed=0` is
**apparatus hold**, not the original spawn. Use
`--scene-preset original,apparatus_hold,apparatus_drive`. Only
`--world-clock pause` zeroes `n_freeze`.

**Presentation.** `--presentation demo` draws the bumper car
(`apparatus=visible`). `--presentation test` (alias `ghost_rail`) keeps
the same ghost rails and hides the mesh (`apparatus=hidden`, presets
`ghost_hold` / `ghost_drive`). Unset leaves the preset unchanged, so
catalog `auto` stays visible. Explicit `--apparatus` overrides the mode.
Recorded videos use demo; scored runs that should not pay the car's
appearance cost use test.

Main table: `--task-names main` (linear identity reconstructions).
Extension: `--task-names extension` (irregular curves, same language).
Do not select either set by accelerator ΔSR. Appendix extras
(`--task-names extra`, five native tasks) use the same async measured-latency
contract and task `check_success`. They record `script_weld` as an aux event
and leave `grasp_hold` unknown. They are not part of the 45 or the 95.

### Policy tick (critical)

RoboTwin `take_action` uses variable-length TOPP at ~250 Hz physics. Absolute
physics time per action is **not** the protocol unit.

| Symbol | Value | Meaning |
|---|---|---|
| `policy_hz` | 20 | Same narrative rate as LIBERO / DOMINO |
| `speed` | m / policy tick | Identical on all accel panels |
| `physics_per_tick` | auto | Median `scene.step` per `take_action`; override with `--physics-per-tick` |

### Hardware-decoupled SR(L) curve

See `eval/run_sr_latency.py`. Inject synthetic \(L\) (including oracle \(L=0\)
and open-loop \(L=\infty\)) to produce a machine-independent \(\mathrm{SR}(L)\)
curve at a fixed `(task, trajectory, speed)`.

### Locked replan + blind-window ratio

RoboTwin locks `replan_steps=8`. Report
\(\mathrm{BWR}=n_{\mathrm{freeze}}/(n_{\mathrm{freeze}}+\mathrm{replan\_steps})\).

### Policy server

```bash
python -m benchmarks.common.policy.serve \
  --benchmark robotwin --checkpoint checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```

Use `--policy-url http://127.0.0.1:8765` in eval CLIs for process-isolated policies.

## 3. Official evaluation contract (frozen)

Use `env/official.py` / `eval/run_official.py` for comparable reports.

| Knob | Official value |
|---|---|
| `protocol` | `real_time_v2` |
| `backend` | `async` |
| `replan_steps` | 8 |
| `policy_hz` | 20 |
| `release_radius` | 0.12 m (per-task overrides OK) |
| `seeds` | `{0,1,2,3,4}` → **N=5** / task |
| Tasks | full v1 whitelist (8 tasks) |
| Trajectories (required) | `linear`, `sine` |
| Speeds (m/tick) | `{0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.006, 0.010}` |
| Rails physics | **ghost sliding** (`rails_collide=false`) |
| Dynamics | `contact_trigger=medium` + `occlusion` + `contact_chain` |
| `secondary_rails` | `auto` (handoff + `secondary_attr`) |

**Primary track** keeps reproducible trajectory-first rails. Demo-only
`rails_collide=true` (physical pushes for toy-car videos) is a **secondary
track**, never set for official benchmark episodes.

Task **diagnostics** before a public write-up: `eval/run_fitness.py`
(default `dynamics-track=rails_only`). Fitness does not choose the main
table; use `--task-names main` for identity language.

## 4. Paired evaluation

For fixed `(task_name, seed, trajectory, speed)`:

- Model weights and RNG seeds are identical across accel presets.
- **Only** the acceleration stack (hence \(L\) / \(n_{\mathrm{freeze}}\)) differs.
- Report paired \(\Delta\mathrm{SR}\), not only pooled means.

## 5. Latency measurement

- Warm up caches for `warmup` samples; take the **median** of the
  remaining samples. **Requires \(K>\mathrm{warmup}\)** (raises otherwise).
- Use **varying observations** between timed calls.
- Report `latency_ms`, fractional `n_freeze`,
  \(\mathrm{pursuit\_lag}\approx v\cdot n_{\mathrm{freeze}}\).

## 6. Difficulty axes

| Axis | Meaning | Official / exploratory |
|---|---|---|
| `speed` | metres per policy tick | Official grid above (`~0.001` is the measured off/full separation band for `place_empty_cup`) |
| `trajectory` | shared with dynamic_libero | Official: `linear`, `sine`. Exploratory: `polyline`, `circle`, `stop_and_go` |
| `task` | dynamic rigid pick targets (whitelist) | see `env/task_registry.py` |
| `dynamics` | §9 coordination stresses | Official mix above; `none` for rails-only fitness |

## 7. Success, escape, timeout

- **Success**: task `eval_success` / `check_success()` (unchanged).
- **Timeout**: `take_action_cnt >= step_lim` without success.
- **Escape** (on by default): object leaves per-task workspace AABB before release.
- Do **not** park the target early so a slow policy can catch it.

## 8. Required metrics

1. \(\mathrm{SR}(\mathrm{speed}, \mathrm{trajectory}, \mathrm{accel})\) with bootstrap CI
2. Paired \(\Delta\mathrm{SR}=\mathrm{SR}(\mathrm{full})-\mathrm{SR}(\mathrm{off})\)
3. **AUC-SR** / **normalized AUC-SR** over the official speed grid; paired ΔAUC
4. **v50**: interpolated speed where SR crosses 0.5
5. Latency / \(n_{\mathrm{freeze}}\) / pursuit lag / BWR
6. Catch-before-escape rate
7. Static sanity: `speed=0` ⇒ \(\mathrm{SR}(\mathrm{off})\approx\mathrm{SR}(\mathrm{full})\)

Aggregate helpers: `metrics/aggregate.py` (re-exports LIBERO AUC / v50 / CI).

## 9. Claim and limitations

**Claim.** Under this protocol, FasterWAM’s acceleration stack improves dynamic
manipulation SR relative to an unaccelerated baseline by reducing blind-window
object drift.

**Limitations.** Absolute SR is protocol-specific (policy-tick definition).
Report cross-config deltas. Official coverage is the dynamic-target whitelist,
not all 50 RoboTwin tasks. Publish fitness filters so low-static-SR tasks do
not dominate headlines.

### 9.1 Rails physics: trajectory priority (ghost sliding)

During active rails the pick target follows the **protocol trajectory**, not a
contact-solved free body. Contact with the robot (or other bodies) is **not
allowed to block or deflect** that path — protocol-level **ghost sliding /
trajectory-first** motion. This preserves reproducible controlled motion for
paired latency comparisons (paper claim), at the cost of possible visual
interpenetration while rails are active.

**Implementation (best-effort, not a correctness gate):**
- Prefer PhysX **kinematic target** (`set_kinematic` + `set_kinematic_target`)
  or **velocity drive** (`v=(xyz_des-xyz)/dt_phys`) over per-step `set_pose`
  teleports. Episode field `rails_drive ∈ {kinematic, velocity, pose_fallback}`.
- When collision shapes are writable, rails clear the target’s contact
  type/affinity groups; release restores them (`ghost_collision_filter`).
- Episode field `ghost_sliding=true` whenever rails ran (`speed>0`).

**Demo-only opt-out (`rails_collide=True`):** standalone toy-car motion videos
may pass `rails_collide=True` so the kinematic rails target physically pushes
scene props. This flag defaults to `False` and is **never** set for official
benchmark episodes. Episode field `rails_collide` records which mode was active.

**On release:** kinematic mode is cleared (if enabled),
`rails_release_velocity` is injected, collision groups are restored, and
PhysX resumes normal dynamic contact for the grasp / post-grasp phase.

## 10. Dynamic-RoboTwin 双臂协调机制（相对论文 §3.2）

论文声明本套件在双臂任务上叠加：moving-object handoffs、transient occlusion、
contact-chain perturbations，以及 **任一侧臂建立接触后目标轨迹改变**。
主要压力是 **bimanual timing / coordination window**，而非单纯 tracking。

术语澄清：旧版 rails 释放时注入的线速度 **不再称为 handoff**，统一改名为
`rails_release_velocity`（见 episode 字段 `rails_release_velocity_m_s`）。
真·臂间 handoff 指 RoboTwin 原生 handover / 双臂传递任务上的协调窗口。

### 10.1 机制语义与强度轴

| 机制 | 语义 | 强度轴 | 与 20 Hz 协议耦合 |
|---|---|---|---|
| `contact_trigger` | 任一侧 TCP/夹爪接近目标触发**轨迹切换** | `contact_switch ∈ {off,mild,medium,strong}` | policy-tick 时间轴；`n_freeze` 期间同样可触发 |
| `occlusion` | 头视相机与目标之间插入 box；**只挡视线** | `duty`、`period_ticks`、`half_size` | 每 policy tick 按占空比显示/收起 |
| `handoff` | 真 handover 任务；被传递物体沿轨运动 | 任务级 `stress=handoff` + `speed` | 与 rails + latency freeze 相同 |
| `contact_chain` | 主目标首次接触后沿辅助物序列传播扰动 | `impulse`、`n_aux`、`spacing`、… | 每 policy tick 推进传播 |

官方默认：`contact_trigger=medium` + `occlusion` + `contact_chain`；
handoff 任务 `secondary_rails=auto`；`speed=0` 时 rails/接触切换关闭。

CLI 默认值从 `configs/default.yaml` → `env/eval_config.py` 读取。

### 10.2 任务白名单

**跟踪 / 抓取类**：`place_empty_cup`, `grab_roller`, `move_playingcard_away`,
`move_pillbottle_pad`, `place_phone_stand`, `place_a2b_left`。

**Handoff / 双臂传递**：`handover_block`（`secondary_attr=target_box`）、
`handover_mic`。

### 10.2.1 Bumper-car 重建套件（统一批次）

官方 `all` **不包含**小车。`--task-names recon` 启用完整重建 catalog
（15 个原版技能 × 5 条 id：`.pick` / `.place` / `.both` 直线，
`.pick.irregular` / `.place.irregular` 为 `s_wave`）。
`seed` / `wave2` 是同一批次的别名，不再分波次。
`--task-names recon-demo` 为 both 直线 + 两条不规则（录 demo 用）。

技能：`place_empty_cup`、`place_phone_stand`、`place_container_plate`、
`place_object_basket`、`place_can_basket`、`place_object_stand`、
`place_object_scale`、`place_bread_skillet`、`move_pillbottle_pad`、
`move_stapler_pad`、`place_a2b_left`、`place_a2b_right`、`move_can_pot`、
`place_mouse_pad`、`place_fan`。

语言与 `check_success` 与原版相同。白名单裸名仍是幽灵轨道。
`all` 仍不含小车。

**Latency slice** (`--task-names latency` / `place_slice`): five recon skills
with FastWAM-off static SR ≥ 0.6. Rails-only (`--dynamic-modes none`), async.
Not official `all`, not the 30-id pool, not a post-hoc ΔSR filter. Use this
to test whether FastWAM Cache converts latency into SR the way it does on
Dynamic-LIBERO pick.

### 10.3 文件与验收

| 组件 | 文件 |
|---|---|
| 官方契约 | `env/official.py` |
| 元数据 / 白名单 | `env/task_registry.py` |
| 接触切换 / 遮挡 / 连锁 | `env/dynamics.py` + `env/driver.py` |
| Episode 接线 | `eval/rollout.py`；CLI：`run_official` / `run_fitness` / `run_sr` / `run_paired` |
| 配置 | `configs/default.yaml` |

验收：
1. `speed=0` ⇒ 无 rails、无轨迹切换；off≈full 静态 sanity 仍成立。
2. `contact_trigger≠off` 且 `speed>0` ⇒ 可记录 `contact_events` / `trajectory_switches`。
3. `occlusion` 开启 ⇒ `occlusion_active_ticks > 0`（占空比>0 时）。
4. 释放速度字段名为 `rails_release_velocity_m_s`。
5. 单元测试不依赖 GPU / 完整策略。

### 10.4 Contact-chain / secondary rails / smoke

与先前 §9.4–9.7 实现一致：多跳 `contact_chain`、`secondary_rails=auto`、
`eval/smoke_dynamics.py` 冒烟，以及 kinematic/velocity rails 驱动注记。

```bash
PYTHONPATH=Benchmarks:src:. python -m benchmarks.dynamic_robotwin.eval.smoke_dynamics \
  --task-name place_empty_cup --steps 30
```

## 11. Hybrid grasp modes（A/B 对照轨）

**官方主轨不变：** `grasp_mode=ghost`（§9.1 ghost sliding）。`hybrid_a` /
`hybrid_b` 仅作抓取物理对照，**不写入** official SR 主表 unless 显式标注。

| 模式 | Pursuit | Engage | Release |
|---|---|---|---|
| `ghost` | kinematic/velocity + ghost 碰撞过滤 | — | TCP `dist < r_release` |
| `hybrid_a` | 同上 | 关 kinematic/ghost filter，停钉轨 | `dist < r_release` ∨ closing@engage |
| `hybrid_b` | 同上 | 可碰撞 + compliant velocity（kp 随接近衰减） | 同 A |

默认：`r_release=0.12` m，`r_engage ≈ 0.21` m（`engage_alpha=1.75`）。
Episode 字段同 LIBERO：`grasp_mode`, `rails_phase`, `engage_t`, `release_t`, …

**CLI：** `--grasp-mode`, `--engage-alpha`（默认 ghost）。与 `rails_collide`
（demo 专用）**分离** — hybrid 不开全开碰撞 demo。

**Demo 录制：**

```bash
python -m benchmarks.dynamic_robotwin.viz.record_sbs \
  --task-name place_empty_cup --trajectory linear --speed 0.001 \
  --accels full --dynamic-modes none \
  --grasp-modes ghost,hybrid_b \
  --out-dir evaluate_results/demo_videos/dynamic_robotwin/grasp_hybrid
```

## 12. Reproducibility entrypoints

```bash
# Freeze banner + official SR grid (primary track)
python -m benchmarks.dynamic_robotwin.eval.run_official \
  --checkpoint checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --mode sr --task-names place_empty_cup --dry-run

# Task fitness (rails-only isolates latency coupling)
python -m benchmarks.dynamic_robotwin.eval.run_fitness \
  --checkpoint checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --task-names all --seeds 0,1

# Aggregate AUC / v50
python -m benchmarks.dynamic_robotwin.metrics.aggregate \
  --inputs 'evaluate_results/dynamic_robotwin/official/**/sr_sweep.json' \
  --out evaluate_results/dynamic_robotwin/SUMMARY.md
```
