# Extra appendix tasks (not Table 1, not recon, not AUC)

RoboTwin **extra** is a demo / appendix catalog of intercept and chase skills.
`--task-names extra` lists them. They never join `all`, `recon`, or AUC-SR.

Current ids: `catch_shuttlecock`, `catch_bunny_toy`, `stop_rolling_orange`,
`pursue_toycar`, `collide_pool_balls`.

`catch_shuttlecock` is the primary latency probe: time-to-contact ≈ 0.36 s.

## What the acceleration stack is allowed to prove

Paired eval uses **the same checkpoint**. Only measured latency
\(L\) / \(n_{\mathrm{freeze}}=L\cdot 20\) changes (`accel=off` vs
`accel=pace`). If two checkpoints are compared, the gap is no longer
about the stack.

Do **not**:

- Train a policy that only works with cached / approximate features
- Open-loop a memorized timed reach from \(t=0\) (eager then also succeeds)
- Degrade the off checkpoint on purpose
- Mix extra episodes into Table-1 AUC

## Fine-tune recipe (FastWAM + π₀.₅)

Both start from the existing RoboTwin checkpoints
(`robotwin_uncond_3cam_384.pt` and `ckpt/pi05_robotwin2`). Adapt **one
shared LoRA** per backbone on extra demos.

1. **Collect oracle demos at \(L=0\)** (privileged, no freeze ticks).
   Record 80–200 episodes per intercept skill with jitter on spawn pose,
   heading, and a ±15% speed scale so the policy must **look**, not time.
2. **SFT / LoRA the action expert only** (freeze VLM / WAN backbone).
   Keep the original instruction strings from `description/task_instruction/`.
3. **Fitness gate at \(L=0\)**: SR ≥ 0.6 on held-out seeds. If this fails,
   the skill is not acquired and a later off/full gap is meaningless.
4. **Measure** median replan latency on the adapted policy:
   \(L_{\mathrm{off}}\) (eager) and \(L_{\mathrm{full}}\) (stack).
5. **Place time-to-contact in the kinematic gap.** A ballistic intercept
   is predictable from frame 0, so \(T_{\mathrm{catch}}<L_{\mathrm{off}}\)
   is not enough — a perfect first chunk can still arrive in time.
   Let \(t_{\mathrm{reach}}\) be home→catch TCP time from the teacher
   (measure it; typically 0.15–0.25 s on the left arm). Require
   \[
   L_{\mathrm{full}} + t_{\mathrm{reach}} < T_{\mathrm{catch}}
   < L_{\mathrm{off}} + t_{\mathrm{reach}}.
   \]
   Measured RoboTwin FastWAM: \(L_{\mathrm{off}}\approx 250\,\mathrm{ms}\),
   \(L_{\mathrm{full}}\approx 19\,\mathrm{ms}\). Shuttlecock
   \(T_{\mathrm{catch}}=0.36\,\mathrm{s}\) sits in that band if
   \(t_{\mathrm{reach}}\gtrsim 0.15\,\mathrm{s}\). π eager is slower
   (~0.3–0.5 s), so the same bird is an easier off-fail cell.
   If the window is too long, raise \(|v_0|\) or catch on the descending
   branch; if even \(L=0\) misses, the LoRA is too weak.
   First replan uses ZOH (no prior chunk), so the bird flies during
   thinking while the arm stays put — that is the intended off failure.
6. **Paired eval** on identical seeds: `accel=off` vs `accel=pace`,
   same checkpoint, `backend=async`, measured latency.
   Report \(\Delta\mathrm{SR}\), not pooled means. Target
   \(\mathrm{SR}(\mathrm{off})\approx 0\) and \(\mathrm{SR}(\mathrm{full})\ge 0.5\)
   on `catch_shuttlecock`.
7. **Static sanity**: freeze the bird (or \(T_{\mathrm{catch}}\to\infty\)).
   Then \(\mathrm{SR}(\mathrm{off})\approx\mathrm{SR}(\mathrm{full})\).
   If off still fails, the LoRA never learned the grasp.

### Per-skill window (start here, then sweep)

| Extra | Native TTC | Role |
|---|---:|---|
| `catch_shuttlecock` | 0.55 s eval (0.36 s cine) | Hard intercept; should be the off-fail / pace-succeed cell |
| `stop_rolling_orange` | 2.4 s | Easier intercept; raise roll speed until off drops |
| `catch_bunny_toy` | 2.4 s | Same idea on hops |
| `pursue_toycar` | ~3 s chase | Horizon stress; less binary than intercepts |
| `collide_pool_balls` | — | Physics demo only; do not LoRA |

### Why a LoRA can still be closed-loop

Demos must condition on the **current** object pose (jittered trajectories).
If every demo is the same parabola, the policy memorizes “wait 0.3 s then
close” and eager succeeds too. Mix:

- ±8 cm launch XY, ±10 cm launch Z
- ±20% \(|v_0|\)
- left-arm intercept with 2–3 cm TCP noise in the teacher

Teacher is the existing cine script (IK + weld), not a second network.

### FastWAM vs π₀.₅

Same extra dataset, two LoRAs. Serve each with `accel=off` and `accel=pace`.
π eager is usually the slower off panel, so the shuttlecock gap should appear
first there; FastWAM off is faster, so you may need a slightly shorter TTC
or a later catch (descending branch) to make off fail.

Policy eval steps the **native extra ODE** during thinking
(`ExtraNativeDriver`): shuttlecock drag, bunny hops, orange roll, toy-car
bicycle path, or the pool-ball PhysX cut. Linear rails are not used.
`--task-names extra` lists all five. `--motion-scale 0` holds the object.
`--world-clock pause` zeroes \(n_{\mathrm{freeze}}\). Success is
`check_success`. `script_weld` is an aux event; `grasp_hold` stays unknown.
These episodes are not part of the 95-task main table.
