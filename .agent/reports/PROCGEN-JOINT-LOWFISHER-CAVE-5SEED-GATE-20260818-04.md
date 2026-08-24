# PROCGEN-JOINT-LOWFISHER-CAVE-5SEED-GATE-20260818-04

## Conclusion

Unique conclusion: `GUARD_NOT_HELPFUL`.

The strict low-Fisher guard is not reproducibly helpful on the bounded
CaveFlyer 1M gate. Across the five paired seeds it has one reward win, one
loss, and three ties. Guard-minus-parent reward is mean `-0.0900`, median
`0.0000`, sample SD `0.3711`, population SD `0.3319`, and range `0.9900`.
Only `1/5` seeds is lower than its parent, so the declared `3/5`
`early-stop-candidate` condition is not met. No early stop, rerun, extension,
or cancellation was performed. This is a 1M causal gate, not a 6M formal
performance result.

## Scope and control-plane snapshots

- Frozen launch commit: `489cc23ba265a0941778399b9a0caaf6b71b00f0`.
- Submission time: `2026-08-18T13:50:25Z`; CSF3 `gpuA`, two arrays
  `18833574` and `18833575`, tasks 1--4, maximum concurrency four per array.
- Pre-launch snapshot (`2026-08-18T13:45Z`): gpuA had 13 mixed and four
  allocated A100-80GB nodes, with no owned Procgen job/trainer. The exact
  seed0 identity had already completed on gpuA. Bede had idle V100s but an OOM
  provenance; ws4090-92/76 were occupied and disk constrained; registered
  dual-5060 had idle 16GB cards without a proven safe FP64 memory envelope.
- Fresh reconciliation (`2026-08-24T10:12:20Z`),
  `login2.csf3.man.alces.network`: neither array has a queue row and no target
  Procgen trainer is live. The only owned queue row was unrelated multicore
  job `19051570`; it was not changed.
- No Jupyter service was used. Quarantined `.54`, `ws4090-31`, and
  `10.49.7.54` were not accessed.

## Frozen identity and strict diff

| Role | SHA256 |
|---|---|
| unguarded trainer `train_shared_jointb_rhsaligned_deterministic.py` | `ff987e0dd5ca1f4c1bb9a91e3794991f5a848bdbfdadc0425d935a72acf3b501` |
| guard trainer `train_shared_jointb_rhsaligned_deterministic_lowfisherguard.py` | `18eea9d75dab6926788673b3bbe9c9ae26468dcbe0688c9a5e9ef150e1751526` |
| unguarded config | `d87a8f648c1c91ee0d260c64ab7dd59d12bb7f9e6b67b0ee0a135389e697fb40` |
| guard config | `7c2ad5efbb004ec36d816143b6ae6b8513d05621c92d21a70a938f48358d06cf` |
| unguarded launcher | `b897e555cefa00f1f1b08e57ce3e0c622acf010c5ba8b4ff6ce1a075c2356096` |
| guard launcher | `7e5db6ed32a6190efa15d8bf828c25e8fa6890071306d62dbfbd5ed7f3482cc6` |

The config diff is exactly these four additions and no deletion or changed
line:

```yaml
joint_low_fisher_actor_critic_guard: true
joint_low_fisher_actor_critic_guard_high: 0.50
joint_low_fisher_actor_critic_guard_low: 0.20
joint_low_fisher_actor_critic_guard_max: 0.05
```

Both sides retain CaveFlyer easy 0--10, IMPALA/ResNet, seed-selected paired
data/evaluation semantics, rollout 4096, minibatch 512, four epochs, nominal
1M, clean all-parameter critic GGN, actor Fisher, full compressed cross terms,
paired-score-residual transformed RHS, `rhs_aligned_rank1_b`, FP64, actor
damping `.003`, critic damping `.5`, block relative damping `.10`, base
actor-from-critic floor `.01`, original clip semantics, momentum zero, and
Kaczmarz false. The verified trainer difference is limited to parsing,
validation, interpolation, and telemetry required for the four guard fields.

Exact command template (the array supplied each literal seed 1--4):

```text
<venv>/python -u <trainer> --config <config> --env_name caveflyer-easy-0-10 --seed <1|2|3|4> --device 0 --total_timesteps 1000000 --joint_ablation_mode full_joint --joint_critic_score_mode clean --joint_critic_param_scope all --joint_critic_reconstruction_scope all --joint_critic_curvature_coef 1.0 --joint_critic_objective_coef 1.0
```

Each root's `command.txt` contains that complete expanded command. Each
`preflight` independently records the frozen commit, exact three hashes,
method, seed, solver and damping identity before training.

## Fresh scheduler and artifact reconciliation

All eight new cells are scientific completions: scheduler `COMPLETED/0:0`,
root `status=PASS`, `rc=0`, 7,872 JSONL rows, and terminal 1,007,616
transitions. Checkpoint policy is `none_by_launcher_design`; absence of model
weights is expected, not artifact loss.

| Method/seed | array/raw | partition/node | elapsed; start--end (CSF3 local) | root/artifacts | classification |
|---|---|---|---|---|---|
| parent s1 | `18833574_1/18833646` | gpuA/node854 | 00:57:09; 14:52--15:49 | `.../unguarded_rhs_aligned_jointb/caveflyer-easy-0-10/seed1`; PASS/rc0/trace/stdout/stderr/command/preflight | scientifically complete |
| parent s2 | `18833574_2/18834133` | gpuA/node855 | 00:57:53; 15:09--16:07 | same root pattern `seed2`; PASS/rc0 | scientifically complete |
| parent s3 | `18833574_3/18835342` | gpuA/node854 | 00:58:22; 15:50--16:48 | same root pattern `seed3`; PASS/rc0 | scientifically complete |
| parent s4 | `18833574_4/18833574` | gpuA/node852 | 00:57:20; 16:07--17:04 | same root pattern `seed4`; PASS/rc0 | scientifically complete |
| guard s1 | `18833575_1/18838990` | gpuA/node863 | 01:00:33; 16:46--17:46 | `.../lowfisher_guard05_rhs_aligned_jointb/caveflyer-easy-0-10/seed1`; PASS/rc0/trace/stdout/stderr/command/preflight | scientifically complete |
| guard s2 | `18833575_2/18839061` | gpuA/node854 | 00:57:38; 16:48--17:46 | same root pattern `seed2`; PASS/rc0 | scientifically complete |
| guard s3 | `18833575_3/18839298` | gpuA/node852 | 00:56:59; 17:04--18:01 | same root pattern `seed3`; PASS/rc0 | scientifically complete |
| guard s4 | `18833575_4/18833575` | gpuA/node852 | 00:59:15; 17:46--18:45 | same root pattern `seed4`; PASS/rc0 | scientifically complete |

Dates for all scheduler rows are `2026-08-18`. Full output base is
`/net/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/gate_1m_cave5seed_20260818_04`.

## Five-seed paired result

`Guard frac` and `floor` are terminal telemetry. Parent guard fraction is not
applicable and its floor is `.01`. Deltas are guard minus parent.

| seed | parent reward | guard reward | reward delta | parent / guard behavior KL | behavior-KL delta | parent / guard current KL | current-KL delta | guard frac | guard floor | parent / guard residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 historical | 2.7800 | 2.0600 | -0.7200 | .00844484 / .00805077 | -.00039406 | 2.75867e-5 / 5.72109e-5 | +2.96242e-5 | .594445 | .0337778 | 5.90343e-13 / 5.35597e-13 |
| 1 | 3.5300 | 3.5300 | 0.0000 | .01389931 / .01389931 | 0 | 9.24369e-5 / 9.24369e-5 | 0 | 0 | .01 | 2.45274e-13 / 2.45274e-13 |
| 2 | 3.9800 | 4.2500 | +0.2700 | .01282985 / .00903619 | -.00379366 | 7.05314e-5 / 8.56306e-5 | +1.50992e-5 | 0 | .01 | 6.40902e-13 / 2.48918e-13 |
| 3 | 4.2800 | 4.2800 | 0.0000 | .00866640 / .00866640 | 0 | 8.05144e-5 / 8.05144e-5 | 0 | 0 | .01 | 3.24049e-13 / 3.24049e-13 |
| 4 | 4.5900 | 4.5900 | 0.0000 | .00383660 / .00383660 | 0 | 4.25832e-5 / 4.25832e-5 | 0 | 0 | .01 | 2.98172e-13 / 2.98172e-13 |

- Paired reward wins/ties/losses for guard: `1/3/1`.
- Parent/guard reward means: `3.8320 / 3.7420`; medians:
  `3.9800 / 4.2500`.
- Paired delta: mean `-0.0900`, median `0`, sample SD `0.3711`, population SD
  `0.3319`, min/max `-0.7200/+0.2700`.
- Guard below parent: `1/5`, not `3/5`; therefore no
  `early-stop-candidate` marker is raised.
- Trace-wide guard activity explains the otherwise zero terminal fractions:
  s1 mean/max/nonzero rows `0/0/0`; s2 `.000897/.267639/72`; s3
  `.000435/.106588/106`; s4 `0/0/0`, each over 7,872 rows. The guard did
  activate transiently in seeds 2 and 3, but only seed2 diverged at the
  terminal metric row. Maximum floors were `.01/.020706/.014264/.01` for
  seeds 1--4. Seed0 had strong terminal activation `.594445`.

## Numerical and auxiliary health

All values below are terminal. `A/C damp` are actor/critic effective median
damping; `Fisher` is categorical Fisher trace; `ret/adv var` are minibatch
return/advantage variance; `A/C/J quad` are joint actor Fisher, critic GGN and
sampled critic quadratics. All are finite.

| method/seed | Fisher | A/C damp | value MSE | ret/adv var | PopArt mean/std | entropy | kernel min/med/max | normalized cross; cross Fro | A/C/J quad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parent 0 | .079228 | 6.279/62.792 | .001762 | .777789/.003454 | 2.43448/2.32649 | .151046 | 8.92e-11/65.545/3214.75 | .053448; 5292.91 | .250882/.942294/.942294 |
| guard 0 | .321666 | 15.515/45.931 | .017222 | .857332/.028406 | 2.28304/2.26267 | .659109 | .001210/409.190/5998.30 | .116585; 14408.72 | .399184/.191190/.191190 |
| parent 1 | .656023 | 36.182/32.416 | .025039 | 1.30258/.050441 | 1.73715/2.16307 | 1.50667 | .115042/333.355/2084.52 | .135145; 10360.17 | .315119/.076212/.076212 |
| guard 1 | .656023 | 36.182/32.416 | .025039 | 1.30258/.050441 | 1.73715/2.16307 | 1.50667 | .115042/333.355/2084.52 | .135145; 10360.17 | .315119/.076212/.076212 |
| parent 2 | .492517 | 32.885/50.517 | .043541 | 1.52722/.073690 | 2.18879/2.32512 | 1.10568 | .162102/466.564/9221.15 | .211645; 28088.10 | .330639/.166209/.166209 |
| guard 2 | .593644 | 67.295/44.346 | .040568 | 1.81296/.069471 | 2.17756/2.31884 | 1.40962 | .054097/494.483/3828.04 | .201664; 27294.99 | .382140/.042856/.042856 |
| parent 3 | .616290 | 35.975/34.820 | .024490 | 1.21134/.036130 | 1.78647/2.07112 | 1.49020 | .017719/351.346/3374.23 | .127975; 11047.91 | .257635/.383763/.383763 |
| guard 3 | .616290 | 35.975/34.820 | .024490 | 1.21134/.036130 | 1.78647/2.07112 | 1.49020 | .017719/351.346/3374.23 | .127975; 11047.91 | .257635/.383763/.383763 |
| parent 4 | .687691 | 38.726/58.450 | .063667 | 1.65101/.098468 | 1.84844/1.94884 | 1.66432 | .107315/520.969/1863.38 | .187209; 21673.37 | .209230/.153587/.153587 |
| guard 4 | .687691 | 38.726/58.450 | .063667 | 1.65101/.098468 | 1.84844/1.94884 | 1.66432 | .107315/520.969/1863.38 | .187209; 21673.37 | .209230/.153587/.153587 |

Additional invariant telemetry for every new cell: total parameter columns
`1,464,544`, critic-head columns `257`, clean critic score noise
min/mean/max `1/1/1` with ESS `512`, `lr_used=.02`, critic ratio min/max
`1/1`, joint clip scale `1`, actor-alone clip scale `0`, and extra actor
attenuation `0`. Actor/critic/cross block Frobenius and separate actor/critic
kernel-diagonal fields are present in each trace; the aggregate table reports
the combined kernel and cross fields while the immutable JSONL preserves every
field. Ratios are finite (new-cell terminal actor ratio envelope overall
`.402575` to `2.702319`). Solver residuals are all below `6.5e-13`.

## Error scan and failure classification

Each new stdout and stderr was scanned case-insensitively for traceback,
standalone NaN/Inf, OOM/out-of-memory, CUDA, NCCL, no-space, disk-quota,
stalled/stuck, and exception signatures: zero hits in all 16 files. All eight
traces end at budget with finite telemetry. Therefore:

- `algorithm-failure`: none among these eight cells.
- `numerical-failure`: none.
- `infrastructure-failure`: none.
- `queued/quota-waiting`: none at reconciliation.
- `unknown/insufficient-evidence`: none among these eight cells.

The scientific conclusion is not inferred from scheduler status alone; it is
supported by PASS/rc0, budget-complete traces, exact identities, finite
terminal rows, and clean error scans.

## Preserved historical ledger

This task adds evidence without deleting or reinterpreting the canonical
ledger in `PROCGEN-JOINT-PROVENANCE-MAP-20260817-03.md`:

- ACTOR_J BossFight seed0 remains `algorithm-failure/EARLY_STOPPED_FAILED`
  (5.7933 versus strict E-v2 10.60, ratio .5465). Original ACTOR_J
  BigFish/CaveFlyer/CoinRun attempts and P1 seed1 roots remain
  `infrastructure-failure`; later recoveries do not overwrite them.
- Bede `1072329_0` remains an infrastructure failure from missing `utils` and
  `1072331_0` remains a V100 CUDA OOM. Completed retry `1072333` erases
  neither failure. Numeric `1072347` remains insufficient evidence for a
  Procgen parent job.
- CSF3 `18642230`, `18624888`, and `18666591` remain
  `cancelled-obsolete-unstarted`: Start=None, no node, zero elapsed and no
  scientific artifacts. They were not released or requeued.
- Historical strict seed0 jobs `18670696_2/18671119` and
  `18672560_2/18673266` remain PASS/rc0 at 1,007,616 and were neither rerun nor
  modified.

## Delivery

- Frozen launch-material commit: `489cc23ba265a0941778399b9a0caaf6b71b00f0`
  (already pushed before submission).
- Evidence/report commit: `2facf8a3c4c444a74ded14ca67570db6a7fa99ba`.
- Delivery HEAD: the follow-up commit containing this immutable delivery
  record; its SHA is reported in the Executor callback after
  `origin/agent-work` is verified.
- Final worktree: required clean after delivery push.
