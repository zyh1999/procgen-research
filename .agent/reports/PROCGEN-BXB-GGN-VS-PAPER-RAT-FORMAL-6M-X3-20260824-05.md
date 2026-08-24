# PROCGEN-BXB-GGN-VS-PAPER-RAT-FORMAL-6M-X3-20260824-05

## Executive status

Unique status: `PRECHECK_BLOCKED`.

No experiment was submitted. The exact original Paper RAT baseline was
recovered and its requested 12 cells are strict reusable completions. The
historical P1 BxB deterministic critic-GGN candidate was also recovered at the
source/config/launcher level, but it is not a strict single-factor match to
Paper RAT. It changes three actor-side optimizer/schedule fields outside the
task's allowed critic-curvature construction, direct solver and telemetry
boundary:

1. initial learning rate `.004` versus Paper RAT `.5`;
2. adaptive-KL learning-rate decisions once per rollout versus after every
   minibatch;
3. SGD momentum/history correction `0/disabled` versus `1e-6/enabled`.

Changing those fields now would create a new P1 method rather than preserve
the frozen historical target, and testing a second deterministic-GGN candidate
is prohibited. In addition, the historical P1 seed0/seed1 artifact host
`procgen-3090` was not DNS-resolvable during this audit, so its previously
reported seed0 completions cannot meet the task's fresh artifact-integrity
reuse requirement. The precheck gate therefore forbids every launch.

## Audit scope and live resources

- Task commit: `cc4e144261dd5e652e3e2399d51f696d795a00c2`.
- Audit window: `2026-08-24T11:05:32Z` onward.
- CSF3 `login2.csf3.man.alces.network`: no owned Procgen queue row or live
  Procgen trainer. Unrelated owned multicore job `19051570` was running and
  untouched. gpuA had mixed/allocated A100-80GB nodes; gpuH had mixed H200
  nodes. No gpuH job or artifact was touched.
- Bede `login2.bede.dur.ac.uk`: owned queue empty; V100-32GB nodes006--031
  were mostly idle. This capacity was not used because the scientific identity
  gate failed.
- Authorized `ws4090-92`, `ws4090-76`, and `procgen-3090` names were queried
  with bounded read-only SSH and were not resolvable from this Executor.
  Their capacity is `unknown/insufficient-evidence`.
- Quarantined `.54`, `ws4090-31`, and `10.49.7.54` were not accessed. No
  Jupyter service was used or created.
- CSF3 corrected-Paper jobs `17794600` and `18229077` are freshly confirmed
  `CANCELLED`, Start=None, no node and zero elapsed. Neither produced training
  artifacts or supplies reusable cells.

## Target identity: historical P1 BxB deterministic critic-GGN

Recovered immutable candidate:

| Role | Path | SHA256 |
|---|---|---|
| trainer | `deterministic_2b_symfp64_20260807/train_shared_rat_exact_deterministic_ggn_symfp64.py` | `2b50f8cc26bb8c85f91e0b394acc7535f5564ac70036403597d3738d3dab9c1b` |
| config | `deterministic_2b_symfp64_20260807/adv_resnet_shared_exact_deterministic_ggn_symfp64.yaml` | `c177ac0948d3d99f5ca2500c21eebb76035f4a80beac09a0fed55f0a98e000d1` |
| historical wrapper | `deterministic_2b_symfp64_20260807/wait_then_run_seed0.sh` | `9c7806fc029905cc1dd52846d97da10c5d39ede6c1503b93d81d1b588d442bcd` |
| historical code root | `/root/procgen_goal1_20260806/code` on `procgen-3090` | current host unavailable |
| historical Python | `/root/procgen_goal1_20260806/.venv/bin/python` | current dependency inventory unavailable |

The exact historical command template was:

```text
/root/procgen_goal1_20260806/.venv/bin/python -u train_shared_rat_exact_deterministic_ggn_symfp64.py --config adv_resnet_shared_exact_deterministic_ggn_symfp64.yaml --env_name <env> --seed 0 --device 0
```

Source/config semantics are unambiguous:

- `adv`, IMPALA/ResNet hidden size256 shared actor/critic, PopArt, GAE,
  entropy0, ratio clamp `.1--10`, 16 envs x 256 steps, 8 minibatches of 512,
  4 epochs, 6M nominal and 5,980,160 last full update;
- separate per-sample policy log-prob Jacobian `A` and deterministic value
  Jacobian `J_v`; stacked `H=[A;sqrt(.1)J_v]`, hence a 2B x 2B dual system;
- actor ratio weights, unit critic ratio, critic residual RHS, critic
  curvature coefficient `.1`, critic objective coefficient `1`;
- symmetric similarity transform, full-FP64 chunked Gram/damping/RHS,
  symmetric Jacobi congruence, Cholesky solve and FP64 direction
  reconstruction; damping `.5`;
- global shared L2 gradient clip `.5`; initial LR `.004`, max LR `.5`,
  momentum `0`, Kaczmarz/history correction false;
- adaptive behavior-KL thresholds `.005/.04`, evaluated once after all four
  epochs of a rollout; checkpoint written after each logged update;
- telemetry includes solver residual, relative residual, Cholesky/Jacobi,
  actor Fisher, critic GGN, clip, damping, ratios and direction norm.

Static AST, YAML and shell parsing passed. Historical controller evidence says
seed0 completed all four environments, while seed1 CoinRun/BigFish/BossFight
ended near 5,529,600 and CaveFlyer near 2,048,000 after the host shutdown,
with stale RUNNING markers and no resumable checkpoint. Seed2 was absent.
Because the host is currently unresolvable, exact seed0 roots, commands,
terminal progress/checkpoints, hashes and error logs cannot be freshly
revalidated and are not promoted to strict reuse.

## Baseline identity: original Paper RAT

The baseline is not ACTOR_K, ACTOR_I, P1, Joint-B, Joint-2B, or a later
7a0698-derived RAT. It is the byte-identical trainer from
`agent-lab/trust-region` commit
`2b5affd64cbb3c624b4bc1f4767f449df231ffb2`, with only the declared Procgen
horizon override from 3M to 6M:

| Role | Path/identity | SHA256 |
|---|---|---|
| trainer | Bede `.../code/train_shared_rat_papercritic.py`; byte-identical to `train_shared.py` and commit trainer | `cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7` |
| formal config | `.../code/configs/adv_resnet_shared_papercritic.yaml`; byte-identical to staged `adv_resnet_shared.yaml` | `1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea` |
| Bede launcher | `formal_rat_papercritic_array.sbatch` | `e368c7afa7ace0044e891d8736aecdaf8a0965cd7d66344bb6d20522e00a319a` |
| local staged Bede launcher | `work/procgen_paper_2b5affd_6m_bede_20260722/run_array.sbatch` | `81a2aa407298a4094f6e4597cf4f660defb2b374add9cd200961dfa299491a8e` |

Bede root:
`/nobackup/projects/bdman37/yihe/procgen_rat_papercritic_ablation_20260724`.
Formal array `1063880` followed successful smoke `1063879`; all 40 original
eight-environment x five-seed cells completed. The current task uses only the
four requested environments and seeds0--2.

Exact formal command template:

```text
/nobackup/projects/bdman37/yihe/ppc64le/envs/procgen_author/bin/python -u train_shared_rat_papercritic.py --config adv_resnet_shared_papercritic.yaml --env_name <env> --seed <seed> --device 0
```

The environment recorded Python 3.11.5, V100-SXM2-32GB, driver 550.54.15 and
CUDA 12.4. Original Paper RAT semantics are shared policy/value sampled-score
rows, two separate B x B FP32 inverse systems, actor rollout ratio, unit
critic pseudo-advantage, damping `.5`, SGD initial LR `.5`, fixed momentum
`1e-6`, enabled original history correction, adaptive real-KL decisions after
every minibatch with thresholds `.005/.04`, global shared L2 clip `.5`, the
same 4096/512/4 schedule, PopArt/GAE/network/reward logging, and 6M horizon.

Every requested baseline wrapper has scheduler `COMPLETED/0:0`, PASS/rc0,
146 progress rows ending at 5,980,160, a 3,766,013-byte terminal checkpoint,
nonempty command/preflight/stdout/stderr/GPU logs, and zero targeted hard-error
hits.

## Strict field diff and gate decision

| Field | P1 target | Original Paper RAT | Allowed? |
|---|---|---|---|
| environment/seed | exact CLI env; seeds0--2 intended | exact CLI env; seeds0--2 | match |
| network | shared ResNet hidden256, same heads | same | match |
| rollout/minibatch/epochs | 4096/512/4 | 4096/512/4 | match |
| horizon/last full update | 6M/5,980,160 | 6M/5,980,160 | match |
| PopArt, GAE, entropy, ratio clamp | true, GAE, 0, `.1--10` | same | match |
| damping/global clip | `.5` / `.5` | `.5` / `.5` | match |
| critic construction | deterministic value Jacobian/residual, lambda `.1`, joint 2B | sampled value score/unit pseudo-advantage, separate B | declared critic difference |
| direct solve/precision | joint symmetric FP64/Jacobi Cholesky | two FP32 `torch.inverse` systems | declared solver difference |
| solver telemetry | extensive residual/Jacobi/GGN fields | legacy loss/KL fields | declared telemetry difference |
| **initial actor LR** | **`.004`** | **`.5`** | **BLOCKING non-critic difference** |
| **adaptive-KL timing** | **once per rollout after four epochs** | **after every minibatch** | **BLOCKING actor schedule difference** |
| **momentum/history** | **momentum0; correction disabled** | **momentum1e-6; original correction enabled** | **BLOCKING actor optimizer difference** |
| adaptive-KL thresholds | `.005/.04`, min `.0001`, max `.5` | `.005/.04`, floor `.0001`, ceiling config LR `.5` | thresholds match; timing does not |
| checkpoint/reward logging | checkpoint each update; `eprewmean` | same | match at source level |

The full configs differ only in the expected solver/critic keys plus the three
blocking optimizer/schedule fields above (the config default environment name
is overridden by the literal command on both sides). The source diff confirms
that the LR timing and momentum/history differences are executed, not dormant
telemetry. Therefore the comparison is not causally attributable only to
critic curvature, and the task's mandatory stop clause applies.

## 24-cell reuse/launch manifest

This is the logical inventory at the blocked gate. No new root was created.

| Method | Environment | Seed | Decision | Evidence/reason |
|---|---|---:|---|---|
| P1 target | BigFish | 0 | `IDENTITY_BLOCKED` | prior completion reported; current root/checkpoint/log/hash unavailable and pair diff fails |
| P1 target | BigFish | 1 | `NOT_STRICT` | infrastructure-interrupted near 5.53M; no checkpoint |
| P1 target | BigFish | 2 | `LAUNCH_MISSING` | absent, but launch forbidden by global identity gate |
| P1 target | BossFight | 0 | `IDENTITY_BLOCKED` | prior completion reported; current artifact proof unavailable and pair diff fails |
| P1 target | BossFight | 1 | `NOT_STRICT` | infrastructure-interrupted near 5.53M; no checkpoint |
| P1 target | BossFight | 2 | `LAUNCH_MISSING` | absent, but launch forbidden |
| P1 target | CaveFlyer | 0 | `IDENTITY_BLOCKED` | prior completion reported; current artifact proof unavailable and pair diff fails |
| P1 target | CaveFlyer | 1 | `NOT_STRICT` | infrastructure-interrupted near 2.05M; no checkpoint |
| P1 target | CaveFlyer | 2 | `LAUNCH_MISSING` | absent, but launch forbidden |
| P1 target | CoinRun | 0 | `IDENTITY_BLOCKED` | prior completion reported; current artifact proof unavailable and pair diff fails |
| P1 target | CoinRun | 1 | `NOT_STRICT` | infrastructure-interrupted near 5.53M; no checkpoint |
| P1 target | CoinRun | 2 | `LAUNCH_MISSING` | absent, but launch forbidden |
| Paper RAT | BigFish | 0 | `REUSE_STRICT_COMPLETE` | `1063880_0/1064035`, PASS/rc0, 5,980,160, checkpoint |
| Paper RAT | BigFish | 1 | `REUSE_STRICT_COMPLETE` | `1063880_1/1064039`, PASS/rc0, 5,980,160, checkpoint |
| Paper RAT | BigFish | 2 | `REUSE_STRICT_COMPLETE` | `1063880_2/1064040`, PASS/rc0, 5,980,160, checkpoint |
| Paper RAT | BossFight | 0 | `REUSE_STRICT_COMPLETE` | `1063880_5/1064047`, PASS/rc0, 5,980,160, checkpoint |
| Paper RAT | BossFight | 1 | `REUSE_STRICT_COMPLETE` | `1063880_6/1064063`, PASS/rc0, 5,980,160, checkpoint |
| Paper RAT | BossFight | 2 | `REUSE_STRICT_COMPLETE` | `1063880_7/1064064`, PASS/rc0, 5,980,160, checkpoint |
| Paper RAT | CaveFlyer | 0 | `REUSE_STRICT_COMPLETE` | `1063880_10/1064067`, PASS/rc0, 5,980,160, checkpoint |
| Paper RAT | CaveFlyer | 1 | `REUSE_STRICT_COMPLETE` | `1063880_11/1064069`, PASS/rc0, 5,980,160, checkpoint |
| Paper RAT | CaveFlyer | 2 | `REUSE_STRICT_COMPLETE` | `1063880_12/1064070`, PASS/rc0, 5,980,160, checkpoint |
| Paper RAT | CoinRun | 0 | `REUSE_STRICT_COMPLETE` | `1063880_15/1064074`, PASS/rc0, 5,980,160, checkpoint |
| Paper RAT | CoinRun | 1 | `REUSE_STRICT_COMPLETE` | `1063880_16/1064075`, PASS/rc0, 5,980,160, checkpoint |
| Paper RAT | CoinRun | 2 | `REUSE_STRICT_COMPLETE` | `1063880_17/1064076`, PASS/rc0, 5,980,160, checkpoint |

Counts: 12 `REUSE_STRICT_COMPLETE`, four `IDENTITY_BLOCKED`, four
`NOT_STRICT`, and four nominal `LAUNCH_MISSING` that are not executable under
this task.

## Recovered formal endpoint table

Target endpoints are deliberately not filled from stale summaries. Paper RAT
`KL` is its terminal legacy `kl` column; it is not silently equated to another
method's behavior/current-step KL.

| Environment | Seed | P1 target | Paper RAT reward | Paper RAT KL | loss_v | grad norm | terminal LR |
|---|---:|---|---:|---:|---:|---:|---:|
| BigFish | 0 | unavailable for strict reuse | 14.71 | .0675053 | .00303462 | .398853 | .000759375 |
| BigFish | 1 | incomplete infrastructure root | 25.95 | .0470041 | .00225192 | .316049 | .00170859 |
| BigFish | 2 | missing | 15.76 | .0650962 | .00429101 | .354365 | .000759375 |
| BossFight | 0 | unavailable for strict reuse | 3.14 | .0595716 | .00310638 | .395786 | .0003375 |
| BossFight | 1 | incomplete infrastructure root | 4.09 | .0392247 | .0349485 | .683844 | .0437894 |
| BossFight | 2 | missing | 5.45 | .0838203 | .0149009 | .423447 | .0003375 |
| CaveFlyer | 0 | unavailable for strict reuse | 6.62 | .0663501 | .00121675 | .329990 | .000225 |
| CaveFlyer | 1 | incomplete infrastructure root | 6.05 | .0624100 | .00360524 | .446900 | .0001 |
| CaveFlyer | 2 | missing | 5.90 | .0649466 | .00731770 | .474516 | .0001 |
| CoinRun | 0 | unavailable for strict reuse | 9.40 | .0633845 | .00525618 | .338527 | .0001 |
| CoinRun | 1 | incomplete infrastructure root | 9.90 | .0514257 | .00922671 | .397367 | .00170859 |
| CoinRun | 2 | missing | 9.70 | .0250056 | .000207654 | .344089 | .0291929 |

Because no strict target endpoints are admissible, per-environment target
mean/std/median, paired differences, critic/solver comparison, and all 12
target/Paper reward ratios are `not-evaluable`. No ratio is labeled an
early-stop candidate and no run was cancelled.

## Failure, cancellation and non-result ledger

- P1 seed1 BigFish/BossFight/CoinRun near 5.53M and CaveFlyer near 2.05M
  remain `infrastructure-failure`; stale RUNNING markers and lack of
  checkpoints are preserved. This audit did not overwrite or relaunch them.
- P1 seed0 remains historical completed evidence, but is
  `unknown/insufficient-evidence` for strict reuse in this task because its
  current artifact host cannot be reached.
- Earlier 7a0698-derived RAT/PPO results remain `NOT_STRICT` for original
  Paper RAT and are not relabeled.
- Bede array `1063573` retains its mixed completed/cancelled prior-campaign
  provenance; it is not substituted for the byte-identical original Paper RAT
  array `1063880`.
- CSF3 Paper submissions `17794600` and `18229077` remain
  `cancelled-obsolete-unstarted`, zero-runtime/no-node/no scientific artifact.
- ACTOR_J BossFight seed0 remains `algorithm-failure/EARLY_STOPPED_FAILED`;
  original ACTOR_J BigFish/CaveFlyer/CoinRun and P1 seed1 roots remain
  infrastructure failures.
- CSF3 `18642230`, `18624888`, `18666591` remain cancelled obsolete/unstarted.
  Bede `1072329_0` missing-utils and `1072331_0` CUDA OOM remain
  infrastructure failures; later completion does not erase them.
- The prior CaveFlyer low-Fisher conclusion remains `GUARD_NOT_HELPFUL` and
  no guard appears in this matrix.

No new algorithm, numerical, infrastructure, queued/quota or unknown run
failure was created, because the mandatory precheck stopped before submission.

## Delivery

- Evidence/report commit: `94d4be2e73844f09650712369b391c8f42b36b23`.
- Push target: `origin/agent-work`.
- Delivery HEAD: the follow-up commit containing this record; its SHA and
  remote verification are reported after push in the Executor callback.
- Final worktree: required clean after delivery push.
