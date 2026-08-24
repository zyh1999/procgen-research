# PROCGEN-PAPER-MATCHED-DETERMINISTIC-GGN-1M-GATE-20260824-06

## Unique conclusion

`GATE_FAIL`

The strict method identity is valid and its numerical solver is healthy, but
the reward gate fails decisively. At the exact same 983,040-transition progress
row, seed0 Target/Paper reward ratios are `.2583` on BigFish, `0` on BossFight,
and `.2188` on CaveFlyer. All three are below the required `.60`; therefore the
3-of-4 success condition is mathematically impossible even if CoinRun were to
pass. The observed failure is `algorithm-failure/step-calibration`, not solver
failure or infrastructure failure.

CoinRun was stopped only under a later explicit user override. This does not
erase its partial artifacts and does not change the gate arithmetic above.

## Exact method definition and identity

The single tested method is `PAPER_MATCHED_DETERMINISTIC_GGN_V1`.

| Role | SHA256 |
|---|---|
| original Paper RAT trainer | `cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7` |
| original Paper RAT config | `1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea` |
| historical P1 solver donor | `2b50f8cc26bb8c85f91e0b394acc7535f5564ac70036403597d3738d3dab9c1b` |
| Target trainer | `41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a` |
| Target config | `69d12937debb8ef8b4531e79b8f9613185b26e4e9056a51e67754129b269391d` |
| diff audit | `5acb70c6b77580ed766414c9d99c5b910fdaca2b115ba054dc57f50dd98451b4` |
| regression test | `0d6e475f42716e4809019faa80a431f8238fd0134b5db8d38e140e9e3a53339b` |
| gpuH bundle launcher | `29987a1f48f3d8df04a0eb4eb9e6179e1d5f82b7fac2a65e5813e1aa75c4ed54` |
| gpuH aggregate preflight | `dfd52c1d18484f8974103d48d6e813d1e1e4a391d53d5605e78156d959caa6e5` |

Original Paper RAT is commit `2b5affd64cbb3c624b4bc1f4767f449df231ffb2`,
formal Bede array `1063880`. Neither its source nor its original artifacts was
modified or rerun.

Exact scientific command per child (with literal environment and seed
substituted by the frozen launcher):

```text
/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python -u train_shared_paper_matched_deterministic_ggn_v1.py --config adv_resnet_shared_paper_matched_deterministic_ggn_v1_1m.yaml --env_name <env>-easy-0-10 --seed <seed> --device 0
```

The gpuH execution environment was PyTorch `2.5.1+cu121`, CUDA `12.1`, driver
`595.71.05`, one H200 per eight-child bundle, eight CPU cores and 188G host
memory. Frozen preflight hashes independently bound the trainer, config,
launcher and compatibility script in every root.

### Machine-auditable Paper to Target diff

Unchanged Paper execution paths:

- shared IMPALA/ResNet hidden256 network and policy/value/PopArt heads;
- actor score, ratios, normalized advantage, entropy and global clip;
- rollout 4096, minibatch 512, four epochs, evaluation and checkpoint rules;
- initial LR `.5`, per-minibatch adaptive-KL checks with thresholds
  `.005/.04`, SGD momentum `1e-6`, and original
  `rhs - H @ momentum_buffer` history correction;
- damping `.5`, global gradient clip `.5`, GAE and reward logging.

The only scientific replacement is inside `Advantage_Update`:

| Field | Paper RAT | Target |
|---|---|---|
| critic rows/RHS | sampled critic score, unit pseudo-advantage | deterministic value Jacobian `J_v`, critic residual |
| critic coefficient | legacy separate system | lambda `.1`, objective coefficient `1` |
| system | two independent B x B inverses | stacked joint-2B, 1024 rows |
| precision/solve | legacy FP32 inverse | symmetric FP64, Jacobi congruence, Cholesky/direct solve |
| additions | legacy metrics | residual, Jacobi, GGN, clip and schedule telemetry |

P1 actor fields `.004` initial LR, rollout-level adaptive KL, momentum0 and
disabled history correction are rejected by config validation and were not
migrated. No second candidate was tested.

### Audit and regression results

Local machine audit:

```text
AUDIT_PASS
paper_trainer_sha256=cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7
paper_config_sha256=1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea
p1_donor_sha256=2b50f8cc26bb8c85f91e0b394acc7535f5564ac70036403597d3738d3dab9c1b
target_trainer_sha256=41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a
target_config_sha256=69d12937debb8ef8b4531e79b8f9613185b26e4e9056a51e67754129b269391d
allowed_scientific_diff=Advantage_Update critic J_v/residual lambda0.1 joint2B FP64 Jacobi Cholesky telemetry
```

CSF3 frozen-environment regression:

```text
REGRESSION_PASS
paper_actor=lr0.5 per_minibatch_KL momentum1e-6 history_rhs_minus_H_buffer
critic=deterministic_residual lambda0.1 joint_rows_2B
solver=FP64_Jacobi_Cholesky relative_residual=2.429e-16
illegal_P1_actor_fields=REJECTED
```

The H200 in-allocation compatibility tests also passed before every scientific
bundle: NVIDIA H200 `150,111,977,472` bytes, eight-child reservation
`124,000,000,000`, headroom `26,111,977,472`, and representative FP64
residuals `6.767e-16`, `6.964e-16`, `6.680e-16` for the first three bundles.

## Planner gate versus explicit user expansion

The Planner-authored matrix was four environments, seed0, 1M. The user later
expanded only execution/scheduling to a gpuL/gpuH race and four gpuH bundles,
each with seeds0--7. This produced 32 logical Target roots. The original gate
is evaluated only from the four seed0 cells; seeds1--7 are supplementary user
evidence and are not represented as Planner-authored.

gpuH began scientific work first. gpuL array `19203054` was then cancelled
while every task had Start=None, elapsed `00:00:00`, no node, no `runs_gpul`
root: `cancelled-race-loser-unstarted`.

## Terminal scheduler and artifact evidence

| Environment | gpuH job | Scheduler | Child evidence | Classification |
|---|---:|---|---|---|
| BigFish | `19203172` | COMPLETED/0:0, node820, 1:35:31 | 8 PASS/rc0; each 7,872 trace rows to 1,007,616, progress at 983,040, checkpoint 3,766,013 bytes | scientific PASS |
| BossFight | `19203173` | COMPLETED/0:0, node820, 1:35:18 | same complete artifact contract for 8 children | scientific PASS |
| CaveFlyer | `19203174` | COMPLETED/0:0, node822, 1:35:17 | same complete artifact contract for 8 children | scientific PASS |
| CoinRun | `19203175` | CANCELLED by 778916, node821, 0:58:17; batch CANCELLED/0:15 | eight stale RUNNING markers, rc absent, copied progress/trace/checkpoint absent; source logs preserved | user-authorized scientific-futility early stop |

CoinRun immutable source logs contain progress at 573,440 and trace endpoints
at 589,824 for six seeds and 593,920 for seeds0/4. All eight had actually
started; none is PASS. Their roots, commands, stdout/stderr, hashes and source
logs remain unmodified.

All 24 completed children and all eight partial CoinRun logs were scanned for
Traceback, standalone NaN/Inf, OOM, CUDA/CUBLAS/NCCL, assertion/runtime, disk
quota/no-space and stall signatures: zero hard-error hits.

## Exact same-transition comparison

Both Target and original Paper rows below are seed0, same environment, same
online episodic-return semantics, and exactly `983,040` transitions. CoinRun
has no Target row at that point because it was stopped earlier.

| Environment | Target reward | Paper reward | ratio | Target KL | Paper KL | Target / Paper LR | Target / Paper loss_v | Judgment |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BigFish | 2.88 | 11.15 | .2583 | .08357 | .05118 | .0001 / .003844 | .02030 / .009832 | below .60 |
| BossFight | 0.00 | 2.72 | 0 | .11615 | .02581 | .0001 / .221684 | .06659 / .04877 | below .60 |
| CaveFlyer | 1.00 | 4.57 | .2188 | .11078 | .06181 | .0001 / .001139 | .04590 / .003969 | below .60 |
| CoinRun | unavailable; stopped at ~.59M | 1.90 | not evaluable | unavailable | .03509 | unavailable / .008650 | unavailable / .008862 | cancelled partial |

The three Target KL values exceed their matched Paper values and all three
Target learning rates have collapsed to the `.0001` floor. This is consistent
with actor step-calibration/adaptive-KL pressure. It is not explained by an
ill-conditioned direct solve: same-row relative residuals are `9.33e-14`,
`6.21e-12`, and `1.46e-13`, with Cholesky info zero.

Every completed seed0 trace ends at 1,007,616 with 7,872 minibatch updates,
7,872 adaptive-KL calls, momentum `1e-6`, history correction active, 1024
joint rows, FP64 Jacobi Cholesky, and finite residuals. Terminal relative
residuals are `1.25e-13`, `1.14e-11`, and `1.32e-13` for the three completed
seed0 cells. The exact Paper-compatible progress schema logs critic `loss_v`
but not an explained-variance column; EV is therefore unavailable rather than
inferred. The matched `loss_v` values are reported in the comparison table.

## User-expanded multi-seed evidence at 983,040

| Environment | Target 8-seed mean / median | Paper RAT 5-seed mean / median | Target rewards | Paper rewards |
|---|---:|---:|---|---|
| BigFish | 6.345 / 4.88 | 9.692 / 9.75 | 2.88, 4.81, 10.96, 11.88, 8.86, 2.71, 4.95, 3.71 | 11.15, 11.43, 9.75, 8.66, 7.47 |
| BossFight | .325 / .06 | 1.914 / 1.87 | 0, .92, 1.37, 0, .02, 0, .10, .19 | 2.72, 1.69, 2.08, 1.21, 1.87 |
| CaveFlyer | .5725 / .34 | 4.584 / 4.57 | 1.00, .28, .40, 0, 2.30, .42, 0, .18 | 4.57, 4.10, 4.00, 5.00, 5.25 |

These aggregates are supplementary robustness evidence only. The causal gate
remains the four seed0 comparisons and rewards are never aggregated across
environments.

## Failure and cancellation ledger

| Evidence | Classification | Preserved reason |
|---|---|---|
| current completed BigFish/BossFight/CaveFlyer cells | `algorithm-failure/step-calibration` for gate value; scientific execution PASS | 3/3 seed0 ratios below .60; finite solver and clean errors |
| CoinRun `19203175`, eight children | `user-authorized scientific-futility early stop` | scheduler CANCELLED after actual training; stale RUNNING markers/absent rc do not mean live or PASS |
| gpuL array `19203054` | `cancelled-race-loser-unstarted` | all Start=None, elapsed0, node none, no root |
| gpuA `19190819`; raw `19201416/19201433/19201447/19190819` | `infrastructure-failure/pre-training-launcher-check` | four node858 FAILED/1:0 in 10--17s; no scientific update/artifact |
| gpuL preflight `19200925` | `infrastructure-failure/preflight-design` | FAILED/1:0, no durable output |
| gpuL preflight `19201660` | `infrastructure-failure/preflight-memory-unit` | genuine 48GB L40S rejected by erroneous 45GiB threshold |
| corrected gpuL preflight `19202370` | compatibility PASS, not scientific result | L40S 47,667,740,672 bytes; residual 6.916e-16 |
| historical P1 seed0 | `unknown/insufficient-evidence` for fresh reuse | host unavailable during formal precheck; historical completion not overwritten |
| historical P1 seed1 four roots | `infrastructure-failure` | host interruption; three near 5.53M, CaveFlyer near 2.05M; no strict checkpoint |
| ACTOR_J BossFight seed0 | `algorithm-failure/EARLY_STOPPED_FAILED` | 5.7933 vs strict E-v2 10.60, ratio .5465 |
| original ACTOR_J BigFish/CaveFlyer/CoinRun | `infrastructure-failure` | original host interruption; later recoveries do not erase provenance |
| `18642230`, `18624888`, `18666591` | `cancelled-obsolete-unstarted` | Start=None, no node, elapsed0, no scientific artifact |
| Bede `1072329_0`, `1072331_0` | `infrastructure-failure` | missing utils; V100 CUDA OOM respectively |
| low-Fisher five-seed gate | `GUARD_NOT_HELPFUL` | one win, three ties, one loss; no reproducible benefit |

No historical failure was deleted, overwritten, relabeled as success, or
silently substituted into this gate.

## Gate decision and evidence boundary

Strict diff and tests passed. At least two environments are below `.60`; in
fact all three completed seed0 comparisons are below. Therefore the only
allowed conclusion is `GATE_FAIL`.

This 1M/partial user-expanded gate is not a 6M formal conclusion. No Paper RAT
baseline was rerun. No second candidate, retry, Jupyter session, quarantined
host access or unrelated gpuH/Isaac mutation occurred.

## Standing user protocol for the next Planner task

- Continue deterministic critic-GGN algorithm search until a RAT-comparable
  run/candidate is obtained.
- Every new candidate test is launched with a 6M intended horizon, not a 1M
  gate.
- Add stage-matched early-stop checks only after at least 2M transitions. If
  target reward is below 60% of original Paper RAT at the same environment,
  seed, evaluation semantics and transition point, cancel that cell and
  preserve configuration/seed/step/baseline/ratio/logs as early-stopped.
- Never compare against RAT 6M terminal when checking an intermediate target.
- Planner owns only algorithm/code/evidence/next unique scientific task.
  Executor owns all live host/GPU/partition/concurrency/queue placement and
  implements the monitor; Planner must not allocate cards.
- Do not use GPU idleness to invent a sweep; no duplicate active READY
  objective.

The Executor must not invent the next method. The same ChatGPT Planner must
return exactly one bounded READY Procgen task before any new experiment.

## Delivery

- Frozen identity commit: `da34ce7c7d964765f336ac02111c9fde95aed1ec`.
- Race package commits: `4bf406dec7619ecbe4e4b120660b8f0895cbd2be`,
  `31db1cb35910afac47121fcb0a2cae04e308a0cd`.
- Evidence/report commit: recorded in the follow-up delivery commit.
- Push target: `origin/agent-work`.
- Delivery HEAD, remote verification and final clean worktree are reported in
  the callbacks after the final push.
