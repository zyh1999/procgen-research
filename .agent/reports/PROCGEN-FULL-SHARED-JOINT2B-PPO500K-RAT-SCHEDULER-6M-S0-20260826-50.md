# PROCGEN-FULL-SHARED-JOINT2B-PPO500K-RAT-SCHEDULER-6M-S0-20260826-50

Status: fully terminal on Bede. BossFight and CaveFlyer stopped at exact 2M;
BigFish and CoinRun completed the endpoint. Unique conclusion:
`CANDIDATE_REJECT`.

Method: `FULL_SHARED_JOINT2B_PPO500K_RAT_ROLLOUT_SCHED_V1`.

## Frozen parent and unique difference

Task50 derives only from Task49 implementation
`e0dc2e5ca4efd85419e974e42561eea11145c96f`, trainer SHA256
`4403ef006f53e8647adbcdb829a442037384f623e66eb69573843f21064db28a`
and config SHA256
`e26f66a616b1d0314561a645ef26111da1b15988aad1391d1ef64b6a146d8135`.

The PPO warmup, fixed transition boundary, network, actor empirical Fisher,
deterministic critic GGN, full strict 2B system/cross blocks, damping, solver,
RHS, reconstruction and all rollout/optimizer/evaluation semantics are
unchanged. The sole difference is that Joint SGD is created cleanly at LR
`.004`; one LR is constant across each complete four-epoch rollout and changes
exactly once afterward using full-class behavior-to-final KL with thresholds
`.005/.04`, multiplier `1.5`, and bounds `[1e-4,.5]`.

Frozen hashes, gate evidence, Bede placement, job/root matrix, exact stages and
the unique conclusion will be appended before delivery.

## Frozen Task50 implementation

| Artifact | SHA256 |
|---|---|
| trainer | `35bb29e82f3d72067cc30431f5794902da01cce345d5d0dd0540193a0e362846` |
| config | `1ebd5f591c9693318eb977775553e06e48f27e1d6f2375d6f33889208eebbbe9` |
| Bede gate wrapper | `ce4c580108e22852d106a30bc2d96f7f928599b980124130bbe818ee404b0f48` |
| Bede science wrapper | `22c57618b2fcbe5925fb674b146e8379b292360bfd381b0df4f66476c079b3ea` |
| Task50 stage monitor | `d8d82ef223ae2c67de6b23e1361f18ff65cd75861ec9fe88ae583b9e7796ae08` |

The bounded local checks are limited to Python compile, shell syntax, frozen
parent hashes and inspection of the parent-to-Task50 source/config diff. No
micro, negative, audit or extra preflight chain was added.

Implementation freeze commit is `e4f8cfc23bf406989f72db61ca8aadf5407d99d4`;
model-free cleanup/delivery base is
`3887f2c2aea9aec3a34bb7b844322c552d90d730`. Both were pushed and verified on
`origin/agent-work` before remote execution.

## Sole Bede gate

Fresh placement checks found Bede `gpu` UP, eight idle V100 nodes, account
`bdman37g`, no Task50 duplicate/root, and all four Task49 jobs
`1074926-1074929` still running unchanged. The sole Task50 gate `1075026` ran
on gpu015 and completed `0:0` in `00:02:18`; root is `PRECHECK_PASS/rc0`.

It performed the PPO update and one phase switch with clean Joint optimizer
state and `joint_lr_at_switch=.004`. Its completed Joint rollouts each contain
32 minibatches with one unique LR and zero LR changes. The first rollout used
`.004`, measured exact full-class KL `0.0003129628`, and updated once to `.006`;
the next used `.006`, measured `0.0012396707`, and updated once to `.009`.
Both reasons are the frozen below-`.005` multiply-by-1.5 rule. Behavior hashes
are distinct and preserved.

The strict 1,024-row Joint-2B solves retained nonzero cross blocks (`9.20` and
`12.09` Frobenius), Cholesky info max `0`, finite scan `1`, relative residuals
`8.97e-16` and `1.05e-15`, and finite reconstructed directions. No hard-error,
OOM, CUDA, NCCL, disk/quota or nonfinite signature exists. The gate will not be
rerun.

## Bede science launch

After gate PASS, all four jobs were submitted together once without dependency,
hold or throttle. They immediately occupied the four V100s on gpu016 and each
formed a fresh root with `RUNNING` and `scientific_started.marker`.

| Environment | Job | Exact root | Initial state |
|---|---:|---|---|
| BigFish | `1075028` | `/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_ppo500k_rat_rollout_sched_6m_s0_20260826_50/runs/FULL_SHARED_JOINT2B_PPO500K_RAT_ROLLOUT_SCHED_V1/bigfish-easy-0-10/seed0/6m` | RUNNING gpu016 |
| BossFight | `1075029` | `/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_ppo500k_rat_rollout_sched_6m_s0_20260826_50/runs/FULL_SHARED_JOINT2B_PPO500K_RAT_ROLLOUT_SCHED_V1/bossfight-easy-0-10/seed0/6m` | RUNNING gpu016 |
| CaveFlyer | `1075030` | `/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_ppo500k_rat_rollout_sched_6m_s0_20260826_50/runs/FULL_SHARED_JOINT2B_PPO500K_RAT_ROLLOUT_SCHED_V1/caveflyer-easy-0-10/seed0/6m` | RUNNING gpu016 |
| CoinRun | `1075031` | `/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_ppo500k_rat_rollout_sched_6m_s0_20260826_50/runs/FULL_SHARED_JOINT2B_PPO500K_RAT_ROLLOUT_SCHED_V1/coinrun-easy-0-10/seed0/6m` | RUNNING gpu016 |

The immutable Paper seed0 baseline was copied from Task49 without modification.
The existing automation `monitor-procgen-task49-ppo-warmup` was updated in
place at 20-minute cadence to monitor the Task49 and Task50 sets with separate
frozen monitors and ledgers. No second automation exists.

Final conclusion: `CANDIDATE_REJECT`; the complete terminal matrix is recorded
below.

## Exact 2M stage and bounded BossFight/CaveFlyer archive

The frozen Task50 monitor (SHA256
`d8d82ef223ae2c67de6b23e1361f18ff65cd75861ec9fe88ae583b9e7796ae08`)
was applied exactly once to each eligible failing cell after the immutable
Paper baseline `SHA256SUMS` fully passed. Comparisons use identical
environment, seed0, evaluation/reward semantics and exact transition
`2,007,040`.

| Environment | Job | Target | Paper | Ratio | Decision / scheduler |
|---|---:|---:|---:|---:|---|
| BigFish | `1075028` | 10.48 | 9.28 | 1.1293103448 | PASS; RUNNING |
| BossFight | `1075029` | 0.39 | 2.92 | 0.1335616438 | `EARLY_STOPPED_ALGORITHM`; CANCELLED |
| CaveFlyer | `1075030` | 2.10 | 4.45 | 0.4719101124 | `EARLY_STOPPED_ALGORITHM`; CANCELLED |
| CoinRun | `1075031` | 8.80 | 3.70 | 2.3783783784 | PASS; RUNNING |

BossFight target SHA256 is
`816d02e776ef684dd193424f81c7125f940632844c2a7a006959dd34eed3089e`
and Paper SHA256 is
`4082868eeec196363e284fd7af68807f20bc0142e7de7e8cf355851a5d89337c`.
CaveFlyer target SHA256 is
`8d01a05bd107c46fd68b2003cb2e4c8e3a339bc2f3a9e27ee239afb63fd7b1c7`
and Paper SHA256 is
`8d10f8614a1cb57d81c7705b7d2373c0c9de6b158c7cd1bdeabba2ca8236e292`.
Both monitor invocations returned rc3 and wrote exactly one
`EARLY_STOPPED_ALGORITHM` row to their own root ledgers.

Both jobs were RUNNING on gpu016 at elapsed `02:14:04` immediately before
apply. Scheduler-authoritative terminal state is `CANCELLED by 639800874`,
main exit `0:0`, elapsed `02:14:05`, start `2026-08-26T17:07:42`, end
`2026-08-26T19:21:47`, node gpu016. Residual root `RUNNING` markers and absent
rc files are stale. Neither root contains a checkpoint; no repeat apply,
retry, requeue or resubmit occurred.

Both phase ledgers record PPO through `503,808`, the one Joint switch at
`507,904`, clean Joint optimizer state and switch LR `.004`. At the exact
stage, BossFight recorded actor/critic raw scales `6803.2593/9606.9424`, cross
Frobenius `1774.8640`, entropy `1.06233`, Cholesky info0 and relative residual
`5.5546e-14`. CaveFlyer recorded actor/critic raw scales
`2267.5227/1073.1359`, cross Frobenius `449.7103`, entropy `1.34891`,
Cholesky info0 and relative residual `8.5355e-15`. Finite scans passed and
hard-error scans are zero. Their rollout ledgers preserve constant minibatch
LR per rollout and exactly one post-rollout scheduler update; telemetry was
not used for any additional intervention.

The bounded model-free Git archive contains exact ledgers, complete progress
and rollout-scheduler tables, phase/frozen identities, terminal trace/log
snapshots, scheduler reconciliation and per-file SHA256 manifests. Complete
source traces/logs remain immutable at the Bede roots; no model/checkpoint was
committed. BigFish and CoinRun continued at that stage. Task49 remained
untouched. No retry, requeue or resubmit occurred.

## BigFish and CoinRun endpoint completions

BigFish `1075028` and CoinRun `1075031` are both scheduler-authoritatively
`COMPLETED/0:0`, elapsed `06:13:26`, start `2026-08-26T17:07:42`, end
`2026-08-26T23:21:08`, node gpu016. Both roots are `PASS/rc0`, and each has
an exact endpoint progress row at `5,980,160`.

| Environment | Transition | Target | Paper | Ratio | Decision |
|---|---:|---:|---:|---:|---|
| BigFish | 2,007,040 | 10.48 | 9.28 | 1.1293103448 | PASS |
| BigFish | 4,014,080 | 10.52 | 13.28 | 0.7921686747 | PASS |
| BigFish | 5,980,160 | 10.55 | 14.71 | 0.7171991842 | PASS |
| CoinRun | 2,007,040 | 8.80 | 3.70 | 2.3783783784 | PASS |
| CoinRun | 4,014,080 | 9.40 | 8.00 | 1.175 | PASS |
| CoinRun | 5,980,160 | 9.50 | 9.40 | 1.0106382979 | PASS |

Both phase ledgers record exactly one switch from PPO at `503,808` to Joint-2B
at `507,904`, with clean Joint optimizer state and LR `.004`. In the final
BigFish rollout, LR stayed `.004` across all minibatches, behavior-final KL
was `.0064782`, the reason was `KL_IN_BAND_UNCHANGED`, and the scheduler update
count was exactly one. Its actor/critic raw scales were
`4586.7227/6113.5381`, natural cross-block Frobenius norm was `179.1353`,
direction norm `.507031`, Cholesky info0, finite scan1 and relative residual
`4.3111e-15`.

In CoinRun's penultimate rollout, KL `.1081465` triggered the single frozen
`.004 -> .0026666667` update. The final rollout then used only
`.0026666667`, recorded zero within-rollout LR changes, KL `.0175938`, reason
`KL_IN_BAND_UNCHANGED`, and exactly one scheduler update. Its actor/critic raw
scales were `5984.0869/22317.7539`, cross Frobenius `446.7604`, direction norm
`.470891`, Cholesky info0, finite scan1 and relative residual `1.3187e-14`.
Both hard-error scans are zero.

Each root contains `model.ckpt` as a regular non-symlink file of `3,766,013`
bytes, mode `664`, one link. Only stat metadata was recorded; checkpoint
contents were not copied, hashed, modified or committed.

## Final Task50 matrix and conclusion

| Environment | Terminal stage | Effective ratio | Classification |
|---|---:|---:|---|
| BigFish | 5,980,160 | 0.7171991842 | COMPLETED / endpoint PASS |
| BossFight | 2,007,040 | 0.1335616438 | `EARLY_STOPPED_ALGORITHM` |
| CaveFlyer | 2,007,040 | 0.4719101124 | `EARLY_STOPPED_ALGORITHM` |
| CoinRun | 5,980,160 | 1.0106382979 | COMPLETED / endpoint PASS |

The four-environment effective-ratio mean is `0.5833273096`. Only two
environments reached the endpoint, two cells were algorithm early stops, only
one endpoint strictly exceeded Paper, and the mean ratio is below one.
Therefore Task50 fails the multi-environment promising criteria and its unique
terminal conclusion is `CANDIDATE_REJECT`.

Task49 delivery `e36750423ff48bfdfc718c6607465a4dd16fe839` was already
verified and remains untouched. Task49 and Task50 are now both fully terminal,
so the sole automation `monitor-procgen-task49-ppo-warmup` may be deleted after
this delivery is verified. No retry, requeue, resubmit, successor or model-byte
action occurred.
