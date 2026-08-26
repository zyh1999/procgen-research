# PROCGEN-FULL-SHARED-JOINT2B-PPO500K-WARMUP-6M-S0-20260826-49

Status: Bede science running with Cave exact-2M and BigFish exact-4M algorithm
early stops; current
conclusion `CANDIDATE_NOT_READY`.

Method: `FULL_SHARED_JOINT2B_PPO500K_WARMUP_V1`.

## Supersession

Task48 is `SUPERSEDED_BEFORE_EXECUTION`. Bounded local and CSF3 searches found
no Task48 implementation, config, report, job, root, process, transition,
artifact or monitor. No Task48 state was modified or created.

## Frozen parent and unique scientific diff

The strict parent is Task06 commit
`da34ce7c7d964765f336ac02111c9fde95aed1ec`, trainer SHA256
`41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a`
and config SHA256
`69d12937debb8ef8b4531e79b8f9613185b26e4e9056a51e67754129b269391d`.

The sole scientific change is standard Paper-matched PPO through the 123rd
complete 4,096-transition rollout (`503,808`), then one switch at the next
rollout to the complete original deterministic full-shared strict Joint-2B.
The PPO identity is Adam LR `.001`, clip `.2`, epochs `4`, minibatches `8`,
value coefficient `1`, entropy coefficient `0`, max gradient norm `.5`, using
the same network/rollout/GAE/PopArt state. Joint-2B optimizer/history starts
clean and does not inherit PPO Adam moments.

Further gate, frozen hashes, placement, jobs, roots, stage results and final
conclusion will be appended by the Executor.

## Frozen implementation and local checks

Implementation commit: `e0dc2e5ca4efd85419e974e42561eea11145c96f`.

| Artifact | SHA256 |
|---|---|
| trainer | `4403ef006f53e8647adbcdb829a442037384f623e66eb69573843f21064db28a` |
| config | `e26f66a616b1d0314561a645ef26111da1b15988aad1391d1ef64b6a146d8135` |
| gate launcher | `081e45c51788c67af901f90bd04006d9ec96e92c98a968dab17d634a5b00f98e` |
| science launcher | `ba0b89440e35f369f297efcca4a69e6d76761e58ca67b6c2fc08109b08c1c873` |
| stage monitor | `4cd6dd6d343a7e79c5b3a49d58d1560c70c5b75330236017991b5d3531148ab5` |

The local bounded checks passed: trainer/monitor compile, config identity,
launcher shell syntax and frozen hashes. No micro/negative/audit chain was
added. The implementation records every minibatch phase, writes exactly one
switch event, asserts the parent optimizer has zero state entries before its
first step, and retains parent Joint-2B Cholesky/residual/direction telemetry
plus raw actor/critic scales and natural cross-block norm.

## Authorized CSF3 to Bede migration

Campaign:
`/scratch/h99859yz/procgen_full_shared_joint2b_ppo500k_warmup_6m_s0_20260826_49`

Before migration, CSF3 job `19441667` was reverified as owned by `h99859yz`,
PENDING with elapsed `00:00:00`, start unknown, node none, and no gate root,
science root, trainer process or artifact. It was then cancelled exactly once.
Final sacct is `CANCELLED by 778916`, exit `0:0`, elapsed zero, node none. The
preserved classification is
`CANCELLED_FOR_USER_AUTHORIZED_ZERO_STEP_BEDE_MIGRATION`; it will never be
restored, retried, requeued or resubmitted.

Bede live refresh found partition `gpu` UP with a two-day limit, 15 idle full
V100 nodes, account `bdman37g`, and no Task49 duplicate. The existing unrelated
`1074901_*` MuJoCo allocations were left untouched. Deployment uses the
Bede-native PPC64LE Procgen environment
`/nobackup/projects/bdman37/yihe/ppc64le/envs/procgen_author/bin/python` and
appends the frozen bundle code root to `PYTHONPATH`; trainer/config bytes and
the normalized science command are unchanged.

| Bede deployment artifact | SHA256 |
|---|---|
| gate wrapper | `27d72ffbb33ec3938e66ce172e25d2f9e9cca7a79d4a46515e8dbf0c938e54f9` |
| science wrapper | `bb44b4cce9c485a03c7491341ef5b4c2992935de113a59560d3cf7e48528f9df` |
| unchanged stage monitor | `4cd6dd6d343a7e79c5b3a49d58d1560c70c5b75330236017991b5d3531148ab5` |

Two scheduler submission probes were rejected before job creation because
Bede's `DefCpuPerGPU=32` disallows a one-task explicit CPUs-per-task request.
No gate code ran and no gate root was created. The wrapper was aligned with the
verified Bede-native request (`nodes=1`, `gres/gpu:1`, default 32 CPUs), after
which the single actual gate job was submitted.

## Bede minimal gate

Gate job `1074924` ran on gpu006 and completed `0:0` in `00:01:34`; root is
`PRECHECK_PASS/rc0`. It constructed the production network/device, performed a
real PPO update, recorded one phase switch, then completed a full strict
Joint-2B solve. The acted solve records 1,024 system rows, nonzero natural
cross-block Frobenius norm `106`, Cholesky info max `0`, finite scan `1`, solve
residual `1.97e-13`, relative residual `3.17e-15`, and finite reconstructed
direction. This is the only actual Bede gate and it will not be rerun.

## Bede science launch

After gate PASS, all four jobs were submitted in one bounded launch action,
without dependency, hold or throttle. Every root was absent beforehand.

| Environment | Job | Exact root | Initial scheduler |
|---|---:|---|---|
| BigFish | `1074926` | `/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_ppo500k_warmup_6m_s0_20260826_49/runs/FULL_SHARED_JOINT2B_PPO500K_WARMUP_V1/bigfish-easy-0-10/seed0/6m` | RUNNING gpu006 |
| BossFight | `1074927` | `/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_ppo500k_warmup_6m_s0_20260826_49/runs/FULL_SHARED_JOINT2B_PPO500K_WARMUP_V1/bossfight-easy-0-10/seed0/6m` | RUNNING gpu006 |
| CaveFlyer | `1074928` | `/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_ppo500k_warmup_6m_s0_20260826_49/runs/FULL_SHARED_JOINT2B_PPO500K_WARMUP_V1/caveflyer-easy-0-10/seed0/6m` | RUNNING gpu006 |
| CoinRun | `1074929` | `/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_ppo500k_warmup_6m_s0_20260826_49/runs/FULL_SHARED_JOINT2B_PPO500K_WARMUP_V1/coinrun-easy-0-10/seed0/6m` | RUNNING gpu007 |

All four actual roots have `RUNNING`, job ID, host/GPU, command,
trainer PID and `scientific_started.marker`; initial PPO minibatches are active
and hard-error scans are clean.

The immutable Paper RAT seed0 baseline was copied model-free, verified with its
original `SHA256SUMS`, and made read-only under the Bede campaign. The existing
automation `monitor-procgen-task49-ppo-warmup` was updated in place to these
four jobs/roots at 20-minute cadence. No second automation exists.

Current bounded conclusion: `CANDIDATE_NOT_READY` pending exact 2M/4M/endpoint
stages and terminal artifact verification.

## Exact 2M stage and CaveFlyer bounded archive

The sole frozen Task49 monitor (SHA256
`4cd6dd6d343a7e79c5b3a49d58d1560c70c5b75330236017991b5d3531148ab5`)
reconciled the first exact common transition at `2,007,040`. All immutable
Paper `SHA256SUMS` entries were verified immediately before the decisions.

| Environment | Job | Target | Paper | Ratio | Decision / scheduler |
|---|---:|---:|---:|---:|---|
| BigFish | `1074926` | 8.64 | 9.28 | 0.9310344828 | PASS; RUNNING |
| BossFight | `1074927` | 1.77 | 2.92 | 0.6061643836 | PASS (strictly above `.60`); RUNNING |
| CaveFlyer | `1074928` | 0.00 | 4.45 | 0.0 | `EARLY_STOPPED_ALGORITHM`; CANCELLED |
| CoinRun | `1074929` | 9.00 | 3.70 | 2.4324324324 | PASS; RUNNING |

For CaveFlyer, the exact target artifact SHA256 is
`5480372a0522a9803225de4004ca77990fcba4b1e1b08d3e39dad14c4ff5d819`
and the immutable Paper row SHA256 is
`8d10f8614a1cb57d81c7705b7d2373c0c9de6b158c7cd1bdeabba2ca8236e292`.
The monitor wrote one `EARLY_STOPPED_ALGORITHM` ledger row and returned rc3.
Scheduler state immediately before apply was RUNNING on gpu006; authoritative
terminal sacct is `CANCELLED by 639800874`, main exit `0:0`, elapsed
`01:56:15`, start `2026-08-26T15:25:13`, end
`2026-08-26T17:21:28`, node gpu006. The root's residual `RUNNING` marker and
absent rc are stale and are not interpreted as a live job. No checkpoint is
present and no repeat apply, retry, requeue or resubmit occurred.

The phase ledger proves PPO ended at transition `503,808`, the first Joint-2B
rollout began at `507,904`, switch count is exactly one, and Joint history
started clean. At the exact stage, Cholesky info was `0`, finite scan passed,
the applied solve residual was `1.6790261e-12`, relative residual was
`9.0928192e-15`, actor raw scale was `2132.2119`, critic raw scale was
`2692.6221`, natural cross-block Frobenius norm was `103.3706`, direction norm
was `0.538161`, entropy was `0.392074`, and no hard-error signature matched.
The bounded model-free archive includes the exact ledger, progress/trace and
phase evidence, command/frozen identity, stdout/stderr snapshots, artifact
hashes and scheduler before/after reconciliation; model/checkpoint bytes are
excluded.

At this 2M archive point, BigFish, BossFight and CoinRun remained live and the
independent Task50 cells were unchanged. The existing sole 20-minute
automation continued; no second automation was created.

## Exact 4M BigFish early stop

BigFish `1074926` retained its exact-2M PASS ledger
(`8.64/9.28=.9310344828`) and reached the next exact common transition at
`4,014,080`. Its Target was `6.42` versus immutable Paper `13.28`, ratio
`.48343373493975905`. Target SHA256 is
`f7ee14e23fecc690dbc3641c140b52df177879251089ca51b489f798048f4927`
and Paper SHA256 is
`caf19809e208f35b8f8bcb41266021d07a6d8ae28f8e1e21d5111268a35961ba`.
The baseline `SHA256SUMS` passed, the frozen Task49 monitor was applied once,
wrote the 4M `EARLY_STOPPED_ALGORITHM` ledger row, and returned rc3.

Scheduler state before apply was RUNNING on gpu006 at elapsed `04:14:32`.
Authoritative terminal state is `CANCELLED by 639800874`, main exit `0:0`,
elapsed `04:14:32`, start `2026-08-26T15:25:13`, end
`2026-08-26T19:39:45`, node gpu006. Root `RUNNING` and absent rc are stale;
there is no checkpoint. No repeat apply, cancellation, retry, requeue or
resubmit occurred.

At the exact stage, actor/critic raw scales were
`11696.1465/5193.6499`, natural cross-block Frobenius norm was `309.3334`,
direction norm was `.493082`, entropy was `.606171`, predicted KL was
`5.6418e-7`, Cholesky info was `0`, finite scan passed and relative residual
was `5.8418e-15`. Hard-error scan was zero. The bounded Git archive preserves
both exact-stage ledger rows, complete progress, phase/frozen identity,
terminal trace/log snapshots, scheduler reconciliation and file hashes;
complete source logs/trace remain at the immutable Bede root and no
model/checkpoint bytes were committed.

BossFight passed exact 4M (`3.92/3.45=1.1362`) and CoinRun passed exact 4M
(`9.50/8.00=1.1875`); both remain RUNNING. CaveFlyer remains the archived 2M
algorithm stop. Task50 Boss/Cave remain archived 2M stops while Task50
BigFish/Coin continue RUNNING. No live cell was touched, and the sole
20-minute automation remains active.
