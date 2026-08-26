# PROCGEN-FULL-SHARED-JOINT2B-PPO500K-WARMUP-6M-S0-20260826-49

Status: Bede science running; current conclusion `CANDIDATE_NOT_READY`.

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
