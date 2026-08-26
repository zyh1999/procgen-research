# PROCGEN-FULL-SHARED-JOINT2B-PPO500K-WARMUP-6M-S0-20260826-49

Status: implementation and minimal gate in progress.

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

## Sole production gate and placement

Campaign:
`/scratch/h99859yz/procgen_full_shared_joint2b_ppo500k_warmup_6m_s0_20260826_49`

The sole gate was submitted once as job `19441667` on gpuH using account
`gpu-h200-fse-pgdr`, QOS `gpu-h200-fse`, one H200, eight CPUs and 180G memory.
It is PENDING with reason `AssocGrpGRES`, elapsed zero, node none and predicted
start unknown. This is an account-level gpuH quota wait, not a code/preflight,
GPU or scientific failure. The user has no compatible alternate gpuH account;
`gpu-aifun` is limited to A100 resources and was not substituted.

Task49 science roots remain absent, no Task49 trainer process or duplicate job
exists, and no science cell has been submitted. The gate was not retried,
requeued, resubmitted or moved. Queue evidence and frozen hashes are preserved
under `evidence_gate_queue`.

The unique 20-minute automation `monitor-procgen-task49-ppo-warmup` was created
by the coordinator and currently binds only gate `19441667`; it will be updated
in place after science IDs exist. No second automation was created.

Current bounded conclusion: `QUEUED_RESOURCE_WAIT`.
