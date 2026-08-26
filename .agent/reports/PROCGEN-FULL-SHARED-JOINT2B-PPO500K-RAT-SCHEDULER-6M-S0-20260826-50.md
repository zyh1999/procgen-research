# PROCGEN-FULL-SHARED-JOINT2B-PPO500K-RAT-SCHEDULER-6M-S0-20260826-50

Status: Bede science running; current conclusion `CANDIDATE_NOT_READY`.

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

Current conclusion: `CANDIDATE_NOT_READY` pending exact 2M/4M/endpoint stages
and terminal artifact verification.
