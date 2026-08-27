# Task55 no-warmup fixed-LR dual-trust quick diagnostic

Task-ID: `PROCGEN-FULL-SHARED-JOINT2B-NOWARMUP-FIXEDLR-DUALTRUST-BETA1-BETA4-BOSS-CAVE-2M-S0-20260827-55`

## Frozen identity

- Parent Task51 implementation: `a9e27fd267aa844f4ea4d140877da7a7417ee5b7`
- Task55 implementation: `3a850cd3870854123c76693a974a2fe45e952203`
- Task55 trainer: `91b835f16989a42293f6566d8fb9893dcd7b9ca969d1685d2d313f3f695f2f81`
- beta1 config: `75fb59290d4bd2399986e372a62e56b4aaa6df7becb205f41ee332538f04425f`
- beta4 config: `f27585bd3a59c0ee67be34f631cbdb9dabaa01cf1342ef56ac4ea3aa1c9bd0b7`
- gate wrapper: `23f2983cfd50596b71625cbdadcaa51379ddd6bffdc60df26f9fdfc7d0d79bab`
- science wrapper: `b348fb06e4ade9bae39762495a4d78ca16893dc9a241adf83636a1c32ac278c4`
- read-only endpoint monitor: `c71a5528e480bb351f1666436de761735ebf89db07db76ff196ffdb3652ad5df`

The only scientific delta from Task51 is removal of the 503,808-transition
PPO warmup. `ppo_warmup_transitions=0`, and the clean Joint SGD path exists at
rollout zero. No PPO update or phase switch occurs. Strict full-shared
Joint-2B, both natural cross blocks, LR `.004`, dual trust, beta1/beta4,
`eta_min=1/64`, damping, clipping, PopArt, objective and evaluation semantics
are otherwise preserved.

## Minimal gate

The single Bede gate was job `1075104`, account `bdman37g`, partition `gpu`,
node gpu029. Scheduler state is `COMPLETED/0:0`, elapsed `00:02:50`. Both
beta roots are `PRECHECK_PASS/rc0` and establish:

- the first and only gate rollout uses `training_phase=joint2b`;
- no `ppo_warmup` row and no nonempty phase-switch ledger exists;
- fixed parameter LR is `.004`, with zero within/between-rollout LR changes;
- actor and critic trust update counts are exactly one;
- strict system identity is `1024` rows and `938976` columns;
- actor-critic and critic-actor natural cross blocks are present;
- Cholesky info is zero, residuals are finite and finite scans pass.

Model-free gate evidence is archived under the Task55 staging `evidence/gate`
directory. No model/checkpoint bytes are included.

## Launch

Campaign:
`/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_nowarmup_fixedlr_dualtrust_beta1_beta4_boss_cave_2m_s0_20260827_55`

All four jobs were submitted once in one bounded action after a fresh Bede
capacity, duplicate and root-absence check. No dependency, hold, throttle,
retry, requeue or resubmit was used.

| arm | environment | job | initial state | node | exact root |
|---|---|---:|---|---|---|
| beta1 | BossFight | 1075105 | RUNNING | gpu029 | `runs/FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_BETA1_V1/bossfight-easy-0-10/seed0/2m` |
| beta1 | CaveFlyer | 1075106 | RUNNING | gpu030 | `runs/FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_BETA1_V1/caveflyer-easy-0-10/seed0/2m` |
| beta4 | BossFight | 1075107 | RUNNING | gpu031 | `runs/FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_BETA4_V1/bossfight-easy-0-10/seed0/2m` |
| beta4 | CaveFlyer | 1075108 | RUNNING | gpu031 | `runs/FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_BETA4_V1/caveflyer-easy-0-10/seed0/2m` |

Each root has `RUNNING`, a trainer PID, hostname and live trace. Initial traces
show Joint2B from the beginning, phase-switch count zero, PPO optimizer state
zero, fixed LR `.004`, nonzero cross blocks, Cholesky info0 and finite strict
residuals. Targeted Traceback/OOM/CUDA/NCCL/disk/quota/NaN/Inf scans are zero.

## Monitoring and conclusion

The existing `monitor-procgen-task51-dual-trust` automation was updated in
place at 20-minute cadence. Task55's exact 2,007,040 endpoint comparison is
read-only: the frozen monitor records Paper and matched Task51/Task52 evidence
but never cancels a scheduler job. Task55 cannot replace the authoritative
Task51 matrix.

Current bounded conclusion: `QUEUED_RESOURCE_WAIT` is not applicable because
all four jobs are RUNNING. Scientific conclusion remains pending exact 2M
terminal evidence.
