# Task-ID: PROCGEN-FULL-SHARED-JOINT2B-PPO500K-WARMUP-6M-S0-20260826-49

Status: RUNNING_ON_BEDE

Method: `FULL_SHARED_JOINT2B_PPO500K_WARMUP_V1`.

Task48 is `SUPERSEDED_BEFORE_EXECUTION`; no Task48 implementation, job, root,
process, artifact or monitor exists and none may be created.

Use frozen Task06 commit `da34ce7c7d964765f336ac02111c9fde95aed1ec`
as the strict deterministic full-shared Joint-2B parent. Preserve its complete
actor empirical-Fisher score rows/RHS, deterministic full-network critic
Jacobian/GGN/RHS, natural actor-critic cross blocks, full shared reconstruction,
raw scales, damping `.5`, rollout/GAE/PopArt, momentum/history, adaptive KL,
global clip, schedule and evaluation semantics.

The only scientific difference is a fixed standard-PPO warmup through the
complete rollout ending at transition `503,808`, followed by exactly one switch
at the next rollout to the untouched parent Joint-2B update. PPO uses its own
Adam state with LR `.001`, clip `.2`, four epochs, eight minibatches, value
coefficient `1`, entropy coefficient `0`, and global gradient clip `.5`. The
same network, PopArt state, environment/rollout state and RNG continue through
the switch; PPO Adam state is not mapped into the clean parent optimizer.

The user authorized an atomic zero-step migration from CSF3 gpuH to Bede.
CSF3 gate `19441667` was verified PENDING with elapsed zero, no node, process,
root or artifact, then cancelled once and classified
`CANCELLED_FOR_USER_AUTHORIZED_ZERO_STEP_BEDE_MIGRATION`.

The only Bede production gate is `1074924`. It completed `0:0` on gpu006 and
proved construction/device, a real PPO update, one boundary switch and a
finite full Joint-2B solve. Exactly one fresh seed0 intended-6M job for each
BigFish `1074926`, BossFight `1074927`, CaveFlyer `1074928` and CoinRun
`1074929` was then submitted together without dependencies or throttling.
Never retry, requeue, resubmit, sweep or add seeds.

Use exactly one 20-minute Task49 monitor, updated in place for Bede. Compare only immutable exact
same-env/seed0/evaluation Paper rows at first common >=2M, >=4M and 5,980,160;
cancel only the individual cell with exact Target/Paper <.60. Commit model-free
evidence only and callback the ordinary ChatGPT Planner and coordinator.
