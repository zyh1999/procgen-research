# Task-ID: PROCGEN-FULL-SHARED-DETGGN-BLOCKDIAG-BXB-6M-S0-20260826-47

Status: READY

Method: `FULL_SHARED_DETGGN_BLOCKDIAG_BXB_V1`.

Use frozen Task06 commit `da34ce7c7d964765f336ac02111c9fde95aed1ec`
as the sole raw deterministic full-shared Joint-2B parent. Preserve its actor
rows/RHS/ratio, deterministic full-network critic Jacobian/RHS/curvature,
ordered complete parameter space, raw scales, damping `.5`, momentum/history,
global clip, adaptive KL, rollout/GAE/PopArt/schedule/evaluation semantics.

The sole scientific delta is to remove the two dual cross blocks and solve
independent raw actor and critic `B x B` systems, reconstruct both complete
parameter directions, and add them. Keep structural zero columns and ensure the
shared trunk receives both contributions. Never form or solve `AJ^T/JA^T`.

Run only frozen hash/diff checks plus one production construction and
no-training shape/finite solve check: two `512 x 512` FP64/Jacobi/Cholesky
systems, info0, finite residuals/directions, correct shared/policy/value
coverage, and no dual cross solve. On failure, conclude `PRECHECK_BLOCKED`
without repair or retry.

After PASS submit exactly once, in fresh disjoint roots, seed0 intended-6M for
BigFish, BossFight, CaveFlyer and CoinRun. Prefer gpuH after live ownership,
capacity, duplicate and root checks; pending is allowed. Never retry, requeue,
resubmit, sweep or add seeds. Preserve Task45 history and all Task46 cells,
especially live Cave/Coin.

Use only the existing `procgen-3090` automation at 20-minute cadence, updated
in place after job IDs exist. Compare only immutable exact same-env/seed0/eval
Paper rows at first common >=2M, >=4M and 5,980,160; cancel only that cell for
exact Target/Paper <.60. Commit model-free evidence only and callback the
ordinary ChatGPT Planner and coordinator.
