# Task-ID: PROCGEN-DETERMINISTIC-JOINT2B-CRITIC-RHS40-DIRECTION-TELEMETRY-2M-S0-20260828-65

Status: SCIENCE_RUNNING

Method: `PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_RHS40_DIRECTION_TELEMETRY_ONLY_V1`

Run one bounded seed0 exact-2,007,040 ablation for BigFish, BossFight,
CaveFlyer and CoinRun on CSF3. The exact parent is frozen Task63/Task06 strict
deterministic full-shared Joint-2B with Task63 read-only post-inverse direction
telemetry.

Preserve `joint_critic_curvature_coef=0.1`, so
`H_C=sqrt(0.1) J_C` remains unchanged. The sole scientific change is
`joint_critic_objective_coef: 1.0 -> 40.0`, making the actual critic RHS
multiplier `40/sqrt(0.1)=126.49110640673517`, exactly forty times the parent
`1/sqrt(0.1)=3.1622776601683795`. Preserve no warmup, adaptive KL/LR, history,
natural cross blocks, strict 1024 rows, damping, clip, actor, seed0,
evaluation/reward, and exact horizon.

Run exactly one real production gate and, only on PASS, submit four fresh
normal one-H200 jobs together exactly once. Do not cancel or modify Task64 Coin
or any unrelated job. No retry/requeue/resubmit, reward early stop, second
candidate, or model/checkpoint content/hash in Git.
