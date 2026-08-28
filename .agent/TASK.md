# Task-ID: PROCGEN-DETERMINISTIC-JOINT2B-CRITIC-RHS10-DIRECTION-TELEMETRY-2M-S0-20260829-66

Status: SCIENCE_RUNNING_WITH_QUEUED_CELLS

Method: `PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_RHS10_DIRECTION_TELEMETRY_ONLY_V1`

Run one bounded seed0 exact-2,007,040 ablation for BigFish, BossFight,
CaveFlyer and CoinRun on Bede. The exact parent is frozen terminal Task65.

Preserve `joint_critic_curvature_coef=0.1`, so
`H_C=sqrt(0.1) J_C` remains unchanged. The sole scientific change is
`joint_critic_objective_coef: 40.0 -> 10.0`, making the actual critic RHS
multiplier `10/sqrt(0.1)=31.622776601683793`. Preserve no warmup, adaptive
KL/LR, history, natural cross blocks, strict 1024 rows, damping, clip, actor,
seed0, evaluation/reward, exact horizon, post-inverse telemetry and aggregator
semantics.

Run exactly one real Bede production gate and, only on PASS, submit four fresh
one-V100 jobs together exactly once. No retry/requeue/resubmit, dependency,
hold, throttle, reward early stop, online tuning, or Task62--65 mutation. Git
contains model-free evidence only and never model/checkpoint bytes or content
hashes.
