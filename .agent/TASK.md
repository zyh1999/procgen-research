# Task-ID: PROCGEN-DETERMINISTIC-JOINT2B-CRITIC-CURVATURE4-DIRECTION-TELEMETRY-2M-S0-20260828-64

Status: SCIENCE_RUNNING_WITH_QUEUED_CELLS

Method: `PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_CURVATURE4_DIRECTION_TELEMETRY_ONLY_V1`

Run one bounded seed0 exact-2,007,040 ablation for BigFish, BossFight,
CaveFlyer and CoinRun on CSF3. The exact scientific parent is Task63/Task06
strict deterministic full-shared Joint-2B. Preserve no warmup, actor rows and
RHS, deterministic critic Jacobian and residual RHS, complete natural cross
blocks, strict 1024-row solve, objective coefficient 1.0, damping .5, Paper
history correction, SGD momentum, global clip and per-minibatch adaptive KL/LR.

The sole scientific change is
`joint_critic_curvature_coef: 0.1 -> 4.0`. Task63-compatible post-inverse
direction telemetry remains read-only and must not alter the installed original
single-RHS update. Run exactly one production gate; on PASS submit all four
fresh one-H200 jobs together exactly once. Never reward-stop, retry, requeue,
resubmit, touch Task63, or copy/commit model/checkpoint bytes or hashes.
