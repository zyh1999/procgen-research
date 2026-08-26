# Task-ID: PROCGEN-FULL-SHARED-JOINT2B-SCALE-RECOVERY-6M-S0-20260826-39

Status: READY

Method: `FULL_SHARED_JOINT2B_SCALE_RECOVERY_V1`

Task38 (`PROCGEN-STANDARD-MSE-GGN-HEAD-D05-6M-S0-20260826-38`) is
`SUPERSEDED_BEFORE_EXECUTION`. Prove that it has no implementation, job, root,
or scientific artifact and do not create any.

Implement exactly one scientific change to the strict full-shared Joint-2B
parent. For the same 512-row minibatch, let `A` be actor Fisher/score rows over
shared+policy parameters (zero value-head columns), `C` the standard value
Jacobian over shared+value parameters (zero policy-head columns), `b_pi` the
parent actor RHS, and `b_v=stopgrad(R_lambda)-V`. Define
`s_pi=||A||_F^2/B`, `s_v=||C||_F^2/B`; require both finite and strictly
positive. Set `Abar=A/sqrt(s_pi)`, `bpi_bar=b_pi/sqrt(s_pi)`,
`Cbar=C/sqrt(s_v)`, `bv_bar=b_v/sqrt(s_v)`, stack `Hbar=[Abar;Cbar]` and
`bbar=[bpi_bar;bv_bar]`, and solve
`(Hbar Hbar^T + 0.5 I_{2B})z=bbar`, reconstructing
`delta=Hbar^T z`. Preserve all cross blocks and full parameter reconstruction.
No floors, clipping-based balance, extra coefficients, environment tuning,
sweep, head-only, sampled-value score, actor weighting, GAE operator, CVLM,
projection, Paper proposal matching, Joint-B, rank reduction, block diagonal,
or cross-zero approximation is allowed.

Keep the strict parent rollout, GAE/lambda return, PopArt, actor definition and
RHS, network, shared trunk/heads, 512 minibatch, four epochs, 6M schedule,
momentum/history, adaptive KL, global clip, evaluation, update order, and
checkpoint protocol unchanged.

Before science, hash and audit `joint2b_diagnosis_20260813.md`, MuJoCo
`export_mujoco_perenv_best.py` and `final_last10_summary.csv`, Procgen strict
Joint-2B/Task06, Task13 and Task37 evidence. Functional preflight must prove:
complete parameter coverage/object identity; exact 1024 rows and all four
direct-reference blocks; full reconstruction with shared actor+critic
contributions and corresponding head contributions; FP64 rescaling invariance
(overall, actor-only, critic-only, opposite scales); normalized block mean
Gram diagonals exactly one with damping 0.5 and spectra/condition/rank; PopArt
affine invariance; symmetric FP64 Jacobi/Cholesky info0, strict residual and
direct reference; and real production-network updates to shared/policy/value
parameters with clip telemetry. Any failure is terminal `PRECHECK_BLOCKED` and
forbids science.

After a complete PASS, commit and push the frozen implementation, refresh live
gpuH ownership/account/QOS/GRES/capacity/duplicates, and submit exactly one
seed0 intended-6M cell for BigFish, BossFight, CaveFlyer and CoinRun to fresh
non-overlapping roots. Never retry, requeue, resubmit, or create a second
configuration. Use the immutable Paper RAT seed0 baseline only at exact common
first >=2M, first >=4M and 5,980,160; cancel only that cell when Target/Paper
is strictly below 0.60. Update the existing `procgen-3090` automation in place;
do not create a second monitor.

Update `.agent/STATE.md`, `.agent/AGENT_REPORT.md`, and
`.agent/reports/PROCGEN-FULL-SHARED-JOINT2B-SCALE-RECOVERY-6M-S0-20260826-39.md`.
Push only model-free code/config/evidence to `origin/agent-work`, verify the
remote SHA, and callback the ordinary ChatGPT Planner and coordinator.

Allowed conclusions: `PRECHECK_BLOCKED`, `QUEUED_RESOURCE_WAIT`,
`RESOURCE_PLACEMENT_BLOCKED`, `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`,
`CANDIDATE_REJECT`, `FULL_SHARED_JOINT2B_SCALE_RECOVERY_SEED0_PROMISING`, or
`CANDIDATE_NOT_READY`.
