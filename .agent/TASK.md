Status: READY

# TASK.md

Task-ID: PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-6M-S0-20260825-34R

## Task replacement

This task fully replaces the unexecuted Task34. Do not implement Task34's same-minibatch ared/pred rule: for a linear value head with frozen features and targets, same-minibatch MSE is exactly quadratic and ared/pred is identically 1, so it cannot calibrate damping.

Task32 and Task33 jobs, roots, artifacts and monitors remain unchanged.

## Unique objective

Implement and test DET_STANDARD_MSE_GGN_HEAD_CVLM_V1: a standard per-sample deterministic critic GGN whose critic objective remains ordinary frozen lambda-return MSE. The sole new mechanism is deterministic cross-minibatch LM calibration of spectrum-relative damping.

## Frozen control identity

Keep Task13/Paper control semantics unchanged for actor, shared-trunk sampled critic, rollout, GAE generation of frozen lambda-return, PopArt, schedule, minibatches, epochs, momentum/history, adaptive KL, global clipping, network, seed, evaluation and stopping. Only replace the update on the 257 critic-exclusive last_v_layer.weight/bias parameters.

## Standard per-sample critic GGN

In frozen PopArt-normalized coordinates:

  e = V_theta - stopgrad(R_lambda)
  D = I
  W = I
  L_v = ||e||^2 / (2B)
  J = dV/dtheta_h
  G = J^T J / B
  g = J^T e / B

Gaussian precision is exactly 1. Do not add Task13's .1 curvature coefficient, actor weighting, a GAE temporal operator, Paper proposal/RHS matching, or any hidden scaling.

Solve:

  (G + mu I) u = -g

with symmetric FP64, Jacobi equilibration and Cholesky.

## Non-degenerate cross-minibatch LM

Retain the original frozen shuffle and all eight complete 512-row minibatches M_0,...,M_7 in every rollout/epoch. For current train block M_i use:

  T_i = M_i
  C_i = M_(i+1 mod 8)

G_i and g_i use all 512 rows of T_i. C_i never enters that trial's solve, remains in the original schedule, and later receives its own complete update. Do not delete, halve, reweight or resample any minibatch. Since D=I, this split creates no cross-episode temporal relationship; episode boundaries were used only when R_lambda was frozen.

Let:

  s_i = trace(G_i) / 257
  mu_i = alpha * max(s_i, epsilon_fp64)

Alpha starts at 1 and is the only persistent LM state.

For each trial:

1. Construct G_i, g_i and u_i from complete T_i.
2. Pass the proposal through the existing head momentum/history and global-clip chain to obtain the actual candidate head change Delta_i.
3. Compute pred_T = -(g_i^T Delta_i + 0.5 Delta_i^T G_i Delta_i).
4. With features, targets and PopArt frozen, temporarily apply only Delta_i and compute ared_C = L_Ci(theta) - L_Ci(theta + Delta_i).
5. Use rho_cv = ared_C / pred_T. Also verify ared_T == pred_T in FP64, but never use that degenerate equality for acceptance.

Decision:

- pred_T <= 0, nonfinite, or rho_cv < .25: reject and alpha <- 4 alpha.
- .25 <= rho_cv <= .75: accept and leave alpha unchanged.
- rho_cv > .75: accept and use alpha <- alpha/2 on the next minibatch.

Allow at most four deterministic trials per minibatch. If all fail, make that head delta zero while committing the actor/shared control update exactly once. Clamp alpha only for numerical safety to [2^-20, 2^20]. This is one fixed algorithm, not an experiment sweep, and no reward/Paper metric may affect alpha or acceptance.

## Rollback and commit semantics

Before every trial snapshot all parameters, optimizer and momentum/history, PopArt, RNG, adaptive-KL and global-clip-related state. A rejected trial must restore every item bit-identically and must not advance any counter or schedule. An accepted head delta must be generated solely from the complete T_i solve; validation rows do not enter it. Actor/shared control updates are committed exactly once and must remain bit-identical to control.

## Mandatory historical audit

Read and align frozen Task07, Task13 and Task32 code/evidence. Derive and numerically verify, rather than infer from config labels:

- objective, 1/B, sign and Gaussian precision;
- Task13 .1 curvature/RHS scaling and its effective standard-coordinate damping;
- fixed .5 relative to each environment's G spectrum and PopArt coordinates;
- ordinary MSE gradient, raw solve and final delta norms/cosines;
- momentum/global-clip effects;
- predicted versus realized reduction;
- coupling to the unchanged shared sampled critic and adaptive-KL path.

## Mandatory preflight

Prove on actual network/data:

- D=I, W=I, K=J and exact standard MSE gradient;
- T_i always has all 512 rows; C_i is disjoint from current T_i but retained in schedule;
- ared_T/pred_T = 1 within FP64 tolerance, documenting the rejected degenerate signal;
- fixed opposing train/validation-gradient cases cause nontrivial LM acceptance and rejection;
- rejected trials roll back bitwise;
- accepted update uses only complete train rows;
- actor/shared directions, deltas and policy logits are bit-identical to control;
- only the 257 head parameters differ scientifically;
- PopArt affine reward-scale regression passes;
- Cholesky info is 0, residual is finite, and no NaN/Inf exists.

Any preflight failure is PRECHECK_BLOCKED and forbids scientific launch.

## Scientific matrix

After preflight PASS, run only BigFish, BossFight, CaveFlyer and CoinRun, seed0, one independent intended-6M cell each, using new verified-absent roots. Do not start seeds1-2. Executor owns all live resource, scheduler, GPU, partition, capacity, concurrency and queue placement decisions. Do not cancel, modify, overwrite or wait for Task32/Task33.

## Exact comparison and early stop

Compare only identical environment, seed0, evaluation and reward semantics at exact common transitions 2,007,040, 4,014,080 and 5,980,160. Cancel only that cell when Target/Paper < .60 and preserve EARLY_STOPPED_ALGORITHM evidence. No exact common row means no action; never compare an intermediate Target with Paper terminal.

## Required telemetry

At each stage record Target/Paper reward and ratio; KL, LR, entropy, MSE, TD error and GAE statistics; PopArt mean/std; G spectrum, trace, condition and effective rank; alpha, mu, trials and decisions; pred_T, ared_T, ared_C and rho_cv; gradient/raw-solve/final-delta norms and cosines; momentum/global-clip changes; validation loss and prediction change; solver residual/info and hard-error scans.

## Allowed conclusions

- PRECHECK_BLOCKED
- QUEUED_RESOURCE_WAIT
- CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE
- CANDIDATE_REJECT
- STANDARD_GGN_CVLM_SEED0_PROMISING
- CANDIDATE_NOT_READY

STANDARD_GGN_CVLM_SEED0_PROMISING requires at least three environments at 5,980,160, at most one algorithm early stop, at least two endpoint rewards above Paper, four-environment mean ratio above 1 with an early-stopped cell counted at its acted-on stage, and healthy LM/numerical evidence.

## Prohibitions

No GAE temporal operator, actor weighting, sampled-score/Paper matching, joint/cross system, NormMatch, low-Fisher guard, second candidate, external hyperparameter sweep, environment/reward tuning, Paper rerun, Task14-31 origin-observer work, Jupyter, quarantined .54/ws4090-31/10.49.7.54 access, MuJoCo or Isaac work. Planner must not specify hardware placement.

## Reporting

Update .agent/STATE.md, .agent/AGENT_REPORT.md and .agent/reports/PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-6M-S0-20260825-34R.md. Include frozen hashes, historical scaling audit, Task13-to-target diff, preflight, jobs/roots, exact-stage results, failure ledger and one allowed conclusion. Commit and push model-free code/config/diff/report/table/log only, never model/checkpoint, and verify origin/agent-work before callback.
