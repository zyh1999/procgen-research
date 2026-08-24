Status: READY
Task-ID: PROCGEN-PAPER-HYBRID-HEAD-DETGGN-6M-S0-20260824-08

## Evidence Interpretation

`PAPER_MATCHED_SEPARATE_B_DET_GGN_V1` failed because removing joint/cross coupling did
not remove the harmful effect of deterministic critic-GGN on shared parameters:

- Paper actor source and actor direction were strictly preserved;
- critic and actor matrices were already separate B×B systems;
- FP64/Jacobi/Cholesky residuals remained approximately `1e-15`–`1e-13`;
- hard-error scans were clean;
- nevertheless BigFish, BossFight and CaveFlyer reached only `.347`, `.017` and
  `.164` of strict Paper reward at exact 2,007,040;
- CoinRun survived to6M, but its ratio narrowed from`1.865` at2M to`.681` at6M,
  while LR also reached`.0001`.

Thus this is not a solver failure and not solely a joint-system failure. The remaining direct
causal channel is the deterministic critic direction on the shared encoder/trunk: those
parameter updates alter policy logits and future advantages even when the explicit actor
solve is unchanged. CoinRun appears unusually tolerant of that interference.

## 唯一科学目标

Construct and test exactly one candidate:

`PAPER_HYBRID_SHARED_PAPER_HEAD_DETGGN_V1`

Retain the original Paper RAT critic direction on every shared encoder/trunk parameter.
Replace only the update of trainable critic-exclusive value-head parameters with an
independent deterministic GGN direction.

This falsifies the shared-trunk-interference hypothesis:

- success means deterministic GGN is viable when confined to parameters that cannot
  directly alter policy logits;
- failure means even critic-head-only deterministic curvature and its downstream value/
  advantage effects are not RAT-comparable under the preserved Paper schedule.

Every cell has a6M intended horizon. No second candidate or sweep is authorized.

## Frozen Algorithm Definition

Start from exact original Paper RAT, not P1, joint-V1 or separate-B V1.

### Preserve exactly

- shared IMPALA/ResNet hidden256 network and all heads;
- Paper actor score, RHS, B×B system, solve and update order;
- initial LR`.5`;
- adaptive KL after every minibatch with thresholds`.005/.04`;
- momentum`1e-6` and original history correction;
- rollout`4096`, minibatch`512`, epochs`4`;
- damping/global clip`.5/.5`;
- PopArt statistics, GAE, entropy, ratio, reward and evaluation semantics;
- checkpoint and formal6M stopping protocol;
- complete original Paper sampled-critic direction on all shared encoder/trunk parameters.

### Parameter partition

Create a frozen manifest with three mutually exclusive groups:

1. `POLICY_EXCLUSIVE`: affects policy output but not value output;
2. `SHARED`: affects both policy and value outputs;
3. `CRITIC_EXCLUSIVE`: trainable parameters affecting value output with exactly zero
   policy-output Jacobian.

The grouping must be established from module ownership plus gradient/Jacobian tests, not
name matching alone. PopArt running statistics retain Paper update semantics and are not
silently treated as curvature parameters.

### Only allowed scientific replacement

For `CRITIC_EXCLUSIVE` parameters only:

- replace the corresponding Paper sampled-critic direction with deterministic
  `J_v`/normalized critic-residual GGN;
- critic lambda`.1`, objective coefficient`1`;
- independent head-only B×B sample-space system;
- symmetric FP64, Jacobi and Cholesky direct solve;
- required head-GGN and relative-residual telemetry.

Form the hybrid critic direction as:

- `SHARED`: exact component of the full original Paper critic direction;
- `CRITIC_EXCLUSIVE`: deterministic head-only GGN direction;
- `POLICY_EXCLUSIVE`: zero critic direction.

Apply the hybrid direction through the original Paper composition, clipping, momentum,
history and update-order semantics.

## Prohibited Algorithm Changes

Do not introduce:

- deterministic GGN on shared parameters;
- joint-2B, cross blocks or RHS-aligned reductions;
- low-Fisher or damping guards;
- policy-null projections or new trust-region hyperparameters;
- LR, KL, momentum, history, damping or clip changes;
- Kaczmarz, normalization changes or another candidate.

## Mandatory Preflight

Before scientific launch, produce machine-auditable evidence that:

1. Paper→Target differences are confined to `CRITIC_EXCLUSIVE` raw direction and
   required telemetry.
2. Parameter groups are exhaustive, mutually exclusive and stable.
3. Every `CRITIC_EXCLUSIVE` parameter has zero policy-logit Jacobian.
4. On a frozen batch, Target and Paper have bit-identical:
   - actor matrix/RHS/direction;
   - Paper sampled-critic shared-trunk direction;
   - LR/KL controller state;
   - shared and policy parameter deltas before applying the changed head direction.
5. A one-step Target update produces policy parameters and policy logits identical to
   Paper within a frozen strict tolerance.
6. Only the value-head parameter delta differs.
7. Head-only deterministic `J_v`, residual, lambda and B×B construction are correct.
8. FP64/Jacobi/Cholesky solve is finite and satisfies the established residual tolerance.
9. Config validation rejects shared deterministic GGN, joint/cross, low-Fisher,
   historical P1 actor fields and Kaczmarz.
10. Source/config/launcher/monitor hashes are frozen.
11. All new roots are absent and no duplicate method/environment/seed/budget objective
    exists.
12. Exact comparison against historical expected/no-cross/block-trace implementations
    confirms this hybrid parameter partition and formula are not duplicates.

If any condition fails, do not train; report `PRECHECK_BLOCKED`.

## Scientific Matrix

Run only:

| Environment | Seed | Intended horizon |
|---|---:|---:|
| `bigfish-easy-0-10` | 0 | 6M |
| `bossfight-easy-0-10` | 0 | 6M |
| `caveflyer-easy-0-10` | 0 | 6M |
| `coinrun-easy-0-10` | 0 | 6M |

Cells not early-stopped must reach the formal endpoint at`5,980,160`.

Do not rerun original Paper RAT. Use its immutable strict-complete seed0 progress and
artifact evidence.

## Frozen Early-Stop Protocol

Evaluate only at:

1. the first exact common logged transition at or above2M;
2. the first exact common logged transition at or above4M;
3. exact terminal transition`5,980,160`.

Every comparison must have identical environment, seed, evaluation semantics and
transition. Never compare an intermediate Target with Paper terminal reward.

When the matched Paper reward is positive:

`ratio = Target reward / Paper RAT reward`

- if `ratio < 0.60`, cancel that Target cell and record
  `EARLY_STOPPED_ALGORITHM`;
- otherwise continue to the next frozen stage or6M endpoint.

If the exact baseline row is absent, semantically ambiguous or nonpositive, record
`not-evaluable` and continue; do not interpolate or substitute another row.

No reward-based cancellation is allowed before2M.

## Computational Requirements and Role Boundary

The task requires at most four independent6M-horizon Procgen cells, FP64 head-only B×B
solves, complete telemetry/checkpoints and a persistent frozen monitor.

The Executor must refresh all authorized resources for scheduler, process, ownership,
capacity, dependency, artifact and duplicate status, then independently choose all live
placement, GPU, partition, concurrency and queue details. Resource choices must not alter
the frozen scientific identity.

Do not use Jupyter. `.54`, `ws4090-31` and `10.49.7.54` remain quarantined.

## Required Evidence

For each cell record:

- frozen git/source/config/launcher/monitor hashes and full command;
- parameter-partition manifest and Jacobian-zero proof;
- environment, seed, intended and actual transitions;
- Executor-owned scheduling provenance;
- unique root, status, rc, progress, trace, logs and checkpoint state;
- at each frozen stage: Target/Paper reward, ratio, KL, LR and entropy;
- actor direction norm and clip;
- Paper sampled shared-trunk critic norm;
- deterministic value-head GGN norm;
- shared/head/actor update-norm ratios;
- critic loss/EV, Jacobi condition and solve residual;
- actor-only, post-shared-critic and post-head-critic policy-logit/KL telemetry;
- hard-error, NaN/Inf, OOM, communication, disk, dependency and stall scans;
- exact terminal or cancellation reason.

Classify failures as:

- `algorithm-failure`
- `EARLY_STOPPED_ALGORITHM`
- `numerical-failure`
- `infrastructure-failure`
- `queued/quota-waiting`
- `cancelled-nonscientific`
- `unknown/insufficient-evidence`

## Falsifiable Conclusion

Return exactly one:

- `CANDIDATE_PROMOTE_TO_3SEED`: all four cells reach6M, PASS/rc0, with no
  algorithm/numerical failure and every frozen comparison ratio at least`.60`.
- `CANDIDATE_REJECT`: any cell triggers scientific early-stop or algorithm/numerical
  failure.
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`: identity passes but infrastructure alone
  prevents completion.
- `PRECHECK_BLOCKED`: Paper actor/shared-critic equivalence, zero-policy-Jacobian
  partition or formula distinctness cannot be proven.

Partial environmental success is not four-environment success.

## Required Outputs

Create:

`.agent/reports/PROCGEN-PAPER-HYBRID-HEAD-DETGGN-6M-S0-20260824-08.md`

Update:

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`

The report and callback must include:

1. exact algorithm definition and all hashes;
2. parameter partition;
3. Paper-equivalence and zero-policy-Jacobian tests;
4. historical formula distinctness;
5. four-environment frozen-stage comparison table;
6. shared/head/actor update and policy-KL decomposition;
7. all failure/cancellation classifications;
8. one falsifiable conclusion;
9. immutable historical ledger preserving:
   - joint-2B `GATE_FAIL`;
   - separate-B `CANDIDATE_NOT_READY`;
   - all associated early stops;
   - low-Fisher `GUARD_NOT_HELPFUL`;
   - P1 and ACTOR_J failures;
   - prior infrastructure failures and obsolete cancellations;
10. final Delivery HEAD, evidence/report commit, verified push and worktree status.

## Acceptance Criteria

- Exactly one hybrid head-only deterministic-GGN candidate is tested.
- Paper actor and Paper sampled shared-trunk critic directions are preserved.
- Deterministic GGN touches only critic-exclusive parameters.
- All four cells begin with a6M intended horizon.
- Early stopping follows only the frozen exact-stage `.60` rule.
- No historical root, artifact or failure record is overwritten.
- Planner makes no live resource-allocation decision.
- Reports are committed and pushed to `origin/agent-work`.

## Prohibited Actions

- Do not rerun joint-V1 or separate-B V1.
- Do not define another candidate or sweep.
- Do not alter Paper actor or shared-trunk critic semantics.
- Do not add joint/cross/shared-GGN/guard/projection/Kaczmarz mechanisms.
- Do not add seeds1/2 or rerun Paper RAT.
- Do not early-stop for reward before2M.
- Do not automatically retry failed cells.
- Do not use Jupyter or quarantined resources.
- Do not plan MuJoCo or Isaac work.
- Do not commit unrelated files.

## Commit and Push

Before launch, commit the frozen implementation, tests, manifest, launcher and monitor with
a message containing:

`PROCGEN-PAPER-HYBRID-HEAD-DETGGN-6M-S0-20260824-08`

After terminal evidence, commit reports/state, push to `origin/agent-work`, verify the
remote HEAD and return the required callback evidence.
