# Task51 paired fixed-LR dual-trust experiment

## Scope and frozen parent

- Task: `PROCGEN-FULL-SHARED-JOINT2B-FIXEDLR-DUALTRUST-BETA1-BETA4-6M-S0-20260827-51`
- Parent implementation/delivery: `e4f8cfc23bf406989f72db61ca8aadf5407d99d4` / `5cbdb7d30c6601bc69a0a4237670b4f141924298`
- Parent trainer/config SHA256: `35bb29e82f3d72067cc30431f5794902da01cce345d5d0dd0540193a0e362846` / `1ebd5f591c9693318eb977775553e06e48f27e1d6f2375d6f33889208eebbbe9`
- Task49/50 and `Bede1074901_*` are immutable and out of scope.

The PPO/Adam warmup ends at transition 503,808. One clean switch creates the
strict full-shared deterministic Joint-2B SGD path. The production actor
Fisher, full critic Jacobian, 1,024 dual rows, 938,976 ordered parameter
columns, both natural cross blocks, damping `.5`, momentum/history, global
clip, PopArt/GAE and evaluation semantics remain the parent implementation.

## Paired scientific delta

The two configs differ in one line: `beta_v: 1.0` versus `beta_v: 4.0`.
Parameter LR is fixed at `.004`; no Task50 LR update remains. Each full Joint
rollout holds `eta_pi` and `eta_v` fixed, measures full-class behavior-to-final
policy KL and fixed-PopArt-coordinate Gaussian value KL, then records exactly
one independent update per coefficient. Metric coefficients strengthen on
divergence above `.04`, weaken below `.005`, use factor `1.5`, and remain in
`[1/64,64]`. Objective coefficients remain one.

Frozen local SHA256 identities before remote work:

| Artifact | SHA256 |
|---|---|
| trainer | `af66fa0aa0115be2b82cad3666c9e91bf705053bfe219151de0689607dd4430d` |
| beta1 config | `57f6ca217bf3f02b0346b42a7797c0546a2a954166c2377667de81de058975c9` |
| beta4 config | `2f802e62fabfa1ce6056b3ea3cde7cd5591a5a8e3239d681d422f67d0eee9a20` |
| Bede gate launcher | `3083a6e09ec9e1e43a9ccfb27ba87eb3e821552da49e2c2695fcff2c7ddc702c` |
| Bede science launcher | `2be97e79821c7baf5f0937c694a819fc7d4c8d5b38e1028cbb2665a105745efb` |
| paired stage monitor | `73451eadf7a64b7ad49d2dbf4d96452463bdb04ca4d9f12cd856cb91478580f2` |

Local syntax/bytecode and both launcher shell checks pass. Each cell uses its
own runtime/log directory, avoiding cross-arm same-environment log collisions.

## Sole paired production gate

Bede job `1075095` completed `0:0` in `00:01:47` on gpu023. Both roots are
`PRECHECK_PASS/rc0`. Each arm ran one real PPO rollout, switched exactly once,
and completed one full Joint rollout. Both use 1,024 rows and 938,976 ordered
parameter columns, retain nonzero equal-transpose natural cross blocks, use
fixed LR `.004` with zero within/between-rollout LR changes, report Cholesky
info0, finite scan PASS and strict FP64 residuals.

| Gate arm | D_pi | D_v | eta_pi | eta_v | cross Frobenius | rel. residual |
|---|---:|---:|---:|---:|---:|---:|
| beta1 | .0006223299 | .0744025186 | 1 -> 2/3 | 1 -> 1.5 | 9.3080698130 | 1.589e-15 |
| beta4 | .0007693351 | .1080075875 | 1 -> 2/3 | 1 -> 1.5 | 9.0059423814 | 1.641e-15 |

Thus both measurements exercised the required directions: low actor
divergence weakened the actor metric and high value divergence strengthened
the critic metric, exactly once per block. PopArt mean/std were fixed within
each Joint rollout (`.3004482388/.3621089160`). Complete model-free gate
JSONL, switch, hash, rc/status and empty scheduler stdout/err are archived
under `evidence_gate/`; no checkpoint/model was copied.

## Capacity and eight-cell launch

The immediate Bede refresh showed three completely idle four-V100 nodes plus
seven mixed nodes, partition `gpu` UP with two-day limit, account `bdman37g`,
and no Task51 job/root/duplicate. This supplied more than the requested six
immediately usable slots, so no CSF3 split or second deployment wrapper was
needed. All eight cells were submitted in one bounded action, without
dependencies, holds or throttling.

| Arm | Environment | Job | Initial state | Node | Exact root suffix |
|---|---|---:|---|---|---|
| beta1 | BigFish | 1075096 | RUNNING | gpu023 | `BETA1_V1/bigfish-easy-0-10/seed0/6m` |
| beta1 | BossFight | 1075097 | RUNNING | gpu023 | `BETA1_V1/bossfight-easy-0-10/seed0/6m` |
| beta1 | CaveFlyer | 1075098 | RUNNING | gpu024 | `BETA1_V1/caveflyer-easy-0-10/seed0/6m` |
| beta1 | CoinRun | 1075099 | RUNNING | gpu024 | `BETA1_V1/coinrun-easy-0-10/seed0/6m` |
| beta4 | BigFish | 1075100 | RUNNING | gpu029 | `BETA4_V1/bigfish-easy-0-10/seed0/6m` |
| beta4 | BossFight | 1075101 | RUNNING | gpu029 | `BETA4_V1/bossfight-easy-0-10/seed0/6m` |
| beta4 | CaveFlyer | 1075102 | RUNNING | gpu030 | `BETA4_V1/caveflyer-easy-0-10/seed0/6m` |
| beta4 | CoinRun | 1075103 | RUNNING | gpu030 | `BETA4_V1/coinrun-easy-0-10/seed0/6m` |

All roots share campaign prefix
`/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_fixedlr_dualtrust_beta1_beta4_6m_s0_20260827_51/runs/FULL_SHARED_JOINT2B_FIXEDLR_DUALTRUST_`.
Every root has `RUNNING`, `scientific_started.marker`, job ID and trainer PID.
Requested allocations: 8. Initial allocated/RUNNING concurrency: 8. PENDING:
0, so there are no pending reasons. Initial hard-error scan is zero; the only
stderr content is the known Gym deprecation notice.

Initial bounded conclusion was `CANDIDATE_NOT_READY` while all eight cells ran
under exact-stage Paper monitoring.

## Exact 2M monitor action (2026-08-27T07:45Z pass)

All immutable Paper files passed `SHA256SUMS`. The first exact common row was
`2,007,040` for every arm/environment:

| Arm | Environment/job | Target/Paper | Ratio | Action |
|---|---|---:|---:|---|
| beta1 | BigFish 1075096 | 10.45/9.28 | 1.1260775862 | PASS, continue |
| beta1 | BossFight 1075097 | .44/2.92 | .1506849315 | EARLY_STOPPED_ALGORITHM |
| beta1 | CaveFlyer 1075098 | 2.50/4.45 | .5617977528 | EARLY_STOPPED_ALGORITHM |
| beta1 | CoinRun 1075099 | 6.50/3.70 | 1.7567567568 | PASS, continue |
| beta4 | BigFish 1075100 | 10.51/9.28 | 1.1325431034 | PASS, continue |
| beta4 | BossFight 1075101 | .92/2.92 | .3150684932 | EARLY_STOPPED_ALGORITHM |
| beta4 | CaveFlyer 1075102 | 2.30/4.45 | .5168539326 | EARLY_STOPPED_ALGORITHM |
| beta4 | CoinRun 1075103 | 7.40/3.70 | 2.0 | PASS, continue |

For the four strict-below-threshold cells, paired monitor SHA
`73451eadf7a64b7ad49d2dbf4d96452463bdb04ca4d9f12cd856cb91478580f2`
was called once with the correct method and distinct root ledger. Every call
wrote `EARLY_STOPPED_ALGORITHM`, returned rc3 and cancelled only its bound job.
Scheduler-authoritative terminal evidence is:

- beta1 Boss `1075097`: `CANCELLED by 639800874`, exit `0:0`, elapsed
  `02:51:45`, gpu023;
- beta1 Cave `1075098`: same classification, elapsed `02:51:46`, gpu024;
- beta4 Boss `1075101`: same classification, elapsed `02:51:46`, gpu029;
- beta4 Cave `1075102`: same classification, elapsed `02:51:46`, gpu030.

The four roots retain stale `RUNNING` markers and absent rc files after
scheduler cancellation. No checkpoint exists. Each evidence directory records
scheduler and accounting before/after/terminal state, input and monitor hashes,
exact ledger, command/rc, artifact metadata, final metric/rollout telemetry and
hard-error scan. The solves were finite, Cholesky info was zero, natural cross
blocks remained nonzero, LR was `.004`, and hard-error matches were zero.

BigFish and CoinRun in both arms remain RUNNING and untouched. The campaign is
not fully terminal, so the bounded conclusion remains `CANDIDATE_NOT_READY`.
