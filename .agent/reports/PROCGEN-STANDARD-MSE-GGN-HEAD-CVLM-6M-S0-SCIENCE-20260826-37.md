# Task37 Standard-MSE GGN Head CVLM Seed0 Science

## Unique conclusion

`CANDIDATE_REJECT`

All four frozen seed0 cells reached the exact first common stage at
2,007,040 transitions, and all four failed the predeclared Paper ratio
threshold of 0.60. The exact ratios are 0.4267241379 for BigFish, 0 for
BossFight, 0.2853932584 for CaveFlyer, and 0 for CoinRun. The four-environment
mean ratio, including the acted-upon early-stop stages, is 0.1780293491.

The frozen monitor recorded `EARLY_STOPPED_ALGORITHM` once for every cell and
returned rc3. Slurm then cancelled only the corresponding job. All four are
scheduler-authoritatively `CANCELLED by 778916`; none is live, PASS,
infrastructure-failed, or numerically failed. Strict hard-error scans have zero
matches. No cell was retried, requeued, or resubmitted.

This outcome cannot meet `STANDARD_GGN_CVLM_SEED0_PROMISING`: zero environments
reached the 5,980,160 endpoint, four were algorithmically early-stopped, no
environment exceeded Paper at an endpoint, and the four-environment acted-stage
mean ratio is far below one.

## Frozen identity

| Artifact | Identity |
|---|---|
| Task37 assignment/control commit | `71f9e17e2fd8411faf34e4c2530800d66301e377` |
| Task37 deployment freeze | `4be726357752b197d2c2fabf0d29500b193e8beb` |
| Task37 deployment wrapper SHA256 | `698480b2b45bbf168c69f10b9a31f387a316bf4f03e67f01d46fd47b693281d7` |
| Trainer | `ca53efd549ae58738c5c215c84dfe5f342c0f801d3eb0f7fb61a2507f31e69fc` |
| Config | `52c133559825947cd233184f2468c4aa715c6c419274bb78f7f715d542718132` |
| Actual-network preflight | `2baac759c26c4fb663ac24ff68a6de8b640bb9bd0c443ec8f65106de3d36759a` |
| Historical scaling audit | `9d4929685bd8368f861770f155c5fcc0b7f86b91e9cfc92481987b0dc8ec2723` |
| Hermetic bundle | `3a9d9720ae7b3c9e6d13a2fd63521d51bba8cb62e7ae7ae2498553b57c00609f` |
| Bundle manifest | `287a744078b10054d107974125bac6b5fac43fd944b6200ec54720cd2695c9af` |
| Original scientific launcher | `6dffce265e55f87fa5be848ec5bd9940fcecb6203f621a1454fb4f6a9c74caca` |
| Task36 audit adapter | `7b8cd684f448b730720e4acd1a9c6762faac95778339471770bd40b11f889dd4` |
| Frozen Task37 stage monitor | `c32c41f863f540256b1817b375329fe9615482ece637d22a1ab9657551e052dc` |

The only Task37 code addition was the deployment-only fresh-root wrapper. Its
normalized scientific command is byte-for-byte equivalent to the frozen
Task34R launcher command. Trainer, config, objective, CVLM, damping, threshold,
trial/rollback, actor/shared/PopArt paths, schedule and evaluation remained
unchanged.

The method remains standard frozen-lambda-return MSE with `D=I`, `W=I`,
`K=J`, `G=J^T J/B`, `g=J^T e/B`, Gaussian precision one, and a solve of
`(G+mu I)u=-g` on only the 257 critic-exclusive value-head parameters.

## Launch checks and placement

Task36's four environment actual-network `PRECHECK_PASS/rc0` roots and
compatibility ledgers were reused without rerunning preflight. Before launch,
the Executor verified every frozen hash, hermetic imports without ambient
fallback, all four Task37 roots absent, no duplicate Task37 process/job, and
live gpuH account/QOS/GRES capacity. gpuH was selected as requested; no
alternate partition or host was used.

Exactly four independent seed0 intended-6M jobs were submitted once:

| Environment | Job | Root |
|---|---:|---|
| BigFish | `19397520` | `/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_6m_s0_science_20260826_37/runs/DET_STANDARD_MSE_GGN_HEAD_CVLM_V1/bigfish-easy-0-10/seed0/6m` |
| BossFight | `19397521` | `/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_6m_s0_science_20260826_37/runs/DET_STANDARD_MSE_GGN_HEAD_CVLM_V1/bossfight-easy-0-10/seed0/6m` |
| CaveFlyer | `19397522` | `/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_6m_s0_science_20260826_37/runs/DET_STANDARD_MSE_GGN_HEAD_CVLM_V1/caveflyer-easy-0-10/seed0/6m` |
| CoinRun | `19397523` | `/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_6m_s0_science_20260826_37/runs/DET_STANDARD_MSE_GGN_HEAD_CVLM_V1/coinrun-easy-0-10/seed0/6m` |

The existing `procgen-3090` automation was converted in place to the sole
five-minute Task37 monitor. No second monitor was created.

## Exact Paper-matched stage decision

All comparisons use the immutable original Paper RAT seed0 progress artifacts,
the same environment, seed, reward/evaluation semantics, and exact transition.
No intermediate Target was compared with the Paper terminal.

| Environment | Transition | Target | Paper | Ratio | Monitor | Decision |
|---|---:|---:|---:|---:|---|---|
| BigFish | 2,007,040 | 3.96 | 9.28 | 0.4267241379 | rc3 | `EARLY_STOPPED_ALGORITHM` |
| BossFight | 2,007,040 | 0.00 | 2.92 | 0 | rc3 | `EARLY_STOPPED_ALGORITHM` |
| CaveFlyer | 2,007,040 | 1.27 | 4.45 | 0.2853932584 | rc3 | `EARLY_STOPPED_ALGORITHM` |
| CoinRun | 2,007,040 | 0.00 | 3.70 | 0 | rc3 | `EARLY_STOPPED_ALGORITHM` |

The target progress SHA256 values recorded in the ledgers are
`4fd832add574fbb695ae9daf0ddb2bcc937a176e5612084ef45cc496e3626bf2`
for BigFish, `6559b10f2e64ba5902a71727e30c8a3392b9db25ef89d229b77f1178f3345120`
for BossFight and CoinRun, and
`3790e342aeb032bf6c6b281846daaf8fd24f58bc72807d1b7b004db0ae3e6313`
for CaveFlyer. The corresponding immutable Paper hashes are retained in each
stage ledger.

Because monitoring is periodic, BigFish, BossFight and CaveFlyer wrote later
progress rows before Slurm cancellation. Their last logged transitions were
2,785,280, 2,744,320 and 2,170,880; CoinRun ended at 2,007,040. These rows are
preserved but were not substituted for the exact 2,007,040 decision and did
not trigger any additional comparison.

## CVLM and numerical diagnostics at the acted stage

| Env | KL | actor LR | entropy | residual | info | CVLM decision | mu | rho_cv | head delta | G condition |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| BigFish | 0.00338562 | 0.5 | 1.41665 | 2.390e-16 | 0 | reject/zero delta | 4.080e3 | -1.60318 | 0 | 1 |
| BossFight | -2.327e-9 | 0.5 | 1.300e-6 | 2.119e-16 | 0 | accept | 2.596e6 | 0.722911 | 6.821e-13 | 2.031e4 |
| CaveFlyer | 5.844e-5 | 0.5 | 2.62427 | 3.745e-16 | 0 | accept | 8.348e-8 | 0.982546 | 0.395306 | 3.258e9 |
| CoinRun | 0.0445886 | 0.0291929 | 1.11788 | 2.978e-17 | 0 | reject/zero delta | 1.459e5 | sentinel | 0 | 1 |

All systems are finite with Cholesky info zero and FP64 relative residuals
between 2.978e-17 and 3.745e-16. Actor/shared identity remains intact at the
stage rows: post-head policy KL and maximum logit difference are exactly zero.
The full exact-stage table records value loss, gradient norm, alpha before and
after, train/calibration ared, predicted reduction, trials, spectrum,
effective rank, trace, prediction change, history correction, Paper clip scale
and policy delta.

The learned relative damping did not reproduce a useful cross-environment
regime relative to Task13's fixed effective standard-coordinate damping 5.
At 2M it spans roughly 8.35e-8 to 2.60e6. BigFish and CoinRun reject to a zero
head delta under very large damping; BossFight accepts only a numerically
negligible 6.82e-13 delta; CaveFlyer accepts a substantial delta under tiny
damping and positive held-out rho, yet reaches only 28.5% of Paper reward.

Consequently, held-out acceptance is not sufficient evidence of subsequent
reward/GAE stability: both BossFight and CaveFlyer accepted at the acted stage
but failed the reward criterion, while BigFish and CoinRun rejected. No cell
continued to the 4M or endpoint checks, so no valid longitudinal correlation
can be inferred. In particular, BigFish did not repeat a 2M-pass/4M-fail
pattern; it failed already at 2M. CaveFlyer did not retain the earlier
successful behavior.

The complete remote metric traces retain per-minibatch MSE, TD residual, GAE,
PopArt, spectrum, CVLM trial, momentum and clipping telemetry. The exact-stage
progress rows do not aggregate every per-minibatch field, so this report does
not invent a false stage-level GAE/TD/PopArt value.

## Scheduler, artifacts and errors

| Environment | Scheduler | Root artifacts | Classification |
|---|---|---|---|
| BigFish | CANCELLED by 778916, 00:37:13, node820 | stale `RUNNING`, no trainer rc/checkpoint; 68 progress rows; exact early-stop directory | `EARLY_STOPPED_ALGORITHM` |
| BossFight | CANCELLED by 778916, 00:37:15, node821 | stale `RUNNING`, no trainer rc/checkpoint; 67 progress rows; exact early-stop directory | `EARLY_STOPPED_ALGORITHM` |
| CaveFlyer | CANCELLED by 778916, 00:28:51, node822 | stale `RUNNING`, no trainer rc/checkpoint; 53 progress rows; exact early-stop directory | `EARLY_STOPPED_ALGORITHM` |
| CoinRun | CANCELLED by 778916, 00:27:22, node820 | stale `RUNNING`, no trainer rc/checkpoint; 49 progress rows; exact early-stop directory | `EARLY_STOPPED_ALGORITHM` |

Scheduler accounting overrides the stale root markers left by killing the
trainer after the scientific decision. Each `early_stop_2007040` directory
contains the exact target/Paper rows, normalized monitor inputs, original
artifact hashes and metadata, action provenance, apply time, scheduler before
and after, checksum manifest and rc3 ledger. Strict stdout/stderr scans found
zero Traceback, OOM, CUDA, NCCL, disk/quota, NaN/Inf, nonfinite, not-SPD or
Cholesky-failure matches.

## Evidence archive

The complete model-free archive, including all metric traces, is preserved at
`/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_6m_s0_science_20260826_37/terminal_model_free_export/task37_model_free.tgz`.
It is 42,223,651 bytes, has SHA256
`14e3cca153da5a90c9463cc7f64c440d9f9688f14b30309d1ad74bf228853e4c`,
and passed `tar -tzf`. Models and checkpoints were explicitly excluded.

The Git evidence directory contains a locally verified compact archive with
SHA256 `74a4233dce11c6fa00e06a0534e2dd939b07d73ede2a39d5cd710ad253a2eb3e`
plus scheduler/artifact tables, exact ledgers, telemetry and hard-error scans.
The compact archive excludes the large metric traces as well as all models and
checkpoints; the content-addressed complete remote archive retains them.

## Preserved history and boundary

- Task13 remains `CANDIDATE_NOT_READY`; this Task37 result does not overwrite
  its separate-B identities, stage rows or artifacts.
- Task32's numerical failure and algorithmic early stops remain unchanged.
- Task33 remains `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`; no Task33 cell was
  rerun or reinterpreted.
- Task34R's missing-`utils` failures and Task35R's audit-path failure remain
  deployment/preflight infrastructure evidence, not science.
- Task36 remains `PRECHECK_RECOVERED`; its four PASS roots were reused, never
  rerun.
- No Paper run, seed1/2, sweep, second candidate, duplicate monitor, retry,
  requeue, resubmission, Jupyter session or quarantined host was used.

The Executor does not infer or implement a successor. Only the ordinary
ChatGPT Planner may issue exactly one next READY or NEED_DECISION task.

CANDIDATE_REJECT
