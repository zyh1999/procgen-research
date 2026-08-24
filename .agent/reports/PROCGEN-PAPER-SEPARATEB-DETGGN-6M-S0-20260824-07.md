# PROCGEN-PAPER-SEPARATEB-DETGGN-6M-S0-20260824-07

## Unique conclusion

`CANDIDATE_NOT_READY`

The frozen `PAPER_MATCHED_SEPARATE_B_DET_GGN_V1` implementation passed its
identity, historical-distinctness, numerical-regression, and H200
compatibility prechecks. It did not pass the scientific gate. At the first
exact common transition at or above 2M (`2,007,040`), Target/Paper reward
ratios were `.3469827586` for BigFish, `.0171232877` for BossFight, and
`.1640449438` for CaveFlyer. The frozen stage monitor therefore early-stopped
those three cells as `EARLY_STOPPED_ALGORITHM`. CoinRun passed at 2M and 4M,
then completed at the formal `5,980,160` endpoint with reward ratio
`.6808510638` and scheduler/artifact PASS/rc0.

Three of four environments failed the predeclared `.60` exact-stage threshold,
so this candidate is not ready for a three-seed promotion. Solver residuals
were finite and small, H200 preflights passed, and hard-error scans were zero.
This is algorithm evidence, not an infrastructure or numerical-solver failure.

## Frozen identity

Frozen Git commit: `8a956130fe661aa41286a9b36ffe10965c223082`.

| Artifact | SHA256 |
|---|---|
| original Paper RAT trainer | `cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7` |
| original Paper RAT config | `1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea` |
| P1 solver donor | `2b50f8cc26bb8c85f91e0b394acc7535f5564ac70036403597d3738d3dab9c1b` |
| prior joint-2B trainer | `41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a` |
| Target trainer | `b0dad110c36dbab4c601aa9128ba51eb437bfc6a3e9cadf87be8fd2172f3729a` |
| Target config | `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda` |
| launcher | `5d3caf579d5203eef44f54f93af7f8c53567a0e423e27ea1cd604be7a9bd0554` |
| stage monitor | `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e` |
| H200 preflight | `72e4c5471e8a7ca9e2c8ada01fd75734ed09806232ff1409bbe164e2e6ad9faa` |
| regression test | `ded706f5c848283b73aa3a0924cadb1359055695d88a584341181814de391ec1` |

Original Paper RAT is Bede source commit
`2b5affd64cbb3c624b4bc1f4767f449df231ffb2`; it was neither modified nor
rerun. The current method retains its sampled actor score/RHS and independent
actor B-by-B solve, LR `.5`, per-minibatch adaptive-KL thresholds `.005/.04`,
momentum `1e-6`, history correction, network, rollout/minibatch/epoch,
evaluation, and checkpoint semantics. Only the independent critic branch is
replaced with deterministic `J_v`/residual GGN, lambda `.1`, its own B-by-B
symmetric FP64/Jacobi/Cholesky solve, and required telemetry. There is no
joint-2B system, cross block, actor change, low-Fisher guard, or sweep.

## Precheck and historical-distinctness gate

The static audit returned `AUDIT_PASS` and `STRICT_LITERAL_PRESERVED` for the
Paper actor path. The frozen CSF3 `.RLvenv` regression returned:

```text
REGRESSION_PASS
actor_direction=BIT_IDENTICAL
adaptive_kl=paper_thresholds_0.005_0.04_per_minibatch
critic_system=independent_BxB_no_cross_blocks
critic=deterministic_Jv_residual_lambda0.1
solver=FP64_Jacobi_Cholesky relative_residual=2.549e-16
illegal_P1_joint_lowfisher_cross_fields=REJECTED
```

The mandatory provenance comparison returned `DISTINCT_FORMULA_PASS`:

| Historical evidence | Full trainer SHA256 | Why it is not this method |
|---|---|---|
| CSF3 `18669377`, block trace | `1881bf7c3fe3f8d29ded23e25976810ab9127d9bc125d9c89332aa39c1ab61dc` | one 512-row actor-plus-critic system using expected-zero cross, not two independent solves |
| CSF3 `18669454/18669615`, expected-relative | `c976c0e563eb3aedb2d306c450d60b44af0c595d0f4a499cf32c65bcec9933d3` | analytic expected-Gaussian shared system and relative damping, not the independent deterministic residual system |
| Bede `1072337`, `1072344/46/49/50` | `0514703d9fb6ca17cc68febabb012defb279ab5a54f57cf95365422164848934` | expected actor-plus-critic B system, dual damping and rollout schedule rather than literal Paper actor plus separate critic B |

Those attempts used altered shared systems, expected/block-trace critic
curvature, initial LR `.004`, rollout-level KL, and momentum0/history disabled.
The current candidate instead uses a literal Paper actor solve plus a second
independent deterministic critic solve with lambda `.1`, Paper LR/schedule,
momentum, and history. Historical rewards are negative provenance, not strict
baselines.

Interpretation constraints were enforced: a clean FP64 residual does not prove
a useful update; joint-2B V1's `.2583/0/.2188` ratios with LR at `.0001`
identify geometry/calibration failure; the five-seed low-Fisher result was
`GUARD_NOT_HELPFUL`; short structural PASS results do not establish
competitiveness; ACTOR_J BossFight's `.5465` ratio at 4.096M motivates the
2M/4M stage checks; and infrastructure failures remain separate.

## Resource placement and immutable baseline

The Executor refreshed authorized resources and selected four independently
auditable CSF3 gpuH jobs. Each requested one H200 and eight CPUs. All ran on
node822 after `GPUH_SEPARATEB_COMPATIBILITY_PASS`. No Jupyter session,
quarantined host, retry, requeue, resubmission, duplicate target, or unrelated
job mutation occurred.

Campaign root:
`/scratch/h99859yz/procgen_paper_separateb_detggn_6m_s0_20260824_07`.

The read-only Paper seed0 progress files came from Bede root
`/nobackup/projects/bdman37/yihe/procgen_rat_papercritic_ablation_20260724`,
source commit `2b5affd64cbb3c624b4bc1f4767f449df231ffb2`, jobs
`1063880_0/1064035`, `_5/1064047`, `_10/1064067`, `_15/1064074`.

| Environment | Paper progress SHA256 |
|---|---|
| BigFish | `caf19809e208f35b8f8bcb41266021d07a6d8ae28f8e1e21d5111268a35961ba` |
| BossFight | `4082868eeec196363e284fd7af68807f20bc0142e7de7e8cf355851a5d89337c` |
| CaveFlyer | `8d10f8614a1cb57d81c7705b7d2373c0c9de6b158c7cd1bdeabba2ca8236e292` |
| CoinRun | `0db1a7538f2ffbcf8c94bec8c84273134b0e08d9eaa5e8366d6b6f15f59e5aeb` |

The originals call their transition field `misc/total_timesteps`; the frozen
monitor expects `transitions_so_far`. Exact normalized monitor inputs added
only that alias, retained every original row and metric, and recorded both
source and input hashes plus adapter provenance. The frozen monitor itself was
not changed.

## Exact stage evidence

All comparisons are same environment, seed0, evaluation/reward semantics, and
exact common transition. No intermediate Target was compared to the Paper 6M
terminal.

| Environment | Stage | Target | Paper | Ratio | Decision |
|---|---:|---:|---:|---:|---|
| BigFish | 2,007,040 | 3.22 | 9.28 | `.3469827586` | `EARLY_STOPPED_ALGORITHM` |
| BossFight | 2,007,040 | .05 | 2.92 | `.0171232877` | `EARLY_STOPPED_ALGORITHM` |
| CaveFlyer | 2,007,040 | .73 | 4.45 | `.1640449438` | `EARLY_STOPPED_ALGORITHM` |
| CoinRun | 2,007,040 | 6.90 | 3.70 | `1.8648648649` | stage PASS |
| CoinRun | 4,014,080 | 7.10 | 8.00 | `.8875` | stage PASS |
| CoinRun | 5,980,160 | 6.40 | 9.40 | `.6808510638` | endpoint PASS |

Target telemetry at the acted-upon/final rows:

| Environment/stage | KL | LR | Entropy | Critic relative residual | Cholesky info |
|---|---:|---:|---:|---:|---:|
| BigFish 2M | `.0908888` | `.0001` | `.579287` | `7.360e-14` | 0 |
| BossFight 2M | `.0554123` | `.0001` | `1.210500` | `7.693e-15` | 0 |
| CaveFlyer 2M | `.181806` | `.00050625` | `.862898` | `6.465e-15` | 0 |
| CoinRun 2M | `2.048213` | `.0001` | `.210112` | `4.948e-14` | 0 |
| CoinRun 4M | `.0282932` | `.00050625` | `.103370` | `1.612e-13` | 0 |
| CoinRun endpoint | `.0431095` | `.0001` | `.683348` | `8.896e-14` | 0 |

The committed `exact_stage_rows.csv` contains reward, KL, LR, entropy,
value loss, gradient/direction norms, clipping/history, residual, GGN,
Jacobi-scale, and Cholesky telemetry for every available Target stage and all
matched Paper rows.

## Terminal scheduler and artifact reconciliation

| Environment | Job | Scheduler evidence | Artifact evidence | Final classification |
|---|---:|---|---|---|
| BigFish | `19210448` | CANCELLED by 778916, node822, 00:30:29 | frozen monitor rc3; `early_stop_2007040`; progress/trace/logs/hashes preserved; no checkpoint; hard-errors0 | `EARLY_STOPPED_ALGORITHM` |
| BossFight | `19210449` | CANCELLED by 778916, node822, 00:30:30 | same evidence contract; no checkpoint; hard-errors0 | `EARLY_STOPPED_ALGORITHM` |
| CaveFlyer | `19210450` | CANCELLED by 778916, node822, 00:28:37 | same evidence contract; no checkpoint; hard-errors0 | `EARLY_STOPPED_ALGORITHM` |
| CoinRun | `19210451` | COMPLETED/0:0, node822, 01:06:17 | PASS/rc0; progress to 5,980,160; trace; checkpoint exists remotely; hard-errors0 | scientific PASS |

The cancelled roots retain stale `RUNNING` markers and lack rc files because
Slurm killed their trainers after the monitor's decision. Scheduler accounting
and the monitor ledgers are authoritative: none is live, failed
infrastructure, or PASS. CoinRun has a terminal PASS marker and rc0. No model
or checkpoint was copied to Git.

The compact evidence export at
`remote_launch_staging/procgen_paper_separateb_detggn_6m_s0_20260824_07/evidence_logs`
contains status/rc, scheduler snapshots, commands, compatibility evidence,
progress CSVs, compressed metric traces and stdout, stderr, exact stage
ledgers, error scans, baseline provenance, and hashes. All 162 payload entries
other than the self-referential checksum manifest itself verified against
`EXPORT_SHA256SUMS`; zero payload mismatches were found.

## Preserved failure and cancellation provenance

- Prior joint-2B V1 is `GATE_FAIL`: exact seed0 ratios `.2583/0/.2188`, LR
  collapsed to `.0001`, finite residuals, and zero hard errors indicate an
  algorithmic geometry/step-calibration failure.
- Its CoinRun bundle `19203175` remains a user-authorized scientific-futility
  early stop, never PASS or infrastructure failure.
- gpuL race loser array `19203054` remains
  `cancelled-race-loser-unstarted`: Start=None, elapsed0, node/root absent.
- gpuA array/raw IDs `19190819`, `19201416`, `19201433`, `19201447` remain
  immutable pre-training launcher-check infrastructure failures.
- gpuL preflights `19200925` and `19201660` remain infrastructure failures;
  corrected `19202370` remains compatibility PASS, not science.
- ACTOR_J BossFight remains algorithm failure; its other original interrupted
  cells, P1 seed1 cells, and Bede missing-utils/OOM attempts remain
  infrastructure failures.
- `18642230`, `18624888`, and `18666591` remain cancelled-obsolete-unstarted.
- The five-seed low-Fisher control remains `GUARD_NOT_HELPFUL` and was not
  reintroduced.

Nothing in this task deletes, overwrites, or promotes those records.

## Decision and Planner boundary

Identity and execution passed, but BigFish, BossFight, and CaveFlyer all
failed the exact 2M `.60` threshold. The unique task conclusion is therefore
`CANDIDATE_NOT_READY`.

The Executor does not infer or implement a successor. The complete evidence
package must be pushed before callback. The same ChatGPT Planner must explain
the failure and return exactly one next bounded scientific/code READY task.
Every next candidate retains a 6M intended horizon and exact same-transition
checks at >=2M, >=4M, and endpoint; only a strict ratio below `.60` authorizes
cell cancellation. Planner owns algorithm/code direction; Executor owns live
host/GPU/partition/concurrency placement and monitoring. Idle hardware does not
authorize a sweep or duplicate objective.

## Delivery

- Frozen implementation commit: `8a956130fe661aa41286a9b36ffe10965c223082`.
- Evidence/report commit: recorded after the terminal evidence push.
- Push target: `origin/agent-work` only.
