# Task33 terminal report

## Identity and scope

- Task: `PROCGEN-GAE-GGN-HEAD-WIDENTITY-6M-S0-20260825-33`
- Method: `DET_GAE_GGN_HEAD_WIDENTITY_V1`
- Assignment: `1ed0aeadd4e31bbf4914ba58a04dbc413f581919`
- Implementation/preflight/two-seed freeze: `6563f98`
- gpuL deployment freeze: `0057469b50cdfa7f6fd504ec146b3f56daf06ecc`
- Trainer SHA256: `9c822949a171d7b4c8148ea9be37c1eb5b6aa3ff8d9d3f1603241bc4873c2665`
- Config SHA256: `4458567bc61bdb85360d55967c65700992fe60cd812819fa1e77cc8638c202af`
- Functional preflight SHA256: `38e588b1c801840280f10c5330701712ad2098f773a75fae09da5ae8902043b9`
- gpuL compatibility wrapper SHA256: `491e02c727d60c41ed82280d2a70045c8daa0eda001350f46c0d53c97dad88bf`
- gpuL launcher SHA256: `a71c7454cebf0c2f0fe91b51f78f544faacc2dc92c8e0991ad41e50c60d30424`
- Stage monitor SHA256: `bb9c898809eee74cc12ba92597f94f27e02bbc3de7d60d32c0d283945b0645c5`

The Planner matrix was four environments at seed0. The user explicitly
expanded it to seeds 0 and 1. No seed2, sweep or second method was launched.
Task34R was not modified or scheduled by this terminalization work.

## Scientific identity and preflight

The exact Task32-to-Task33 code, field and AST audit proves the only scientific
delta is removal of Task32 actor-score weighting: Task33 uses `K=D J_h`,
`r=q`, and an unweighted GAE objective, with implicit weight exactly one for
every row. The actor, sampled shared critic, GAE operator, PopArt, network,
optimizer/history, schedule, global clip, adaptive KL and evaluation semantics
remain unchanged. The Task32 max-weight-512 concentration path is absent.

Four actual-network gpuH preflights `19286267`--`19286270` completed rc0. The
gpuL/L40S compatibility job `19319577` completed rc0 on node887 and reproduced
exact W=I, actor/shared and policy-logit identity, direct-reference GAE-GGN,
PopArt affine identity, Cholesky info 0 and FP64 relative residual
`4.650e-16`. The only warning was the benign initial cuBLAS context warning.

## Placement and terminal scheduler state

The eight original gpuH jobs `19314824`--`19314831` were cancelled while
unstarted after held gpuL replacements were validated. They had no node,
elapsed time or scientific root. The replacements retained the exact method,
environment, seed, 6M horizon and roots.

| Environment | Seed | gpuL job | Terminal state | Elapsed | Node |
|---|---:|---:|---|---:|---|
| BigFish | 0 | 19319678 | COMPLETED/0:0 | 02:00:03 | node887 |
| BigFish | 1 | 19319679 | COMPLETED/0:0 | 02:03:45 | node883 |
| BossFight | 0 | 19319680 | COMPLETED/0:0 | 02:03:11 | node878 |
| BossFight | 1 | 19319681 | COMPLETED/0:0 | 02:02:35 | node876 |
| CaveFlyer | 0 | 19319682 | COMPLETED/0:0 | 01:59:49 | node883 |
| CaveFlyer | 1 | 19319683 | COMPLETED/0:0 | 02:06:30 | node877 |
| CoinRun | 0 | 19319684 | COMPLETED/0:0 | 02:00:29 | node878 |
| CoinRun | 1 | 19319685 | COMPLETED/0:0 | 01:59:25 | node876 |

All eight trainer processes reached `1466/1466`, printed a final
`misc/total_timesteps` of `5.98e+06`, returned rc0 and had zero hard-error
matches. The scheduler is empty for these job IDs. The existing Task33
automation `procgen-3090` was paused after terminal confirmation.

## Artifact split and failure ledger

All seed1 roots are `PASS/rc0` with 147 progress rows including the exact
5,980,160 endpoint, 46,912 trace rows through 6,004,736, and one 3,766,013-byte
`model.ckpt`. Checkpoint hashes were recorded as metadata only; no checkpoint
or model is committed.

All seed0 roots are `FAIL/rc0`, with empty `progress.csv` and no checkpoint.
Their root traces stop at 286,720 (BigFish), 77,824 (BossFight), 237,568
(CaveFlyer), and 86,016 (CoinRun), despite direct stdout/stderr proof that each
trainer reached the full horizon.

This is a systematic deployment/artifact-routing failure. The frozen launcher
selects a global `${ENV_NAME}.*_${SEED}` source directory using
`find | sort | tail -1`. The frozen compatibility preflight invokes the
production path with hard-coded seed0. During the two-seed concurrent run,
newer empty seed0 preflight directories were created by other jobs. Every
seed0 root's final `source_log_dir` therefore points to a different node from
its scientific job and contains an empty progress file, a 40-byte event file,
and no trace or checkpoint. The partial traces left in the roots were copied
before the source pointer changed and cannot be treated as exact-stage
scientific evidence. Scheduler/trainer completion is preserved separately
from root finalization failure. No retry, repair, relabel or resubmission was
performed.

## Exact-stage evidence

Seed0 has no eligible exact 2,007,040, 4,014,080 or 5,980,160 progress row, so
there was no Paper comparison and no reward-based cancellation. The prior
partial root traces are not substituted for progress rows.

Seed1 has complete exact rows, but no verified immutable original Paper RAT
seed1 baseline with matching environment, evaluation semantics and transition.
Consequently no ratio or cancellation is valid for seed1. Raw target rewards
are retained for Planner interpretation only:

| Environment | 2,007,040 | 4,014,080 | 5,980,160 |
|---|---:|---:|---:|
| BigFish seed1 | 6.02 | 4.69 | 2.08 |
| BossFight seed1 | 0.00 | 0.00 | 0.00 |
| CaveFlyer seed1 | 1.10 | 2.20 | 0.90 |
| CoinRun seed1 | 9.70 | 0.00 | 0.00 |

At all twelve seed1 exact rows, critic Cholesky info is 0 and the relative
residual is finite (`9.98e-16` to `9.70e-14`). Exact KL, LR, entropy, value
loss, GAE loss/statistics, TD/return residuals, spectrum, rank and condition
are archived in `evidence/terminal/seed1_exact_stage_telemetry.tsv`.

No exact same-stage Task32-to-Task33 seed0 comparison is possible because the
Task33 seed0 rows were not finalized. Seed1 cannot replace seed0 for that
comparison.

## Evidence inventory

Model-free terminal evidence is under
`remote_launch_staging/procgen_gae_ggn_head_widentity_6m_s0_20260825_33/evidence/terminal/`:

- `scheduler_terminal.tsv`
- `artifact_terminal.tsv`
- `seed0_completion_and_routing.tsv`
- `seed1_exact_stage_telemetry.tsv`

The gpuL migration and compatibility evidence remains under
`evidence/gpul_migration/`. No model or checkpoint is included.

## Conclusion

The complete intended decision matrix cannot be evaluated: all four matching
seed0 performance artifacts failed finalization, while the four complete
seed1 cells lack a matching immutable Paper seed1 control. Process completion
and healthy available solver telemetry do not repair the missing comparison
evidence.

`CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`
