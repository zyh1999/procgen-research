# Task46 launch report

Task-ID: `PROCGEN-FULL-SHARED-JOINT2B-ACTOR-SCALE-FLOOR-6M-S0-20260826-46`

Method: `FULL_SHARED_JOINT2B_CRITIC_ANCHORED_ACTOR_SCALE_FLOOR_V1`

Current conclusion: `QUEUED_RESOURCE_WAIT`

## Frozen implementation

- Implementation/delivery base commit: `829c58773c2b6a9bc01db2546f0145c24fb118d0`
- Parent trainer SHA256: `b13b3eae445f83f0e1d6565cb942c8390b6cd143fcc7f366410406c370aa0815`
- Task46 trainer SHA256: `10524f5f5f5072206b20c4e830d8eff997751d54e0c3a0b1dd00fb1880ea6e89`
- Task46 config SHA256: `16bee61090b168b4e4f175f3e2f533bbc79ea98336e98e4790773356be206c86`
- Task46 gpuH launcher SHA256: `f1e2752764671d3afd0231d2ac85761b45f1635fb8146815c088445b684c7685`
- Task46 exact-stage monitor SHA256: `91c00e35ed3ab8c4ec78553c7947a4ec01047c2308d147fc24109066d97ec989`
- Parent oracle remains `62b1b10a81fc89ec621f0ceaf60735864cc899fafd6bda074c7414744506d303`.

The parent-to-Task46 trainer diff is 75 insertions and 33 deletions and the
config diff is one line. The only scientific change is
`s_pi_eff=max(s_pi_raw, 0.01*s_v_raw)` for actor row/RHS normalization. Critic
normalization continues to use positive finite `s_v_raw`; strict 2B rows,
natural cross blocks, full reconstruction and relative damping `.5` remain
unchanged. The remaining additions are required telemetry. No hysteresis or
adaptive guard was added. The previously rejected `GUARD_NOT_HELPFUL` variant
is distinct and is not present.

## Minimal launch checks

At `2026-08-26T09:40:19Z` the Executor verified only the authorized bounded
checks:

- local syntax compilation and launcher shell syntax;
- remote frozen-Python trainer import and YAML load;
- exact remote hashes listed above;
- normalized command
  `python -u <trainer> --config <config-basename> --env_name <env> --seed 0 --device 0`;
- gpuH ownership/account/QOS/GRES were accepted by Slurm;
- Task46 campaign and all four science roots were absent before submission;
- no Task46 trainer process or named Slurm job existed;
- Task45 roots/jobs were disjoint and were not modified.

The first remote YAML assertion used the wrong top-level lookup and raised a
local `KeyError`; the immediately corrected read used the existing
`algo_config` mapping and passed. This was an Executor smoke-command mistake,
not a trainer/config/preflight/scientific failure. No model construction,
micro-test, negative test, oracle/Jacobian reference, production preflight or
audit chain was run.

## Live placement and launch matrix

Campaign:
`/scratch/h99859yz/procgen_full_shared_joint2b_actor_scale_floor_6m_s0_20260826_46`

Exactly four seed0 intended-6M cells were submitted once on gpuH:

| Environment | Job | Initial scheduler state | Initial root state | Root |
|---|---:|---|---|---|
| BigFish | `19424173` | RUNNING, node820 | RUNNING; `scientific_started.marker` present | `/scratch/h99859yz/procgen_full_shared_joint2b_actor_scale_floor_6m_s0_20260826_46/runs/FULL_SHARED_JOINT2B_CRITIC_ANCHORED_ACTOR_SCALE_FLOOR_V1/bigfish-easy-0-10/seed0/6m` |
| BossFight | `19424174` | PENDING, `AssocMaxJobsLimit` | not created | `/scratch/h99859yz/procgen_full_shared_joint2b_actor_scale_floor_6m_s0_20260826_46/runs/FULL_SHARED_JOINT2B_CRITIC_ANCHORED_ACTOR_SCALE_FLOOR_V1/bossfight-easy-0-10/seed0/6m` |
| CaveFlyer | `19424175` | PENDING, `AssocMaxJobsLimit` | not created | `/scratch/h99859yz/procgen_full_shared_joint2b_actor_scale_floor_6m_s0_20260826_46/runs/FULL_SHARED_JOINT2B_CRITIC_ANCHORED_ACTOR_SCALE_FLOOR_V1/caveflyer-easy-0-10/seed0/6m` |
| CoinRun | `19424176` | PENDING, `AssocMaxJobsLimit` | not created | `/scratch/h99859yz/procgen_full_shared_joint2b_actor_scale_floor_6m_s0_20260826_46/runs/FULL_SHARED_JOINT2B_CRITIC_ANCHORED_ACTOR_SCALE_FLOOR_V1/coinrun-easy-0-10/seed0/6m` |

The pending cells are `queued/quota-waiting`, not scientific results. No job
was retried, requeued or resubmitted.

## Task45 preservation

At submission, Task45 BigFish `19409681`, BossFight `19409682` and CoinRun
`19409684` remained RUNNING on node820. Task45 Cave `19409683` remained
FAILED/1:0 after 00:10:08 and was not touched. Its authoritative classification
remains algorithm/numerical: actor Fisher scale about `2.646e-71`, critic scale
`3.6126e5`, direction/quadratic `Inf`, predicted KL `NaN`, prior solver residual
about `1e-15`, and no infrastructure error.

## Monitoring contract

The existing sole automation `procgen-3090` must be updated in place, not
duplicated, to monitor Task45 and Task46 independently. For Task46 it must use
the frozen monitor above and immutable original Paper RAT seed0 rows. It may
act only at the first exact common transition at or above 2M, the first exact
common transition at or above 4M, and endpoint `5,980,160`; only an exact
same-environment/seed/evaluation Target/Paper ratio below `.60` permits
cancelling that individual cell with a complete ledger. Scheduler state
overrides stale root markers.

No model or checkpoint is included in Git evidence.
