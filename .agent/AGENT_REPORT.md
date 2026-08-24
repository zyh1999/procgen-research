# Executor Report

## Metadata

- Task-ID: `PROCGEN-HYBRID-HEAD-TRAINABLE-GRAD-PREFLIGHT-AND-6M-S0-20260824-11`
- Assignment: `0a7b19a44f78224e6da829d671bf5fb5052b35d0`
- Harness freeze: `26b2252527076df4bfe537a8612446317cbdcf3a`
- Preflight evidence freeze: `dcfd7b08e1827de1cb23dec0241149dd30632d79`
- Repository target: `origin/agent-work`

## Result

The only authorized production preflight, gpuH job `19227905`, completed rc0
and proved the exact trainable/production-update parameter identity, PopArt
state exclusion/invariance, canonical config/model path, exact parameter
partition, zero critic-head policy Jacobian, Paper actor/shared-critic and
one-step policy/logit equivalence, H200 memory headroom, and finite
FP64/Jacobi/Cholesky solve. Scientific files remained byte-identical.

Four frozen seed0 6M jobs were then submitted. BigFish `19228676` started
science and passed exact2,007,040 at `6.53/9.28=.7036637931`. It failed the
exact4,014,080 stage at `6.23/13.28=.4691265060`; the frozen monitor returned
rc3 and Slurm records the mandated `EARLY_STOPPED_ALGORITHM` cancellation.
Solver telemetry was finite, Cholesky info was0, and hard-error scan was clean.

BossFight `19228677`, CaveFlyer `19228678`, and CoinRun `19228679` failed
their immutable per-job preflight before scientific start because a full-file
partition-manifest SHA assertion was environment-sensitive. Scheduler states
are FAILED/70:0 after24/31/31 seconds. They provide infrastructure evidence
only and were not retried or repaired.

Three cells therefore lack scientific evidence, preventing promotion, while
only one environment triggered algorithm early stop, which is insufficient
for the task's two-environment rejection criterion. Full model-free evidence
was committed; no model/checkpoint, retry, requeue, Jupyter, quarantined access,
duplicate candidate or unrelated mutation occurred.

CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE
