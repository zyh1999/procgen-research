# Executor report

Task: `PROCGEN-FULL-SHARED-JOINT2B-SCIENCE-LAUNCH-20260826-45`

Current conclusion: `CANDIDATE_NOT_READY`

The user explicitly authorized direct Task45 science with no further preflight
or audit. Frozen trainer/config/science-launcher/oracle identities and the
normalized command remain exact. Deployment freeze
`9f0fcc2b076693964ac331477e4d1b8977660313` routes only fresh Task45 roots.

Exactly four gpuH seed0 intended-6M cells were submitted once: BigFish
`19409681`, BossFight `19409682`, CaveFlyer `19409683`, CoinRun `19409684`.
All are initially `RUNNING` on node820 with scientific-start markers, trainer
PIDs and active minibatches; no immediate hard error is present. Task43's
unresolved preflight discrepancies remain recorded and were not called PASS.

Full launch evidence and monitoring identities are in
`.agent/reports/PROCGEN-FULL-SHARED-JOINT2B-SCIENCE-LAUNCH-20260826-45.md`
and `remote_launch_staging/procgen_full_shared_joint2b_science_launch_20260826_45/`.
# Executor report

Task: `PROCGEN-FULL-SHARED-JOINT2B-ACTOR-SCALE-FLOOR-6M-S0-20260826-46`

Current conclusion: `QUEUED_RESOURCE_WAIT`

The critic-anchored `.01` actor scale floor is implemented and frozen at
`829c58773c2b6a9bc01db2546f0145c24fb118d0`. Minimal syntax/import/hash,
command, duplicate, root and live gpuH placement checks passed. Exactly four
seed0 intended-6M jobs were submitted once: BigFish `19424173`, BossFight
`19424174`, CaveFlyer `19424175`, and CoinRun `19424176`. BigFish initially
runs on node820; the other three wait on `AssocMaxJobsLimit`.

Task45 remains disjoint and untouched. Full frozen identities, roots and the
replacement monitoring contract are in
`.agent/reports/PROCGEN-FULL-SHARED-JOINT2B-ACTOR-SCALE-FLOOR-6M-S0-20260826-46.md`.
