# Task-ID: PROCGEN-FULL-SHARED-JOINT2B-PRODUCTION-SHAPE-RECOVERY-AND-6M-S0-20260826-40

Status: READY

Recover only Task39's production-network preflight image-shape construction.
Keep trainer `b13b3eae445f83f0e1d6565cb942c8390b6cd143fcc7f366410406c370aa0815`,
config `1b0cf73885dcb7c61078e463f826d6b296abfedb5a6a019f6782c6b899896b52`,
science launcher `59dea11cb0fed21974f4985d2eb2016703dd33f06d2e11daa7ca419d5a888fb4`,
method `FULL_SHARED_JOINT2B_SCALE_RECOVERY_V1`, and all scientific semantics
byte-identical. Do not retry or relabel Task39 job `19407505`. Task38 remains
`SUPERSEDED_BEFORE_EXECUTION`.

The preflight must obtain the real Procgen observation space through the same
production environment/model-construction path, audit HWC to CHW layout, and
pass the true spatial dimension to ResNet. Reject channel-as-image-size,
missing/swapped/nonproduction dimensions, config/network drift, mock/reduced
models and parameter-manifest differences. After minimal shape regressions,
run exactly one corrected production preflight. Failure is terminal
`PRECHECK_BLOCKED` with no repair/retry.

Only after `PRECHECK_PASS`, submit exactly one fresh seed0 intended-6M cell for
BigFish, BossFight, CaveFlyer and CoinRun. Preserve the exact Paper seed0
same-transition >=2M, >=4M and 5,980,160 comparison; cancel only a cell with
Target/Paper < .60. No sweep, second candidate, extra seeds, retry, requeue,
resubmit, Paper rerun, provenance observer, Jupyter or quarantined access.

Write the Task40 report, update STATE/AGENT_REPORT, push model-free evidence to
origin/agent-work, verify the delivery SHA, and callback Planner/coordinator.
