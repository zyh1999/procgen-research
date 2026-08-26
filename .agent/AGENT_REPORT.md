# Executor Report

## Metadata

- Task-ID: `PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-AUDIT-PATH-RECOVERY-20260826-36`
- Method: `DET_STANDARD_MSE_GGN_HEAD_CVLM_V1`
- Control assignment: `2d35acca43e6d5f9f274354861f42bc7df503798`
- Path-adapter freeze: `bc8d2f44dbebffe6a8119abae682a26ff9d325b3`
- Frozen Task34R implementation: `55984df39bf883685583f22894edd5eb615f95ea`
- Repository target: `origin/agent-work`

## Result

Unique Task36 conclusion: `PRECHECK_RECOVERED`.

The path-only recovery preserved the immutable Task35R archive/manifest and
all Task34R scientific bytes. The adapter resolves the exact trainer and
config identities from the manifest under `bundle/code/`, validates repository
path, Git blob, SHA256, size, mode, regular non-symlink type and containment,
and checks identical pre/post device/inode/hash identity. It runs the exact
historical audit SHA while replacing only its two stale target path expressions
in memory. The audit file is neither edited nor copied.

All required negative tests passed locally and under remote Python 3.9. The
single complete local gate then passed once, including immutable archive and
manifest verification, empty-CWD manifest-backed imports, standard objective,
`G=J^T J/B`, `g=J^T e/B`, precision one, Task13 effective damping 5 and RHS
multiplier 10.

gpuH was refreshed and preferred. With the four-H200 user limit, no duplicate
job and four absent roots, exactly four one-shot actual-network preflights were
submitted. BigFish `19395683`, BossFight `19395684`, CaveFlyer `19395685` and
CoinRun `19395686` all completed `0:0` on node821 and all roots are
`PRECHECK_PASS/rc0` with compatibility markers and complete scientific
identity ledgers.

Every cell proves the 938,979-parameter production network, exact
257-parameter value-head partition, standard `D=I/W=I/K=J`, full 512-row
train and disjoint calibration blocks, cross-minibatch CVLM acceptance and
rejection, bitwise rollback, train-only accepted direction,
actor/shared/logit identity, PopArt regression, Cholesky info 0 and finite
residual. Strict hard-error scans are clean; only the benign first-use cuBLAS
context warning appears.

No 6M science, monitor, transition row, checkpoint or model was created.
Task34R/35R were not retried, and Task32/33 were untouched. Full model-free
evidence is in
`.agent/reports/PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-AUDIT-PATH-RECOVERY-20260826-36.md`
and
`remote_launch_staging/procgen_standard_mse_ggn_head_cvlm_audit_path_recovery_20260826_36/evidence_remote/`.

TASK_COMPLETE
