# Executor Report

## Metadata

- Task-ID: `PROCGEN-NORMMATCH-V2-HERMETIC-BUNDLE-AND-6M-S0-20260825-15`
- Assignment: `2d0932e37884584601d53318ece9cb16f400fba6`
- Method: `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`
- Repository target: `origin/agent-work`

## Result

The deployment-only recovery froze a 32-file content-addressed archive from
Git commit `c2470eac...`. Manifest SHA256 is `99191542...`, archive SHA256 is
`3da17520...`, and the reachable 23-file local import closure includes exact
original-Paper `utils` and `vec_env`. A deterministic rebuild was
byte-identical. Frozen trainer/config/preflight/regression/monitor and
original-launcher hashes remain unchanged. The deployment launcher audit
proved the normalized scientific command byte-identical.

The single mandatory no-training clean-room job `19241161` ran on node820 and
ended `FAILED/1:0` after four seconds. Bundle and manifest verification passed,
then the audit rejected its empty temporary working directory as an extra
`sys.path` entry before trainer import. Under the explicit no-repair/no-retry
gate, no four-environment real-network preflight or scientific job was
submitted.

Final reconciliation found no Task 15 queue row, accepted preflight, science
root, trainer process, transition, progress, trace, checkpoint or model. Task
14 jobs `19238126`--`19238129` remain preserved unchanged. Full model-free
evidence and the exact traceback are tracked in the Task 15 staging directory.

PRECHECK_BLOCKED
