# Immutable preflight failure ledger

## Job 19220448

- Scheduler: `FAILED`, exit `1:0`, elapsed `00:00:15`, node820, start
  `2026-08-24T20:27:41+01:00`.
- Allocation: one genuine NVIDIA H200; `gpu.txt` was written.
- Regression: `HYBRID_HEAD_REGRESSION_PASS`, including bit-identical Paper
  actor/shared-critic and exact zero-disconnected head policy Jacobian.
- Compatibility test: did not begin. Importing the frozen target failed with
  `ModuleNotFoundError: No module named 'utils'` because the preflight launcher
  did not add the staged code directory to Python's import path.
- Classification: `infrastructure-failure/preflight-design`, not an algorithm,
  numerical, compatibility, or scientific result.
- Scientific work: no root, marker, trainer process, transition, progress,
  trace, checkpoint, or model was created.
- Preservation: original durable files remain directly under the remote
  campaign `preflight/`; corrected attempts write to a job-ID subdirectory and
  cannot overwrite them.

The correction only exports the already frozen staged code directory through
`PYTHONPATH`, adds an exit trap for durable rc/status, and isolates output by
job ID. Trainer and config hashes are unchanged.

## Job 19220752

- Scheduler: `FAILED`, exit `1:0`, elapsed `00:00:15`, node820, start/end at
  CSF3 local `2026-08-24T20:30`.
- Allocation: one genuine NVIDIA H200; durable job-ID-isolated output and rc1
  status `INFRASTRUCTURE_PREFLIGHT_FAIL` were written.
- Regression: `HYBRID_HEAD_REGRESSION_PASS` again, including all equivalence,
  partition-algebra, and solver checks.
- Compatibility test: target import succeeded, then actual-network
  construction stopped before partition/Jacobian or memory testing because
  the harness's `SimpleNamespace` omitted required `norm_obs`:
  `AttributeError: 'types.SimpleNamespace' object has no attribute 'norm_obs'`.
- Classification: `infrastructure-failure/preflight-design`, not an algorithm,
  numerical, compatibility, or scientific result.
- Scientific work: no root, marker, trainer process, transition, progress,
  trace, checkpoint, or model was created.

After this second persistent preflight-design failure, no further correction,
retry, allocation, or scientific submission was attempted. The mandatory
actual-network partition and H200 memory proof remains unestablished, so the
task terminates `PRECHECK_BLOCKED`.
