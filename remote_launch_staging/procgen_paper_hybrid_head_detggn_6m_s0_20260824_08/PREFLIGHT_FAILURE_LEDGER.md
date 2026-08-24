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

After this second persistent preflight-design failure, task 08 made no further
correction, retry, allocation, or scientific submission. Planner task 09 later
authorized exactly one canonical recovery, recorded below.

## Job 19225085

- Authorization: the single canonical recovery authorized by
  `PROCGEN-HYBRID-HEAD-CANONICAL-PREFLIGHT-AND-6M-S0-20260824-09`.
- Scheduler: `FAILED`, exit `1:0`, elapsed `00:00:20`, node820, start/end at
  CSF3 local `2026-08-24T21:17--21:18`.
- Scientific identity: trainer/config/scientific-launcher/monitor hashes
  remained exactly frozen.
- Canonical path: PASS through the trainer's own `main()` parsing/default merge
  and original production `train_fn()`/`SharedActorCritic` construction.
- Resolved configurations: preflight, scientific-launcher dry-run and trainer
  entry JSON files are byte-identical, each file SHA256
  `61f8ebe38443acbdbf141981f4e9921435dccd5d4abb6a63959e3d4bdb9232ab`;
  canonical compact-payload SHA256 is
  `0f34624bdbb1eae181cbfd35a08e1e413c7c9aea73c10f63b75dec0cedb9afdc`.
- Actual production partition reached and proved policy-exclusive 3,855,
  shared 934,864, critic-exclusive 257. The critic head is exactly
  `last_v_layer.weight/bias`, with policy autograd disconnected and Jacobian
  probe L2 exactly zero; partition manifest SHA256 is
  `b45298be8fc5bdccfa36ce653c7dbc0c41f2d013f23d4f4db0a4a580035f3087`.
- Failure: the recovery harness retained a stale, non-scientific assertion
  `SHARED.numel > 1_000_000`. The exact frozen production network has 934,864
  shared trainable parameters and 938,979 reported total parameters, so the
  assertion failed after the successful canonical construction/partition.
- Unreached checks: actual-network one-step equivalence, production-scale H200
  memory footprint, and final head FP64 solve did not execute.
- Classification: `infrastructure-failure/preflight-design`, not algorithm,
  numerical, solver, config mismatch, partition failure, or hardware
  incompatibility.
- Scientific work: no root, marker, trainer process, transition, progress,
  trace, checkpoint, or model was created.

The one authorized recovery allocation is exhausted. No resubmission or
scientific launch followed. Mandatory preflight remains incomplete and the
task terminates `PRECHECK_BLOCKED`.

## Job 19225707

- Authorization: the single final assertion-fix recovery authorized by
  `PROCGEN-HYBRID-HEAD-ASSERTION-FIX-AND-6M-S0-20260824-10`.
- Scheduler: `FAILED`, exit `1:0`, elapsed `00:00:17`, node820, CSF3 local
  start/end `2026-08-24T21:29`.
- Exact assertion fix: PASS. Total 938,979; policy 2/3,855; shared22/934,864;
  critic2/257; exact critic names; exact manifest SHA
  `b45298be8fc5bdccfa36ce653c7dbc0c41f2d013f23d4f4db0a4a580035f3087`.
- Canonical config/model/partition/Jacobian: PASS. Three resolved JSON files
  again had exact SHA
  `61f8ebe38443acbdbf141981f4e9921435dccd5d4abb6a63959e3d4bdb9232ab`.
- Failure: the next actual-network one-step harness called `autograd.grad`
  over every `model.parameters()` entry, including non-trainable PopArt state.
  PyTorch raised `RuntimeError: One of the differentiated Tensors does not
  require grad` before actor/critic equality or update checks completed.
- Unreached: actual-network one-step equality, production-scale H200 memory,
  final FP64/Jacobi/Cholesky residual and full hard-error/nonfinite pass.
- Classification: `infrastructure-failure/preflight-design`, not algorithm,
  numerical, solver, config, partition, Jacobian, or hardware incompatibility.
- Scientific work: no root, marker, trainer process, transition, progress,
  trace, checkpoint, or model was created.

Task 10 explicitly allowed no field repair or retry after any failure. No
resubmission or scientific launch followed. The terminal conclusion remains
`PRECHECK_BLOCKED`.
