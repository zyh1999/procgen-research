# Task36 Audit-Path Recovery

## Identity

- Task-ID: `PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-AUDIT-PATH-RECOVERY-20260826-36`
- Method: `DET_STANDARD_MSE_GGN_HEAD_CVLM_V1`
- Planner control assignment: `2d35acca43e6d5f9f274354861f42bc7df503798`
- Path-adapter freeze: `bc8d2f44dbebffe6a8119abae682a26ff9d325b3`
- Frozen Task34R implementation: `55984df39bf883685583f22894edd5eb615f95ea`
- Adapter SHA256: `7b8cd684f448b730720e4acd1a9c6762faac95778339471770bd40b11f889dd4`
- One-shot local-gate launcher SHA256: `50354a81175d6c6e8abf343d7592ba693e672389acdb05b2b79fcdf735f2e6c4`
- gpuH preflight launcher SHA256: `e6e6ae2fc12add496e2598c544339bc09686a39dd3972e27f5baed790fa06f0d`

The Task34R trainer, config, preflight, historical audit, CVLM mathematics,
damping, trial/rollback logic, actor/shared paths, PopArt, schedule and frozen
scientific launcher remained byte-identical. No science job was launched.

## Frozen inputs

| Input | SHA256 |
|---|---|
| Task35R bundle archive | `3a9d9720ae7b3c9e6d13a2fd63521d51bba8cb62e7ae7ae2498553b57c00609f` |
| Bundle manifest | `287a744078b10054d107974125bac6b5fac43fd944b6200ec54720cd2695c9af` |
| Trainer | `ca53efd549ae58738c5c215c84dfe5f342c0f801d3eb0f7fb61a2507f31e69fc` |
| Config | `52c133559825947cd233184f2468c4aa715c6c419274bb78f7f715d542718132` |
| Actual-network preflight | `2baac759c26c4fb663ac24ff68a6de8b640bb9bd0c443ec8f65106de3d36759a` |
| Historical scaling audit | `9d4929685bd8368f861770f155c5fcc0b7f86b91e9cfc92481987b0dc8ec2723` |
| Scientific launcher | `6dffce265e55f87fa5be848ec5bd9940fcecb6203f621a1454fb4f6a9c74caca` |

## Path-adapter diff and identity

The immutable audit had two stale expressions under its `paths` dictionary:

- `root / "train_shared_det_standard_mse_ggn_head_cvlm_v1.py"`
- `root / "adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml"`

The versioned adapter first verifies the exact audit SHA. It parses that exact
source and replaces only those two in-memory AST expressions with explicit
manifest-resolved `Path` objects. The audit file is neither copied nor edited;
no compatibility file or symlink exists under `bundle/frozen/`.

The trainer is resolved by the exact repository path
`remote_launch_staging/procgen_standard_mse_ggn_head_cvlm_6m_s0_20260825_34r/train_shared_det_standard_mse_ggn_head_cvlm_v1.py`
to `bundle/code/train_shared_det_standard_mse_ggn_head_cvlm_v1.py`. Its
manifest identity is Git blob `f480eb3b509fb693a45d3264ae44aa383cc5f2e6`,
SHA256 `ca53efd5...`, size 74,577 and mode `100755`. Every local/preflight
ledger proves regular non-symlink type, containment under the verified
`bundle/code/`, and identical device/inode/mode/size/SHA before and after the
audit. Config identity is checked in the same way against its exact manifest
entry. `ambient_fallback=false` and `audit_math_source_modified=false`.

## Negative tests

The local and corrected remote Python 3.9 suites passed all seven groups:

- the historical `bundle/frozen/...trainer.py` false path;
- symlink and path escape;
- same bytes with a different repository manifest identity;
- wrong Git blob, SHA256, size and mode;
- missing and duplicate manifest entries;
- ambient checkout/file fallback;
- any numerical-audit source mutation.

The first remote test process had already passed the identity assertions but
its test teardown hit an NFS delayed-file cleanup error. Before the complete
local gate began, only that test fixture was changed to avoid the extra shared
temporary directory. The adapter and every frozen input were unchanged; the
remote Python 3.9 suite then passed 7/7. This pre-gate harness event is retained
in `evidence_remote/negative_tests.txt` and is not a gate or scientific result.

## Single complete local gate

Exactly one complete local gate ran from `2026-08-26T05:13:09Z` through
`05:13:23Z` and returned `LOCAL_GATE_PASS/rc0`.

| Gate | Result |
|---|---|
| Original archive/manifest verification | PASS, immutable hashes above |
| Exact trainer/config manifest identity | PASS, pre/post identity equal |
| Empty-CWD import | PASS, manifest-backed module-origin ledger |
| Ambient repository fallback | false |
| Frozen historical audit | `TASK34R_HISTORICAL_SCALING_AUDIT_PASS` |
| Task36 path adapter | `TASK36_AUDIT_PATH_ADAPTER_PASS` |
| Four new preflight roots | all absent before submission |

The recovered frozen audit proves
`||V-stopgrad(R_lambda)||^2/(2B)`, `G=J^T J/B`, `g=J^T e/B`, Gaussian
precision one, Task13 effective standard-coordinate damping 5 and RHS
multiplier 10. The direct-versus-standard transformed maximum absolute error
is `4.440892098500626e-16`, within the existing FP64 tolerance.

## gpuH placement

After the local gate passed, live refresh showed the user association
`gpu-h200-fse-pgdr`, QOS `gpu-h200-fse`, a four-H200 user maximum and an UP
32-H200 `gpuH` partition across `node820`--`node823`. There was no user gpuH
job, duplicate Task34R/35R/36 job, or existing Task36 preflight root. gpuH was
used as requested; no alternative queue was selected.

## Four-environment actual-network preflight

| Environment | Job | Scheduler | Root | Key numerical evidence |
|---|---:|---|---|---|
| BigFish seed0 | `19395683` | `COMPLETED/0:0`, 54s, node821 | `PRECHECK_PASS/rc0` | ared-pred `5.551e-17`; reject rho `-1.58377`; residual `3.162e-16` |
| BossFight seed0 | `19395684` | `COMPLETED/0:0`, 54s, node821 | `PRECHECK_PASS/rc0` | ared-pred `2.168e-19`; reject rho `-1.58714`; residual `2.118e-16` |
| CaveFlyer seed0 | `19395685` | `COMPLETED/0:0`, 46s, node821 | `PRECHECK_PASS/rc0` | ared-pred `8.327e-17`; reject rho `-1.29050`; residual `5.877e-16` |
| CoinRun seed0 | `19395686` | `COMPLETED/0:0`, 46s, node821 | `PRECHECK_PASS/rc0` | ared-pred `4.066e-20`; reject rho `-1.38927`; residual `1.166e-15` |

Every root contains `GPUH_STANDARD_MSE_GGN_HEAD_CVLM_COMPATIBILITY_PASS`, a
precheck marker, canonical three-way resolved config, production 938,979
parameter network and exact 257-parameter value-head partition. All four prove
`D=I`, `W=I`, `K=J`, precision one, full 512-row train and disjoint 512-row
calibration blocks, eight-block schedule, aligned acceptance rho 1, rejected
trial bitwise rollback, train-only accepted direction, actor/shared directions,
deltas and policy logits bit-identical to control, PopArt affine regression,
Cholesky info 0 and finite FP64 residual. No NaN/Inf or fallback occurred.

The only stderr line in each actual-network preflight is PyTorch's benign
first-use cuBLAS primary-context warning. Strict scans found no Traceback,
assertion/runtime error, OOM, CUDA/NCCL failure, disk/quota error or kill. Each
isolated preflight setup created a zero-byte `progress.csv`; there are no
transition rows, metric trace, checkpoint or model. Scheduler completion was
not used alone: root status, rc, PASS marker and all ledgers were verified.

## Failure ledger and boundaries

- Task34R jobs `19319418`--`19319421` were not retried.
- Task35R's single path-layout failure remains immutable and was not rerun.
- The pre-gate remote test teardown issue was fixture/NFS cleanup only and was
  corrected before the one complete local gate, with no adapter/science delta.
- Task32 and Task33 jobs, roots and histories were untouched.
- No 6M science job, monitor, reward comparison, transition, checkpoint or
  model exists for Task36.

Model-free evidence is under
`remote_launch_staging/procgen_standard_mse_ggn_head_cvlm_audit_path_recovery_20260826_36/evidence_remote/`;
extracted bundle directories and empty working directories were intentionally
excluded because the immutable Task35R archive and manifest are already
content-addressed in Git.

PRECHECK_RECOVERED
