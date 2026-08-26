# Task35R Hermetic Preflight Recovery

## Identity

- Task-ID: `PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-HERMETIC-PREFLIGHT-20260826-35R`
- Method: `DET_STANDARD_MSE_GGN_HEAD_CVLM_V1`
- Planner assignment commit: `189c5f0bff3a1a058042863c033667cd6cf25742`
- Hermetic implementation freeze: `cbbd7dc812f97e436e459cf7910acb3f62f47d2d`
- Frozen Task34R scientific implementation: `55984df39bf883685583f22894edd5eb615f95ea`
- Frozen trainer SHA256: `ca53efd549ae58738c5c215c84dfe5f342c0f801d3eb0f7fb61a2507f31e69fc`
- Frozen config SHA256: `52c133559825947cd233184f2468c4aa715c6c419274bb78f7f715d542718132`
- Frozen actual-network preflight SHA256: `2baac759c26c4fb663ac24ff68a6de8b640bb9bd0c443ec8f65106de3d36759a`
- Frozen historical audit SHA256: `9d4929685bd8368f861770f155c5fcc0b7f86b91e9cfc92481987b0dc8ec2723`
- Frozen Task34R scientific launcher SHA256: `6dffce265e55f87fa5be848ec5bd9940fcecb6203f621a1454fb4f6a9c74caca`

The trainer, config, CVLM mathematics, damping, acceptance thresholds, rollback,
actor/shared paths, PopArt, schedule, evaluation semantics and all scientific
files remained byte-identical.  No trainer source was modified to bypass an
import.

## Hermetic bundle

- Archive: `task35r_source_3a9d9720ae7b3c9e6d13a2fd63521d51bba8cb62e7ae7ae2498553b57c00609f.tar`
- Archive SHA256: `3a9d9720ae7b3c9e6d13a2fd63521d51bba8cb62e7ae7ae2498553b57c00609f`
- Manifest SHA256: `287a744078b10054d107974125bac6b5fac43fd944b6200ec54720cd2695c9af`
- Frozen source commit for every payload entry: `55984df39bf883685583f22894edd5eb615f95ea`
- Manifest entries: 31
- Statically reachable repository-local closure entries: 23
- Per entry: repository path, Git blob, SHA256, byte size and Git mode are recorded.

Two independent builds produced byte-identical archives and manifests.  The
verifier recomputes archive, manifest, file, mode and Git-blob/content identity.
Negative gates rejected missing `utils.logger`, altered file content, a changed
Git blob claim and ambient-path fallback.

## Module origins

The empty-CWD import ran with no `PYTHONPATH`; its repository-local import root
was only the extracted bundle `code/` directory.  The complete 11,004-byte
ledger is `evidence_terminal/module_origins.json` with SHA256
`8a5e94e17554f7a270f2d28f188153272fa17c8767c0059970406691c476365c`.

| Module | Manifest path | Git blob | SHA256/result |
|---|---|---|---|
| frozen trainer | `code/train_shared_det_standard_mse_ggn_head_cvlm_v1.py` | `f480eb3b509fb693a45d3264ae44aa383cc5f2e6` | `ca53efd5...` |
| `utils` | bundle namespace `code/utils` | namespace | exact bundle search location |
| `utils.logger` | `code/utils/logger.py` | `94f8df269c78ce1b30b5c3dbcc8d8a5a18199335` | `5dce7313...` |
| `utils.runners` | `code/utils/runners.py` | `1b2f90e2ca85021f8d22e58b250f9fdd2738a408` | `2a17f6c6...` |
| `utils.utils` | `code/utils/utils.py` | `2a84458f7488c594328fc2c8efa8a6147c8bf169` | `3c39421c...` |
| `vec_env` | `code/vec_env/__init__.py` | `9fbb2a6b19e8df06c045789ca9cf7357e2a6efda` | `4eaff3aa...` |

All observed local origins were manifest-backed and under the bundle; ambient
repository fallback was false.  The config parsed with the expected five
top-level sections and exact frozen hash.  The full production three-way
resolved-config check was not reached because the mandatory frozen historical
audit failed first.

## Launcher equivalence

The normalized preflight argument vector is unchanged:

`$PY $PREFLIGHT $TRAINER $CONFIG $EVIDENCE/parameter_partition.json $TRAINER_SHA $CONFIG_SHA`

The frozen scientific command remains:

`$PY -u $TRAINER --config $(basename $CONFIG) --env_name $ENV_NAME --seed 0 --device 0`

Only bundle verification/extraction, import root, fresh preflight root and
deployment provenance differ.  The equality ledger is
`evidence_local/launcher_equivalence.json`; the new launcher SHA256 is
`5195d92df2a797f9878c779c17eb9ee82fbd79c616643b7dec205dbd505d38be`.

## Local gate matrix

| Gate | Result | Evidence |
|---|---|---|
| Python compile and shell syntax | PASS | local test and `bash -n` |
| Two independent Git-object bundle builds | PASS | archive/manifest hashes above |
| Bundle verifier and safe extraction | PASS | 31 files verified |
| Four required negative tests | PASS | all four rejected |
| Empty-CWD frozen trainer import | PASS | `TASK35R_EMPTY_CWD_IMPORT_PASS` |
| Local module origins | PASS | manifest-backed ledger; no ambient fallback |
| Frozen trainer/config hashes and config parse | PASS | exact SHA256 and expected keys |
| Launcher normalized-command equality | PASS | equality ledger |
| Four fresh preflight roots | PASS | all absent before the gate |
| Task34R science root/process/duplicate | PASS | absent; no duplicate |
| Frozen Task34R historical scaling audit | FAIL | path-layout assertion before numerical audit |

The first and only frozen audit invocation exited 1:

```text
Traceback (most recent call last):
  File "/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_hermetic_preflight_20260826_35r/local_gates/bundle/frozen/audit_task34r.py", line 33, in <module>
    assert path.is_file() and path.stat().st_size > 0, path
AssertionError: /net/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_hermetic_preflight_20260826_35r/local_gates/bundle/frozen/train_shared_det_standard_mse_ggn_head_cvlm_v1.py
```

The unchanged audit derives its target trainer/config location from its own
`frozen/` directory.  The verified bundle places the frozen trainer/config at
`code/`, which is required for the repository import root.  This is a bounded
deployment/path-layout gate failure, not algorithmic, numerical, solver, GPU or
scientific evidence.  Task rules prohibit a field repair or second invocation.

## gpuH placement and four-environment matrix

Live refresh confirmed association `gpu-h200-fse-pgdr`, QOS
`gpu-h200-fse`, per-user maximum four H200 GPUs, an UP 32-H200 `gpuH`
partition, mixed nodes `node820`--`node823`, no user gpuH job, and no duplicate
Task35R process/root.  gpuH was therefore compatible in principle and was not
silently replaced by another queue.

| Environment | Job ID | Scheduler | Root | Actual-network artifacts |
|---|---:|---|---|---|
| BigFish seed0 | none | NOT_SUBMITTED | absent | none |
| BossFight seed0 | none | NOT_SUBMITTED | absent | none |
| CaveFlyer seed0 | none | NOT_SUBMITTED | absent | none |
| CoinRun seed0 | none | NOT_SUBMITTED | absent | none |

No actual-network preflight, scientific job, monitor, transition, progress,
metric trace, checkpoint or model was created.  Task34R jobs `19319418`--
`19319421` were not retried; Task32 and Task33 were untouched.

## Evidence paths

- `remote_launch_staging/procgen_standard_mse_ggn_head_cvlm_hermetic_preflight_20260826_35r/bundle/`
- `remote_launch_staging/procgen_standard_mse_ggn_head_cvlm_hermetic_preflight_20260826_35r/evidence_local/`
- `remote_launch_staging/procgen_standard_mse_ggn_head_cvlm_hermetic_preflight_20260826_35r/evidence_terminal/`
- `remote_launch_staging/procgen_standard_mse_ggn_head_cvlm_hermetic_preflight_20260826_35r/SHA256SUMS`

PRECHECK_BLOCKED
