# User-authorized gpuA to gpuL migration

Task: `PROCGEN-PAPER-MATCHED-DETERMINISTIC-GGN-1M-GATE-20260824-06`

The scientific trainer, config, method, environments, seed, budget, command,
evaluation, trace, checkpoint, and artifact acceptance logic are unchanged.
The gpuL launcher differs from the frozen gpuA launcher only in scheduler and
placement fields: partition/account/QOS, L40S GRES, node/task declaration,
CPU/memory/time reservation, no-requeue, scheduler-log directory, job name,
and the required non-colliding `runs_gpul` placement root.

Array `19190819` is terminal and cannot be cancelled or restarted. The gpuL
recovery may be submitted only after the persistent compatibility preflight
passes and the exact failure/recovery materials are pushed.

## Immutable infrastructure-failure ledger

- gpuA array `19190819` started before the migration cancellation gate. All
  four cells ran on node858 and failed before environment construction or a
  scientific update: task0/raw `19201416` in 17s, task1/raw `19201433` in
  10s, task2/raw `19201447` in 11s, task3/raw `19190819` in 10s; scheduler
  state `FAILED/1:0`. Per-cell rc is 1, progress/trace/checkpoint are absent.
  The launcher passed an absolute config path to the frozen Paper CLI, which
  prepended `configs/` and raised `FileNotFoundError`. Its later optional
  `find code/logs` check also emitted a missing-directory error. Classification:
  `infrastructure-failure/pre-training-launcher-check failure`.
- Connection-bound gpuL preflight `19200925` ran on node886 for 55s and
  finished `FAILED/1:0` without durable stdout/stderr or a compatibility
  artifact. Classification: `infrastructure-failure/preflight-design failure`.
- Persistent gpuL preflight `19201660` received exactly one L40S on node869
  (`AllocTRES gres/gpu:l40s=1`) and failed before its tensor/solver checks in
  73s. The preflight incorrectly required `total_memory >= 45 * 1024**3`;
  NVIDIA's 48 GB decimal L40S reports about 44.7 GiB. Durable stderr ends at
  that assertion and stdout is empty. Classification:
  `infrastructure-failure/preflight-memory-unit failure`.
- The one user-authorized corrected preflight retains the L40S name check and
  uses the hardware-accurate threshold `47,000,000,000` bytes. It prints the
  exact detected name/bytes before validation and retains the frozen hashes,
  imports/config checks, exact 1024-by-1,464,544 joint footprint, native FP64
  Gram/Jacobi/Cholesky path, peak below 18 GiB and headroom above 25 GiB.
- The user explicitly authorized exactly one gpuL infrastructure recovery.
  Neither failure is an algorithmic or numerical result, and neither root may
  be overwritten or removed.

## Recovery-only launcher corrections

- scheduler placement changes gpuA/A100 to gpuL/L40S with account/QOS,
  conservative 96G host memory, 12 CPUs, one-day limit, no-requeue, at most
  four concurrent cells, and non-colliding `runs_gpul` roots;
- pass only the config basename because the unchanged Paper CLI prepends
  `configs/`;
- create/check the optional log directory safely so absent logs cannot abort
  artifact reconciliation.

No trainer or config byte changes are permitted.

## Corrected gpuL compatibility result

Persistent corrected preflight `19202370` completed `0:0` on node879 in 32s.
Durable stdout records `GPUL_COMPATIBILITY_PASS`, NVIDIA L40S,
`47,667,740,672` total bytes, `14,996,930,560` peak allocated bytes,
`32,670,810,112` conservative headroom bytes, exact joint shape
`(1024, 1464544)`, FP64/Jacobi/Cholesky relative residual `6.916e-16`,
PyTorch `2.5.1+cu121`, CUDA `12.1`, and driver `595.71.05`. Stderr contains
only Pillow/torchvision deprecation warnings. This preflight is compatibility
evidence, not a scientific race start or result.

## Read-only gpuH capacity audit

- User association: account `gpu-h200-fse-pgdr`, QOS `gpu-h200-fse`,
  `MaxTRESPU=4` H200 GPUs.
- At audit time gpuH had five physically unallocated H200s (node821 two,
  node822 two, node823 one), but a correct test-only four-cell request projected
  start `2026-09-11 22:01`, so gpuH was not an immediate alternative.
- gpuH permits at most eight CPU cores per GPU; a 12-core test request is
  rejected.
- No gpuH job was submitted. Test-only pseudo ID `19202431` does not exist in
  `squeue`. No gpuH or unrelated Isaac scheduler state was mutated.
