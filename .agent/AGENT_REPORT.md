# Executor Report

## Metadata

- Task-ID: `PROCGEN-NORMMATCH-V2-MP-MAIN-INPATH-CAPTURE-READONLY-20260825-31R`
- Assignment: `b345ad9e22619c5f2f26fd0c8eca3722c065ad49`
- Frozen capture implementation: `ae93ca3990168c058a2d9b87662a10ca0d9e0511`
- Repository target: `origin/agent-work`

## Result

Unique Task31R conclusion: `OBSERVER_PERTURBED`.

Exactly one bounded capture activity was launched as gpuH job `19279429` on
node821. The first capture-on process completed the unchanged production CUDA
construction and Task27 runtime semantic-binding proof. The versioned wrapper
then reached the unchanged Task28R origin scan through nested `runpy`; this
altered the frozen Task23 probe module's live `spec/package` state. Task28R
rejected it with `exact probe spec/package mismatch` before the expected
`__mp_main__` origin-scan frame-local record existed, and the capture routine
reported `Task31R required existing origin/runpy frames unavailable`.

The proposed observation mechanism therefore changed the state it was meant
to observe. The mandatory hard stop was honored: no correction or rerun was
performed, no second capture-on or either capture-off process started, and no
capture JSON was emitted. This precludes the planned on/off comparison and is
classified as
`capture-failure/task31r-nested-runpy-spec-package-perturbation`, not
algorithm, numerical, GPU, or scientific evidence.

Scheduler terminal state is `FAILED/1:0`, elapsed 18 seconds, node821. Root
status/rc are `READONLY_CAPTURE_FAIL/1`; there is no live process and no real
OOM, CUDA, NCCL, disk, NaN, or Inf failure. Complete model-free evidence is
archived in
`remote_launch_staging/procgen_normmatch_v2_mp_main_inpath_capture_readonly_20260825_31r/evidence/task31r_model_free_evidence_19279429.tar.gz`
with SHA256
`fea85c23140260188668fa77a3ea49150125046b149a021ef1f8735717a9bfbd`.

No classifier, policy, allowlist, manifest, bundle, frozen scientific file,
formal audit, four-environment preflight, science, scientific root,
transition, checkpoint/model, cancellation, or monitor was created or
modified. Task29 and Task30 ledgers remain preserved.

Report:
`.agent/reports/PROCGEN-NORMMATCH-V2-MP-MAIN-INPATH-CAPTURE-READONLY-20260825-31R.md`.

TASK_COMPLETE
