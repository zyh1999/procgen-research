# Executor Report

## Metadata

- Task-ID: `PROCGEN-NORMMATCH-V2-EXACT-PROBE-ALIAS-RECOVERY-AND-6M-S0-20260825-28R`
- Assignment: `0d913f8d82611fa1ee659f0071994e4a18a2d0de`
- Validator/closure freeze: `9174b00ab74d317c28897348b0c6c74020dcae3d`
- Method: `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`
- Repository target: `origin/agent-work`

## Result

Task28R added only `APPROVED_EXACT_FROZEN_CLOSURE_PROBE_ALIAS`, bound to
frozen probe commit `baab71b...`, blob `e4c63952...`, SHA256 `c3529cb1...`,
and exact CSF3 raw/resolved same-file identity. Validator SHA is
`96da9c8ee7497ce01df3230f6fa0875a81d1175c958c1a5d8276390090c118ad`.
Local and actual Python 3.9.25 positive/negative tests passed; all scientific
and prior audit identities stayed frozen.

The single closure job `19277045` ran on gpuH node821 and ended `FAILED/1:0`
after `00:00:22`. Both bundle checks and the real CUDA Task27 preflight passed.
The prior closure-probe `/net/scratch` rejection did not recur, so the exact
alias classification succeeded. The unchanged downstream audit then failed
on multiprocessing alias `__mp_main__` with `RuntimeError: bundle module
absent from manifest or hash mismatch: __mp_main__`.

Failure ledger:
`precheck-failure/task28r-frozen-mp-main-bundle-manifest-alias`. The exception
occurred before the post-scan exact-probe ledger and first reproduction JSON
could be emitted. Task28R forbids closure repair/retry, so no formal audit,
environment preflight, science, stage comparison, cancellation, or monitor
was created.

Report:
`.agent/reports/PROCGEN-NORMMATCH-V2-EXACT-PROBE-ALIAS-RECOVERY-AND-6M-S0-20260825-28R.md`.

BLOCKED
