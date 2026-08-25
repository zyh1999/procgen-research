# Executor Report

## Metadata

- Task-ID: `PROCGEN-NORMMATCH-V2-MP-MAIN-EXACT-ALIAS-AND-6M-S0-20260825-29`
- Assignment: `28b1585808ce136fc48cd664bca5209a2f5239cf`
- Method retained: `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`
- Repository target: `origin/agent-work`

## Result

The mandatory read-only Python 3.9.25 proof did not establish the strict
`__main__`/`__mp_main__` alias required to add the new classifier.

Actual stdlib identities were verified: import-time assignment
`multiprocessing/__init__.py:37` has SHA256 `a5a42976...`, and spawn reset
assignments `multiprocessing/spawn.py:262,290` have SHA256 `16ce6d81...`.
Initial proof job `19277384` demonstrated that importing `multiprocessing`
from the observer artificially makes the two keys the same object, but it also
changes the frozen scan and is preserved as an observer-effect infrastructure
failure. The corrected harness contains no early multiprocessing import.

With natural frozen construction timing, job `19277433` passed the bundle and
complete production CUDA/Task27 preflight, then stopped before module scan
because `sys.modules["__main__"] is sys.modules["__mp_main__"]` was false.
Scheduler state is `FAILED/2:0`, elapsed 22 seconds, node821. This is the exact
mandatory proof failure, not trainer, NormMatch, deterministic GGN, numerical,
solver, H200, reward, or scientific evidence.

No classifier or allowlist was implemented; Task28R and all frozen scientific
identities are unchanged. No closure, formal audit, four-environment preflight,
scientific job/root/transition/checkpoint, stage comparison, cancellation, or
monitor was created. Failure ledger:
`precheck-failure/task29-natural-mp-main-not-exact-main-object-alias`.

Report:
`.agent/reports/PROCGEN-NORMMATCH-V2-MP-MAIN-EXACT-ALIAS-AND-6M-S0-20260825-29.md`.

BLOCKED
