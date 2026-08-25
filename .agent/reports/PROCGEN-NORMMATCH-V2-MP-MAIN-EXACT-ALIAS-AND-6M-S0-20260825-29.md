# PROCGEN-NORMMATCH-V2-MP-MAIN-EXACT-ALIAS-AND-6M-S0-20260825-29

## Conclusion

`PRECHECK_BLOCKED`

The actual Python 3.9.25 natural-context proof did not establish the strict
`__main__`/`__mp_main__` object alias required by Task29. The task therefore
stopped before classifier implementation. No bundle manifest or allowlist was
broadened, and no closure, formal audit, environment preflight, or science was
run.

## Frozen identity and scope

- Assignment: `28b1585808ce136fc48cd664bca5209a2f5239cf`
- Task28R delivery: `99fd6086b017e1082a16a63d15c3edbe4c81b67a`
- Task28R exact-probe validator SHA256:
  `96da9c8ee7497ce01df3230f6fa0875a81d1175c958c1a5d8276390090c118ad`
- Frozen probe SHA256:
  `c3529cb171306d7b3b0517974a682ddffb91d65dc015c102b1e84658e9eeb1f5`
- Frozen probe device/inode: `3592384858/144122242274496637`
- Method retained: `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`

Task28R, trainer, config, scientific preflight/regression, bundle/manifest,
launchers, stage monitor, Task23 hook, Task25 classifier, Task26 AST helper,
and Task27 binding preflight were not edited.

## CPython 3.9 source provenance

The actual CSF3 interpreter is Python `3.9.25`, GCC 11.5.0. Its relevant
standard-library source is:

| Role | Source | SHA256 | Exact semantics |
|---|---|---|---|
| import-time alias | `/usr/lib64/python3.9/multiprocessing/__init__.py` | `a5a42976033c7d63ee2740acceef949a3582dcb0e0442845f9717e1be771c68b` | line 37 assigns `sys.modules['__mp_main__'] = sys.modules['__main__']` |
| spawn child reset | `/usr/lib64/python3.9/multiprocessing/spawn.py` | `16ce6d81f8b5ef7228e5500bff04b37bdceb3d7dfc8d6de3ad523598798c43f4` | `spawn_main -> _main -> prepare -> _fixup_main_from_path`; lines 262 and 290 assign both keys to one reconstructed module |

Both files were parsed and compiled, and their exact alias assignments and
call edges were recorded by the proof harness.

## Proof executions

### Observer-effect attempt `19277384`

- Scheduler: `FAILED/1:0`, node821, 18 seconds.
- Bundle verification and complete production CUDA/Task27 preflight passed.
- The observer imported `multiprocessing` before the frozen construction. It
  then observed `__main__ is __mp_main__`, exact frozen probe backing,
  identical dictionaries, raw/resolved samefile, device/inode
  `3592384858/144122242274496637`, UID/GID `778916/10049`, mode `0644`, size
  4,558 and exact probe SHA.
- That import changed alias creation timing. Task28R consequently saw two
  modules with the raw probe origin and stopped with `exact probe origin must
  map to exactly one loaded module` rather than reproducing the natural scan.
- Classification:
  `infrastructure-failure/proof-observer-import-timing`. This attempt is
  retained as negative evidence: the desired exact alias appears only after
  an observer mutation prohibited by the task.

### Natural-context proof `19277433`

- The only harness correction removed the premature `multiprocessing` import;
  no frozen file or acceptance logic changed.
- Scheduler: `FAILED/2:0`, node821, 22 seconds.
- Bundle verification passed. The exact 938,979-parameter CUDA network,
  structural/connectivity checks, Task27 direct semantic binding, config
  identity, actor/shared/head update checks and solver telemetry all passed.
- At the pre-scan proof boundary, naturally imported `multiprocessing` was
  present, but `sys.modules["__main__"] is sys.modules["__mp_main__"]` was
  false. The proof stopped immediately with:

```text
RuntimeError: live __main__/__mp_main__ exact object alias not established
```

This fails Task29's mandatory strict relationship. No classifier may convert
the stale/different object into an approved alias merely by key, basename,
SHA, import presence, or Python version.

## Preserved scientific telemetry

Both proof executions reached the unchanged Task27 preflight before the alias
gate. The frozen telemetry remains:

- deterministic/Paper/target head proposal norms:
  `.6050832272/.9192549586/.9192548990`
- norm-match scale: `1.519220710`
- deterministic/Paper cosine: `.8612535000`
- FP64 relative residual: `8.627e-16`
- Cholesky info: `0`

These values confirm unchanged production construction only. They are not
reward or training evidence.

## Failure and non-actions

Failure ledger increment:

```text
precheck-failure/task29-natural-mp-main-not-exact-main-object-alias
```

No `APPROVED_CPYTHON39_MULTIPROCESSING_MAIN_ALIAS` classifier, atomic
acceptance ledger, generic framework, allowlist, manifest entry, formal
closure, second clean process, normalized closure, formal audit, per-environment
preflight, science job/root, transition, progress, trace, checkpoint/model,
stage comparison, cancellation, retry/requeue/resubmit, Jupyter session,
quarantined-host access, duplicate objective, sweep, or unrelated mutation
exists.

Model-free evidence is under
`remote_launch_staging/procgen_normmatch_v2_mp_main_exact_alias_6m_s0_20260825_29/evidence_remote/`.
