# PROCGEN-NORMMATCH-V2-HERMETIC-BUNDLE-AND-6M-S0-20260825-15

## Conclusion

`PRECHECK_BLOCKED`

The content-addressed source bundle and normalized launcher-equivalence audit
passed, but the single mandatory clean-room job failed its `sys.path` gate
before importing the trainer. The task explicitly forbids repair or retry
after a failed gate. Therefore no four-environment real-network preflight and
no scientific cell was submitted.

## Assignment and immutable scientific identity

- Assignment/origin commit: `2d0932e37884584601d53318ece9cb16f400fba6`
- Frozen bundle/launcher commit: `0623ecff91fc856d1fe42254ea4d881af55b5c5f`
- Method: `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`
- trainer: `0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b`
- config: `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- preflight: `b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc`
- regression: `f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c`
- monitor: `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`
- preserved original launcher: `85e12886ce5cf81fd98647aa5163319a50174a39210cbeea1ccfde015aaf9d19`
- deployment science launcher: `ec60864aaa9940fd61eb1391008b50f2f48402eeae8f78c91c0b0c1fc313a398`
- deployment preflight launcher: `374d24881d108bbdd08dee0880b7392a7cc3adf6af177f6eb4deaac15535b228`

No frozen scientific file changed. Task 14 jobs `19238126`--`19238129`
remain immutable `infrastructure-failure/deployment-package-incomplete`
records with their original missing-`utils` tracebacks.

## Hermetic bundle

The builder reads every payload byte from frozen Git commit
`c2470eac175ee7a05904a73fa2d93bac1f643cf7`. It records repository path,
Git blob SHA-1, SHA256, size, imports and resolved repository-local edges.

- archive SHA256: `3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f`
- manifest SHA256: `99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa`
- manifest files: 32
- reachable repository-local closure: 23 files
- `utils`/`vec_env` lineage: `work/procgen_paper_2b5affd_6m_bede_20260722/source`

The closure contains the V2 trainer; 13 required `utils` modules; and all nine
runtime `vec_env` modules. Bundled `code/utils/utils.py` is SHA256
`3c39421c055fc3414b9fe55e17cd339576fb5567520c940254addaf0afb1b358`,
matching the previously successful Hybrid runtime and an immutable Git blob.
A second build produced a byte-identical archive and manifest. Independent
local extraction verified the complete file set and every hash.

## Launcher equivalence

The variants only verify/extract the archive, set `PYTHONNOUSERSITE=1`, and
set the explicit bundle code directory as `PYTHONPATH`/working directory.
Original and deployment science launchers contain the same command:

```text
CMD=("$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name "$ENV_NAME" --seed 0 --device 0)
```

Command, argument order, environment, seed, device, config and 6M budget are
byte-identical after path normalization. `LAUNCHER_EQUIVALENCE.json` records
`DEPLOYMENT_LAUNCHER_EQUIVALENCE_PASS`.

## Mandatory clean-room gate

Exactly one no-training audit was submitted:

| Job | Node | State | Exit | Elapsed | Bundle verification | Trainer import |
|---:|---|---|---|---:|---|---|
| 19241161 | node820 H200 | FAILED | 1:0 | 00:00:04 | PASS | not reached |

Verifier output:

```text
HERMETIC_BUNDLE_VERIFY_PASS
bundle_sha256=3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f
manifest_sha256=99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa
files=32
```

The next assertion failed before trainer import because Python retained the
deliberately empty working directory as its first search path:

```text
RuntimeError: non-hermetic sys.path entry:
/mnt/iusers01/fatpou01/compsci01/h99859yz/tmp/procgen-nm2-empty-19241161.WFurfV
```

This is a clean-room harness/design failure, not algorithm, solver, numerical,
environment, H200, reward or training evidence. The task mandates terminal
`PRECHECK_BLOCKED` with no repair or retry after any clean-room failure.

## Unreached execution and safety reconciliation

- No real-network preflight job/file or accepted-preflight directory exists.
- No scientific job, root, process, transition, progress, trace, checkpoint or model exists.
- No exact 2M/4M/5,980,160 Target/Paper comparison was eligible.
- No monitor was created; final `squeue` contained no Task 15 target row.
- No retry, requeue, resubmit, field repair, second candidate, Jupyter,
  quarantined access, Paper rerun, sweep, overwrite or unrelated mutation occurred.

Model-free evidence is under
`remote_launch_staging/procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15/evidence_clean_room/`.
It includes accounting, H200 identity, timestamps, status/rc, successful
bundle verification and the exact traceback. No checkpoint/model is committed.

PRECHECK_BLOCKED
