# PROCGEN-NORMMATCH-V2-PY39-LSTAT-COMPATIBILITY-AND-6M-S0-20260825-21

## Conclusion

`PRECHECK_BLOCKED`

The bounded Python 3.9 stat compatibility correction passed every mandatory
compatibility and path-identity gate. The single authorized clean-room audit
then failed because the unchanged origin policy rejected a newly observed
Torch-generated `_classes.py` module inside the designated-empty directory.
Task21 forbids repair or retry after any audit failure, so no real-network
preflight or scientific cell was submitted.

## Assignment and immutable scientific identity

- assignment/origin: `5e041cd82ae5a4a078baaa0aa8991cc2b861ee41`
- compatibility freeze/origin: `2230ef6485e5e8f7f5529d3595c65aec0241b056`
- method: `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`
- trainer: `0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b`
- config: `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- scientific preflight/regression/monitor: `b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc` / `f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c` / `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`
- bundle/manifest: `3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f` / `99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa`
- science/preflight launchers: `ec60864aaa9940fd61eb1391008b50f2f48402eeae8f78c91c0b0c1fc313a398` / `374d24881d108bbdd08dee0880b7392a7cc3adf6af177f6eb4deaac15535b228`
- Task18 origin policy: `889b914a792132358f90572d5c4f16b561c60bf0615eb8492c9ecf781f601fc1`
- Torch generated module: `8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`

Task14--20 evidence and failure ledgers remain immutable. No algorithm,
bundle, origin policy, scientific file, deployment launcher, Torch provenance,
monitor, baseline or scientific identity changed.

## Bounded compatibility correction

Only the Task20 audit validator was versioned. `Path.stat(follow_symlinks=...)`
was replaced by `os.lstat`, `os.stat`, `os.path.samefile`, `os.open`, and
`os.fstat`. The original device/inode, ordinary non-symlink, UID/GID,
mode, size, SHA, fd identity, and pre/post replacement guards remain.

| Task21 file | SHA256 |
|---|---|
| `policy_namespace_support.py` | `7e168404a858b28c841b23e812cf0ae1ffe3f5ffc4ed16e9de3792ef727a65e8` |
| `test_policy_path_identity_py39.py` | `779c9a04e92aab73a3ce17124a19810389218f5f3617ef15d4d48386dd1b876d` |
| `test_frozen_identity.py` | `ac2db70b3196cad1160bc8a392d647d6398d740951079f451aeb5fd960b054ca` |
| `clean_room_bundle_audit_task21_gpuh.sbatch` | `85ed9bdefc00060e4c9a0c3b1065cb3bfe2009c538e534e64c4724ec4c052688` |

Local Task16--20 safety and frozen-identity regressions passed where their
dependencies were available. The authoritative remote compatibility suite used
the actual interpreter and frozen Torch provenance environment:

```text
Python 3.9.25
pathlib.Path.stat signature: (self)
O_NOFOLLOW_AVAILABLE=True
O_NOFOLLOW_VALUE=131072
TASK21_UNSUPPORTED_PATH_STAT_STATIC_SCAN_PASS
TASK21_PY39_POLICY_PATH_IDENTITY_TESTS_PASS
PY39_COMPATIBILITY_PASS rc=0
```

The positive same-inode storage alias passed. Final symlink, identical bytes on
a different inode, missing path, wrong device/inode/UID/GID/mode/size/SHA,
replacement after resolve, and replacement after policy execution were all
rejected.

## Identity and fd ledger

The audit prestart ledger proves raw `/scratch/.../origin_safety.py` and
resolved `/net/scratch/.../origin_safety.py` are the same file. Raw/resolved
lstat and stat, opened fd, post-exec fd, and post-exec path all report device
`3592384858`, inode `144122242006038476`, UID `778916`, GID `10049`, mode
`0644`, size `13605`, regular file, non-symlink. `O_NOFOLLOW` was available and
applied with value `131072`. Pre-exec, post-exec-fd, and post-exec-path SHA256
all equal the frozen policy SHA `889b914a...`. The designated directory was
UID-owned mode `0700` and empty before interpreter start.

## Single remote audit

Exactly one audit was submitted after all compatibility gates:

| Job | Owner | Placement | Node | State | Exit | Elapsed |
|---:|---|---|---|---|---|---:|
| `19263636` | `h99859yz` | `gpuH`, account `gpu-h200-fse-pgdr`, 1 H200 | node820 | FAILED | `1:0` | `00:00:14` |

Bundle/manifest verification passed. The stat/fd identity ledger passed and the
policy executed. During the unchanged exhaustive loaded-module origin scan,
the audit rejected:

```text
RuntimeError: module resolved from designated empty directory:
/mnt/iusers01/fatpou01/compsci01/h99859yz/tmp/
procgen-nm2-task21-empty-19263636.8dOEUz/_classes.py
```

This is `infrastructure-failure/clean-room-loaded-module-origin-policy`. It is
not a Python stat compatibility failure, file-identity failure, algorithm,
numerical, solver, H200, memory, reward, or training result. Task21 authorizes
no field repair and no second audit.

## Unreached gates and immutable ledger

- four-environment real-network preflight: not submitted;
- accepted preflight: none;
- scientific jobs, roots, processes, transitions, logs, traces, checkpoints,
  models: none;
- exact 2M/4M/5,980,160 Target/Paper comparisons: none eligible;
- scientific monitor: not created; and
- Task21 failure-ledger addition: job `19263636`,
  `infrastructure-failure/clean-room-loaded-module-origin-policy`.

No repair, retry, requeue, resubmit, second audit, second candidate, sweep,
Jupyter, quarantined access, Paper rerun, overwrite, or unrelated mutation
occurred. Model-free evidence is under
`remote_launch_staging/procgen_normmatch_v2_py39_lstat_compatibility_6m_s0_20260825_21/evidence_remote/`.

PRECHECK_BLOCKED
