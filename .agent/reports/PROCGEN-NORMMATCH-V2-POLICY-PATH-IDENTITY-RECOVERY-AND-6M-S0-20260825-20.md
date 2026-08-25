# PROCGEN-NORMMATCH-V2-POLICY-PATH-IDENTITY-RECOVERY-AND-6M-S0-20260825-20

## Conclusion

`PRECHECK_BLOCKED`

The bounded path-identity implementation and all specified local regressions
passed, but the single authorized remote clean-room audit failed in prestart
before an identity ledger was written. Task20 forbids repair or retry after
that audit failure, so no four-environment real-network preflight or scientific
cell was submitted.

## Assignment and frozen identity

- Assignment/origin: `60c195be34bdcd3853770dfe00aa62e2cbef3350`
- Audit-only freeze/origin: `c9518163c7eef295f3acbd632e4935bd09f9dfdf`
- Method: `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`
- trainer: `0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b`
- config: `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- scientific preflight: `b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc`
- regression: `f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c`
- monitor: `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`
- archive/manifest: `3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f` /
  `99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa`
- science/preflight launchers: `ec60864aaa9940fd61eb1391008b50f2f48402eeae8f78c91c0b0c1fc313a398` /
  `374d24881d108bbdd08dee0880b7392a7cc3adf6af177f6eb4deaac15535b228`
- Task18 origin policy: `889b914a792132358f90572d5c4f16b561c60bf0615eb8492c9ecf781f601fc1`
- generated Torch module: `8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`

Task14--19 ledgers and every scientific/bundle/deployment/provenance identity
remain unchanged. Task20 adds only the audit-only path identity validator,
its tests and a versioned audit launcher; it reuses Task19's frozen prestart
and clean-room probe byte-for-byte.

## Bounded identity correction and local gates

The validator removes raw-string equality and instead requires strict
resolution, `samefile`, equal raw/resolved device and inode, ordinary
non-symlink final target, frozen owner/gid/mode/size and exact SHA. It opens the
resolved target read-only with `O_NOFOLLOW`, proves the fd identity, executes
the fd bytes, then revalidates fd/path identity and SHA. No storage spelling,
path search, glob or filename fallback is present.

All specified local gates passed:

- parent storage alias to the same file/device/inode and SHA: PASS;
- byte-identical different inode: rejected;
- final-component symlink and missing path: rejected;
- device/inode/owner/mode/size/SHA mismatch: rejected;
- replacement after resolve and after execution: rejected;
- Task19 bare namespace, Task18 Torch provenance, Task17 interpreter zip and
  Task16 designated-empty protections: retained PASS; and
- all frozen identity, compilation and Slurm syntax checks: PASS.

Task20 audit-only SHA256 values:

| File | SHA256 |
|---|---|
| `policy_namespace_support.py` | `2e65f545ccdec2b5306b0c2d1366784b20de38e7d6d141d994a2c1d0aa8baf92` |
| `test_policy_path_identity.py` | `2fae8f67ad232e0d63e3b65adff97eda9fa917180ea59484b29f22457dd4f6af` |
| `test_frozen_identity.py` | `54d7a8761d457f193bf05eae097672f1cb24c43c55d925686a461ab7b6f11c86` |
| `clean_room_bundle_audit_task20_gpuh.sbatch` | `ec5c812d0b10abbc1f6e5632c46b653b8510be914f0998c152ff5b0b6728a5be` |

## Single clean-room audit

Exactly one audit was submitted after freeze and local regression:

| Job | Owner | Placement | Node/GPU | State | Exit | Elapsed |
|---:|---|---|---|---|---|---:|
| `19260683` | `h99859yz` | `gpuH` / `gpu-h200-fse-pgdr` / `gpu-h200-fse` | node820 / NVIDIA H200 | FAILED | `1:0` | `00:00:04` |

The immutable bundle gate passed:

```text
HERMETIC_BUNDLE_VERIFY_PASS
bundle_sha256=3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f
manifest_sha256=99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa
files=32
```

Prestart strictly resolved the supplied path and then failed at the first
raw-stat call:

```text
TypeError: stat() got an unexpected keyword argument 'follow_symlinks'
```

The frozen remote interpreter is Python `3.9.25`; its
`pathlib.Path.stat` signature is `(self)`. The local regression interpreter is
Python 3.13, where `Path.stat(follow_symlinks=False)` is supported. The remote
policy target itself remained the exact Task19 record: regular file, mode
`0644`, UID `778916`, GID `10049`, device `3592384858`, inode
`144122242006038476`, size `13605`, SHA256 `889b914a...`, with raw
`/scratch/...` and resolved `/net/scratch/...` spellings.

This is `infrastructure-failure/clean-room-prestart-python-api-compatibility`,
not a path-identity rejection and not algorithm, numerical, solver, H200,
memory, reward or training evidence. Task20's no-repair/no-retry rule makes it
terminal.

## Unreached execution and ledger

- Path-identity ledger: not emitted; failure occurred before raw/resolved stat
  records, fd open, policy execution and post-exec revalidation.
- Designated-empty snapshot: not emitted.
- Audited interpreter/import-origin manifest: not reached.
- Four-environment real-network preflight: not submitted.
- Task15 `preflight`, `accepted_preflight`, and `runs`: all zero entries.
- Scientific jobs/roots/processes/transitions/progress/traces/checkpoints/models: none.
- Exact 2M/4M/5,980,160 Target/Paper comparisons: none eligible.
- Scientific monitor: not created.
- New failure ledger row: `19260683`,
  `infrastructure-failure/clean-room-prestart-python-api-compatibility`.

No repair, retry, requeue, resubmit, second audit, field mutation, second
candidate, Jupyter, quarantined access, Paper rerun, sweep, overwrite or
unrelated job mutation occurred. Complete model-free evidence is under
`remote_launch_staging/procgen_normmatch_v2_policy_path_identity_recovery_6m_s0_20260825_20/evidence_remote/`.

PRECHECK_BLOCKED
