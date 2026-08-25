# PROCGEN-NORMMATCH-V2-BARE-EXEC-NAMESPACE-RECOVERY-AND-6M-S0-20260825-19

## Conclusion

`PRECHECK_BLOCKED`

The bounded explicit-policy namespace implementation and every local gate
passed, but the single authorized remote clean-room audit failed in the
prestart path before an origin-policy ledger was written. Task19 forbids field
repair or retry after that audit failure, so no four-environment real-network
preflight or scientific cell was submitted.

## Assignment and frozen identity

- Assignment/origin: `4db20d741410c95d708ddbee3840de7c1323204e`
- Audit-only freeze/origin: `bec45a4a15d3c25d648000727842b4e953899c70`
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
- Task18 provenance probe: `917faf124f35ca7a1c4ceef4a8dc43183500cbc2131c71f95f8d2186c51f6c23`
- generated module: `8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`

Task14--18 failure ledgers and all scientific/bundle/deployment identities
remain unchanged. Task19 changes only the audit-only policy namespace support,
prestart/probe wrappers, audit launcher and their tests.

## Bounded namespace implementation and local gates

The audit launcher supplies the required absolute policy path through
`NORMMATCH_V2_ORIGIN_POLICY_PATH`. The audit-only support validates a regular
non-symlink file, current owner, exact mode `0644`, exact Task18 SHA256, and
separation from the designated empty cwd. It records path/device/inode/owner/
mode/size/SHA and injects the validated policy filename only as execution
metadata for the otherwise bare policy namespace. It performs no cwd search,
glob, filename guessing, fallback or missing-variable suppression.

All mandatory local gates passed:

- explicit path with a bare caller namespace: PASS;
- missing variable, nonexistent path, symlink and wrong SHA: rejected;
- eager `__file__` default under a truly bare namespace: reproduced failure;
- ordinary-module and explicit bare-loader API/snapshot results: identical;
- Task16 designated-empty and Task17 interpreter-zip regressions: PASS;
- Task18 exact Torch generator/content/lifecycle regression in the frozen
  remote environment: PASS; and
- frozen scientific, bundle, launcher, policy and provenance hashes: PASS.

Task19 audit-only SHA256 values:

| File | SHA256 |
|---|---|
| `clean_room_bundle_audit_task19_gpuh.sbatch` | `51bdeee8d101e26e61bf92f7c68efc84477a8c659bb96cbfa6cbaabcadaaf719` |
| `policy_namespace_support.py` | `5d5fd71f81bb7678399607fe7fa3881bb153507b00fdb43d875394aaf407c4c3` |
| `prepare_designated_empty_task19.py` | `cfff25b3c7e5170e062a49b38f6f666aedafefd76ebac7ed7993c387f0f6b2cc` |
| `clean_room_probe_task19.py` | `549a911e96ac47d1beb1db6e429d1cef82ec65add2ef2504ee4608e52aec13df` |
| `test_policy_namespace_support.py` | `d6673e040be4f0787c84385f5bcee9b644d800e6bb33f724a6e26992082085ab` |
| `test_frozen_identity.py` | `b444fffe9bd13fe57a8e63ea2db270edc0259faa211e1b4d42eb349dc5a07f55` |

## Single clean-room audit

Exactly one audit was submitted after freeze and regression:

| Job | Owner | Placement | Node/GPU | State | Exit | Elapsed |
|---:|---|---|---|---|---|---:|
| `19258476` | `h99859yz` | `gpuH` / `gpu-h200-fse-pgdr` / `gpu-h200-fse` | node820 / NVIDIA H200 | FAILED | `1:0` | `00:00:03` |

The immutable bundle gate passed:

```text
HERMETIC_BUNDLE_VERIFY_PASS
bundle_sha256=3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f
manifest_sha256=99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa
files=32
```

The prestart then rejected the supplied path at the audit-only canonical-string
check:

```text
RuntimeError: origin-policy path must already be canonical
```

The launcher supplied the ordinary non-symlink path
`/scratch/h99859yz/.../tools/origin_safety.py`. On this CSF3 node,
`Path.resolve(strict=True)` canonicalized the mount spelling to
`/net/scratch/h99859yz/.../tools/origin_safety.py`. Both spellings identify the
same regular `0644` file owned by UID `778916`, device `3592384858`, inode
`144122242006038476`, size `13605`, and SHA256 `889b914a...`. The extra
raw-string-equals-resolved-string assertion therefore failed before the ledger
write.

This is `infrastructure-failure/clean-room-prestart-path-canonicalization`, not
algorithm, numerical, solver, H200, memory, reward or training evidence. The
Task19 no-repair/no-retry rule makes it terminal.

## Unreached execution and ledger

- Policy prestart ledger: not emitted; failure occurred immediately before it.
- Designated-empty snapshot: not emitted.
- Audited interpreter/import-origin manifest: not reached.
- Four-environment real-network preflight: not submitted.
- Task15 `preflight`, `accepted_preflight`, and `runs`: all zero entries.
- Scientific jobs/roots/processes/transitions/progress/traces/checkpoints/models: none.
- Exact 2M/4M/5,980,160 Target/Paper comparisons: none eligible.
- Scientific monitor: not created.
- New failure ledger row: `19258476`,
  `infrastructure-failure/clean-room-prestart-path-canonicalization`.

No repair, retry, requeue, resubmit, second audit, field mutation, second
candidate, Jupyter, quarantined access, Paper rerun, sweep, overwrite or
unrelated job mutation occurred. Complete model-free evidence is under
`remote_launch_staging/procgen_normmatch_v2_bare_exec_namespace_recovery_6m_s0_20260825_19/evidence_remote/`.

PRECHECK_BLOCKED
