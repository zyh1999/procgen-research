# PROCGEN-NORMMATCH-V2-TORCH-GENERATED-ORIGIN-AUDIT-AND-6M-S0-20260825-18

## Conclusion

`PRECHECK_BLOCKED`

Strict generator provenance and every mandatory regression passed, but the
single authorized clean-room audit failed in the prestart harness before the
audited interpreter. Task18 forbids repair or retry after that audit failure,
so no four-environment preflight or scientific cell was submitted.

## Assignment and immutable identity

- Assignment/origin: `1e8c8aa56bc5e9d242e13586c2af9bda3d054c2e`
- Auditor/provenance freeze: `793a49d35699ca755c18f45c3ea080c8850bab03`
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

Task14--17 files and failure ledgers remain unchanged. No bundle was rebuilt or
repacked, and no scientific identity changed.

## PyTorch generator/loader provenance

Two independent clean processes in the exact frozen environment produced
separate temporary directories and identical stable evidence:

| Field | Evidence |
|---|---|
| distribution | `torch` `2.5.1+cu121`, installer `pip`, no `direct_url.json` |
| trigger | `torch.distributed.nn.api.remote_module`, import-time `instantiate_non_scriptable_remote_module_template()` |
| generator/loader | `torch.distributed.nn.jit.instantiator.instantiate_non_scriptable_remote_module_template -> _do_instantiate_remote_module_template -> _write -> importlib.import_module` |
| template | `get_remote_module_template(True)` with fixed non-scriptable substitutions |
| generated name | `_remote_module_non_scriptable` |
| loader | `_frozen_importlib_external.SourceFileLoader` |
| package | empty string |
| content | 2,355 bytes; SHA256 `8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8` |
| content proof | byte-identical installed deterministic template; AST and compile PASS |
| lifecycle | parent/file created after process start; UID `778916`; parent mode `0700`; ordinary non-symlink file |

Installed source provenance:

| Role | Distribution path | SHA256 |
|---|---|---|
| trigger | `torch/distributed/nn/api/remote_module.py` | `55c9c44ba25a2b5edf105fbd740ceff771f937147d7d6a9d6232f05681e7eeaf` |
| generator/loader | `torch/distributed/nn/jit/instantiator.py` | `440a619c764e4133564d7956ba060a7223e94664854b94a4a2074d095756db7e` |
| template | `torch/distributed/nn/jit/templates/remote_module_template.py` | `0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc` |

All three paths, sizes and SHA256-encoded hashes match Torch's installed
distribution RECORD. The two reproduction manifests differ only in expected
process timestamps and randomized temporary paths; module content, generator,
loader, distribution and semantic metadata are identical.

## Narrow category and regressions

The only added category is
`APPROVED_RUNTIME_GENERATED_THIRDPARTY_MODULE`. It is reachable only for the
exact module after simultaneous proof of installed generator sources/RECORD,
module/spec/package/loader, post-policy parent and file lifecycle, owner and
permissions, exact deterministic template/hash, AST/compile, approved import
roots, absence of repository/network references, and post-import
origin/inode/metadata/hash revalidation. No temporary directory or general
filename category is permitted.

The exact frozen-environment regression passed the real module and rejected:

- same-name preexisting module/file;
- content, AST, hash or loader mismatch;
- symlink parent and symlink file;
- non-PyTorch generator/loader;
- post-import file replacement; and
- bundle-external repository or network references.

Task16 designated-empty and Task17 interpreter-zip/out-of-bundle tests remain
PASS. Frozen identity, compilation and launcher syntax checks also passed.

Task18 audit file SHA256 values:

| File | SHA256 |
|---|---|
| `clean_room_bundle_audit_task18_gpuh.sbatch` | `87f8a4dd3075219e21607c883b60b3469bde57903120e71dd043f015db264ee3` |
| `origin_safety.py` | `889b914a792132358f90572d5c4f16b561c60bf0615eb8492c9ecf781f601fc1` |
| `provenance_probe.py` | `917faf124f35ca7a1c4ceef4a8dc43183500cbc2131c71f95f8d2186c51f6c23` |
| `test_torch_generated_origin_safety.py` | `69bb6ed1bfa6fc961b48976178f726305e3649ef93da997355de4595ad8cf697` |

## Single clean-room audit

Exactly one audit was submitted after the provenance and regression freeze:

| Job | Owner | Placement | Node/GPU | State | Exit | Elapsed |
|---:|---|---|---|---|---|---:|
| `19254931` | `h99859yz` | `gpuH` / `gpu-h200-fse-pgdr` / `gpu-h200-fse` | node820 / NVIDIA H200 | FAILED | `1:0` | `00:00:04` |

The immutable bundle gate passed:

```text
HERMETIC_BUNDLE_VERIFY_PASS
bundle_sha256=3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f
manifest_sha256=99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa
files=32
```

The next prestart command used Task17's generic bare-namespace executor. While
evaluating Task18 origin policy, the environment-provided base path was valid,
but Python still evaluated the default argument expression containing
`Path(__file__)`; the bare namespace has no `__file__`:

```text
NameError: name '__file__' is not defined
```

The failure occurred before `DESIGNATED_EMPTY_PRESTART_PASS`, the audited
interpreter, runtime-generated category, import-origin manifest, clean-room
manifest or input hash ledger. This is a prestart audit-harness namespace
failure. It does not invalidate the separately completed generator provenance
or regression evidence, but Task18 explicitly forbids repairing/retrying the
single audit.

## Unreached execution and ledger

- Four-environment real-network preflight: not submitted.
- Task15 `preflight`, `accepted_preflight`, and `runs`: zero entries.
- Scientific jobs/roots/processes/transitions/progress/traces/checkpoints/models: none.
- Exact 2M/4M/5,980,160 Target/Paper comparisons: none eligible.
- Scientific monitor: not created.
- New ledger row: `19254931`,
  `infrastructure-failure/clean-room-prestart-namespace`; bundle PASS;
  bare-exec `__file__` NameError; designated-empty and audited interpreter not reached.

No retry, requeue, resubmit, field repair, second candidate, Jupyter,
quarantined access, Paper rerun, sweep, overwrite or unrelated mutation
occurred. Complete model-free evidence is under
`remote_launch_staging/procgen_normmatch_v2_torch_generated_origin_audit_6m_s0_20260825_18/`.

PRECHECK_BLOCKED
