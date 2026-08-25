# PROCGEN-NORMMATCH-V2-RUNTIME-GENERATED-CLOSURE-AUDIT-AND-6M-S0-20260825-22

## Conclusion

`PRECHECK_BLOCKED`

The required complete runtime-generated closure could not be stably reproduced.
Two independent clean Python 3.9 imports consistently proved that the observed
`_classes.py` is a synthetic `torch.classes.__file__` spelling with no physical
artifact, spec, loader, package, origin, or file lifecycle. The subsequent
full production-model closure provenance job failed in its first process when
the filesystem audit hook recursively audited its own stack-source reads.
Task22 explicitly requires `PRECHECK_BLOCKED` when the complete normalized
closure cannot be reproduced and forbids retry. No formal clean-room audit,
real-network preflight, or scientific cell was submitted.

## Assignment and immutable identity

- assignment/origin: `8eb97a9f489268644d88ac069ab0c2d6fac23f32`
- closure-gate freeze/origin: `6c0d6f1f359c7e0b9f022faf5d9682798cbe53b7`
- method: `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`
- Task21 terminal delivery: `782790a2708b355d8b928d004eed5d3df50be0b4`
- trainer/config: `0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b` / `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- scientific preflight/regression/monitor: `b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc` / `f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c` / `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`
- bundle/manifest: `3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f` / `99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa`
- science/preflight launchers: `ec60864aaa9940fd61eb1391008b50f2f48402eeae8f78c91c0b0c1fc313a398` / `374d24881d108bbdd08dee0880b7392a7cc3adf6af177f6eb4deaac15535b228`
- Task18 origin policy: `889b914a792132358f90572d5c4f16b561c60bf0615eb8492c9ecf781f601fc1`
- Task21 Python 3.9 path-identity validator: `7e168404a858b28c841b23e812cf0ae1ffe3f5ffc4ed16e9de3792ef727a65e8`

Task14--21 evidence and ledgers are unchanged. The algorithm, trainer/config,
preflight/regression/monitor, bundle/manifest, science/preflight launchers,
Task18 origin policy, prior `_remote_module_non_scriptable.py` provenance,
Task21 stat/fd/path/SHA logic, and scientific identity remain byte-identical.

## Frozen closure gate

| File | SHA256 |
|---|---|
| `runtime_generated_closure_probe.py` | `0dbcc8bf3214528756748304b938420fdd7dbddf4e4f08c634331e4717c5cefb` |
| `analyze_closure_reproductions.py` | `bec31761444c0ccd82ec20ac2b33ba7b603eb66622c56c268b6ceb98298937de` |
| `test_closure_analysis.py` | `f8351132bb41a5d1e12b47436001ff12f56cd658c8ae7f43d818bdf483c45a3b` |
| `test_frozen_identity.py` | `1e6d750163cc35bbdf9051d4b349a00caf182f733f95cd8605e1fb633121d362` |
| `closure_provenance_task22_gpuh.sbatch` | `63eb34454c76988ce0efca63ca13ff41ae09e9f89d7fe4802cc4495683f93bdc` |

Local static negative-contract, frozen-identity, Python compilation, and diff
checks passed. The probe was designed to run the immutable production H200
preflight twice in separate bundle extractions and Python processes, with
separate UID-owned mode-0700 designated directories, filesystem events and
Python call stacks, physical-file metadata/content checks, installed Torch
distribution/RECORD evidence, and normalized closure equality.

## Independent clean-import observations

Two separate frozen Python `3.9.25` / PyTorch `2.5.1+cu121` processes began in
distinct empty directories. Both produced the exact same observation:

| Field | Value |
|---|---|
| sys.modules key / module name | `torch.classes` / `torch.classes` |
| type | `torch._classes._Classes` |
| `__file__` | `_classes.py` |
| `__spec__`, loader, package, origin | all absent/None |
| prestart / post-import directory entries | `[]` / `[]` |
| physical `_classes.py` | absent |

The installed source is Torch `2.5.1+cu121` file `torch/_classes.py`, SHA256
`2a3dd93d72e9f19450670b89f3a57b5b5adf245709f6a49a551bbfad33c434bf`,
size `1721`, with RECORD hash
`sha256=Kj3ZPXLp8ZRQZwuJ86V7W1rfJFcJ9qSaVRu_rTPENL8`. Lines 19--20 define
`class _Classes(types.ModuleType)` and assign `__file__ = "_classes.py"`.

Thus the rejected path was a synthetic relative pseudo-origin, not a created
file. It cannot satisfy Task22's mandatory ordinary-file/non-symlink,
module-spec/loader, create/write/rename/delete lifecycle, content/template,
inode, and post-import stability conditions. No filename or directory was
whitelisted.

## Production closure provenance terminal evidence

One bounded no-training provenance job was submitted after freeze:

| Job | Owner | Placement | Node | State | Exit | Elapsed |
|---:|---|---|---|---|---|---:|
| `19266959` | `h99859yz` | `gpuH`, account `gpu-h200-fse-pgdr`, 1 H200 | node820 | FAILED | `1:0` | `00:00:38` |

Both independent bundle extractions passed exact bundle and manifest hashes.
The first Python process then installed its filesystem audit hook and entered
the frozen production preflight through `runpy`. The hook captured an `open`
event and requested `traceback.extract_stack`; Python's linecache/tokenizer
opened stack source, which emitted another audited `open` and recursively
re-entered the hook until `RecursionError: maximum recursion depth exceeded`.

No first reproduction JSON was emitted; production trainer/model construction,
the second independent process, complete artifact closure, negative artifact
validation, and normalized equality were not reached. This is
`infrastructure-failure/closure-provenance-audit-hook-recursion`. Under Task22,
failure to stably reproduce the complete closure is itself a terminal precheck
block; the instrumentation was not repaired and the provenance job was not
retried.

## Unreached gates and ledger

- `APPROVED_RUNTIME_GENERATED_THIRDPARTY_MODULE` closure: not established;
- formal remote clean-room audit: not submitted;
- four-environment real-network preflight: not submitted;
- scientific jobs/roots/processes/transitions/progress/traces/checkpoints/models: none;
- exact 2M/4M/5,980,160 Target/Paper comparisons: none eligible;
- scientific monitor: not created; and
- Task22 failure-ledger addition: `19266959`,
  `infrastructure-failure/closure-provenance-audit-hook-recursion`.

No repair, retry, requeue, resubmit, second audit, second candidate, sweep,
Jupyter, quarantined access, Paper rerun, overwrite, or unrelated mutation
occurred. Model-free evidence is under
`remote_launch_staging/procgen_normmatch_v2_runtime_generated_closure_audit_6m_s0_20260825_22/evidence_remote/`.

PRECHECK_BLOCKED
