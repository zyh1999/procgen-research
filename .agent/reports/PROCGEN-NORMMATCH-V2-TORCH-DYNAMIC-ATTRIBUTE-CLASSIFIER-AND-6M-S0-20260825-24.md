# PROCGEN-NORMMATCH-V2-TORCH-DYNAMIC-ATTRIBUTE-CLASSIFIER-AND-6M-S0-20260825-24

## Conclusion

`PRECHECK_BLOCKED`

The mandatory actual-Python positive contract contradicts the frozen installed
Torch implementation. `torch.classes` has no instance-dictionary `__file__`,
but `inspect.getattr_static(torch.classes, "__file__", sentinel)` returns
`"_classes.py"`, not the sentinel, because exact type `_Classes` declares a
class-level `__file__` at installed source line 20. Public `getattr` resolves
that ordinary class attribute; `_Classes.__getattr__` is not its provider.

Task24 requires both the static-sentinel result and the dynamic-provider claim,
forbids weakening or reinterpretation, and requires local actual-environment
PASS before a closure job. No classifier code was changed and no closure job,
formal audit, real-network preflight, or scientific cell was submitted.

## Assignment and preservation

- assignment/origin: `b0b08faab99afc5581eadafe218de157fa9e749f`
- Task23 terminal delivery: `4adfe8eaf5943ba550636bb54c8c34c9814a5598`
- Task23 non-reentrant hook: `8d9206a6defc4525114398a952d29ffdd4872cd933dc5c9b96fc838bd1273dbe`
- method: `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`
- trainer/config: `0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b` / `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- scientific preflight/regression/monitor: `b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc` / `f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c` / `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`
- bundle/manifest: `3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f` / `99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa`
- science/preflight launchers: `ec60864aaa9940fd61eb1391008b50f2f48402eeae8f78c91c0b0c1fc313a398` / `374d24881d108bbdd08dee0880b7392a7cc3adf6af177f6eb4deaac15535b228`

Task14--23 evidence and ledgers remain immutable. There is no Task24 code
freeze because the required positive invariant failed before an authorized
change could be validated.

## Actual Python 3.9 / Torch evidence

The frozen environment is Python `3.9.25`, Torch `2.5.1+cu121`. Installed
`torch/_classes.py` has SHA256
`2a3dd93d72e9f19450670b89f3a57b5b5adf245709f6a49a551bbfad33c434bf`,
size `1721`, and RECORD hash
`sha256=Kj3ZPXLp8ZRQZwuJ86V7W1rfJFcJ9qSaVRu_rTPENL8`.

| Required field | Actual | Gate |
|---|---|---|
| `"__file__" not in vars(module)` | true | PASS |
| `vars(module).get("__file__")` | None | PASS |
| `inspect.getattr_static(..., sentinel)` | `_classes.py` | FAIL |
| public `getattr(module, "__file__")` | `_classes.py` | PASS |
| exact type | `torch._classes._Classes` | PASS |
| class dictionary `__file__` | `_classes.py` | proves static provider |
| requested dynamic provider | `_Classes.__getattr__` | FAIL |
| spec/loader/package/origin | all None | PASS |

Installed lines 19--20 are `class _Classes(types.ModuleType)` followed by
`__file__ = "_classes.py"`. Lines 25--28 show `__getattr__` creates a
`_ClassNamespace` only for a missing name. Since `__file__` exists on the
class, this method is not invoked for the public value.

## Unreached gates

- classifier implementation and Task16--23 regression suite: not run because
  the mandatory positive invariant is false before implementation;
- closure provenance job: not submitted;
- two-process production construction and normalized closure: absent;
- formal clean-room audit: not submitted;
- four-environment real-network preflight: not submitted;
- scientific jobs/roots/processes/transitions/traces/checkpoints/models: none;
- exact 2M/4M/5,980,160 comparisons: none eligible;
- monitor: not created;
- failure-ledger addition:
  `precheck-failure/task-spec-static-vs-dynamic-provider-contradiction`.

No retry, requeue, resubmit, field repair, second candidate, sweep, Jupyter,
quarantined access, Paper rerun, overwrite, or unrelated mutation occurred.

PRECHECK_BLOCKED
