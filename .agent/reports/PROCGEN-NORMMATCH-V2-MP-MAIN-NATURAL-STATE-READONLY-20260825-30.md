# PROCGEN-NORMMATCH-V2-MP-MAIN-NATURAL-STATE-READONLY-20260825-30

## Unique conclusion

`INSUFFICIENT_EVIDENCE`

Three independent natural-state observations exposed a repeatable
`__main__`/`__mp_main__` transition, but the mandatory whole-process
reproduction and observer-nonperturbation gates did not pass. The stable
transition is evidence for a future design decision, not approval of an alias
classifier.

## Identity and immutable scope

- Assignment: `2151b00d8cfeed33f8cf5f3466a2fcb0c2114806`
- Read-only observer implementation/evidence freeze:
  `06448412720a504f55ba14d77e01e902152be655`
- Method retained: `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`
- Frozen observer SHA256:
  `b6fca687d468955914de352270d386a5234677bdc01f84afd06a5847b953de9b`
- Analyzer SHA256:
  `0d3997808ab220b89cd7b6f802a2512ca27fa7943ff85ca1d07862b7abad33af`
- Job wrapper SHA256:
  `737f2f28e9bdcd5797b4882085724e3062c954691e632b0764c4b2ba0f98e3da`
- Frozen-identity/read-only test SHA256:
  `6bb10d8c84c2fb96c529712d6b490d9dc7f192abac1241744b0e281540e59be9`

The observer never imported `multiprocessing`, assigned either module key, or
made an acceptance decision. It removed its own `sitecustomize` module before
the closure-probe body and disabled tracing before the unchanged origin scan.
The acceptance policy, bundle manifest, frozen probes, trainer, config,
regression, monitor and launchers were not edited.

Protected identities remained:

| Artifact | SHA256 |
|---|---|
| NormMatch V2 trainer | `0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b` |
| config | `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda` |
| Task14 preflight | `b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc` |
| regression | `f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c` |
| stage monitor | `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e` |
| bundle archive | `3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f` |
| bundle manifest | `99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa` |
| Task23 hook | `8d9206a6defc4525114398a952d29ffdd4872cd933dc5c9b96fc838bd1273dbe` |
| frozen Task23 closure probe | `c3529cb171306d7b3b0517974a682ddffb91d65dc015c102b1e84658e9eeb1f5` |
| Task25 classifier | `f80de2abbcbce29e7a57ef456156c86636798c4e1ea37171922b3b466b6790fc` |
| Task26 AST helper | `c753b38c229a65dcecd54eb376aeabbcbd45586426a000970ea905f2982674b6` |
| Task27 preflight | `e43fe7e730e840de07cc467bbc56900591c581fd24d4520abe129b0bad3d2cfb` |
| Task28R exact-probe validator | `96da9c8ee7497ce01df3230f6fa0875a81d1175c958c1a5d8276390090c118ad` |

## Execution and terminal state

- Exactly one bounded read-only provenance job: `19278072`.
- Scheduler: `FAILED/1:0`, node821, start `2026-08-25T07:04+01:00`, elapsed
  45 seconds.
- Four independent clean processes ran sequentially: `obs1`, `obs2`, `obs3`
  with the process-start observer, and `control` without it.
- Each bundle verification passed, built the actual production network and
  emitted config/partition/structural/connectivity/AST/Task27 artifacts.
- All four intentionally reached the unchanged Task28R terminal rejection:
  `bundle module absent from manifest or hash mismatch: __mp_main__`.
- The analyzer completed and wrote
  `TASK30_UNIQUE_CONCLUSION=INSUFFICIENT_EVIDENCE`.
- The wrapper then failed only while constructing its final checksum list:
  the historical Task15 science-launcher pathname was absent on CSF3. This is
  a post-analysis evidence-packaging failure. It did not change or erase any
  process ledger. No repair or second job was attempted.

## Natural state-transition table

Every observer process produced the following same transition sequence:

| Milestone | `__mp_main__` | Object identity | `__main__` backing | `__mp_main__` backing |
|---|---:|---:|---|---|
| child-process entry | absent | false | no file yet | absent |
| closure-probe start | absent | false | exact Task23 frozen probe | absent |
| trainer import before | present | true | deployed Task27 preflight | same deployed Task27 preflight |
| trainer module entry | present | true | deployed Task27 preflight | same deployed Task27 preflight |
| trainer import after | present | true | deployed Task27 preflight | same deployed Task27 preflight |
| production model constructed | present | true | deployed Task27 preflight | same deployed Task27 preflight |
| immediately before origin scan | present | false | exact Task23 frozen probe | deployed Task27 preflight |

At the final boundary, each reproduction had:

- `__main__.__file__` equal to the exact frozen Task23 probe, SHA256
  `c3529cb1...`, classified `TASK28R_FROZEN_PROBE`;
- `__mp_main__.__file__` equal to that reproduction's temporary
  `frozen/gpuh_preflight_normmatch_v2.py`, classified
  `DEPLOY_NONMANIFEST`;
- different objects, dictionaries, files, loader/package fields and code
  bodies;
- the same normalized dictionary difference table in all three processes.

This directly refutes treating the final natural state as exact object or
same-backing identity. It does not by itself prove that the two-file
relationship is observer-independent or safe enough for a classifier.

## CPython 3.9 transition mapping

| State operation | Frozen source | SHA256 | Lines |
|---|---|---|---|
| import-time `__mp_main__` alias | `/usr/lib64/python3.9/multiprocessing/__init__.py` | `a5a42976033c7d63ee2740acceef949a3582dcb0e0442845f9717e1be771c68b` | 37 |
| child preparation chain and main reconstruction | `/usr/lib64/python3.9/multiprocessing/spawn.py` | `16ce6d81f8b5ef7228e5500bff04b37bdceb3d7dfc8d6de3ad523598798c43f4` | `spawn_main -> _main -> prepare -> _fixup_main_from_path`; 125, 234, 236, 262, 290 |

The snapshots are consistent with reconstruction making both keys point at
the deployed preflight during its execution, followed by restoration of the
outer closure probe as `__main__` while the reconstructed module remains
under `__mp_main__`. Because the observer nonperturbation gate failed, this
mapping is recorded as an observed correspondence, not an accepted natural
classification theorem.

## Reproduction and observer controls

- Final module relationship: consistent in all three reproductions.
- Whole normalized observation SHA256 values differed:
  `260cab37...`, `bdd0f717...`, `e77e06f5...`.
- Resolved configs, structural manifest, connectivity probe and AST ledger:
  normalized equal across all three observers and the control.
- Critical stdout: equal across all four processes.
- Original Task28R failure: equal across all four processes.
- Task27 wrapped/unwrapped RNG/output/parameter/optimizer/telemetry equality:
  true within every process.
- Runtime semantic-binding ledgers: not normalized equal across independently
  initialized processes/control.
- Import order after excluding the observer module: not equal to control.

The last two failures prevent the required proof that observation did not
perturb the natural state. Therefore no stronger positive or negative
relationship conclusion is justified under Task30's frozen acceptance order.

## Evidence and preserved ledgers

Complete model-free evidence archive:

`remote_launch_staging/procgen_normmatch_v2_mp_main_natural_state_readonly_20260825_30/evidence/task30_model_free_evidence_19278072.tar.gz`

Archive SHA256:
`882c82f5a13aad30931f452a4ae2176b7b1eec632282153669452180e7e13909`.
It contains 74 provenance files plus scheduler stdout/stderr, including all
three full milestone ledgers, the control artifacts, import-time traces,
analyzer output, terminal rc/status and the post-analysis packaging error. It
contains no model or checkpoint.

Task29 failures remain unchanged:

- `infrastructure-failure/proof-observer-import-timing` (`19277384`);
- `precheck-failure/task29-natural-mp-main-not-exact-main-object-alias`
  (`19277433`).

All Task14--29 infrastructure, precheck, algorithm early-stop and cancellation
ledgers remain historical and were neither retried nor relabeled.

## Exact remaining evidence gap

A future Planner task would need a demonstrably nonperturbing capture method
whose import order and normalized runtime artifacts match a no-observer
control while retaining the full milestone backing/file/fd/code-object
evidence. Task30 does not authorize designing that method or a classifier.

No classifier, allowlist, policy, manifest, frozen probe, formal audit,
four-environment preflight, science, scientific root, transition, progress,
trace, checkpoint/model, early-stop action, or scientific monitor was created
or changed.
