# Executor Report

## Metadata

- Task-ID: `PROCGEN-NORMMATCH-V2-RUNTIME-SPY-SEMANTIC-BINDING-AND-6M-S0-20260825-27`
- Assignment: `a670a49f8be6fc69d2773d45e72647bc2d0f73ad`
- Binding/closure freeze: `84de09cda16f2d75f172fd704b15a8ed1108ae32`
- Method: `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`
- Repository target: `origin/agent-work`

## Result

The Task27 correction uses an immutable semantic-role mapping and direct
object capture at the real frozen preflight boundary. It maps trainer AST
`head_direction` to the actual `det_proposal` tensor and maps the Paper role
to the actual `paper_head_proposal` tensor. It does not query strings,
`locals()`, trainer names or equal values. Static/frozen identity tests and
actual Python 3.9.25 / Torch 2.5.1+cu121 positive and required negative tests
passed. New preflight SHA is
`e43fe7e730e840de07cc467bbc56900591c581fd24d4520abe129b0bad3d2cfb`;
all scientific and prior audit identities stayed frozen.

The one authorized closure job `19276602` ran on gpuH node821 and ended
`FAILED/1:0` after `00:01:56`. Both bundle checks and the real CUDA preflight
passed. Its runtime ledger proves exact direct-object identity, one call,
unchanged inputs, norm match and wrapped/unwrapped equivalence, with proposal
norms `.6050832272/.9192549586`, scale `1.519220710`, target norm
`.9192548990`, cosine `.8612535000`, and solver residual `8.627e-16`.

After that PASS, the unchanged exhaustive origin scan rejected the frozen
closure probe itself because it was observed through its canonical
`/net/scratch/.../runtime_closure_probe_task23.py` spelling. This is
`precheck-failure/task27-closure-probe-self-origin-storage-alias-policy`, a
closure/audit infrastructure failure. No second clean process or normalized
closure exists. Task27 forbids repair or retry, so no formal audit,
environment preflight, science, stage comparison, cancellation, or monitor
was created.

Report:
`.agent/reports/PROCGEN-NORMMATCH-V2-RUNTIME-SPY-SEMANTIC-BINDING-AND-6M-S0-20260825-27.md`.

BLOCKED
