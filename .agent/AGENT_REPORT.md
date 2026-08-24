# Executor Report

## Metadata

- Task-ID: `PROCGEN-HYBRID-HEAD-CANONICAL-PREFLIGHT-AND-6M-S0-20260824-09`
- Method: `PAPER_MATCHED_HYBRID_HEAD_DETGGN_V1`
- Assignment: `acec34e38a3df7c785b1be3e54ce26c9809e2721`
- Scientific freeze: `fe4b8a58812e80689705abec11364457cae31e26`
- Canonical recovery freeze: `9a56a6aba6d70ce7a16b9e81cf105dccbd43d638`
- Repository target: `origin/agent-work`

## Result

The one authorized canonical recovery preflight was job `19225085`. It invoked
the frozen trainer's own `main()` parser/default merge, original production
`train_fn()`, and real `SharedActorCritic`; no hand-built namespace remained.
Preflight, scientific-launcher dry-run and trainer-entry resolved configs were
byte-identical. The real network partition proved policy-exclusive 3,855,
shared 934,864 and critic-exclusive 257 parameters. Both value-head parameters
were autograd-disconnected from policy logits with exact zero Jacobian probes.

The job then failed on a stale harness-only assertion requiring shared numel
above 1,000,000. The exact frozen production network reports 934,864 shared
parameters and 938,979 total parameters. Scheduler evidence is FAILED/1:0,
20 seconds, node820. This is
`infrastructure-failure/preflight-design`, not an algorithm, numerical,
solver, config, partition, or hardware incompatibility result.

The assertion stopped execution before actual-network one-step equivalence,
production-scale H200 memory and final head-solver checks. Because task 09
authorized exactly one recovery preflight, no resubmission or scientific job
was made. Final reconciliation found no target queue row, run root, trainer,
transition, progress, trace, checkpoint, or model.

All scientific hashes remained immutable. Full raw evidence and the three-job
failure ledger are recorded in
`.agent/reports/PROCGEN-HYBRID-HEAD-CANONICAL-PREFLIGHT-AND-6M-S0-20260824-09.md`
and the frozen staging evidence directory.

PRECHECK_BLOCKED
