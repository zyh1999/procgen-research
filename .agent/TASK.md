# Task-ID: PROCGEN-FULL-SHARED-JOINT2B-FIXEDLR-DUALTRUST-BETA1-BETA4-6M-S0-20260827-51

Status: READY

Paired methods:

- `FULL_SHARED_JOINT2B_FIXEDLR_DUALTRUST_BETA1_V1`
- `FULL_SHARED_JOINT2B_FIXEDLR_DUALTRUST_BETA4_V1`

Parent Task50 implementation is `e4f8cfc23bf406989f72db61ca8aadf5407d99d4`
and terminal delivery is `5cbdb7d30c6601bc69a0a4237670b4f141924298`.
Preserve the PPO/Adam warmup through exactly 503,808 transitions, the one
switch to the full-shared strict deterministic Joint-2B path, all actor and
critic rows, all 938,976 trainable columns, both natural cross blocks,
damping `.5`, global clip `.5`, PopArt/GAE, rollout/minibatch/epoch, momentum,
evaluation and checkpoint semantics.

Joint parameter LR is fixed at `.004` for every minibatch and rollout. The
only paired scientific difference is `beta_v=1` versus `beta_v=4`. Both arms
start `eta_pi=eta_v=1`; within a rollout they are constant. The strict system
uses `lambda_pi=eta_pi`, `lambda_v=beta_v*eta_v`, scaled rows
`[sqrt(lambda_pi) A; sqrt(lambda_v) C]`, and inverse-scaled actor/critic RHS,
with objective weights fixed at one. After each complete Joint rollout,
measure exact full-distribution policy KL and
`beta_v/2 * mean((Vbar_final-Vbar_behavior)^2)` in a fixed PopArt coordinate.
Update each eta once with band `.005/.04`, factor `1.5`, bounds `[1/64,64]`;
higher divergence strengthens the corresponding metric.

Run exactly one concise Bede production gate covering both arms. Only a full
PASS permits eight fresh seed0 intended-6M cells, four environments per arm,
each submitted exactly once. Target six simultaneous GPUs if live capacity
allows: prefer Bede, and only if fewer than six Bede slots are immediately
usable may the remaining independent cells use freshly verified CSF3 gpuH
deployment. Keep disjoint roots/logs and identical normalized scientific
commands. No retry, requeue, resubmit, extra beta, sweep or unrelated mutation.

Use immutable Paper RAT seed0 and act only at exact common first >=2M, first
>=4M, and 5,980,160. Only a per-cell ratio below `.60` permits one frozen
monitor application. Return all identities, gate evidence, jobs, roots,
partitions, nodes, pending reasons, allocation count and actual RUNNING
concurrency. The coordinator owns the sole 20-minute automation.
