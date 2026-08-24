# PROCGEN-PAPER-HYBRID-HEAD-DETGGN-6M-S0-20260824-08

## Conclusion

`PRECHECK_BLOCKED`. The scientific matrix was not launched because the
mandatory actual-network parameter-partition/H200 compatibility proof did not
complete after two separately preserved preflight-design failures.

## Frozen scientific identity

`PAPER_HYBRID_SHARED_PAPER_HEAD_DETGGN_V1` starts from exact original Paper
RAT. It preserves the complete Paper actor and sampled critic direction on all
shared encoder/trunk parameters. Policy-exclusive parameters receive no critic
direction. Only the 257 trainable critic-exclusive PopArt value-head
weight/bias parameters replace Paper sampled critic direction with normalized
residual J_v GGN, lambda `.1`, objective coefficient1, one independent
head-only BxB system, symmetric FP64/Jacobi/Cholesky. PopArt statistics retain
Paper update semantics and are non-curvature state.

Initial LR `.5`, per-minibatch KL `.005/.04`, momentum `1e-6`, Paper history,
rollout4096, minibatch512, epochs4, damping/clip `.5/.5`, network, GAE,
entropy, ratio, evaluation, checkpoint, and formal 6M semantics are unchanged.

## Frozen hashes

| Artifact | SHA256 |
|---|---|
| exact Paper trainer | `cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7` |
| exact Paper config | `1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea` |
| P1 donor | `2b50f8cc26bb8c85f91e0b394acc7535f5564ac70036403597d3738d3dab9c1b` |
| target trainer | `7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54` |
| target config | `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda` |
| builder | `bb286d556c51465676b058c826755f027a207c1b4eb0b76e48670a753a8610a8` |
| identity audit | `d6d268e4fc8e28f34ec4a7c60ae97c8a0e85183ae96f4bd7135d5258a20e1bca` |
| regression | `8a774ee31e49157556a2e4454227114f033a76d410acd796c240d43c6bae5465` |
| gpuH compatibility test | `4bcbff44137ddf66c76a8ad06a357459411726d6f9dc1fa7d10897a473027292` |
| corrected persistent preflight launcher | `39bd8a40891c8c1dcfd46440b499b626be92172a7bdb1ef84c577113040ce64d` |
| scientific launcher | `ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e` |
| stage monitor | `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e` |

Frozen implementation commit is
`fe4b8a58812e80689705abec11364457cae31e26`; the versioned import-path
correction and first failure ledger are commit
`896f54459b53f9f489951fb3c9f9ed5fec32c11e`.

## Identity, partition, and one-step proof

Static audit returned `HYBRID_HEAD_AUDIT_PASS` and
`DISTINCT_FORMULA_PASS`. CSF3 `.RLvenv` regression returned:

```
HYBRID_HEAD_REGRESSION_PASS
partition=EXHAUSTIVE_MUTUALLY_EXCLUSIVE_STABLE
critic_exclusive_policy_jacobian=EXACT_ZERO_DISCONNECTED
paper_actor_matrix_rhs_direction=BIT_IDENTICAL
paper_sampled_shared_critic_direction=BIT_IDENTICAL
one_step_policy_parameters=BIT_IDENTICAL
one_step_policy_logits=BIT_IDENTICAL
only_value_head_delta=DIFFERS
head_solver=FP64_Jacobi_Cholesky relative_residual=2.616e-16
illegal_joint_sharedggn_cross_guard_projection_kaczmarz_fields=REJECTED
```

The algebraic partition is policy head, shared trunk, and 257-parameter value
head. The mandatory actual Procgen-network realization was intended to prove
`POLICY_EXCLUSIVE=3855`, `CRITIC_EXCLUSIVE=257`, `SHARED>1000000`, exact
head policy disconnection, H200 memory headroom, and FP64 residual `<1e-10`.
That realization did not complete, so the preflight gate cannot be promoted
from algebraic proof to actual-network proof.

## Historical distinctness

The target trainer differs from CSF3 block-trace `1881bf7c...`, CSF3 expected
`c976c0e5...`, Bede expected `0514703d...`, joint-2B `41334b59...`, and
separate-B `b0dad110...`. Unlike each deterministic shared-parameter formula,
this candidate preserves Paper sampled shared-trunk critic direction and
restricts deterministic J_v curvature to the critic-exclusive head. Historical
rewards are provenance only, not baselines.

## Resource and duplicate audit

At CSF3 `2026-08-24T20:21:50+01:00`, the campaign/root, target scheduler
history, trainer process, and duplicate objective were absent. gpuH was live;
the Executor selected one-H200/eight-CPU independently auditable jobs. Unrelated
gpuL jobs `19210338/42/43/44` and multicore `19051570` were not touched. Bede
was reachable as `yihe` with three idle GPU nodes but was not selected. Named
4090 aliases were unresolved. No Jupyter or quarantined host was used.

## Preflight failures

| Job | Scheduler | Durable result | Classification |
|---|---|---|---|
| `19220448` | FAILED/1:0, 15s, node820 | regression PASS; target import failed: `No module named 'utils'` | infrastructure-failure/preflight-design |
| `19220752` | FAILED/1:0, 15s, node820 | regression PASS; actual-model construction failed: missing `norm_obs` in harness namespace | infrastructure-failure/preflight-design |

Both allocations identified a genuine NVIDIA H200. Neither began training or
produced a scientific root/artifact. The first failure was preserved before a
single import-path-only correction. The second failure's job-ID-isolated
status is `INFRASTRUCTURE_PREFLIGHT_FAIL`, rc1. No third preflight was
attempted because the mandatory gate remained unproven and the task forbids
continuing past a failed precheck.

Raw text evidence is under
`remote_launch_staging/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/evidence/`;
no model/checkpoint is included.

## Scientific matrix and stages

| Environment | Seed | Intended horizon | Launch/status | 2M | 4M | 5,980,160 |
|---|---:|---:|---|---|---|---|
| BigFish | 0 | 6M | not launched / precheck blocked | N/A | N/A | N/A |
| BossFight | 0 | 6M | not launched / precheck blocked | N/A | N/A | N/A |
| CaveFlyer | 0 | 6M | not launched / precheck blocked | N/A | N/A | N/A |
| CoinRun | 0 | 6M | not launched / precheck blocked | N/A | N/A | N/A |

There is consequently no target reward/KL/LR/entropy/update decomposition,
solver telemetry, progress, trace, checkpoint, or early-stop ledger. This is
insufficient preflight evidence, not negative scientific evidence.

## Final reconciliation and immutable ledger

At CSF3 `2026-08-24T20:32:01+01:00`, both preflights were terminal FAILED,
the target queue was empty, campaign `runs/` was absent, and no target trainer
process existed. No retry, requeue, scientific submission, cancellation,
Paper rerun, second candidate, or unrelated mutation occurred.

Prior joint-2B `GATE_FAIL`, separate-B `CANDIDATE_NOT_READY` and its three
algorithm early stops, low-Fisher `GUARD_NOT_HELPFUL`, P1/ACTOR_J failures,
gpuA/gpuL/preflight infrastructure failures, and obsolete/unstarted
cancellations remain immutable in `.agent/STATE.md` and their dedicated
reports. Clean FP64 residual is not interpreted as update usefulness.

## Falsifiable conclusion

`PRECHECK_BLOCKED`: mandatory actual-network zero-policy-Jacobian partition
and H200 memory/solver realization were not proven. No scientific conclusion
about the hybrid-head method is authorized.
