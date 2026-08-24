# Precheck evidence

## Identity and formula gate

The deterministic source audit returned `HYBRID_HEAD_AUDIT_PASS`. It proved
that the exact Paper actor and sampled shared-trunk critic are literally
preserved and that the only scientific replacement is the critic-exclusive
value-head raw direction plus required telemetry. Historical formula identity
returned `DISTINCT_FORMULA_PASS`; see `HISTORICAL_PROVENANCE.md`.

The numerical regression ran in the authorized CSF3 `.RLvenv` and returned:

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

## Live placement and duplicate audit

At CSF3 `2026-08-24T20:21:50+01:00`, no target scheduler record, trainer
process, campaign root, or duplicate objective existed. The owned GPU jobs
were unrelated gpuL work `19210338/42/43/44` and were not touched. All four
gpuH nodes were live in mixed state. The Executor selected gpuH with one H200,
eight CPUs, and one independently auditable job per environment. Bede was
reachable as `yihe`, with three idle GPU nodes, but was not selected because
the frozen hardware compatibility gate and launcher target H200. The named
authorized 4090 aliases did not resolve from CSF3. The quarantined host was not
contacted. No Jupyter was used.

## Mandatory hardware gate

The persistent non-training gpuH preflight must still prove the actual network
partition (`POLICY_EXCLUSIVE=3855`, `CRITIC_EXCLUSIVE=257`, `SHARED>1000000`),
exact zero policy-logit Jacobian for the critic head, Procgen imports, frozen
hashes, H200 identity/memory, aggregate Paper/head row footprint, and finite
FP64/Jacobi/Cholesky residual below `1e-10`. A scientific job may be submitted
only after its durable status is `PRECHECK_PASS` with rc0.

Initial preflight job `19220448` acquired a genuine H200 and reran the
regression successfully, but the compatibility test did not begin because the
launcher omitted the staged code directory from Python's import path. It is
preserved as `infrastructure-failure/preflight-design`; see
`PREFLIGHT_FAILURE_LEDGER.md`. The corrected launcher changes only import-path
and evidence-output handling. Scientific trainer/config hashes remain
`7bcf9bb6...` / `9497be42...`.

Corrected job `19220752` also returned scheduler FAILED/1:0 after 15 seconds
on node820. Target import succeeded and the regression passed again, but the
actual model constructor stopped because the harness namespace omitted required
`norm_obs`. Thus the actual-network partition/Jacobian and H200 memory proof
never ran. No scientific root or trainer existed after either preflight. The
mandatory gate therefore terminates `PRECHECK_BLOCKED`; no third preflight or
scientific submission was attempted.
