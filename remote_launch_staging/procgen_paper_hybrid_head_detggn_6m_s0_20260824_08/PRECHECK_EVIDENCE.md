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

## Task 11 trainable-gradient recovery

Planner task `PROCGEN-HYBRID-HEAD-TRAINABLE-GRAD-PREFLIGHT-AND-6M-S0-20260824-11`
authorized exactly one further preflight after preserving jobs
`19220448/19220752/19225085/19225707` as immutable pre-training harness
failures. The allowed harness-only fix is frozen at commit
`26b2252527076df4bfe537a8612446317cbdcf3a`:

- harness SHA256 `df297a9305312cd8dc8e4b0811331ade762e3487f33548fc296b8ce667d080fd`;
- static-test SHA256 `1115f6c534bdcb695d5fb56e53ce81f245b50157f656f8b4431cac76da2697ef`;
- trainer/config/scientific-launcher/monitor hashes remain byte-identical to
  the frozen scientific identity.

The single authorized production preflight was CSF3 gpuH job `19227905`. It
completed `0:0` in `00:02:02` on node822 and wrote durable status
`PRECHECK_PASS` with rc0. Its complete model-free output is preserved under
`evidence/preflight_19227905/`. Mandatory results:

- all three resolved JSON files are byte-identical, each SHA256
  `61f8ebe38443acbdbf141981f4e9921435dccd5d4abb6a63959e3d4bdb9232ab`;
- production partition is total938,979, policy2/3,855, shared22/934,864,
  critic2/257, with manifest SHA256
  `b45298be8fc5bdccfa36ce653c7dbc0c41f2d013f23d4f4db0a4a580035f3087`;
- the ordered 26-tensor/938,976-element `requires_grad=True` set is identical
  item-by-item to the audited production update set in name, order, shape,
  dtype, device and object identity;
- PopArt `mean`, `mean_sq`, and `debiasing_term` remain non-trainable model
  state, are excluded from optimizer/autograd/direction/update, and are
  unchanged across both audited one-step paths;
- actual-network Paper actor and sampled shared-critic directions are
  bit-identical; one-step policy parameters and logits are bit-identical; only
  the value-head delta differs;
- the critic-exclusive head is exactly `last_v_layer.weight/bias`, its policy
  Jacobian is zero/disconnected, and its value path is connected;
- on a genuine NVIDIA H200, peak allocated memory was 2,045,893,120 of
  150,111,977,472 bytes; head Cholesky info max was0 and FP64 relative residual
  was `8.627e-16` with no fallback or hard error.

Immediately before this job, no target scheduler job, process or run root
existed. The preflight created no scientific root. A scientific submission is
therefore permitted under Task11 only after this PASS evidence is committed
and pushed.
