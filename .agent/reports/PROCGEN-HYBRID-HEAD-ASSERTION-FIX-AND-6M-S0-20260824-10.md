# PROCGEN-HYBRID-HEAD-ASSERTION-FIX-AND-6M-S0-20260824-10

## Unique conclusion

`PRECHECK_BLOCKED`. The exact assertion fix passed, but the one final
authorized preflight failed in a later actual-network one-step harness check.
No scientific cell was launched.

## Commits and immutable identity

- Assignment: `30fb08b6791c64cf5fde9e1de5355cb3e72a24c2`
- Scientific freeze: `fe4b8a58812e80689705abec11364457cae31e26`
- Canonical preflight freeze: `9a56a6aba6d70ce7a16b9e81cf105dccbd43d638`
- Assertion-fix freeze: `a22f1a51bbcc953881e780f4dc00da16b2fc317f`

| Artifact | SHA256 | Status |
|---|---|---|
| trainer | `7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54` | immutable |
| config | `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda` | immutable |
| scientific launcher | `ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e` | immutable |
| monitor | `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e` | immutable |
| assertion-fixed harness | `120b8d5f41b54782836c63cdb6678cf84d92181cfddf7f5b67c27607807fe939` | allowed preflight only |
| preflight launcher | `3c1356f5387226f40e2c6a5692d59a232cbec5ab5f3048f4265252d321ab05e3` | unchanged preflight only |
| invariant static test | `88d4d1fae7e1e7a0e0e29184a0efc6a3a89d4909bb54834469ad80ceeaba8798` | allowed test only |

## Exact line-level assertion fix

The only harness logic change replaced the stale
`SHARED.numel > 1_000_000` assertion with all required exact invariants:

- total parameter count `938_979`;
- policy tensors/numel `2/3_855`;
- shared tensors/numel `22/934_864`;
- critic tensors/numel `2/257`;
- critic names exactly `last_v_layer.weight`, `last_v_layer.bias`;
- manifest file SHA exactly
  `b45298be8fc5bdccfa36ce653c7dbc0c41f2d013f23d4f4db0a4a580035f3087`.

The static test asserts that the stale comparison is absent and every exact
replacement literal is present. No other harness path or scientific file was
modified.

## Final preflight 19225707

Scheduler: `FAILED/1:0`, elapsed `00:00:17`, node820, CSF3 local
`2026-08-24T21:29`; genuine NVIDIA H200 allocation.

### Passed before failure

- canonical static test and prior scientific algebraic regression;
- trainer `main()` config/default path and original production `train_fn()`;
- real `SharedActorCritic` construction, reported total938,979;
- three byte-identical resolved configs, each SHA
  `61f8ebe38443acbdbf141981f4e9921435dccd5d4abb6a63959e3d4bdb9232ab`;
- exact partition manifest SHA
  `b45298be8fc5bdccfa36ce653c7dbc0c41f2d013f23d4f4db0a4a580035f3087`;
- policy2/3,855, shared22/934,864, critic2/257;
- critic-exclusive exact value-head names, policy autograd disconnected and
  Jacobian probe L2 exactly0, value connected;
- complete Planner-specified assertion replacement.

### Exact failure

The next `raw_grads()` one-step proof built `params = list(net.parameters())`.
That list includes non-trainable PopArt mean/mean_sq/debiasing state. Passing
those tensors to `torch.autograd.grad(actor_loss, params, ...)` raised:

```
RuntimeError: One of the differentiated Tensors does not require grad
```

This occurred before any one-step update or scientific training. It is an
`infrastructure-failure/preflight-design`. It does not contradict the already
proven partition/Jacobian and is not algorithm, numerical, solver, config, or
H200 incompatibility evidence.

### Unreached mandatory checks

- actual-network Paper/Target actor and shared-critic gradient equivalence;
- actual one-step policy parameter/logit equality and head-only delta;
- production-scale H200 memory footprint;
- final FP64/Jacobi/Cholesky info/residual and full nonfinite/error scan.

Any unreached mandatory item blocks scientific launch. Task10 explicitly
forbids another field repair or retry.

## Four-job immutable preflight ledger

| Job | Scheduler | Exact failure | Classification |
|---|---|---|---|
| `19220448` | FAILED/1:0,15s,node820 | import path omitted staged utils | infrastructure-failure/preflight-design |
| `19220752` | FAILED/1:0,15s,node820 | hand-built namespace omitted norm_obs | infrastructure-failure/preflight-design |
| `19225085` | FAILED/1:0,20s,node820 | stale shared numel >1M assertion | infrastructure-failure/preflight-design |
| `19225707` | FAILED/1:0,17s,node820 | one-step autograd included non-trainable PopArt state | infrastructure-failure/preflight-design |

Raw model-free evidence is under
`remote_launch_staging/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/evidence/preflight_19225707/`.
No prior evidence was overwritten.

## Resource/no-launch reconciliation

Before job submission, gpuH was live and target roots/processes/duplicates were
absent. Bede was reachable with idle nodes but not selected; unrelated gpuL
jobs and multicore work were untouched. No Jupyter or quarantined host was
used. At `2026-08-24T21:31:21+01:00`, job19225707 was terminal, `runs/` was
absent, and no target trainer existed.

## Four-cell stage table

| Environment | Seed | Intended horizon | Status | >=2M | >=4M | 5,980,160 |
|---|---:|---:|---|---|---|---|
| BigFish | 0 | 6M | not launched / precheck blocked | N/A | N/A | N/A |
| BossFight | 0 | 6M | not launched / precheck blocked | N/A | N/A | N/A |
| CaveFlyer | 0 | 6M | not launched / precheck blocked | N/A | N/A | N/A |
| CoinRun | 0 | 6M | not launched / precheck blocked | N/A | N/A | N/A |

No reward/KL/LR/entropy/solver/transition/checkpoint or early-stop evidence
exists for this task. Historical algorithm failures, infrastructure failures,
cancellations and negative results remain unchanged.
