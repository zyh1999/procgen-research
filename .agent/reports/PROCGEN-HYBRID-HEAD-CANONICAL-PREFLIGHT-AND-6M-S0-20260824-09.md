# PROCGEN-HYBRID-HEAD-CANONICAL-PREFLIGHT-AND-6M-S0-20260824-09

## Unique conclusion

`PRECHECK_BLOCKED`. The single authorized canonical recovery preflight reached
the real production config/model/partition path, then stopped on a stale
harness-only shared-parameter-count assertion before all mandatory checks.
No scientific cell was launched.

## Commits and immutable scientific identity

- Assignment: `acec34e38a3df7c785b1be3e54ce26c9809e2721`
- Original scientific freeze: `fe4b8a58812e80689705abec11364457cae31e26`
- Canonical recovery freeze: `9a56a6aba6d70ce7a16b9e81cf105dccbd43d638`

| Artifact | SHA256 | Classification |
|---|---|---|
| trainer | `7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54` | immutable scientific |
| config | `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda` | immutable scientific |
| scientific launcher | `ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e` | immutable scientific |
| monitor | `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e` | immutable scientific |
| canonical recovery harness | `9cc29a14083dcec8640f5822128a11e2913d997ae2d331b070d68a13a4072b32` | allowed preflight-only |
| persistent preflight launcher | `3c1356f5387226f40e2c6a5692d59a232cbec5ab5f3048f4265252d321ab05e3` | allowed preflight-only |
| canonical static test | `854bd05c02e2d80a2fb325fee091c5c7713527d8a71eb2b063fc04e74ab4e342` | allowed preflight-only |

The line-level classification is frozen in
`remote_launch_staging/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/CANONICAL_PREFLIGHT_RECOVERY.md`.
The hand-built `SimpleNamespace` was removed; no scientific file changed.

## Preflight job 19225085

Scheduler accounting: `FAILED/1:0`, elapsed `00:00:20`, node820, CSF3 local
start/end `2026-08-24T21:17--21:18`. A genuine NVIDIA H200 was allocated.
Static canonical guard and the prior scientific regression both passed.

### Canonical resolved-configuration proof

The harness invoked the trainer's own `main()` three times. The production
model capture continued through original `train_fn()` and intercepted only at
`learn()` before rollout or optimizer work. The following files are
byte-identical, each SHA256
`61f8ebe38443acbdbf141981f4e9921435dccd5d4abb6a63959e3d4bdb9232ab`:

- `resolved_config_preflight.json`
- `resolved_config_scientific_launcher_dry_run.json`
- `resolved_config_trainer_entry.json`

Their canonical compact payload SHA256 is
`0f34624bdbb1eae181cbfd35a08e1e413c7c9aea73c10f63b75dec0cedb9afdc`.
They resolve exact environment BigFish seed0, device0, one process, `adv`, LR
`.5`, per-minibatch adaptive KL, epochs4, minibatches8, damping/clip `.5/.5`,
normalized-residual head coefficients `.1/1`, ResNet hidden256,
`norm_obs=false`, PopArt, rollout4096 and intended 6M.

### Real production model/partition proof

Production `SharedActorCritic` construction succeeded and reported 938,979
parameters. Partition manifest SHA256 is
`b45298be8fc5bdccfa36ce653c7dbc0c41f2d013f23d4f4db0a4a580035f3087`:

| Group | Tensors | Trainable parameters | Policy/value connectivity |
|---|---:|---:|---|
| POLICY_EXCLUSIVE | 2 | 3,855 | policy connected, value disconnected |
| SHARED | 22 | 934,864 | policy and value connected |
| CRITIC_EXCLUSIVE | 2 | 257 | policy disconnected/exact Jacobian L2 0; value connected |

The critic-exclusive names are exactly `last_v_layer.weight` and
`last_v_layer.bias`. PopArt mean/mean_sq/debiasing remain non-curvature state.

### Blocking failure and unreached checks

Immediately after writing the valid partition, line167 asserted
`manifest["SHARED"]["numel"] > 1_000_000`. That stale preflight-only threshold
came from the prior hand-built network assumption and is not a requirement in
TASK.md. The exact production shared count is 934,864, so the assertion raised.

Consequently these mandatory checks did not execute:

- actual-production-network Paper actor and shared sampled-critic direction
  equality after the partition point;
- actual-network one-step policy parameter/logit equality and head-only delta;
- production-scale H200 row footprint and memory headroom;
- final FP64/Jacobi/Cholesky info/residual/nonfinite scan.

The valid partial proof does not authorize scientific launch. Classification
is `infrastructure-failure/preflight-design`; it is not algorithm, numerical,
solver, config mismatch, partition failure, or H200 incompatibility evidence.

## Immutable three-job failure ledger

| Job | Terminal evidence | Exact failure | Classification |
|---|---|---|---|
| `19220448` | FAILED/1:0, 15s, node820 | staged `utils` absent from import path | infrastructure-failure/preflight-design |
| `19220752` | FAILED/1:0, 15s, node820 | hand-built namespace omitted `norm_obs` | infrastructure-failure/preflight-design |
| `19225085` | FAILED/1:0, 20s, node820 | stale shared-numel >1M harness assertion after canonical path/partition PASS | infrastructure-failure/preflight-design |

No success overwrites any prior failure. Raw model-free evidence is in
`remote_launch_staging/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/evidence/preflight_19225085/`.

## Resource and no-launch reconciliation

Before submission, CSF3 had no target history beyond the two preserved failed
preflights, no root and no target trainer. gpuH was live; Bede was reachable
with five idle GPU nodes but not selected because the frozen recovery targeted
H200. Unrelated gpuL jobs `19210338/42/43/44` and multicore `19051570` were
not touched. Named authorized 4090 aliases remained unavailable. No Jupyter or
quarantined host was used.

At `2026-08-24T21:19:47+01:00`, job19225085 was terminal, campaign `runs/`
was absent, and no target trainer process existed. No retry, requeue,
cancellation, scientific submission, Paper rerun, second candidate, sweep, or
unrelated mutation occurred.

## Four-cell stage table

| Environment | Seed | Intended horizon | Status | >=2M | >=4M | 5,980,160 |
|---|---:|---:|---|---|---|---|
| BigFish | 0 | 6M | not launched / precheck blocked | N/A | N/A | N/A |
| BossFight | 0 | 6M | not launched / precheck blocked | N/A | N/A | N/A |
| CaveFlyer | 0 | 6M | not launched / precheck blocked | N/A | N/A | N/A |
| CoinRun | 0 | 6M | not launched / precheck blocked | N/A | N/A | N/A |

There is no target reward/KL/LR/entropy/solver telemetry, progress, trace,
checkpoint, hard-error result or early-stop ledger. This is incomplete
preflight evidence, not negative scientific evidence.

## Preserved historical evidence

Joint-2B `GATE_FAIL`, separate-B `CANDIDATE_NOT_READY` and its early stops,
low-Fisher `GUARD_NOT_HELPFUL`, P1/ACTOR_J failures, prior infrastructure
failures and obsolete/unstarted cancellations remain unchanged in STATE and
their dedicated reports.

## Blocking item

The one Planner-authorized recovery was consumed. A new explicit Planner
decision/task is required before any further preflight correction or
scientific launch. The Executor does not silently remove the assertion or
resubmit.
