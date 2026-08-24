# PROCGEN-HYBRID-HEAD-TRAINABLE-GRAD-PREFLIGHT-AND-6M-S0-20260824-11

## Unique conclusion

`CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`.

The full trainable-gradient recovery preflight passed. Of the four frozen
scientific jobs, BigFish started and later triggered the mandated algorithm
early stop at the exact4M stage. The other three cells failed their immutable
per-job preflight before scientific start. Thus promotion is impossible, but
the task's two-environment scientific rejection threshold is not met.

## Frozen identities

| Artifact | SHA256 / Git identity | Result |
|---|---|---|
| Assignment | `0a7b19a44f78224e6da829d671bf5fb5052b35d0` | immutable |
| Harness freeze | `26b2252527076df4bfe537a8612446317cbdcf3a` | pushed before preflight |
| Preflight evidence freeze | `dcfd7b08e1827de1cb23dec0241149dd30632d79` | pushed before science |
| Trainer | `7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54` | unchanged |
| Config | `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda` | unchanged |
| Scientific launcher | `ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e` | unchanged |
| Stage monitor | `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e` | unchanged |
| Recovered preflight harness | `df297a9305312cd8dc8e4b0811331ade762e3487f33548fc296b8ce667d080fd` | allowed harness only |
| Static test | `1115f6c534bdcb695d5fb56e53ce81f245b50157f656f8b4431cac76da2697ef` | allowed test only |

No scientific trainer, config, launcher, monitor, method, seed, horizon,
evaluation or output semantic changed.

## Authorized preflight 19227905

Scheduler: `COMPLETED/0:0`, `00:02:02`, node822. Durable status was
`PRECHECK_PASS`, rc0.

- Three resolved JSON files are byte-identical at SHA256
  `61f8ebe38443acbdbf141981f4e9921435dccd5d4abb6a63959e3d4bdb9232ab`.
- Production partition is total938,979; policy2/3,855; shared22/934,864;
  critic2/257. Manifest SHA is
  `b45298be8fc5bdccfa36ce653c7dbc0c41f2d013f23d4f4db0a4a580035f3087`.
- Ordered trainable set is26 tensors/938,976 elements and matches the audited
  production update set item-by-item in name, order, shape, dtype, device and
  object identity.
- PopArt `mean`, `mean_sq`, `debiasing_term` remain non-trainable state,
  excluded from optimizer/autograd/direction/update and unchanged before/after
  both audited one-step paths.
- Critic-exclusive parameters are exactly `last_v_layer.weight/bias`, policy
  Jacobian zero/disconnected and value connected.
- Actual-network Paper actor and sampled shared-critic directions, one-step
  policy parameters/logits and shared delta are bit-identical. Only the
  value-head delta differs.
- Genuine H200 total memory was150,111,977,472 bytes, audited peak
  2,045,893,120. Cholesky info max0; head FP64 relative residual `8.627e-16`;
  no fallback or hard error.

Complete preflight files are under `evidence/preflight_19227905/`. Prior jobs
`19220448/19220752/19225085/19225707` remain immutable
`infrastructure-failure/preflight-design` provenance and are not erased by the
successful recovery.

## Four-cell terminal matrix

| Environment | Job | Scheduler / root reconciliation | Scientific evidence | Exact stages |
|---|---:|---|---|---|
| BigFish | `19228676` | `CANCELLED by 778916`, elapsed44:52,node820; root RUNNING/absent rc stale | `EARLY_STOPPED_ALGORITHM` | 2,007,040 PASS `.7036637931`; 4,014,080 early stop `.4691265060` |
| BossFight | `19228677` | `FAILED/70:0`,24s,node820 | none; per-job preflight failure | N/A |
| CaveFlyer | `19228678` | `FAILED/70:0`,31s,node822 | none; per-job preflight failure | N/A |
| CoinRun | `19228679` | `FAILED/70:0`,31s,node822 | none; per-job preflight failure | N/A |

All roots are unique and preserved. No cell was retried, requeued, resubmitted
or overwritten.

## Exact BigFish Paper comparison

Immutable Paper source `bigfish_seed0_progress.csv` has SHA256
`caf19809e208f35b8f8bcb41266021d07a6d8ae28f8e1e21d5111268a35961ba`.
The adapter changed only the header alias `misc/total_timesteps` to
`transitions_so_far`; all rows and other fields were preserved, with source
and input hashes recorded at each stage.

| Transition | Target | Paper | Ratio | Decision | KL | LR | Entropy | Head relative residual |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 2,007,040 | 6.53 | 9.28 | `.7036637931` | PASS | `.0581182614` | `.00050625` | `.580910385` | `1.355e-15` |
| 4,014,080 | 6.23 | 13.28 | `.4691265060` | `EARLY_STOPPED_ALGORITHM` | `.0112067182` | `.0656840836` | `.815481305` | `3.646e-15` |

At4M, head solve residual was `5.031e-14`, Cholesky info max0, Paper shared
critic relative residual `7.671e-06`, and all values were finite. The hard-error
scan found zero Traceback, NaN/Inf, OOM, CUDA/NCCL, disk or quota signatures.
This is algorithm stage evidence, not solver or hardware failure. Post-cancel
artifact synchronization reached4,177,920; it is not used for the decision.

## Three per-job infrastructure failures

Each failed root constructed the exact938,979-parameter production network,
then stopped at immutable `gpuh_preflight.py` line175 before
`scientific_started.marker`:

```text
assert sha(manifest_path) ==
  b45298be8fc5bdccfa36ce653c7dbc0c41f2d013f23d4f4db0a4a580035f3087
AssertionError
```

The structural partition remained environment-invariant, but the full JSON
contains observation-derived connectivity probe magnitudes. Its SHA therefore
varied: BossFight `c22a3e2a...`, CaveFlyer `53bc06f3...`, CoinRun
`77dd50d1...`, rather than BigFish `b45298be...`. Frozen scientific files and
allocated H200s were correct. These are
`infrastructure-failure/per-job-preflight-design`, not algorithm, numerical,
hardware, or scientific results. Task11 authorizes no repair or retry.

## Evidence and restrictions

Complete model-free evidence is in
`remote_launch_staging/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/evidence_task11/`:
final scheduler accounting; Paper provenance/hashes; progress and compressed
trace/logs; exact stage ledgers, snapshots, adapters and hashes;
trainable/optimizer/PopArt and partition manifests; and all three per-job
failures. No model or checkpoint is included.

No Jupyter, quarantined host, second candidate, duplicate monitor, sweep,
unrelated mutation, retry, repair, requeue or resubmission occurred. Three
environments have no scientific evidence, so promotion is impossible. Only one
environment triggered scientific early stop, so the at-least-two-environment
rejection criterion is not satisfied. The supported conclusion is
`CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`.
