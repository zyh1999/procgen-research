# Task47 — full-shared deterministic GGN block-diagonal BxB

Task-ID: `PROCGEN-FULL-SHARED-DETGGN-BLOCKDIAG-BXB-6M-S0-20260826-47`

Method: `FULL_SHARED_DETGGN_BLOCKDIAG_BXB_V1`

Current conclusion: `QUEUED_RESOURCE_WAIT`

## Frozen identity and causal delta

The sole parent is Task06 commit
`da34ce7c7d964765f336ac02111c9fde95aed1ec`, not Task39/45/46.

- Parent trainer Git blob: `3385a5ef038e9e5740acf3a6bea01b46e04b52f3`
- Parent trainer SHA256: `41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a`
- Parent config Git blob: `51d63053c1c5c094086497ba3602df6f0800b8dc`
- Parent config SHA256: `69d12937debb8ef8b4531e79b8f9613185b26e4e9056a51e67754129b269391d`
- Parent report SHA256: `2a22205933109985c4388cdcfe64f0aeda23e44c2122fa1dec13757acb0f6251`
- Task47 implementation/origin commit:
  `8f9abc3687434c96bf9786fca29051dd084bc6f6`
- Task47 trainer SHA256:
  `fac7ae81734837faa8b3c3c2c62e671423636f5b4b43143d032db49d9950aea4`
- Task47 config SHA256:
  `7fd09cf24622b6556174bdc735e93320cfb607cf4862f299e3374e1a5e5eaf92`
- Preflight SHA256:
  `3ff16c92d8cd897a1a3953ac314995f4000c0f79553643435fc1996d31cead72`
- Preflight launcher SHA256:
  `88b027f7fb7b90006f0f890c4f664f283956bbec6e1276c961332722d0688180`
- Science launcher SHA256:
  `d04fae6c54751e86565eaedf1e85c2441dfffd61d6e666805705f0d32f74f328`
- Stage monitor SHA256:
  `c7a68afec033c04ef14251d17aef1bcaee141688fd79829de3cd26863abb6ae9`
- Parent-to-Task47 trainer diff SHA256:
  `bbc9dcfdd58a0c008a2e1ec7b40d0b530fa0cdfad5171c2f31a312e1307fcdbc`
- Parent-to-Task47 config diff SHA256:
  `443029bf7f9c444134ac7d3d7688abab97d9a6b9a3fa2a690cf603ef8e31fd3c`

The parent raw actor rows/RHS/rollout ratio, deterministic full-network value
Jacobian/RHS, critic curvature `.1`, critic objective coefficient `1`, damping
`.5`, complete ordered parameter space, history correction, momentum, global
clip, adaptive KL, rollout/GAE/PopArt and evaluation semantics are retained.
The 6M config changes only the intended easy-environment horizon.

The sole scientific change removes `AJ^T` and `JA^T` from the dual solve:

```text
(AA^T/B D_A + .5 I) alpha_A = b_A
(JJ^T/B       + .5 I) alpha_C = b_C
d_A = A^T D_A alpha_A / B
d_C = J^T alpha_C / B
d   = d_A + d_C
```

The actor and critic retain the same full `P` columns. Actor value-head columns
and critic policy-head columns remain structural zeros; shared columns receive
both directions. No normalization, actor floor, CVLM, projection, sampled
critic, head-only path or extra stabilizer is present. The implementation adds
only required per-block solve/direction telemetry around this change.

## Single production preflight

The only production preflight was Slurm job `19425914`, which completed
`COMPLETED/0:0` in 17 seconds on node820. Root status is `PRECHECK_PASS`, rc0.

It used the actual Procgen construction path:

- observation space HWC `(64,64,3)`;
- model input CHW `(3,64,64)` and ResNet image size `64`;
- 26 ordered trainable tensors, `P=938,976`;
- actor and critic matrices each `(512,938976)`;
- actor value-head structural-zero columns: `257`;
- critic policy-head structural-zero columns: `3,855`.

The two independent dual systems were exactly `(512,512)` and `(512,512)`.
No dual cross block was assembled or solved. Actor/critic Cholesky info values
were `0/0`; relative residuals were `5.1941303647114013e-14` and
`1.1461023091918497e-13`. All directions were finite. Direction norms were:

| Block | actor | critic | sum |
|---|---:|---:|---:|
| shared | .11710155 | .46707293 | .48224455 |
| policy head | .02594219 | 0 | .02594219 |
| value head | 0 | .03543279 | .03543279 |

The full summed direction norm was `.48423988`. This is the one authorized
no-training production solve check; no micro/negative/audit chain was run.

## Placement and science launch

Campaign:
`/scratch/h99859yz/procgen_full_shared_detggn_blockdiag_bxb_6m_s0_20260826_47`

At the single launch refresh, account `gpu-h200-fse-pgdr` and QOS
`gpu-h200-fse` permitted at most four H200 jobs for this user. Task46 CoinRun
`19424176` already occupied one slot. All four Task47 roots were absent and no
Task47 job/process existed.

All four cells were submitted once in one bounded command. There was no
dependency, array throttle, hold, serial release, retry, requeue or resubmit.

| Environment | Job | Initial scheduler/root state | Root |
|---|---:|---|---|
| BigFish | `19425987` | RUNNING node820; root RUNNING, scientific marker | `/scratch/h99859yz/procgen_full_shared_detggn_blockdiag_bxb_6m_s0_20260826_47/runs/FULL_SHARED_DETGGN_BLOCKDIAG_BXB_V1/bigfish-easy-0-10/seed0/6m` |
| BossFight | `19425988` | RUNNING node821; root RUNNING, scientific marker | `/scratch/h99859yz/procgen_full_shared_detggn_blockdiag_bxb_6m_s0_20260826_47/runs/FULL_SHARED_DETGGN_BLOCKDIAG_BXB_V1/bossfight-easy-0-10/seed0/6m` |
| CaveFlyer | `19425989` | RUNNING node821; root RUNNING, scientific marker | `/scratch/h99859yz/procgen_full_shared_detggn_blockdiag_bxb_6m_s0_20260826_47/runs/FULL_SHARED_DETGGN_BLOCKDIAG_BXB_V1/caveflyer-easy-0-10/seed0/6m` |
| CoinRun | `19425990` | PENDING `AssocMaxJobsLimit`; no root until start | `/scratch/h99859yz/procgen_full_shared_detggn_blockdiag_bxb_6m_s0_20260826_47/runs/FULL_SHARED_DETGGN_BLOCKDIAG_BXB_V1/coinrun-easy-0-10/seed0/6m` |

Requested Task47 concurrency is four; initial Task47 concurrency is three
because Task46 Coin occupies the fourth H200 allowance. The pending Task47
Coin cell remains queued to start naturally when that slot releases. Task46
was not cancelled, moved or modified. The three started roots produced early
progress and had zero immediate hard-error matches.

## Monitoring and comparison

The existing sole automation `procgen-3090` was updated in place at its
20-minute cadence to monitor Task46 Coin plus these four Task47 cells. No
second automation exists. The frozen Task47 monitor may compare only immutable
same-environment/seed0/evaluation Paper rows at first exact common >=2M, first
exact common >=4M and endpoint `5,980,160`; only Target/Paper `<.60` permits
one-cell cancellation and `EARLY_STOPPED_ALGORITHM` evidence.

No exact Task47 comparison row exists yet. No model or checkpoint is included
in Git. The current bounded conclusion is `QUEUED_RESOURCE_WAIT`: three cells
are scientifically running and the fourth is queued solely on the live user
H200 concurrency limit.

## Exact 2M decisions: BossFight and CaveFlyer

At the first exact common row, 2,007,040, the frozen Task47 monitor wrote one
decision per cell and was applied exactly once to each failing cell:

| Environment | Job | Target | Paper | Ratio | Action |
|---|---:|---:|---:|---:|---|
| BigFish | `19425987` | 8.28 | 9.28 | .8922413793 | PASS; remains RUNNING |
| BossFight | `19425988` | .07 | 2.92 | .02397260274 | `EARLY_STOPPED_ALGORITHM` |
| CaveFlyer | `19425989` | 2.50 | 4.45 | .5617977528 | `EARLY_STOPPED_ALGORITHM` |

BossFight and CaveFlyer both became scheduler-authoritative `CANCELLED by
778916`, exit `0:0`, elapsed 00:34:55 on node821. Their root `RUNNING` marker
and absent launcher rc are stale effects of scheduler cancellation. Neither
cell has a checkpoint. No repeat apply, retry, requeue or resubmit occurred.

The exact-stage numerical evidence is healthy and distinguishes these reward
failures from infrastructure or solver failures:

| Metric at 2,007,040 | BossFight | CaveFlyer |
|---|---:|---:|
| raw actor scale | 114106.859 | 45954.703 |
| raw critic-J scale | 311227.125 | 93429.281 |
| weighted critic scale | 31122.713 | 9342.928 |
| actor direction norm | .494923 | .448758 |
| critic direction norm | .608464 | .486435 |
| summed direction norm | .885544 | .717909 |
| actor/critic direction cosine | .280614 | .177263 |
| shared actor/critic norm ratio | .853081 | .944758 |
| global clip scale | .564625 | .696467 |
| actor/critic relative residual max | 1.6665e-13 | 1.1197e-13 |
| Cholesky info max | 0 | 0 |
| finite scan | PASS | PASS |
| hard-error matches | 0 | 0 |

At the stage row, Boss entropy was `.3210`, logged behavior KL
`7.735e-7`, and LR `.0001`; Cave entropy was `.3911`, KL `1.442e-7`, and LR
`.0001`. Both maintained separate actor/critic BxB solves and
`blockdiag_no_dual_cross_solve=1`.

Because the 20-minute monitor acted after the exact rows were already durable,
the processes had advanced to later transitions before scheduler cancellation.
Those later rows are retained only as pre-cancel numerical evidence; the
scientific decision remains the exact 2,007,040 comparison above. Full progress,
logs, selected exact/final trace rows, artifact hashes, scheduler state and
zero hard-error scans are under
`evidence_monitor_20260826_1300z/{bossfight,caveflyer}-easy-0-10`.

Task47 CoinRun `19425990` started naturally on node822 immediately after the
two slots released; BigFish and CoinRun remain RUNNING. Task46 CoinRun
`19424176` independently passed its exact 4,014,080 comparison,
`6.4/8.0=.8`, and remains RUNNING. No live cell was cancelled or otherwise
modified by this archive. Task47 is nonterminal and its current bounded
conclusion is `CANDIDATE_NOT_READY`.
