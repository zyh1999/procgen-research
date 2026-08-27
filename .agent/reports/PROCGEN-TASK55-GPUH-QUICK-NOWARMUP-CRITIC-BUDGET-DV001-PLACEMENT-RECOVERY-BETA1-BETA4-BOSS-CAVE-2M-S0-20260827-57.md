# Task57 placement recovery launch report

Task-ID: `PROCGEN-TASK55-GPUH-QUICK-NOWARMUP-CRITIC-BUDGET-DV001-PLACEMENT-RECOVERY-BETA1-BETA4-BOSS-CAVE-2M-S0-20260827-57`

## Identity

Task56 remains immutable `RESOURCE_PLACEMENT_BLOCKED`. Task57 changes only the
deployment campaign/root and Slurm step request; all scientific files are
byte-identical to Task56.

- Task56 trainer SHA256: `db3357f9f828a5c0753065b87e7edde9b76f7710374f15e188221558c20ea31a`
- beta1 config SHA256: `ac6034d09c06df24170c24d09311122778d7bc8183f8fa250ed7f91139d7a304`
- beta4 config SHA256: `9b71ff8447221fda788eb1bc8a3442fa9b26e823bd69059de6dc0c7ea12b49a9`
- Task56 bundle SHA256: `e2c961036d39557420d417f7d175d206335a76c00736ef7172d92d8f419f0578`
- Task57 wrapper SHA256: `62fceea0761869c7a9270036b4de7cda5f413596fa74ba9cbfcc3453ef6c490a`

The wrapper diff from Task56 is limited to Task-ID, fresh campaign/root routing,
and frozen-wrapper provenance filename.

## Resource resolution

Slot A allocation `19487251` is RUNNING on node820 with
`cpu=8,mem=64G,gres/gpu:h200:1`. Historical successful Task52 step
`19487251.1` used eight CPUs and allocated the parent's `64G` plus one H200;
its step request recorded no explicit `ReqMem`. Task57 therefore requests
eight CPUs and one H200 with no explicit memory flag, inheriting no more than
the parent allocation's `64G`.

## Launch

The single Task57 step creation attempt succeeded:

- Parent allocation: `19487251`, node820
- Persistent step: `19487251.9`, `RUNNING/0:0`
- Launch time: `2026-08-27T12:18:49+01:00`
- Resolved step TRES: `cpu=8,mem=64G,gres/gpu:h200=1`
- Explicit memory flag: absent
- Campaign: `/scratch/h99859yz/procgen_task55_gpuh_quick_nowarmup_critic_budget_dv001_placement_recovery_beta1_beta4_boss_cave_2m_s0_20260827_57`

All four processes were launched concurrently exactly once without MPS:

| arm | environment | trainer PID | exact root suffix |
|---|---|---:|---|
| beta1 | BossFight | 1985952 | `runs/FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_DV001_BETA1_V1/bossfight-easy-0-10/seed0/2m_quick_nowarmup_critic_budget_dv001` |
| beta1 | CaveFlyer | 1985945 | `runs/FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_DV001_BETA1_V1/caveflyer-easy-0-10/seed0/2m_quick_nowarmup_critic_budget_dv001` |
| beta4 | BossFight | 1985973 | `runs/FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_DV001_BETA4_V1/bossfight-easy-0-10/seed0/2m_quick_nowarmup_critic_budget_dv001` |
| beta4 | CaveFlyer | 1985944 | `runs/FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_DV001_BETA4_V1/caveflyer-easy-0-10/seed0/2m_quick_nowarmup_critic_budget_dv001` |

Each root is `RUNNING`, has its own PID, log, runtime directory and
scientific-start marker. At the initial snapshot every cell had reached a real
Joint2B trace at transition `16,384`: `training_phase=joint2b`, phase switch
zero, PPO optimizer state zero, LR `.004`, `critic_trust_upper=.01`, 1024 rows,
938,976 columns, nonzero natural cross blocks, Cholesky info0 and finite scans.
Relative residuals were `2.917e-16` through `8.094e-16`; hard-error matches were
zero.

The shared H200 reported 143,771 MiB total, 78,618 MiB used, 64,541 MiB free
and 100% utilization. Slot B `19487252/node822` remained RUNNING and untouched.
No retry, requeue or resubmit occurred.

The existing sole automation was updated in place at its unchanged 20-minute
cadence to monitor Task51, Task55 and Task57. No second automation was created.
Current conclusion: `RUNNING_QUICK_READ_ONLY`.

## Actionable beta1 BossFight failure at 1.31M

The bounded 2026-08-27 13:28Z monitor pass found beta1 BossFight terminal at
the cell level while allocation `19487251` and step `19487251.9` remain
RUNNING for the other three independent cells. Root state is `FAIL/rc1`, PID
`1985952` is dead, the last progress row is transition `1,310,720` reward
`.35`, and the last trace is transition `1,318,912`. There is no exact
2,007,040 row and no checkpoint, so no Paper comparison or scheduler action
is eligible.

The terminal exception is `RuntimeError: Task51 natural actor-critic cross
blocks vanished` at the frozen trainer's `Advantage_Update`. Immediately
before failure, the actor raw scale was exactly zero, actor Fisher quadratic
and policy delta were zero, and both natural cross norms had collapsed to
`1.4155319819142736e-38`, while the critic raw scale remained `834463.8125`.
The preceding solve itself was finite with Cholesky info0 and relative
residual `2.8879584719616574e-14`. Targeted scans found no OOM, CUDA, NCCL,
disk or quota error. This is therefore an `algorithm/numerical` failure caused
by actor/cross collapse, not infrastructure, GPU or linear-solver failure.

The full bounded metadata and artifact hashes are in
`evidence/actionable_beta1_boss_1318912/summary.txt`. Beta1 Cave, beta4 Boss
and beta4 Cave remain RUNNING and were not modified. Current campaign
conclusion remains `RUNNING_QUICK_READ_ONLY_WITH_ONE_ALGORITHM_FAILURE`.

## Terminal read-only archive

Step `19487251.9` is terminal `FAILED/1:0` after `03:01:43` on node820. That
aggregate state is caused solely by the already archived beta1 BossFight
`FAIL/rc1`; beta1 CaveFlyer, beta4 BossFight and beta4 CaveFlyer each finished
`PASS/rc0` with 49 progress rows ending at exact transition `2,007,040`.

| arm | environment | Task57 | Paper | ratio | Task55 no-warmup | Task52 warmup H200 |
|---|---|---:|---:|---:|---:|---:|
| beta1 | BossFight | failed at 1.31M | 2.92 | ineligible | .19 | .70 |
| beta1 | CaveFlyer | 0.00 | 4.45 | 0 | 3.10 | 4.07 |
| beta4 | BossFight | .29 | 2.92 | .0993150685 | .26 | .62 |
| beta4 | CaveFlyer | 0.00 | 4.45 | 0 | 0.00 | 3.94 |

All immutable baseline checks passed. BossFight and CaveFlyer Paper progress
hashes are respectively `4082868e...5d89337c` and
`8d10f861...8236e292`. Task57 is read-only: even though all three endpoint
ratios are below `.60`, no cell was cancelled and Task51 was not mutated.

The completed endpoints preserve rollout-zero Joint2B, phase switch0, PPO
state0, fixed LR `.004`, critic upper `.01`, nonzero cross blocks, Cholesky
info0 and finite residuals (`6.52e-14` to `7.00e-13`). Hard-error scans are
zero. Each completed checkpoint is recorded only as regular non-symlink,
3,766,013 bytes, mode0640; no checkpoint bytes or hashes enter Git.

Relative to matched Task55, tighter critic upper `.01` improves only beta4
Boss by `.03`, collapses beta1 Cave from `3.10` to zero, leaves beta4 Cave at
zero, and beta1 Boss fails before endpoint. It is also below Task52 in every
comparable cell. The bounded conclusion is
`QUICK_DV001_TERMINAL_READ_ONLY_WITH_ONE_ALGORITHM_FAILURE`; it does not
replace authoritative Task51 evidence.
