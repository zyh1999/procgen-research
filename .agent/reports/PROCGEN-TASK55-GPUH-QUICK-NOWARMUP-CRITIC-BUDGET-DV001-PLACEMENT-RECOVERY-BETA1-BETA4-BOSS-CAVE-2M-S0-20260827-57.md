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
