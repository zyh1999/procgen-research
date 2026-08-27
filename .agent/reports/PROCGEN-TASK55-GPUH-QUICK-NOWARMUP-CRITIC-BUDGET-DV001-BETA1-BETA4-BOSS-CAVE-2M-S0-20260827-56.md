# Task56 no-warmup critic-budget `.01` quick diagnostic

Task-ID: `PROCGEN-TASK55-GPUH-QUICK-NOWARMUP-CRITIC-BUDGET-DV001-BETA1-BETA4-BOSS-CAVE-2M-S0-20260827-56`

## Frozen scope

Task56 is a four-cell, seed0, exact-2,007,040 read-only quick diagnostic on
CSF3 Slot A. It preserves Task55 rollout-zero strict full-shared Joint2B,
fixed LR `.004`, beta1/beta4, both natural cross blocks, actor band
`.005/.04`, eta bounds `[1/64,64]`, multiplier `1.5`, damping, clipping,
PopArt, objective and evaluation semantics. Its sole scientific change is the
critic trust upper threshold `.04 -> .01`; the critic lower threshold remains
`.005`.

The paired warmup `.01/.005` Task56 plan was superseded before launch. No step,
root or process from that plan exists. Slot B allocation `19487252` remains
untouched.

## Frozen implementation

- Parent Task55 trainer SHA256: `91b835f16989a42293f6566d8fb9893dcd7b9ca969d1685d2d313f3f695f2f81`
- Task56 trainer SHA256: `db3357f9f828a5c0753065b87e7edde9b76f7710374f15e188221558c20ea31a`
- beta1 config SHA256: `ac6034d09c06df24170c24d09311122778d7bc8183f8fa250ed7f91139d7a304`
- beta4 config SHA256: `9b71ff8447221fda788eb1bc8a3442fa9b26e823bd69059de6dc0c7ea12b49a9`
- Slot A wrapper SHA256: `b55124a689819ef5849d9fc301f2436c435ffc4ee78aa6b958577b32cda891d0`

Compile, shell syntax and exact Task55-to-Task56 source/config diffs pass. The
trainer change adds a separately validated `dualtrust_critic_upper=.01`, uses
it only for eta_v adaptation/direction validation, and records the threshold
and accurate action reason. Actor adaptation remains bound to the original
`.005/.04` fields.

## Placement and launch

The remote code bundle was frozen under:

`/scratch/h99859yz/procgen_task55_gpuh_quick_nowarmup_critic_budget_dv001_beta1_beta4_boss_cave_2m_s0_20260827_56`

- Bundle SHA256: `e2c961036d39557420d417f7d175d206335a76c00736ef7172d92d8f419f0578`
- Parent allocation: `19487251`, `RUNNING`, node820
- Parent allocation TRES: `cpu=8,mem=64G,gres/gpu:h200=1`
- Slot B `19487252`: `RUNNING`, node822, untouched

The single authorized `srun` command requested `100G`. Slurm rejected step
creation before the wrapper or any cell process ran:

`srun: error: Unable to create step for job 19487251: Memory required by task is not available`

The error file SHA256 is
`dc4597d3954a85f0dde13393818620914a0329c35baec2f9b92363eea3b51508`.
Accounting contains no Task56 step ID; the campaign has no `runs/` directory,
root, PID, progress, marker, checkpoint, or scientific artifact. A process scan
also found no Task56 process. The failed launch command was not retried with a
smaller request, moved to Slot B, requeued, or resubmitted.

This is a pre-science Slurm step-creation/resource-request failure, not an
algorithmic or numerical result. Terminal conclusion:
`RESOURCE_PLACEMENT_BLOCKED`.
