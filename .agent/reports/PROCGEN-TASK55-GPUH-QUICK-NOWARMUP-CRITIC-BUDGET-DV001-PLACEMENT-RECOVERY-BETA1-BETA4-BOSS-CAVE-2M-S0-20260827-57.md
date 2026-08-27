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

Pending the single exactly-once step creation attempt.
