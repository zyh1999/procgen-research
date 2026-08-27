# Task-ID: PROCGEN-TASK55-GPUH-QUICK-NOWARMUP-CRITIC-BUDGET-DV001-PLACEMENT-RECOVERY-BETA1-BETA4-BOSS-CAVE-2M-S0-20260827-57

Status: RUNNING_QUICK_READ_ONLY

Task56 is terminal `RESOURCE_PLACEMENT_BLOCKED` and immutable. Task57 is a
fresh placement-only recovery using byte-identical Task56 trainer/config
science: no PPO warmup, rollout-zero strict full-shared Joint2B, fixed LR
`.004`, full 1024 rows and both natural cross blocks, actor thresholds
`.005/.04`, critic thresholds `.005/.01`, eta bounds `[1/64,64]`, multiplier
`1.5`, beta1/beta4 and matched solver/PopArt/objective/evaluation semantics.

Create one fresh persistent step inside Slot A allocation `19487251/node820`.
Match the successful Task52 step shape: eight CPUs, one H200, and inherited
parent memory with no explicit request above its fixed `64G`. Validate the
resolved request is at most `64G` before exactly one launch attempt. Run four
concurrent beta1/beta4 BossFight/CaveFlyer seed0 processes to exact 2,007,040,
with distinct roots/logs/PIDs and no MPS.

No preflight suite, science change, retry, requeue, resubmit, Slot B use,
credential exposure, model/checkpoint Git content, or Task51/55/56 mutation.
On successful launch update the existing sole automation in place; never create
a second automation.

Task57 persistent step `19487251.9` is RUNNING on node820. All four roots and
trainer PIDs exist, each has a scientific-start marker and a finite rollout-zero
Joint2B trace. The existing sole automation now monitors Task51, Task55 and
Task57 at the unchanged 20-minute cadence.
