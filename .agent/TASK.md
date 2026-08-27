# Task-ID: PROCGEN-TASK55-GPUH-QUICK-NOWARMUP-CRITIC-BUDGET-DV001-BETA1-BETA4-BOSS-CAVE-2M-S0-20260827-56

Status: IMPLEMENTATION_FROZEN

Run exactly four read-only quick cells on CSF3 Slot A allocation `19487251`
(`node820`): beta1/beta4 BossFight and CaveFlyer, seed0, exact horizon
2,007,040. The parent is frozen Task55: Joint2B begins at rollout zero with
no PPO update or phase switch, parameter LR remains `.004`, both natural cross
blocks and the full 1024-row system remain present, and eta bounds, multiplier,
damping, clipping, PopArt, objectives, evaluation and reward semantics remain
unchanged.

The sole scientific difference is a critic trust upper budget of `.01` instead
of `.04`. The actor band remains `.005/.04`; the critic lower threshold remains
`.005`; eta_min remains `1/64`. Run all four processes concurrently on the
single allocated H200 without MPS, using fresh roots and one persistent Slurm
step. Minimal compile/config/hash/CUDA/start checks only. No retry, requeue,
resubmit, cancellation, negative suite, audit, second budget, or Task51/55
mutation. Slot B `19487252` remains untouched.

Task56 is a quick read-only diagnostic and cannot replace Task51 or Task55.
At exact 2,007,040 archive Paper and matched Task55/Task52 comparisons without
scheduler cancellation. Keep model/checkpoint/token bytes out of Git. Update
the existing sole automation in place; never create a second automation.
