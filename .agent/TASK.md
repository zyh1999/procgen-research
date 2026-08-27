# Task-ID: PROCGEN-TASK55-QUICK-NOWARMUP-POSTSOLVE-ENTROPY-GRAD001-BETA1-BOSS-CAVE-2M-S0-20260827-61

Status: RUNNING_QUICK_READ_ONLY_GPUH_MIGRATED

Method: `FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_POSTSOLVE_ENTGRAD001_BETA1_V1`

Execute exactly one bounded beta1 seed0 quick diagnostic for BossFight and
CaveFlyer to exact transition `2,007,040`.

Strict parent is Task55 freeze `3a850cd3870854123c76693a974a2fe45e952203`,
trainer SHA256 `91b835f16989a42293f6566d8fb9893dcd7b9ca969d1685d2d313f3f695f2f81`
and beta1 config SHA256
`75fb59290d4bd2399986e372a62e56b4aaa6df7becb205f41ee332538f04425f`.
Preserve rollout-zero Joint2B, PPO state zero, fixed LR `.004`, beta1,
dual-trust bands `.005/.04`, eta bounds `[1/64,64]`, damping `.5`, global clip
`.5`, PopArt, complete natural cross blocks and strict `1024x938976` system.

The sole scientific change is `postsolve_entropy_coef=.01`: compute the
standard current-policy entropy ascent gradient only after the unchanged
Joint system/RHS/history/solve/direction, add `.01` times that gradient to the
applied direction, then use the parent's single global clip and fixed LR.
Entropy must be exactly zero on critic-exclusive value-head parameters and
must not enter the system, RHS, solve, eta controller or Joint history buffer.
The unchanged actual rollout `D_pi/D_v` measurements drive eta feedback.

The sole Bede gate `1078146` is `COMPLETED/0:0` and `PRECHECK_PASS`. Under the
explicit user speed-migration authorization, low-progress Bede jobs `1078147`
and `1078148` were cancelled exactly once and classified
`CANCELLED_FOR_USER_AUTHORIZED_GPUH_SPEED_MIGRATION`. Exact fresh CSF3 jobs
`19507047` BossFight and `19507048` CaveFlyer were submitted together once and
are RUNNING on node822. Idle Procgen allocations `19487251/19487252` were
released only after all scientific child work was terminal, classified
`RELEASED_AFTER_TERMINAL_PROCGEN_QUICK_WORK_TO_UNBLOCK_TASK61_NORMAL_GPUH`.
Never retry/requeue/resubmit, touch Task51--60, create another arm or
coefficient, or commit model/checkpoint bytes. At endpoint compare read-only
against immutable Paper and Task55 beta1; never cancel for reward.
