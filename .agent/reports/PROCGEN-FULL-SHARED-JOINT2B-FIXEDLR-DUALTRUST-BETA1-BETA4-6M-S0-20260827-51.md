# Task51 paired fixed-LR dual-trust experiment

## Scope and frozen parent

- Task: `PROCGEN-FULL-SHARED-JOINT2B-FIXEDLR-DUALTRUST-BETA1-BETA4-6M-S0-20260827-51`
- Parent implementation/delivery: `e4f8cfc23bf406989f72db61ca8aadf5407d99d4` / `5cbdb7d30c6601bc69a0a4237670b4f141924298`
- Parent trainer/config SHA256: `35bb29e82f3d72067cc30431f5794902da01cce345d5d0dd0540193a0e362846` / `1ebd5f591c9693318eb977775553e06e48f27e1d6f2375d6f33889208eebbbe9`
- Task49/50 and `Bede1074901_*` are immutable and out of scope.

The PPO/Adam warmup ends at transition 503,808. One clean switch creates the
strict full-shared deterministic Joint-2B SGD path. The production actor
Fisher, full critic Jacobian, 1,024 dual rows, 938,976 ordered parameter
columns, both natural cross blocks, damping `.5`, momentum/history, global
clip, PopArt/GAE and evaluation semantics remain the parent implementation.

## Paired scientific delta

The two configs differ in one line: `beta_v: 1.0` versus `beta_v: 4.0`.
Parameter LR is fixed at `.004`; no Task50 LR update remains. Each full Joint
rollout holds `eta_pi` and `eta_v` fixed, measures full-class behavior-to-final
policy KL and fixed-PopArt-coordinate Gaussian value KL, then records exactly
one independent update per coefficient. Metric coefficients strengthen on
divergence above `.04`, weaken below `.005`, use factor `1.5`, and remain in
`[1/64,64]`. Objective coefficients remain one.

Frozen local SHA256 identities before remote work:

| Artifact | SHA256 |
|---|---|
| trainer | `af66fa0aa0115be2b82cad3666c9e91bf705053bfe219151de0689607dd4430d` |
| beta1 config | `57f6ca217bf3f02b0346b42a7797c0546a2a954166c2377667de81de058975c9` |
| beta4 config | `2f802e62fabfa1ce6056b3ea3cde7cd5591a5a8e3239d681d422f67d0eee9a20` |
| Bede gate launcher | `3083a6e09ec9e1e43a9ccfb27ba87eb3e821552da49e2c2695fcff2c7ddc702c` |
| Bede science launcher | `2be97e79821c7baf5f0937c694a819fc7d4c8d5b38e1028cbb2665a105745efb` |
| paired stage monitor | `73451eadf7a64b7ad49d2dbf4d96452463bdb04ca4d9f12cd856cb91478580f2` |

Local syntax/bytecode and both launcher shell checks pass. Each cell uses its
own runtime/log directory, avoiding cross-arm same-environment log collisions.

## Gate and launch matrix

Pending the sole two-arm Bede production gate. No science job or root exists
at this report stage. Gate PASS will be followed immediately by all eight
once-only submissions, targeting six simultaneous GPUs based on the immediate
live capacity refresh. Allocation count and actual RUNNING concurrency will
be reported separately.

Current bounded conclusion: `QUEUED_RESOURCE_WAIT` pending the sole gate.
