# Task61: no-warmup Joint2B plus post-solve entropy gradient

## Status

`CANDIDATE_NOT_READY` — implementation and minimal local gates are complete;
the sole production gate has not yet run.

## Frozen parent

- Task55 implementation: `3a850cd3870854123c76693a974a2fe45e952203`
- Trainer SHA256: `91b835f16989a42293f6566d8fb9893dcd7b9ca969d1685d2d313f3f695f2f81`
- beta1 config SHA256: `75fb59290d4bd2399986e372a62e56b4aaa6df7becb205f41ee332538f04425f`
- Matched controls: frozen Task55 beta1 BossFight/CaveFlyer exact-2M roots;
  they remain untouched.

## Sole scientific change

`postsolve_entropy_coef=0.01`. The current-policy entropy ascent gradient is
computed in the exact ordered full parameter space only after the unchanged
Joint system, RHS, history correction, dual solve and primal reconstruction.
The applied direction is `joint_dir + .01 * entropy_ascent`, followed by the
parent's single global Euclidean clip and fixed LR `.004`.

Entropy is absent from the system/RHS/solve/eta controller and from the Joint
history buffer. Policy/shared/value entropy norms are recorded separately;
value-head-exclusive entropy must equal zero exactly. Joint-only and applied
actor/critic projections, quadratics and predicted divergences are distinct
telemetry fields. Actual rollout `D_pi/D_v` continues to drive the unchanged
dual-trust controller.

## Minimal local gates

- parent trainer/config SHA match: PASS
- Python compile: PASS
- gate/science shell syntax: PASS
- config contains only the explicit `.01` post-solve coefficient: PASS
- method is beta1 only; no warmup, LR or trust-band change: PASS

## Remote gate and science matrix

Pending exactly one production gate. On PASS only:

| Environment | Seed | Horizon | Job | Root | State |
|---|---:|---:|---|---|---|
| BossFight | 0 | 2,007,040 | pending | fresh | not submitted |
| CaveFlyer | 0 | 2,007,040 | pending | fresh | not submitted |

No retry, requeue, resubmit or reward-based cancellation is allowed. Model and
checkpoint bytes/hashes remain outside Git.
