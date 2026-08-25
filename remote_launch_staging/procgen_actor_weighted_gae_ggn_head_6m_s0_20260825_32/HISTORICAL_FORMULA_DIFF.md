# Task32 formula provenance and distinctness

| Method | Head curvature rows | Head residual/RHS | Damping | Actor/shared path | Cross blocks / rows |
|---|---|---|---:|---|---|
| Task32 `DET_ACTOR_WEIGHTED_GAE_GGN_HEAD_V1` | `diag(sqrt(w)) D_gamma,lambda,m J_h` | `diag(sqrt(w)) D_gamma,lambda,m (V-return)`; primal negative gradient | `.5` | exact Paper sampled actor and shared-critic systems, per-minibatch KL and momentum/history | head-only 257; no cross; minibatch output rows with full-rollout temporal operator |
| Hybrid-head V1 | `J_h` | scaled normalized return residual | `.5` | exact Paper actor/shared | head-only, B rows |
| NormMatch V2 | `J_h`, then proposal rescaling | Hybrid V1 residual plus Paper proposal-norm match | `.5` | exact Paper actor/shared | head-only, B rows |
| Separate-B | deterministic critic Jacobian in an independent B system | normalized value residual | `.1` critic damping | separate actor B solve | no cross, B rows |
| Joint-2B V1 | actor score and deterministic critic rows | paired joint actor/critic RHS | method-specific actor/critic damping | joint actor/critic solve | joint 2B with cross blocks |
| Historical expected/no-cross | expected or relative critic formulas | expected-score/relative RHS variants | historical method-specific | expected/relative actor variants | no-cross or expected cross-zero |

Task32 contains no sampled-value proposal similarity gate, proposal norm/RHS/
inverse matching, low-Fisher guard, projection, multi-epsilon quadrature, joint
or cross block. Paper RAT is retained only as the unchanged actor/shared causal
control and the exact-stage reward baseline.
