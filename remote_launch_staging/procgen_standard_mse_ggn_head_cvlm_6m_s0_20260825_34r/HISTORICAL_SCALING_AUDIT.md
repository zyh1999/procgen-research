# Task34R historical scaling audit

This audit aligns the critic-head systems in frozen PopArt-normalized coordinates. It does not infer semantics from method names.

## Target standard MSE system

For error `e = V - stopgrad(R_lambda)` and complete minibatch size `B=512`:

- objective: `e^T e / (2B)`;
- gradient: `g = J^T e / B`;
- GGN: `G = J^T J / B`;
- Gaussian precision: exactly one;
- solve: `(G + mu I)u = -g`;
- initial `mu = trace(G)/257`, because persistent `alpha` starts at one.

There is no Task13 curvature coefficient, GAE temporal operator, actor weighting, Paper RHS matching, norm matching, or hidden scaling.

## Task07 and Task13

Both frozen sources construct `head_rows = sqrt(0.1) J` and `head_rhs = (1/sqrt(0.1))(R-V)`, with fixed damping `0.5`. Their primal system is therefore

`(0.1 G + 0.5 I)u = -g`.

In standard coordinates this is

`(G + 5 I)u = -10 g`.

Thus the historical fixed `0.5` is not standard-MSE damping `0.5`: after the `.1` curvature scaling it is effective damping `5`, while the RHS is multiplied by `10`. Task07 applies its separate-B construction more broadly; Task13 preserves the Paper sampled critic on shared parameters and applies deterministic GGN only to the 257 critic-exclusive value-head parameters.

## Task32

Task32 is not a scalar rescaling of ordinary MSE. It forms a GAE temporal error/Jacobian operator and then applies normalized actor-score weights. Its fixed damping and observed max-weight concentration therefore cannot be transferred to the target standard `D=I, W=I, K=J` geometry.

## Momentum, clipping, and LM reductions

The actor/shared Paper gradient and its global clipping coefficient are computed exactly as in the frozen control. For each target trial, the raw train-only solve is passed through the existing head SGD momentum/history and that frozen global clip coefficient to form the actual candidate parameter delta.

Because a linear head with frozen features and targets has an exactly quadratic MSE, same-minibatch `ared_T` equals `pred_T` in FP64. This equality is audited but is rejected as an LM calibration signal. Acceptance uses only `rho_cv = ared_C / pred_T` on the disjoint next complete minibatch. The validation minibatch never enters `G`, `g`, the solve, momentum, clipping, or candidate construction.

The generated `historical_scaling_ledger.json` records source hashes and a deterministic numerical comparison of gradient, raw-solve norms, cosines, and direct-versus-standard-coordinate Task13 equivalence.
