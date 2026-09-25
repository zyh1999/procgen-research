# Procgen Shared Leverage 127+1: Exact Source Snapshot

These two files are unmodified copies of the audited shared large-batch
leverage experiment, published for code and mathematical review.

Source campaign:
`procgen_largebatch_shared_dual127p1_leverage2000_capx3_first4_s012_10m_bede_20260924`

Snapshot date: 2026-09-25. This is Procgen, shared actor-critic, not MuJoCo,
not no-shared, and not the newer fixed-tail ordinary128 experiment.

## Read These Functions

- `streamed_dual.py:select_streamed_leverage`: sampled 2000 parameter columns,
  ratio-weighted shared scores, rank127 randomized PCA, 5% uniform mixture,
  fixed-size pivotal sampling without replacement; no HT correction.
- `streamed_dual.py:solve_streamed`: full-minibatch non-anchor aggregation,
  shared sampled score `logpi + 2 * noise * value`, selected Gram and cross terms.
- `streamed_dual.py:solve_statistics`: the actual ratio-consistent 127+1
  coefficient systems and Schur solve.
- `train_large.py:train`: construction of actor/critic targets and ratios,
  detached coefficients, weighted policy/value losses, gradient accumulation,
  clipping and optimizer step.

## Exact Matrix Convention

Let H contain the shared sampled-score gradient rows. S contains 127 anchors;
N contains all remaining samples. In this implementation Q=128 (total reduced
degrees of freedom), B=16384, and damping mu=0.5. The source variable `q`
in train_large.py means Q, not the number of anchors.

Solve separately for actor (b=advantage, w=clamped policy ratio) and critic
(b=ones, w=ones), using the same H and anchor indices:

```text
base = H_N.T @ (w_N * b_N)
E = sum(w_N * b_N**2)
s = sqrt(w_S)
A = diag(s) @ (H_S @ H_S.T / Q) @ diag(s) + mu * I
c = s * (H_S @ base) / Q
C = (base.T @ base) / Q + mu * E
h = s * b_S

[ A   c ] [ y   ] = [ h ]
[ c.T C ] [ rho ]   [ E ]

alpha_S = y / s
alpha_N = rho * b_N
```

For E=0, the code sets rho=0 and solves the anchor block. Otherwise the Schur
denominator must be finite and positive. Actor and critic have separate rho.

The parameter update is reconstructed through the original losses, NOT by
directly assigning a joint-score parameter direction:

```text
L_pi = -sum(clamp(exp(logpi-logpi_old), .1, 10) * stopgrad(alpha_actor)) / B
L_v  = sum((value-return)**2 * stopgrad(alpha_critic)) / B
g = grad_theta(L_pi + L_v)
g = L2_clip(g, max_norm=.5)
theta <- theta - learning_rate * g
```

The ratio clamp is differentiated as written in train_large.py; it is not
silently replaced by a detached-ratio surrogate in the shared branch.
Physical chunks of128 accumulate into one logical minibatch gradient before
the optimizer step. Rollout=131072, four epochs, 32 updates per rollout.

## Provenance and Limitations

SHA-256 of the unmodified files:

```text
7804eacba0b1e804c2b299b7c4a6945dcb162a92ab7543715f6b02c477ab1a03  streamed_dual.py
2afacd97fff2cde764c941cd3f5bc6aa115ec52b5c0fef33e3ee4f0bd24dca4d  train_large.py
```

These hashes matched the audited Bede deployment. The files retain legacy
Task266 labels alongside the later leverage-specific metadata override;
see the executed override and endpoint in train_large.py.

This is a two-file audit snapshot, not a standalone runnable distribution.
Imported project utilities, environment packages, logs, checkpoints and
credentials are not included. No training-performance claim is made here.
The equations above document the implemented algorithm; they are not a claim
that its update equals a full parameter-space natural gradient.
