# Validation — updated 2026-09-27

Validated with PyTorch 2.4.1 and NumPy 1.26.4 on macOS ARM64 CPU.
All21 public unit tests pass. All six explicit training entries import with
`--help`; each corresponding shell script passes `bash -n`.

- Full-RHS parameter directions agree with an independently formed dense
  parameter-space solve (FP32 contraction tolerance 2e-5).
- Dual coefficients agree with an independently formed dense Galerkin solve,
  including positive non-unit ratios, signed targets and the zero-tail case.
- Conv2d d_in/d_out tail contractions agree with explicit per-example gradients.
- Pivotal samples preserve fixed cardinality, uniqueness and certainty units.
- Both methods execute B=512 / reduced-budget128 updates with finite parameters.
- KL controller boundary/floor/cap checks pass.
- Shared ResNet outputs and PopArt unnormalized-value preservation pass.

The frozen Task253 `Advantage_Update` function was also executed directly
beside the exported update on identical B=512 inputs with the same model and
RNG state. All four combinations (Dual127+1 / Full-RHS128 × uniform / PCA-pivotal)
produced **exactly equal parameters (maximum difference 0), preclip gradient
norm, behavior KL, next LR and final RNG state** in this CPU fixture.
The exported shared ResNet/heads also have exactly equal initialized state
dicts to the frozen model under the same seed.

## Current coefficient128 and B8192 profiles

The exported coefficient128 normal update was additionally compared directly
with the frozen Task280 `Advantage_Update` function on identical B512 inputs,
for uniform and leverage sampling. Parameters, preclip norm, behavior KL,
next LR and final RNG state are **exactly equal** in both CPU fixtures.
These are extraction checks, not new leverage training results.

- Coefficient128 with ray on/off agrees with independently formed dense
  symmetric sample systems and fixed-tail affine restrictions (FP32 tolerance).
- Streaming Dual127+1 and coefficient128 agree with full-factor reference
  solvers, with physical chunks37 and128, including convolutional factors.
- Streamed and normal B512 real-loss updates agree within FP32 tolerance,
  retain equal final RNG state and choose the same next LR.
- B8192 synthetic real-loss updates for both methods remain finite, keep
  system/reconstruction denominators8192 and inherit LR across updates.
- Ray-off sets gamma exactly1; ray-on sample residual never increases.
- Four profiles have exact endpoints1466×4096 or229×65536 and32 updates/rollout.
- CLI regression checks freeze residual-ray ON by default for both128
  profiles, with explicit on/off overrides;127+1 does not use that scaling.

The newly added B8192 coefficient128 stream is a mathematical implementation
port, **not a completed full RL result**. Large-batch synthetic update tests
use small networks; real ResNet shape tests and convolution algebra checks
do not establish full8192 ResNet training memory or speed on every GPU.
No cluster jobs, frozen training sources, checkpoints or existing campaigns
were changed while packaging. The old combined train.py entry was removed;
the previous2026-09-26 export zip remains available outside this package.

## Fixed-LR PPO additions

- B512 and B8192 shared PPO streamed gradients, combined norm clip and one
  Adam step match independent dense mean/B reference losses in FP64
  (parameter atol1e-12, rtol1e-11; preclip norm and losses agree to11places).
- One logical minibatch increments every Adam state by exactly one step,
  independent of physical chunk size; regression fixture uses chunk37.
- Low/high finite behavior KL never changes LR3e-4. A changed LR is rejected.
- Nonfinite gradients fail before the optimizer step and preserve parameters.
- Plain PPO linear value head returns raw values; default RAT PopArt remains
  enabled, with existing shape/algebra/PopArt invariance tests unchanged.
- Both fixed PPO profiles have exact endpoints,32updates/rollout,entropy0,
  no PopArt or adaptive-LR/sampling/ray options.
- A two-rollout synthetic runner fixture checks driver status,64trace events,
  constant logged LR, endpoints and checkpoint creation without reading it.
  This fixture does not instantiate Procgen or start an RL benchmark.

The PPO port uses Task266/278 shared PPO loss/optimizer conventions. It is
not a completed end-to-end reproduction, and does not include the separate
no-shared Task284 training implementation. No benchmark results are added.

This checks algebra and extraction fidelity, not learning curves or
cross-hardware bitwise reproduction. No new short RL run or full benchmark
was launched to package this release. The simplified Procgen wrapper/training
entry point has not completed an end-to-end Procgen run in this validation.
