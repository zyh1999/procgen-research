# Frozen manifest: PAPER_HYBRID_SHARED_PAPER_HEAD_DETGGN_V1

- Task: `PROCGEN-PAPER-HYBRID-HEAD-DETGGN-6M-S0-20260824-08`
- Source base: exact original Paper RAT trainer/config.
- Scientific replacement: critic-exclusive value-head raw direction only.
- Shared trunk: exact original Paper sampled critic direction.
- Policy exclusive: zero critic direction.
- Head solver: deterministic normalized residual `J_v`, lambda `.1`, one
  independent head-only B-by-B system, symmetric FP64/Jacobi/Cholesky.
- Matrix: BigFish, BossFight, CaveFlyer, CoinRun; seed0; 6M intended horizon.
- Stages: first exact common >=2M, >=4M, then 5,980,160; strict ratio `<.60`
  alone authorizes algorithmic early stop.
- No Paper rerun, second candidate, sweep, retry, Jupyter, shared GGN, joint or
  cross block, guard, projection, Kaczmarz, or quarantined host.

## Frozen artifact SHA256

| Artifact | SHA256 |
|---|---|
| exact Paper trainer | `cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7` |
| exact Paper config | `1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea` |
| P1 donor | `2b50f8cc26bb8c85f91e0b394acc7535f5564ac70036403597d3738d3dab9c1b` |
| target trainer | `7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54` |
| target config | `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda` |
| deterministic builder | `bb286d556c51465676b058c826755f027a207c1b4eb0b76e48670a753a8610a8` |
| identity audit | `d6d268e4fc8e28f34ec4a7c60ae97c8a0e85183ae96f4bd7135d5258a20e1bca` |
| regression | `8a774ee31e49157556a2e4454227114f033a76d410acd796c240d43c6bae5465` |
| gpuH compatibility test | `4bcbff44137ddf66c76a8ad06a357459411726d6f9dc1fa7d10897a473027292` |
| gpuH persistent preflight launcher | `6cf90a9811b9c9b271cc787c468724bc70e05bd173ec6db26e92f2043ea43e28` |
| gpuH scientific launcher | `ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e` |
| frozen stage monitor | `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e` |
| historical provenance | `4c361d494792caf4ed76f5dc556bb5e1e67e11ad9f6b3b2bbfe8da315672caa1` |

The actual-network parameter partition is written by the persistent gpuH
preflight. Scientific submission is forbidden unless that preflight returns
`HYBRID_HEAD_PREFLIGHT_PASS` and `GPUH_HYBRID_HEAD_COMPATIBILITY_PASS`.
