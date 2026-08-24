# Frozen scientific identity

- Method: `PAPER_MATCHED_DETERMINISTIC_GGN_V1`
- Environments: BigFish, BossFight, CaveFlyer, CoinRun easy 0-10
- Seed/budget: seed 0, 1,000,000 requested transitions only
- Paper base commit: `2b5affd64cbb3c624b4bc1f4767f449df231ffb2`
- Paper trainer SHA256: `cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7`
- Paper config SHA256: `1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea`
- P1 solver donor SHA256: `2b50f8cc26bb8c85f91e0b394acc7535f5564ac70036403597d3738d3dab9c1b`
- Target trainer SHA256: `41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a`
- Target config SHA256: `69d12937debb8ef8b4531e79b8f9613185b26e4e9056a51e67754129b269391d`
- Diff audit SHA256: `5acb70c6b77580ed766414c9d99c5b910fdaca2b115ba054dc57f50dd98451b4`
- Regression test SHA256: `0d6e475f42716e4809019faa80a431f8238fd0134b5db8d38e140e9e3a53339b`
- Launcher SHA256: `d7581c58e89f52eb38603ac435e8c09abe2bbb6a0dbb651e1bc96a0e9912b48f`
- Corrected gpuL launcher SHA256: `d0e7c7928ceefc9ba7f6eec78a17606d050c25467f05a532c49ab8d8ad17266e`
- Corrected persistent gpuL preflight script SHA256:
  `024ebe2b1ba715b618f755986c811cd265b5d737953e9c6ca91a53a6cbbceaba`
- Persistent gpuL preflight sbatch SHA256:
  `a0ffe6bb0f6896be1cfc81a857fffe557f46a5b151ba557452638381c47615ce`
- H200 eight-child aggregate preflight SHA256:
  `dfd52c1d18484f8974103d48d6e813d1e1e4a391d53d5605e78156d959caa6e5`
- H200 environment-bundle launcher SHA256:
  `29987a1f48f3d8df04a0eb4eb9e6179e1d5f82b7fac2a65e5813e1aa75c4ed54`
- Explicit user race override SHA256:
  `94820e10e8029399734964ff0327c4f22d2fe50220b691bc12b775a75e759473`
- Scheduler-only diff SHA256:
  `1ca369021fa73b45cdd848f9e660f252367366761b1fad18c486d29f2a24e75b`
- Scientific delta: sampled critic score and two independent Paper inverses are
  replaced by deterministic value Jacobian/residual, critic curvature lambda
  0.1, stacked joint-2B kernel, symmetric FP64 Jacobi Cholesky solve, and
  solver telemetry.
- Paper invariants: ResNet hidden256, actor score/ratio/advantage normalization,
  SGD LR .5, momentum 1e-6, Paper `rhs - H @ momentum_buffer` history rule,
  per-minibatch adaptive KL thresholds .005/.04, rollout 4096, minibatch 512,
  four epochs, damping .5, global clip .5, PopArt/GAE/log/checkpoint semantics.
- Forbidden: P1 LR .004, rollout adaptive KL, momentum 0, disabled history,
  Jupyter, retries, Paper rerun, seeds1/2, 6M, or a second candidate.
