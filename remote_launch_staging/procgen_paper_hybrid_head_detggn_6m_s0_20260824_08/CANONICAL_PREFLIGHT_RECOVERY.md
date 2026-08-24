# Canonical preflight recovery

- Task: `PROCGEN-HYBRID-HEAD-CANONICAL-PREFLIGHT-AND-6M-S0-20260824-09`
- Authorization: exactly one canonical non-scientific recovery preflight.
- Prior immutable failures: `19220448` import-path design failure and
  `19220752` incomplete hand-built namespace (`norm_obs`) design failure.

## Immutable scientific identity

| Artifact | Required and observed SHA256 |
|---|---|
| trainer | `7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54` |
| scientific config | `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda` |
| scientific launcher | `ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e` |
| frozen stage monitor | `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e` |

## Allowed recovery files

| Artifact | SHA256 |
|---|---|
| canonical gpuH harness | `9cc29a14083dcec8640f5822128a11e2913d997ae2d331b070d68a13a4072b32` |
| persistent preflight launcher | `3c1356f5387226f40e2c6a5692d59a232cbec5ab5f3048f4265252d321ab05e3` |
| canonical static guard | `854bd05c02e2d80a2fb325fee091c5c7713527d8a71eb2b063fc04e74ab4e342` |

## Line-level change classification

- `gpuh_preflight.py`: removes `SimpleNamespace` and direct
  `SharedActorCritic` construction. It invokes the frozen trainer's own
  `main()` argument parser and default/override merge three times. The
  production-model capture calls the original `train_fn()` and intercepts at
  `learn()` before any rollout or optimizer step. The three resolved configs
  (preflight, scientific-launcher dry-run, trainer entry) are canonical JSON
  and must be byte-identical. Actual-network partition/Jacobian, one-step
  policy isolation, Paper actor/shared-critic direction identity, FP64 head
  solve, and H200 memory checks then run on that production-constructed model.
- `hybrid_head_preflight_gpuh.sbatch`: adds only the frozen canonical static
  guard, its hash capture, and an explicit BigFish environment for the
  non-scientific construction. Scheduler identity is unchanged.
- `test_canonical_preflight.py`: asserts all four scientific hashes, absence of
  the hand-built namespace, presence of trainer `main()`/original `train_fn()`
  invocation, and three-way configuration comparison.

No trainer, scientific config, scientific launcher, monitor, algorithm,
environment, seed, budget, evaluation, root, or scientific retry semantics are
changed.
