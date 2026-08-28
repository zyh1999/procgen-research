# PROCGEN-DETERMINISTIC-JOINT2B-CRITIC-RHS10-DIRECTION-TELEMETRY-2M-S0-20260829-66

Status: `IMPLEMENTATION_FROZEN_PRECHECK_PENDING`

Method: `PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_RHS10_DIRECTION_TELEMETRY_ONLY_V1`

## Scientific identity

The exact parent is terminal Task65. Curvature remains `0.1`; the sole
scientific change is critic objective coefficient `40 -> 10`, yielding critic
RHS weight `31.622776601683793`. No-warmup strict deterministic full-cross
Joint-2B, 1024 rows, actor definition, history correction, adaptive KL/LR,
damping, global clip, PopArt/GAE, seed0, evaluation/reward, exact 2,007,040
horizon, post-inverse direction telemetry and aggregation semantics are
unchanged.

## Frozen implementation

Local compile, wrapper syntax and exact Task65-to-Task66 diff checks pass. The
trainer/config/aggregator/wrapper/monitor hashes, implementation commit, Bede
gate result, science job/root mapping and first-update telemetry are recorded
below after each bounded stage completes. No model/checkpoint bytes or content
hashes enter Git.

- trainer `eb66bc51b8489001eb7ee3849843045fb1426dfd46c8286b8c44ac871e067dcd`
- gate config `7eb4cae54b43e1399d2363712fbdf6927e71337f588a3468c317a0b0fa2e7c82`
- science config `cde225f5b92be719d8a05eb04954e9e39429504ed53b57438bdb18c493154771`
- aggregator `e1f881b18fe596e88c1034849af82edb05661a1d41dc7b5cd3dd4050edf7ee8d`
- read-only monitor `6abe8d8dad83ba7b720d1dbe32178c6461d2575480a64abe84799c95ebc04381`
- Bede gate wrapper `2898e9f14bec5806c10977818c16eb793e2eb594687c693d8eeb897205b5a7f9`
- Bede science wrapper `02f509ea9b350277d31830c5004e78a76467c373f1131452ff7d0202bd3ccecf`

## Bede gate and launch

Pending the sole authorized production gate.
