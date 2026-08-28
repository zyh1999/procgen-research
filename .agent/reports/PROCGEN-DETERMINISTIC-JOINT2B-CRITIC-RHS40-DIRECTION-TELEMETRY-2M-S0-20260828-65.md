# PROCGEN-DETERMINISTIC-JOINT2B-CRITIC-RHS40-DIRECTION-TELEMETRY-2M-S0-20260828-65

Status: `RESOURCE_PLACEMENT_BLOCKED`

Method: `PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_RHS40_DIRECTION_TELEMETRY_ONLY_V1`

## Scientific identity

The exact parent is Task63/Task06 strict deterministic full-shared Joint-2B.
Curvature remains `0.1`, objective coefficient changes only from `1` to `40`,
and therefore the critic RHS multiplier changes from
`3.1622776601683795` to `126.49110640673517` exactly. Actor rows/RHS,
deterministic critic Jacobian, lambda-return residual, full natural cross
blocks, strict 1024-row solve, FP64 Jacobi/Cholesky, damping `.5`, no warmup,
adaptive KL/LR, history correction, momentum, global clip, PopArt, GAE,
seed/evaluation/reward and exact 2,007,040 horizon are unchanged. Task63
post-inverse direction decomposition remains telemetry-only.

## Frozen files

- trainer `18ea62829cb40dbc842c2c557be53a73ae3f3c8424ba9f9dec9ce00feb97e21c`
- gate config `d7228010aed2b807a87e619f51fa4b9753134521e399be51086e74530fa16c3c`
- science config `f283ad2442c88a1ff0c6500dca2a1a243af9bab47d32c5d028e362293e943f03`
- gate wrapper `9567a435f48573f8f31912e9b70b0e8477420e370ef46ee157cb35ed35465105`
- science wrapper `836cde1dc6715d69557dd27cf78b1dff9e2b491d730d078d0b318c91e534970c`
- aggregator `730e0cdfff095294ba9dcb325c1a969de33614b7f7609c1e714a318e27be896e`
- read-only monitor `6abe8d8dad83ba7b720d1dbe32178c6461d2575480a64abe84799c95ebc04381`

Local compile, config identity, wrapper syntax and exact parent-to-Task65 diff
checks pass. No model/checkpoint content or hash is included.

## Bounded placement result

Implementation commit `acf1664b66e3aacb798eec4f84d5529e4facbb38` was
pushed and matched `origin/agent-work`. The immediate CSF3 refresh could not
authenticate: the prior control socket no longer existed, the scoped SSH call
returned `Permission denied (keyboard-interactive)`, and an interactive reopen
stopped at the user password/MFA prompt. Therefore no live capacity/root check
could be completed, no campaign was deployed, and no gate or science job was
submitted. This is external authentication/resource-placement blocking, not a
scientific, numerical, code, or gate failure. Task64 Coin and all unrelated
jobs remained untouched.
