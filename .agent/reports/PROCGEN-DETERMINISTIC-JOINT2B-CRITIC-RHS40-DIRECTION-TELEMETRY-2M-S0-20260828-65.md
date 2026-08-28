# PROCGEN-DETERMINISTIC-JOINT2B-CRITIC-RHS40-DIRECTION-TELEMETRY-2M-S0-20260828-65

Status: `TERMINAL_TELEMETRY_COMPLETE_TASK65_AGGREGATION_PASS`

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

## Authorized Bede recovery

The user authorized Bede as Task65's first and only deployment. The existing
authenticated Bede connection is live; account `bdman37g`, partition `gpu`,
and idle V100 capacity are available. The exact Bede campaign and duplicate
Task65 job/process checks are absent. Deployment-only wrappers are frozen:

- Bede gate wrapper `d0c47d86145650ff7c77439ffad8b32a59954384eebea3fb8cb343ed01601dc0`
- Bede science wrapper `e903997ba08fbca1a5c6f08427358344c489030645b52f56a12ea4b2f14397fc`

Trainer/config/aggregator bytes and the RHS40 scientific identity are unchanged.
Task64 Coin remains on CSF3 and is not migrated or duplicated.

## Bede gate and science launch

The sole Bede gate `1078982` completed `COMPLETED/0:0` in `00:01:51` on
`gpu017`, with `PRECHECK_PASS/rc0`. Its one real update proves curvature `.1`,
objective `40`, critic RHS weight `126.49110640673517`, actor/critic rows
`512/512`, strict system rows `1024`, natural cross Frobenius `.00365258`,
Cholesky info `0`, relative residual `7.993e-16`, RHS reconstruction `0`, alpha
reconstruction `1.332e-15`, direction reconstruction `3.423e-8`, finite PASS,
and zero hard-error matches.

After a fresh root/duplicate check, all four seed0 exact-2,007,040 jobs were
submitted together exactly once without dependency, hold or throttle:

| Environment | Job | Initial state | Node | Exact root |
|---|---:|---|---|---|
| BigFish | `1078983` | RUNNING | gpu017 | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_RHS40_DIRECTION_TELEMETRY_ONLY_V1/bigfish-easy-0-10/seed0/2m` |
| BossFight | `1078984` | RUNNING | gpu017 | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_RHS40_DIRECTION_TELEMETRY_ONLY_V1/bossfight-easy-0-10/seed0/2m` |
| CaveFlyer | `1078985` | RUNNING | gpu022 | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_RHS40_DIRECTION_TELEMETRY_ONLY_V1/caveflyer-easy-0-10/seed0/2m` |
| CoinRun | `1078986` | RUNNING | gpu022 | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_RHS40_DIRECTION_TELEMETRY_ONLY_V1/coinrun-easy-0-10/seed0/2m` |

Each job owns one V100 and a distinct root. All four produced a complete first
transition-4096 telemetry record with exact frozen coefficients/1024 rows,
nonzero cross, Cholesky `0`, finite scan PASS and residuals in
`[6.256e-16,7.183e-16]`; hard-error scans are zero. The initial full actor norm
shares are `.0181/.00615/.0122/.00383`, showing the intended RHS40 critic-heavy
direction at the first update, but this is only an initial snapshot. Reward is
read-only and never early-stops. Task64 Coin was not touched.

## Final Task65 terminal result

All four jobs completed naturally with scheduler `COMPLETED/0:0`, root
`PASS/rc0`, exact transition `2,007,040`, 49 progress rows and 15,680 complete
telemetry records per cell. BigFish/BossFight/CaveFlyer/CoinRun elapsed times
were `02:44:19/02:46:49/02:45:53/02:45:53` on `gpu017/gpu017/gpu022/gpu022`.
Their endpoint rewards were `2.11/0.00/3.20/8.20`, versus immutable Paper
`9.28/2.92/4.45/3.70` and Task63 parent `5.08/.04/0/10.00`. Reward remained a
read-only sanity check.

Every frozen aggregate reports `TASK65_AGGREGATION_PASS`. Overall median raw
actor metric energy shares are `.96743/.96203/.98083/.97166`, but median
post-inverse full actor norm shares fall to
`.03791/.03309/.03775/.03369`, and actor signed projection shares fall to
`.001724/.001057/.001517/.002361`. Shared actor signed projection medians are
only `.003951/.004156/.003553/.003873`. Thus the exact RHS40 intervention
reliably makes the coupled solved direction critic-dominant despite an
actor-heavy raw metric; unlike Task64, this changes the intended RHS rather
than curvature. It does not provide a general reward rescue.

All endpoint coefficients remain curvature `.1`, objective `40`, RHS weight
`126.49110640673517`, strict1024 and full-cross. Cholesky info is zero, relative
residuals are `6.848e-16`--`3.976e-15`, RHS reconstruction is exact zero, alpha
reconstruction is at `1.214e-15`--`1.506e-15`, direction reconstruction is
`1.520e-8`--`9.465e-8`, finite scans pass, and scoped hard-error scans are
empty. Each remote checkpoint is a regular 3,766,013-byte mode0664 file; only
stat metadata is archived, never model bytes or content hashes.

Complete model-free evidence is under
`remote_launch_staging/procgen_deterministic_joint2b_critic_rhs40_direction_telemetry_2m_s0_20260828_65/evidence/terminal/`.
