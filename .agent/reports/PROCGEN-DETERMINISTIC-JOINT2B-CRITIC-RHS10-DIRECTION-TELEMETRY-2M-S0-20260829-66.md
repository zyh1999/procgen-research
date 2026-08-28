# PROCGEN-DETERMINISTIC-JOINT2B-CRITIC-RHS10-DIRECTION-TELEMETRY-2M-S0-20260829-66

Status: `SCIENCE_RUNNING_WITH_QUEUED_CELLS`

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

Implementation commit `f07a7f1f2027ad0dda4e1675996af1f1db450fe7` was
pushed and matched `origin/agent-work`. The fresh Bede campaign is:

`/nobackup/projects/bdman37/yihe/procgen_deterministic_joint2b_critic_rhs10_direction_telemetry_2m_s0_20260829_66`

The sole gate `1084426` completed `0:0` in `00:00:52` on `gpu018` with
`PRECHECK_PASS/rc0`. Its real update reports curvature `.1`, objective `10`,
RHS weight `31.622776601683793`, actor/critic rows `512/512`, strict system
rows `1024`, cross Frobenius `.003652576`, Cholesky info `0`, relative residual
`7.411e-16`, RHS reconstruction `0`, alpha reconstruction `1.280e-15`,
direction reconstruction `3.273e-8`, installed identity maxabs `0`, finite
PASS and no refined hard-error match.

After a fresh duplicate/root/capacity check, all four cells were submitted
together exactly once without dependency, hold or throttle:

| Environment | Job | Initial state | Node | Exact root suffix |
|---|---:|---|---|---|
| BigFish | `1084427` | RUNNING | `gpu018` | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_RHS10_DIRECTION_TELEMETRY_ONLY_V1/bigfish-easy-0-10/seed0/2m` |
| BossFight | `1084428` | RUNNING | `gpu018` | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_RHS10_DIRECTION_TELEMETRY_ONLY_V1/bossfight-easy-0-10/seed0/2m` |
| CaveFlyer | `1084429` | PENDING Resources | none | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_RHS10_DIRECTION_TELEMETRY_ONLY_V1/caveflyer-easy-0-10/seed0/2m` |
| CoinRun | `1084430` | PENDING Priority | none | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_RHS10_DIRECTION_TELEMETRY_ONLY_V1/coinrun-easy-0-10/seed0/2m` |

At the launch archive snapshot, BigFish and BossFight have exact first-update
transition-4096 traces. Their relative residuals are `7.411e-16/7.339e-16`,
Cholesky info is `0`, finite scan is PASS, and full actor norm/signed-projection
shares are `.013345/.000189` and `.284375/.141625`. Cave/Coin remain zero-step
queued with absent roots and may start naturally. Reward is read-only and never
early-stops. The sole 20-minute automation is
`monitor-procgen-task66-rhs10-direction-telemetry`.
