# PROCGEN-DETERMINISTIC-JOINT2B-CRITIC-CURVATURE4-DIRECTION-TELEMETRY-2M-S0-20260828-64

## Status

`SCIENCE_RUNNING_PARTIAL_TERMINAL`

## Scientific identity

Method:
`PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_CURVATURE4_DIRECTION_TELEMETRY_ONLY_V1`.
The parent is Task63 instrumentation over Task06 strict deterministic
full-shared Joint-2B. Parent Task06 implementation is
`da34ce7c7d964765f336ac02111c9fde95aed1ec`; parent scientific trainer/config
SHA256 are `41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a`
and `69d12937debb8ef8b4531e79b8f9613185b26e4e9056a51e67754129b269391d`.
The immediate Task63 telemetry trainer/config SHA256 are
`067a3c9fc6f309aa13cba9d54ab6c29252ed318c4398df8de01256e5c439415a`
and `4e2d18ae97084cf3ffaabb601ee63472408b77bd63fbc82d76fe1c385701db77`.

The only scientific delta is
`joint_critic_curvature_coef: 0.1 -> 4.0`. The critic objective coefficient
remains exactly `1.0`; no beta/eta/dual-trust path exists. Actor semantics,
no-warmup rollout0, complete actor and critic rows, both natural cross blocks,
strict `1024`-row system, damping `.5`, adaptive KL/LR, history correction,
momentum, global clip, PopArt, GAE, seed/evaluation/reward and exact endpoint
remain unchanged. Direction decomposition remains telemetry-only.

## Frozen files

- trainer `1fd8c1d7f2dc7529930b976ea96cd8adbf1365bc3ec1569c864e1e5991b61947`
- gate config `60c31084bc92afe9f1077601de0923d1d968309609909e5a5c6197ea30cca9a7`
- science config `b4348d16e6bcc78a171d9eb1d115eeb5ea02932537889b8b3bd04e5868c14a69`
- aggregator `a9eea0b9d33f91d7f298f9e2ddf6340b72c9d0358eba541c091155b0e0763420`
- gate wrapper `0c29008b589648eb3bedb03b10cf6795d2adaa4a7e16b351ccfd1dfcfe685bbd`
- science wrapper `9dadd550ab59922dcb1f79fd0ee01e6df4c9325933ececf158f3593f7a5ce31f`
- read-only monitor `6abe8d8dad83ba7b720d1dbe32178c6461d2575480a64abe84799c95ebc04381`

Local compile, shell syntax and exact frozen-field checks pass. No model or
checkpoint content is part of the bundle or Git evidence.

## Placement precheck

CSF3 `gpuH`, account `gpu-h200-fse-pgdr`, QOS `gpu-h200-fse` are live. The
association and QOS limit are four H200 GPUs. Three unrelated owned H200 jobs
are currently running, so the immediate safe capacity is one H200; the four
Task64 science cells will still be submitted together without dependencies or
throttling and may queue naturally. The Task64 campaign, roots and duplicate
jobs are absent. Task63 Bede jobs `1078181`--`1078184` are out of scope and
remain untouched.

## Gate

Implementation/origin freeze commit is
`122b5ab02203524dcd98330666ec74c015391808`. The sole production gate job
`19531850` naturally ran on CSF3 `node823` and completed `COMPLETED/0:0` in
`00:00:18`; root status is `PRECHECK_PASS`, rc `0`, with one complete real
update. It proves curvature `4.0`, objective coefficient `1.0`, actor/critic
rows `512/512`, strict system rows `1024`, exact deterministic kernel, nonzero
natural cross Frobenius `.0231009`, Cholesky info `0`, relative residual
`6.827e-16`, RHS reconstruction `0`, alpha reconstruction `1.635e-15`,
direction reconstruction `4.286e-8`, structural zeros, finite telemetry and
first-update installed identity max-absolute difference `0`. Hard-error scan is
zero. No gate model/checkpoint was written or archived.

## Science launch

All four seed0 exact-2,007,040 cells were submitted together exactly once,
without dependencies, holds or throttling:

| Environment | Job | Initial state | Node | Exact root |
|---|---:|---|---|---|
| BigFish | `19531929` | RUNNING | node823 | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_CURVATURE4_DIRECTION_TELEMETRY_ONLY_V1/bigfish-easy-0-10/seed0/2m` |
| BossFight | `19531930` | PENDING `AssocGrpGRES` | none | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_CURVATURE4_DIRECTION_TELEMETRY_ONLY_V1/bossfight-easy-0-10/seed0/2m` |
| CaveFlyer | `19531931` | PENDING `AssocGrpGRES` | none | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_CURVATURE4_DIRECTION_TELEMETRY_ONLY_V1/caveflyer-easy-0-10/seed0/2m` |
| CoinRun | `19531932` | PENDING `AssocGrpGRES` | none | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_CURVATURE4_DIRECTION_TELEMETRY_ONLY_V1/coinrun-easy-0-10/seed0/2m` |

BigFish root is RUNNING with trainer PID `2311560` on one H200 and has healthy
telemetry through transition `16,384`: curvature `4.0`, objective `1.0`, rows
`1024`, Cholesky info `0`, relative residual `6.584e-15`, direction
reconstruction `4.364e-8`, full actor norm/projection share `.4968/.4932`,
shared actor norm/projection share `.5181/.5367`, and hard-error scan zero. The
other three roots remain absent while queued. No retry/requeue/resubmit or
unrelated mutation occurred.

Current conclusion is science running with three naturally queued cells. Reward
comparison is read-only only; no early stop is authorized.

## BigFish terminal

BigFish19531929 naturally completed `COMPLETED/0:0` after00:45:43 on node823.
The PASS/rc0 root has exact2,007,040, reward1.97, 15,680 records and a clean
frozen aggregation. Overall actor metric norm/energy is `.24180/.09232`, while
post-inverse full actor norm/projection is `.73590/.92205` and shared is
`.73390/.91931`. This is the opposite of the intended critic-contribution
increase: curvature4 makes the metric critic-heavy but the solved direction
more actor-dominant than Task63 BigFish (`.44048/.37842`), while reward falls
from5.08 to1.97. Solver/reconstruction/hard-error evidence is healthy.

Checkpoint bytes remain remote; only regular-file size/mode metadata is
recorded. Boss/Cave remain RUNNING, and Coin has naturally started RUNNING;
all remain untouched. Model-free details are in
`evidence/partial_terminal_bigfish_20260828.md`.

## BossFight and CaveFlyer terminal

BossFight `19531930` and CaveFlyer `19531931` naturally completed
`COMPLETED/0:0` on `node820` and `node821`, with PASS/rc0 roots at exact
`2,007,040`, 15,680 complete trace records each, checkpoint stat metadata only,
and zero hard-error matches. Rewards are `0.01` and `1.49`.

Both frozen complete-trace aggregations pass. BossFight's actor metric
norm/energy shares are `.26690/.11703`, but its post-inverse full actor
norm/projection shares are `.72938/.94268` and shared shares are
`.73163/.94311`. CaveFlyer's corresponding values are `.22846/.08061`,
`.73329/.92628`, and `.73123/.92103`. Thus curvature `4` makes the raw metric
critic-heavy while the coupled inverse remains strongly actor-dominant in all
three completed Task64 environments. Relative to curvature `.1`, Boss reward
falls `.04 -> .01`; Cave improves `0 -> 1.49` but remains below Paper `4.45`.

CoinRun `19531932` naturally started and remains RUNNING on `node821`; at the
09:57Z refresh it was at transition `696,320`, with finite strict-1024 telemetry,
and remains untouched. Status is `SCIENCE_RUNNING_PARTIAL_TERMINAL`. Full
model-free evidence is in `evidence/partial_terminal_boss_cave_20260828.md`.
