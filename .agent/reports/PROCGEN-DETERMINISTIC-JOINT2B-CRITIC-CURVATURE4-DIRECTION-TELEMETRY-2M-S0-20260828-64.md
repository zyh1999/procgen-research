# PROCGEN-DETERMINISTIC-JOINT2B-CRITIC-CURVATURE4-DIRECTION-TELEMETRY-2M-S0-20260828-64

## Status

`IMPLEMENTATION_FROZEN_PRECHECK_PENDING`

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

## Gate and science

Pending the sole production gate.
