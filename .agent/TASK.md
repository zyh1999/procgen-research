# Task-ID: PROCGEN-DETERMINISTIC-JOINT2B-ACTOR-CRITIC-DIRECTION-TELEMETRY-2M-S0-20260828-63

Status: IMPLEMENTATION_FROZEN_PENDING_GATE

Method: `PAPER_MATCHED_DETERMINISTIC_GGN_ACTOR_CRITIC_DIRECTION_TELEMETRY_ONLY_V1`

Run one instrumentation-only exact-2,007,040 replay of frozen Task06 strict
deterministic full-shared Joint-2B for BigFish, BossFight, CaveFlyer and
CoinRun seed0. Parent implementation is
`da34ce7c7d964765f336ac02111c9fde95aed1ec`, trainer SHA256
`41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a`
and config SHA256
`69d12937debb8ef8b4531e79b8f9613185b26e4e9056a51e67754129b269391d`.

Preserve the original installed single-RHS direction and every scientific
control. Add only post-inverse actor/critic RHS solves using the already
factorized full coupled system, direction/role decomposition telemetry and
terminal Early/Middle/Late aggregation. One Bede production gate is permitted;
PASS allows four fresh one-V100 jobs submitted together exactly once. Never
reward-stop, retry/requeue/resubmit, touch Task62 or commit model bytes.
