# Task-ID: PROCGEN-RAT-SHARED-ACTOR-CRITIC-CONTRIBUTION-TELEMETRY-2M-S0-20260828-62

Status: QUEUED_RESOURCE_WAIT

Method: `RAT_SHARED_PAPER_ACTOR_CRITIC_TELEMETRY_ONLY_V1`

Execute exactly one instrumentation-only replay of frozen original shared
Procgen Paper RAT for BigFish, BossFight, CaveFlyer and CoinRun, seed0, exact
`2,007,040`. Parent source commit is
`2b5affd64cbb3c624b4bc1f4767f449df231ffb2`, trainer SHA256
`cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7`
and config SHA256
`1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea`.

Preserve the stochastic Gaussian critic score, combined B-row H, both parent
solves, damping `.5`, history correction, ent_coef0, vf_coef1, PopArt, ratio
clamp, advantage normalization, global clip `.5`, SGD momentum `1e-6`, and
per-minibatch policy-KL LR controller byte/semantically. The only runtime
changes are the exact 2M horizon and side-effect-free telemetry: deterministic
policy-only rows after original H, realized value rows by subtraction,
side-effect-free actor/value autograd gradients before the unchanged total
backward, complete role decomposition and JSONL trace. Telemetry must neither
consume RNG nor populate `.grad` before the original backward.

Run one minimal production gate exactly once. PASS permits four fresh CSF3
gpuH jobs submitted together once; failure is terminal without retry. Never
reward-stop this diagnostic, touch Task51--61, create another method/seed, or
commit checkpoint/model bytes.

The sole gate is job `19528173`, submitted once to CSF3 gpuH. It is zero-step
`PENDING (AssocGrpGRES)` with no node or gate root yet. Do not cancel, migrate,
duplicate, retry, requeue or resubmit it. Science remains unlaunched until this
exact gate completes `PRECHECK_PASS`.
