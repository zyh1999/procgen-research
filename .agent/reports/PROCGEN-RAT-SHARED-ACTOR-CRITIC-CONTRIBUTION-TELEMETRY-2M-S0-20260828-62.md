# Task62 Paper RAT actor/critic contribution telemetry

## Status

`QUEUED_RESOURCE_WAIT`

## Frozen parent

- source commit: `2b5affd64cbb3c624b4bc1f4767f449df231ffb2`
- parent trainer SHA256: `cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7`
- parent config SHA256: `1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea`
- method: `RAT_SHARED_PAPER_ACTOR_CRITIC_TELEMETRY_ONLY_V1`

The original combined stochastic Paper H, both solves, loss/backward, global
clip, optimizer step and adaptive-LR controller remain authoritative. Added
computations are read-only policy-row, realized value-row and actor/value
gradient decompositions; they do not replace H or populate `.grad`.

## Frozen Task62 files

- trainer: `3d78ea8d985ed2ece4dee4b79be79590802d483800786fd8b1d2bdd754da9a55`
- 2M config: `1fc395c5434eb2d842c9f089e36778ed3b58cde3850d2e0c5ddcc8f7eac09b26`
- aggregator: `6bc146ea2fea372bb68b9b1f6b9d34b27eb3f51616264996e056a4c20a586c23`
- gpuH gate wrapper: `40b170a38e06f4f37028a72b08d7a5de05f56c91e25a905fe204417c20ae547e`
- gpuH science wrapper: `4fe47cd7c0ee274378ba3fb8240bb20c14e2541c740ba931bbeb314f599b46ad`

Local Python compile and shell syntax checks pass. Remote gate, placement,
job/root matrix and terminal Early/Middle/Late aggregates will be appended
after their bounded events.

## Placement and gate submission

Live CSF3 refresh found gpuH UP and compatible, 32 H200s across node820--823,
but the shared association GRES was saturated by unrelated account users.
The Task62 campaign/root and duplicate process/job checks passed. Bede also
had idle compatible V100 nodes, but the user preference was CSF3 and the
exactly-once gate had not yet been submitted.

The frozen bundle was deployed to
`/scratch/h99859yz/procgen_rat_shared_actor_critic_contribution_telemetry_2m_s0_20260828_62`.
All five remote frozen hashes match the implementation commit. The sole gate
job `19528173` was submitted once, requesting one H200, 8 CPUs and 64G. It is
zero-step `PENDING (AssocGrpGRES)`, elapsed `00:00:00`, node none; the gate
root remains absent until Slurm starts the job. This is resource queueing, not
preflight, infrastructure or scientific failure. No retry/requeue/resubmit or
science job exists.
