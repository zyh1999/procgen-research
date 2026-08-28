# Task62 Paper RAT actor/critic contribution telemetry

## Status

`SCIENCE_RUNNING`

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

## User-authorized atomic Bede migration

CSF3 gate `19528173` was confirmed zero-step before cancellation: PENDING on
`AssocGrpGRES`, elapsed zero, start unknown, no node, root, process, progress,
trace or artifact. It was cancelled exactly once and is terminal
`CANCELLED/0:0`, classified
`CANCELLED_FOR_USER_AUTHORIZED_ZERO_STEP_BEDE_MIGRATION`. It was not restored,
retried, requeued or resubmitted.

The deployment-only Bede wrappers preserve the frozen trainer, config and
aggregator bytes. Their SHA256 identities are:

- gate wrapper: `91c5f3c3c0871f6ef5dc47a39378c12af5e09628d4b18e1e65d839e3379209bd`
- science wrapper: `e1c450a12677a30ae60ce375044aa61b0282ed74f2235faddbf6251a353baa71`

The Bede campaign is
`/nobackup/projects/bdman37/yihe/procgen_rat_shared_actor_critic_contribution_telemetry_2m_s0_20260828_62`.
The sole Bede production gate `1078175` completed `COMPLETED/0:0` in 91 seconds
on `gpu002`, with root `gate/production`, `PRECHECK_PASS`, rc0 and one complete
production update. It retained the original combined stochastic Paper H and
unchanged update while proving finite telemetry, exact structural zeros and
small read-only reconstruction errors: H relative residual
`4.1936037975531804e-11`, gradient relative residual
`4.213032056554766e-08`, policy-exclusive critic-gradient norm `0`, and
value-exclusive actor-gradient norm `0`.

## Exactly-once Bede science launch

After the gate passed and a fresh duplicate/root/capacity check, all four jobs
were submitted together once, without dependencies, holds or throttling. Each
requests and receives one V100:

| Environment | Job | Node | Root |
|---|---:|---|---|
| BigFish | 1078176 | gpu002 | `runs/RAT_SHARED_PAPER_ACTOR_CRITIC_TELEMETRY_ONLY_V1/bigfish-easy-0-10/seed0/2m` |
| BossFight | 1078177 | gpu002 | `runs/RAT_SHARED_PAPER_ACTOR_CRITIC_TELEMETRY_ONLY_V1/bossfight-easy-0-10/seed0/2m` |
| CaveFlyer | 1078178 | gpu003 | `runs/RAT_SHARED_PAPER_ACTOR_CRITIC_TELEMETRY_ONLY_V1/caveflyer-easy-0-10/seed0/2m` |
| CoinRun | 1078179 | gpu005 | `runs/RAT_SHARED_PAPER_ACTOR_CRITIC_TELEMETRY_ONLY_V1/coinrun-easy-0-10/seed0/2m` |

At the archived launch snapshot all four jobs and roots were RUNNING with live
trainer PIDs and distinct V100 UUIDs. Complete contribution records already
existed: 224 BigFish, 203 BossFight, 124 CaveFlyer and 190 CoinRun records.
Every latest record had `finite_scan_pass=1`, `batch_rows=512`,
`parameter_columns=938976`, zero policy-exclusive critic gradient and zero
value-exclusive actor gradient. H reconstruction relative residuals were
`1.93e-8`, `1.81e-8`, `1.82e-8`, `3.52e-9`; gradient reconstruction relative
residuals were `3.61e-7`, `4.48e-7`, `3.00e-7`, `3.02e-8`. Precise scans for
Traceback, runtime, OOM, CUDA, NCCL, disk/quota and nonfinite failures found
zero matches in every cell.

This is an instrumentation campaign, so reward is read-only and no cell may
be reward-cancelled. Current operational conclusion is `SCIENCE_RUNNING`.
Terminal Early/Middle/Late aggregation remains pending. No checkpoint/model
bytes or hashes are included in Git, and no Task51--61 state was touched.
