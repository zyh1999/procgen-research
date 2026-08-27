# Task61: no-warmup Joint2B plus post-solve entropy gradient

## Status

`CANDIDATE_NOT_READY` — the sole production gate passed and both authorized
quick cells are running after the user-authorized atomic speed migration to
CSF3 gpuH.

## Frozen parent

- Task55 implementation: `3a850cd3870854123c76693a974a2fe45e952203`
- Trainer SHA256: `91b835f16989a42293f6566d8fb9893dcd7b9ca969d1685d2d313f3f695f2f81`
- beta1 config SHA256: `75fb59290d4bd2399986e372a62e56b4aaa6df7becb205f41ee332538f04425f`
- Matched controls: frozen Task55 beta1 BossFight/CaveFlyer exact-2M roots;
  they remain untouched.

## Sole scientific change

`postsolve_entropy_coef=0.01`. The current-policy entropy ascent gradient is
computed in the exact ordered full parameter space only after the unchanged
Joint system, RHS, history correction, dual solve and primal reconstruction.
The applied direction is `joint_dir + .01 * entropy_ascent`, followed by the
parent's single global Euclidean clip and fixed LR `.004`.

Entropy is absent from the system/RHS/solve/eta controller and from the Joint
history buffer. Policy/shared/value entropy norms are recorded separately;
value-head-exclusive entropy must equal zero exactly. Joint-only and applied
actor/critic projections, quadratics and predicted divergences are distinct
telemetry fields. Actual rollout `D_pi/D_v` continues to drive the unchanged
dual-trust controller.

## Minimal local gates

- parent trainer/config SHA match: PASS
- Python compile: PASS
- gate/science shell syntax: PASS
- config contains only the explicit `.01` post-solve coefficient: PASS
- method is beta1 only; no warmup, LR or trust-band change: PASS

## Frozen Task61 identities

- Implementation commit: `60948b47a5d8d36ce91305118662ead7e83cbefc`
- Trainer SHA256: `3f4946dbb7fa674d3996fc9dc27fc5cea080ebcca23989e1c1491562198d56dd`
- Config SHA256: `eba2d9f6a18d06839087fefa3e77ce047735adcf1c5a792ee528be0516006856`
- Gate wrapper SHA256: `1b3e14128714fd6fe3cd03e065b2afb780bacee873dd33c95485cc39d2341fc4`
- Science wrapper SHA256: `45f82f2b078e32367cb48240083b1ac474ca6dab6d2d6a60170c24eb49a7afc7`
- Read-only monitor SHA256: `ab00df29ae106506649c37051c559879bd917606cb8e1c82331fd57fda81af8d`

## Sole production gate

Job `1078146` is scheduler-authoritatively `COMPLETED/0:0`, elapsed
`00:02:01`, node gpu011; root is `PRECHECK_PASS/rc0`. The gate proves the real
production model/device path, strict `1024x938976` Joint system, nonzero
natural cross blocks, Cholesky info0 and relative residual
`6.587411024709103e-16`.

The final minibatch records entropy gradient norm `.0851079`, policy norm
`.0292955`, shared norm `.0799070`, value-exclusive norm exactly `0`, scaled
entropy norm `.000851079`, Joint/entropy cosine `-.0938228`, finite applied
actor/critic projections and all three separation flags (system, RHS,
history) equal to one. Fixed LR is `.004`; actual rollout `D_pi/D_v` is
`3.2259e-05/.0056531`. Precise hard-error scan is zero. The Gym deprecation
text is benign and contains no traceback, OOM, CUDA/NCCL, disk, NaN or Inf.

## Science launch

Live Bede refresh showed five idle compatible nodes, no Task61 job/process,
and absent fresh roots. Both cells were submitted in the same bounded action,
with no dependency, hold or throttle.

| Environment | Seed | Horizon | Job | Root | State |
|---|---:|---:|---|---|---|
| BossFight | 0 | 2,007,040 | 1078147 | `runs/FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_POSTSOLVE_ENTGRAD001_BETA1_V1/bossfight-easy-0-10/seed0/2m_quick` | RUNNING gpu011 |
| CaveFlyer | 0 | 2,007,040 | 1078148 | `runs/FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_POSTSOLVE_ENTGRAD001_BETA1_V1/caveflyer-easy-0-10/seed0/2m_quick` | RUNNING gpu011 |

Each allocation has one V100, its own Slurm job/root/PID/log chain and frozen
identity. Initial traces reach `16,384`: LR `.004`, phase switch zero, beta1,
natural cross nonzero, Cholesky info0, residuals near `5e-16`, entropy
value-head norm zero, all finite scans PASS and hard-error scans zero.

No retry, requeue, resubmit or reward-based cancellation is allowed. Model and
checkpoint bytes/hashes remain outside Git.

## User-authorized Bede to CSF3 speed migration

A read-only placement audit measured only about 135--143 transitions/second on
the Bede V100 cells, with roughly four hours remaining, while the frozen
Task61 identity was compatible with the established CSF3 Procgen environment.
The user explicitly authorized an atomic fresh-root migration.

The deployment-only CSF3 wrapper SHA256 is
`ea04718ee9c2ccc3c222a1e5571c3df14430f690a4b55bb6e4ed366557a8b3c2`.
Its normalized scientific command is identical to the frozen Bede command;
only scheduler/deployment paths and fresh roots differ. It is frozen in commit
`7d7ab8576a4f782daf28e1e460cf77e180f4624d`.

Jobs `1078147/1078148` were cancelled exactly once at low progress, after
healthy finite traces and before any endpoint. Both became scheduler-terminal
`CANCELLED by 639800874/0:0` at `2026-08-27T17:06:36+01:00`; their trainer
PIDs were gone. This is solely
`CANCELLED_FOR_USER_AUTHORIZED_GPUH_SPEED_MIGRATION`, not a retry and not an
algorithm, numerical or infrastructure failure.

Exactly two fresh normal gpuH jobs were submitted together once. They initially
waited zero-step on `AssocGrpGRES`, with fresh roots still absent. Immediate
reconfirmation proved the two old Procgen Jupyter parent allocations had only
batch/extern live and every scientific child step was terminal. Under the
explicit follow-up authorization, exactly allocations `19487251/19487252`
were released once at `17:12+01:00`, classified
`RELEASED_AFTER_TERMINAL_PROCGEN_QUICK_WORK_TO_UNBLOCK_TASK61_NORMAL_GPUH`.
No Task61 job was cancelled, modified, requeued or resubmitted, and no other
job/allocation was touched.

| Environment | Job | Exact CSF3 root | Initial authoritative state |
|---|---:|---|---|
| BossFight | 19507047 | `/scratch/h99859yz/procgen_task55_quick_nowarmup_postsolve_entropy_grad001_beta1_boss_cave_2m_s0_20260827_61/runs/FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_POSTSOLVE_ENTGRAD001_BETA1_V1/bossfight-easy-0-10/seed0/2m_quick` | RUNNING node822, root RUNNING, PID51634 |
| CaveFlyer | 19507048 | `/scratch/h99859yz/procgen_task55_quick_nowarmup_postsolve_entropy_grad001_beta1_boss_cave_2m_s0_20260827_61/runs/FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_POSTSOLVE_ENTGRAD001_BETA1_V1/caveflyer-easy-0-10/seed0/2m_quick` | RUNNING node822, root RUNNING, PID51643 |

Both started naturally at `17:12+01:00`. Initial traces reached at least
61,440/65,536 transitions. They verify LR `.004`, rollout-zero Joint2B,
phase-switch zero, PPO state zero, strict `1024x938976`, nonzero natural cross
blocks, Cholesky info0 and finite residuals. Post-solve entropy coefficient is
`.01`; value-exclusive entropy norm is exactly zero and all three
entropy-absent-from-system/RHS/history flags equal one. The H200 was 94 percent
utilized during verification. Precise hard-error scan was zero; only benign
torchvision deprecation warnings occurred.

The existing sole Task61 automation must be updated in place to bind jobs
`19507047/19507048` and these two exact CSF3 roots. No second automation is
authorized.
