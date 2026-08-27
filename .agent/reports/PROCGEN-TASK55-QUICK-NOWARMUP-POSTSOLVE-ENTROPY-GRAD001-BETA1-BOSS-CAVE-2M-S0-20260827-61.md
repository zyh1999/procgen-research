# Task61: no-warmup Joint2B plus post-solve entropy gradient

## Status

`CANDIDATE_NOT_READY` — the sole production gate passed and both authorized
quick cells are running.

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
