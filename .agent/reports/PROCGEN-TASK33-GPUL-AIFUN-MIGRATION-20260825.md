# Task33 gpuL / gpu-aifun migration

Task: `PROCGEN-GAE-GGN-HEAD-WIDENTITY-6M-S0-20260825-33`

This report covers deployment placement only.  The scientific method,
trainer, config, four environments, seeds 0/1, intended 6M horizons, artifact
roots and comparison protocol are unchanged.

## Compatibility

The deployment-only freeze is commit
`0057469b50cdfa7f6fd504ec146b3f56daf06ecc`.  Compatibility job `19319577`
completed on gpuL node887 using account/QOS `gpu-aifun` and one L40S.  The
original H200 preflight source remained byte-identical; its versioned wrapper
changed only the GPU-name and minimum-memory acceptance clauses.

All scientific gates passed, including exact W=I, exact absence of policy
probability weighting, actor/shared one-step identity, GAE-GGN direct
reference, PopArt affine identity, Cholesky info 0 and relative residual
`4.650e-16`.  Peak allocation was 15.52 GB on a 47.67 GB L40S.

## Atomic mapping

| Environment | Seed | Old gpuH | New gpuL |
|---|---:|---:|---:|
| BigFish | 0 | 19314824 | 19319678 |
| BigFish | 1 | 19314825 | 19319679 |
| BossFight | 0 | 19314826 | 19319680 |
| BossFight | 1 | 19314827 | 19319681 |
| CaveFlyer | 0 | 19314828 | 19319682 |
| CaveFlyer | 1 | 19314829 | 19319683 |
| CoinRun | 0 | 19314830 | 19319684 |
| CoinRun | 1 | 19314831 | 19319685 |

Every replacement was first submitted held and verified for account, QOS,
partition, L40S GRES, environment, seed, launcher and still-absent root.  The
old jobs were then cancelled and verified as unstarted (`Elapsed=00:00:00`,
`Start=None`, `Node=None assigned`).  Only afterward were replacements
released.  This prevents duplicate live cells and preserves the old
cancelled-pending ledger separately.

The new jobs are pending under `gpu-aifun`.  The requested placement is up to
four concurrent L40S cards, one cell per card; remaining cells follow as quota
and scheduling allow.  The user has no gpuA job, and gpuA is not treated as a
gpuL allocation.

Model-free evidence is under
`remote_launch_staging/procgen_gae_ggn_head_widentity_6m_s0_20260825_33/evidence/gpul_migration/`.
