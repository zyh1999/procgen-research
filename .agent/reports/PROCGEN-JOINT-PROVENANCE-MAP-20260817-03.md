# Procgen Joint Provenance Map

Task: `PROCGEN-JOINT-PROVENANCE-MAP-20260817-03`  
Evidence window: `2026-08-18T13:11:16Z` to `2026-08-18T13:22:09Z`  
Control host: `login1.csf3.man.alces.network`  
Bede host: `login1.bede.dur.ac.uk`

## Executive conclusion

`STRICT_PARENT_COMPLETE`

The completed CSF3 low-Fisher-guard successor/control `18672560` is a strict
single-causal-ablation match to the completed seed0 1M RHS-aligned Joint-B gate
`18670696`. The only scientific change is the predeclared low-Fisher
actor-from-critic damping guard. All required environment, seed, architecture,
data geometry, budget, stopping, algorithm, precision, and artifact protocol
fields otherwise match. Every target/control cell is PASS/rc0.

No 100k/250k/500k PAP/FADP/RAT/Joint-2B/Schur run is a strict parent for the
1M Joint-B gate because its budget and/or method, source, RHS, solver,
reduction, cross-block, damping, or hardware identity differs. The 250k and
500k RHS-aligned CSF3 gates share the target method but fail the equal-budget
requirement. The conclusion identifies a causal control; it is not a promotion
decision or a 6M performance claim.

## Queries, freshness and non-action attestation

Read-only evidence sources were `squeue`, `sacct -X`, `scontrol` where retained,
`nvidia-smi`, `pgrep`, launcher/config/source SHA256, preflight/command/status/
rc files, terminal JSONL rows, and bounded log/checkpoint/error scans. CSF3 was
refreshed at `2026-08-18T13:22:09Z`; Bede at `2026-08-18T13:19:36Z`.

No Procgen scheduler row or trainer was live on CSF3. Login A2 was 0% and
116/15356 MiB; no capacity was inferred. No new Procgen artifact was created by
this task. No experiment was started, resumed, cancelled, released, requeued,
or early-stopped. Jupyter was not used. `.54`, `ws4090-31`, and `10.49.7.54`
were not accessed. An unrelated Bede job exposed through raw-ID collision was
not investigated.

Environment/cell order throughout is `0=BigFish`, `1=BossFight`,
`2=CaveFlyer`, `3=CoinRun`; all requested scientific cells use seed 0.

## CSF3 cancellation ledger

| Job | Exact identity and command | Cell evidence | Artifact/replacement | Classification |
|---|---|---|---|---|
| `18642230` | `pg-j2b-acguardA`; launcher `jupyter_joint2b_actorcriticguard01_3m_gpul.sbatch` `5630d130...`; trainer `train_shared_joint2b_actor_critic_guard.py` `58e73af1...`; config `adv_resnet_shared_joint2b_actorcriticguard01_lr02.yaml`; full Joint-2B clean/all, 3M, rollout 4096, minibatch 512, 4 epochs; root `gate_3m_seed0_jupyter_actorcriticguard01_gpua_v1/acguard01` | raw `18642271/18642272/18642445/18642230`; all `CANCELLED by 778916`, Start=None, no node, `00:00:00`, exit `0:0`; cancellation CSF3 local 2026-08-18 14:08 | Exact root absent. Older separate root `gate_3m_seed0_jupyter_actorcriticguard01_v1` has stale RUNNING markers and traces at 2.073M-2.810M; it is not an artifact of this cancelled array. | `cancelled-obsolete-unstarted` |
| `18624888` | `pg-j2b-block05`; dependency `afterany:18624264`; launcher `jupyter_joint2b_block_damping_01_lr02_gpua.sbatch` `0f90689a...`; trainer `train_shared_joint2b_block_damping.py` `a36d0e22...`; config `adv_resnet_shared_joint2b_block_damping_005_lr03.yaml` `66fac681...`; block005_lr03, floor .05, LR max .03, 1M; root `gate_1m_seed0_jupyter_block_damping_v1/block005_lr03` | compressed raw `18624888_[0-3%4]`; all cancelled, Start=None, no node, `00:00:00`, exit `0:0`; cancellation local 2026-08-18 14:08 | Exact root absent; obsolete held control, no scientific artifact | `cancelled-obsolete-unstarted` |
| `18666591` | `pg-j2b-papklg`; gpuH submission of the PAP KL-guard launcher later used by `18666610` | compressed `_[0-3%4]`; all cancelled local 2026-08-16 00:12, Start=None, no node, `00:00:00`, exit `0:0` | immediately replaced by gpuA `18666610`, whose four cells completed | `cancelled-obsolete-unstarted` |

## CSF3 scheduler-cell matrix

All rows have exit `0:0`. Each tuple is
`raw-id/node/runtime/scheduler-state`; workdir is
`/net/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1`.

| Job | 0 BigFish | 1 BossFight | 2 CaveFlyer | 3 CoinRun |
|---|---|---|---|---|
| `18666610` PAP KL guard | `18666611/node858/5:38/COMPLETED` | `18666614/node849/6:34/COMPLETED` | `18666619/node847/5:37/COMPLETED` | `18666610/node858/5:31/COMPLETED` |
| `18667225` PAP KL backtrack | `18667226/node856/5:47/COMPLETED` | `18667227/node856/5:51/COMPLETED` | `18667228/node858/5:54/COMPLETED` | `18667225/node856/5:51/COMPLETED` |
| `18667467` PAP rollout-LR | `18667468/node856/6:41/COMPLETED` | `18667469/node856/7:23/COMPLETED` | `18667470/node858/7:24/COMPLETED` | `18667467/node856/7:40/COMPLETED` |
| `18667627` PAP progressive | `18667628/node854/7:45/COMPLETED` | `18667629/node856/6:45/COMPLETED` | `18667663/node856/6:39/COMPLETED` | `18667627/node855/7:32/COMPLETED` |
| `18667792` PAP full columns | `18667793/node850/5:52/COMPLETED` | `18667794/node854/6:08/COMPLETED` | `18667795/node856/5:57/COMPLETED` | `18667792/node849/6:17/COMPLETED` |
| `18667941` Fisher-adaptive guard | `18667942/node849/5:12/COMPLETED` | `18667964/node856/5:28/COMPLETED` | `18667976/node849/5:14/COMPLETED` | `18667941/node856/5:22/COMPLETED` |
| `18668461` direct Joint-2B | `18668496/node849/5:14/COMPLETED` | `18668497/node850/5:23/COMPLETED` | `18668589/node855/5:33/COMPLETED` | `18668461/node855/5:25/COMPLETED` |
| `18669377` RAT block trace | `18669382/node858/5:54/COMPLETED` | `18669401/node858/5:34/COMPLETED` | `18669445/node858/5:31/COMPLETED` | `18669377/node847/5:30/COMPLETED` |
| `18669429` direct relative | `18669465/node847/5:15/COMPLETED` | `18669476/node849/7:33/COMPLETED` | `18669490/node847/5:16/COMPLETED` | `18669429/node849/5:13/COMPLETED` |
| `18669454` expected relative | `18669535/node849/4:44/COMPLETED` | `18669556/node849/4:52/COMPLETED` | `18669601/node849/4:49/COMPLETED` | `18669454/node849/4:43/COMPLETED` |
| `18669530` exact Schur | `18669632/node849/5:18/COMPLETED` | `18669657/node849/5:23/COMPLETED` | `18669676/node861/10:56/COMPLETED` | `18669530/node849/5:20/COMPLETED` |
| `18669613` actor-relative direct | `18669727/node861/5:20/COMPLETED` | `18669741/node849/5:24/COMPLETED` | `18669754/node861/5:22/COMPLETED` | `18669613/node847/5:27/COMPLETED` |
| `18669615` actor-relative expected | `18669787/node847/4:51/COMPLETED` | `18669817/node847/4:59/COMPLETED` | `18669819/node855/5:03/COMPLETED` | `18669615/node847/4:53/COMPLETED` |
| `18669725` RHS 250k | `18669858/node847/14:54/COMPLETED` | `18669922/node847/15:16/COMPLETED` | `18669962/node861/15:41/COMPLETED` | `18669725/node847/14:53/COMPLETED` |
| `18670437` RHS 500k | `18670438/node847/28:54/COMPLETED` | `18670442/node858/29:48/COMPLETED` | `18670536/node847/28:52/COMPLETED` | `18670437/node858/29:09/COMPLETED` |
| `18670696` RHS 1M | `18670697/node847/57:45/COMPLETED` | `18670698/node847/57:34/COMPLETED` | `18671119/node847/57:47/COMPLETED` | `18670696/node847/57:41/COMPLETED` |
| `18672560` RHS 1M low-Fisher guard | `18672708/node847/57:25/COMPLETED` | `18672928/node847/58:34/COMPLETED` | `18673266/node847/57:43/COMPLETED` | `18672560/node847/57:38/COMPLETED` |

## CSF3 exact launch and source identities

All launch commands are the literal launcher paths below, optionally with the
recorded `afterany` dependency or `PROCGEN_METHOD`; roots append
`/<environment>/seed0`. Every run uses rollout 4096, minibatch 512, 4 epochs,
momentum 0, Kaczmarz false and float64 linear solve.

| Job | Launcher SHA / artifact root | Trainer SHA / config SHA | Exact method |
|---|---|---|---|
| `18666610` | PAP KL-guard launcher `a2ceb7ea...`; `...pi_klguard_v1/paperselective_pi_klguard_u002_b05` | `b5cbdd20...` / `06890b9d...` | 100k full Joint-2B, clean/all, PAP selective, paper-weight RHS, PI KL guard |
| `18667225` | PAP KL-backtrack `072b3504...`; `...klbacktrack_v1/paperselective_pi_klbacktrack_u005_b05` | `0cb26b1e...` / `3ea28112...` | PAP KL backtracking |
| `18667467` | PAP rollout-LR `9dcb913e...`; `...rolloutlr_v1/paperselective_pi_klbacktrack_u005_rolloutlr` | `c2e63bf2...` / `4b5a1353...` | PAP KL backtrack, rollout LR |
| `18667627` | PAP progressive `1c45cb83...`; `...progressive_v1/paperselective_pi_klbacktrack_u005_progressive` | `4c4604f3...` / `37a79150...` | PAP progressive reconstruction |
| `18667792` | PAP full columns `3f168f9e...`; `...paperfullcolumns...v1/paperfullcolumns_pi_klbacktrack_u005_progressive` | `e2a2d92a...` / `02541127...` | full-column PAP progressive |
| `18667941` | Fisher-adaptive guard `5db27254...`; `...fisheradaptive_v1/fisheradaptive_f085_f050_b001_b010_g000_g001_lr02` | `e32556e3...` / `dd040e25...` | full Joint-2B, paired RHS, adaptive block damping .85/.50 |
| `18668461` | direct dual damping `53c63165...`; `...dualdamping_v1/direct_2b` | `5b5c3078...` / `287d44e2...` | full direct 2B, paired RHS/full cross |
| `18669377` | RAT block trace `ddd08efb...`; `...rat_blocktrace_v1/deterministic_task_blocktrace_b_dualdamping` | `1881bf7c...` / `a77a94e2...` | deterministic RAT block-trace B, expected cross zero |
| `18669429` | direct relative `0f62048e...`; `...relative_dualdamping_v1/direct_2b_relative_block` | `5b5c3078...` / `7cf514cd...` | direct 2B relative damping |
| `18669454` | expected relative `0c0e1308...`; `...expected_relative_dualdamping_v1/expected_gaussian_score_b_relative` | `c976c0e5...` / `003fb112...` | analytic expected-Gaussian RAT B, no cross |
| `18669530` | exact Schur `fd9226a0...`; `...exact_schur_v1/schur_critic_b` | `5b5c3078...` / `e3905718...` | full Joint-2B, critic-B Schur |
| `18669613` | shared launcher `0424912c...`, method direct; `...actorrelative_criticfloor05_v1/direct_2b` | `5b5c3078...` / `078b528e...` | direct Joint-2B, actor-relative/critic floor .05 |
| `18669615` | shared launcher `0424912c...`, method expected; `...actorrelative_criticfloor05_v1/expected_b` | `c976c0e5...` / `e5fa92dc...` | expected-Gaussian RAT B |
| `18669725` | RHS 250k `01c09560...`; `gate_250k.../rhs_aligned_rank1_b` | `ff987e0d...` / `801203fc...` | deterministic RHS-aligned rank-1 B Joint-B, full compressed cross |
| `18670437` | RHS 500k `b86f09a2...`; `gate_500k.../rhs_aligned_rank1_b` | `ff987e0d...` / `aac919d8...` | same exact Joint-B method, 500k |
| `18670696` | RHS 1M `78f253f2...`; `gate_1m.../rhs_aligned_rank1_b` | `ff987e0d...` / `d87a8f64...` | same exact Joint-B method, 1M target |
| `18672560` | RHS guard 1M `903a481e...`; `gate_1m...lowfisherguard05_v1/rhs_aligned_rank1_b` | `18eea9d7...` / `7c2ad5ef...` | exact target plus single low-Fisher damping guard |

### Literal CSF3 submit/command registry

The following are the complete accounting submit lines; `TRAINER`, `CONFIG`,
and `RUN_ROOT` are the literal launcher assignments (the two-method launcher
selects its trainer/config/root from `PROCGEN_METHOD`). `$ROOT` is the workdir
printed above. Per-cell `command.txt` adds the environment and seed0 to these
identities.

| Job | Literal submit line | Literal trainer; config; run root |
|---|---|---|
| `18666610` | `sbatch --parsable jupyter_joint2b_correlation_dualanchor_paperselective_entropy_pi_klguard_smoke100k_gpua.sbatch` | `train_shared_joint2b_rownorm_correlation_dualanchor_entropy_pi_klguard.py`; `adv_resnet_shared_joint2b_correlation_dualanchor_paperselective_entropy_pi_klguard_p01_i002_d03_lr05_100k.yaml`; `smoke_100k_seed0_jupyter_correlation_dualanchor_paperselective_entropy_pi_klguard_v1/paperselective_pi_klguard_u002_b05` |
| `18667225` | `sbatch --parsable jupyter_joint2b_correlation_dualanchor_paperselective_entropy_pi_klbacktrack_smoke100k_gpua.sbatch` | `train_shared_joint2b_rownorm_correlation_dualanchor_entropy_pi_klbacktrack.py`; `adv_resnet_shared_joint2b_correlation_dualanchor_paperselective_entropy_pi_klbacktrack_p01_i002_d03_lr05_100k.yaml`; `smoke_100k_seed0_jupyter_correlation_dualanchor_paperselective_entropy_pi_klbacktrack_v1/paperselective_pi_klbacktrack_u005_b05` |
| `18667467` | `sbatch jupyter_joint2b_correlation_dualanchor_paperselective_entropy_pi_klbacktrack_rolloutlr_smoke100k_gpua.sbatch` | `train_shared_joint2b_rownorm_correlation_dualanchor_entropy_pi_klbacktrack_rolloutlr.py`; `adv_resnet_shared_joint2b_correlation_dualanchor_paperselective_entropy_pi_klbacktrack_rolloutlr_p01_i002_d03_lr05_100k.yaml`; `smoke_100k_seed0_jupyter_correlation_dualanchor_paperselective_entropy_pi_klbacktrack_rolloutlr_v1/paperselective_pi_klbacktrack_u005_rolloutlr` |
| `18667627` | `sbatch jupyter_joint2b_correlation_dualanchor_paperselective_entropy_pi_klbacktrack_progressive_smoke100k_gpua.sbatch` | `train_shared_joint2b_rownorm_correlation_dualanchor_entropy_pi_klbacktrack_progressive.py`; `adv_resnet_shared_joint2b_correlation_dualanchor_paperselective_entropy_pi_klbacktrack_progressive_p01_i002_d03_lr05_100k.yaml`; `smoke_100k_seed0_jupyter_correlation_dualanchor_paperselective_entropy_pi_klbacktrack_progressive_v1/paperselective_pi_klbacktrack_u005_progressive` |
| `18667792` | `sbatch jupyter_joint2b_correlation_dualanchor_paperfullcolumns_entropy_pi_klbacktrack_progressive_smoke100k_gpua.sbatch` | `train_shared_joint2b_rownorm_correlation_dualanchor_entropy_pi_klbacktrack_progressive_fullcolumns.py`; `adv_resnet_shared_joint2b_correlation_dualanchor_paperfullcolumns_entropy_pi_klbacktrack_progressive_p01_i002_d03_lr05_100k.yaml`; `smoke_100k_seed0_jupyter_correlation_dualanchor_paperfullcolumns_entropy_pi_klbacktrack_progressive_v1/paperfullcolumns_pi_klbacktrack_u005_progressive` |
| `18667941` | `sbatch jupyter_joint2b_actorcriticguard_fisheradaptive_smoke100k_gpua.sbatch` | `train_shared_joint2b_actorcriticguard_fisheradaptive.py`; `adv_resnet_shared_joint2b_actorcriticguard_fisheradaptive_lr02_100k.yaml`; `smoke_100k_seed0_jupyter_actorcriticguard_fisheradaptive_v1/fisheradaptive_f085_f050_b001_b010_g000_g001_lr02` |
| `18668461` | `sbatch jupyter_joint2b_dualdamping_direct_smoke100k_gpua.sbatch` | `train_shared_joint2b_dualdamping_schur.py`; `adv_resnet_shared_joint2b_dualdamping_fisheradaptive_direct_100k.yaml`; `smoke_100k_seed0_jupyter_dualdamping_v1/direct_2b` |
| `18669377` | `sbatch jupyter_rat_blocktrace_dualdamping_smoke100k_gpua.sbatch` | `train_shared_rat_blocktrace_dualdamping.py`; `adv_resnet_shared_rat_blocktrace_dualdamping_100k.yaml`; `smoke_100k_seed0_jupyter_rat_blocktrace_v1/deterministic_task_blocktrace_b_dualdamping` |
| `18669429` | `sbatch --dependency=afterany:18669377 jupyter_joint2b_relative_dualdamping_smoke100k_gpua.sbatch` | `train_shared_joint2b_dualdamping_schur.py`; `adv_resnet_shared_joint2b_relative_dualdamping_direct_100k.yaml`; `smoke_100k_seed0_jupyter_relative_dualdamping_v1/direct_2b_relative_block` |
| `18669454` | `sbatch --dependency=afterany:18669429 jupyter_rat_expected_relative_dualdamping_smoke100k_gpua.sbatch` | `train_shared_rat_expectedcritic_relative_dualdamping.py`; `adv_resnet_shared_rat_expectedcritic_relative_dualdamping_100k.yaml`; `smoke_100k_seed0_jupyter_expected_relative_dualdamping_v1/expected_gaussian_score_b_relative` |
| `18669530` | `sbatch --dependency=afterany:18669454 jupyter_joint2b_exact_schur_smoke100k_gpua.sbatch` | `train_shared_joint2b_dualdamping_schur.py`; `adv_resnet_shared_joint2b_dualdamping_fisheradaptive_schur_100k.yaml`; `smoke_100k_seed0_jupyter_exact_schur_v1/schur_critic_b` |
| `18669613` | `sbatch --dependency=afterany:18669530 --job-name=pg-j2b-cf05 --export=ALL,PROCGEN_METHOD=direct_2b jupyter_jointb_actorrelative_criticfloor05_smoke100k_gpua.sbatch` | method `direct_2b`; exact identities/hashes in preceding table; `smoke_100k_seed0_jupyter_actorrelative_criticfloor05_v1/direct_2b` |
| `18669615` | `sbatch --dependency=afterany:18669613 --job-name=pg-rat-cf05 --export=ALL,PROCGEN_METHOD=expected_b jupyter_jointb_actorrelative_criticfloor05_smoke100k_gpua.sbatch` | method `expected_b`; exact identities/hashes in preceding table; `smoke_100k_seed0_jupyter_actorrelative_criticfloor05_v1/expected_b` |
| `18669725` | `sbatch --dependency=afterany:18669615 jupyter_jointb_rhsaligned_actorrelative_criticfloor05_gate250k_gpua.sbatch` | `train_shared_jointb_rhsaligned_deterministic.py`; `adv_resnet_shared_jointb_rhsaligned_actorrelative_criticfloor05_250k.yaml`; `gate_250k_seed0_jupyter_rhsaligned_actorrelative_criticfloor05_v1/rhs_aligned_rank1_b` |
| `18670437` | `sbatch jupyter_jointb_rhsaligned_actorrelative_criticfloor05_gate500k_gpua.sbatch` | same trainer; `adv_resnet_shared_jointb_rhsaligned_actorrelative_criticfloor05_500k.yaml`; `gate_500k_seed0_jupyter_rhsaligned_actorrelative_criticfloor05_v1/rhs_aligned_rank1_b` |
| `18670696` | `sbatch jupyter_jointb_rhsaligned_actorrelative_criticfloor05_gate1m_gpua.sbatch` | same trainer; `adv_resnet_shared_jointb_rhsaligned_actorrelative_criticfloor05_1m.yaml`; `gate_1m_seed0_jupyter_rhsaligned_actorrelative_criticfloor05_v1/rhs_aligned_rank1_b` |
| `18672560` | `sbatch jupyter_jointb_rhsaligned_actorrelative_criticfloor05_crossguard05_gate1m_gpua.sbatch` | `train_shared_jointb_rhsaligned_deterministic_lowfisherguard.py`; `adv_resnet_shared_jointb_rhsaligned_actorrelative_criticfloor05_crossguard05_1m.yaml`; `gate_1m_seed0_jupyter_rhsaligned_actorrelative_criticfloor05_lowfisherguard05_v1/rhs_aligned_rank1_b` |

## CSF3 terminal scientific metrics and integrity

Each bracket is `[reward, behavior-KL, current-step-KL, solve-residual]` in
BigFish/BossFight/CaveFlyer/CoinRun order. All cells listed below have PASS,
rc0, nonempty command/preflight/stdout/trace, and no targeted Traceback,
nonfinite, OOM, CUDA/NCCL, disk-full, or stall signature. Actual transitions
are 106,496 for 100k, 258,048 for 250k, 507,904 for 500k, and 1,007,616 for
1M. No checkpoint exists in these roots by launcher design.

| Job | Terminal metrics by environment | Scientific status |
|---|---|---|
| `18666610` | `[2.04,.000567,3.70e-6,2.98e-13]`; `[.01,.006492,6.85e-6,6.04e-13]`; `[1.00,.002065,2.94e-6,5.33e-13]`; `[2.70,.009213,1.31e-5,3.02e-13]` | complete smoke |
| `18667225` | `[1.65,.000236,1.73e-6,4.98e-13]`; `[.01,.003432,3.87e-6,3.38e-13]`; `[1.30,.000030,5.03e-8,3.67e-13]`; `[3.00,.000017,1.32e-8,2.50e-13]` | complete smoke |
| `18667467` | `[1.83,.004936,0,2.59e-13]`; `[.01,.004978,0,6.86e-13]`; `[1.30,.004836,0,3.67e-13]`; `[3.50,.004945,0,3.00e-13]` | complete smoke |
| `18667627` | `[2.41,.004921,2.15e-6,1.63e-13]`; `[0,.004966,9.29e-7,6.20e-13]`; `[1.10,.004988,8.84e-7,5.91e-13]`; `[2.90,.004968,6.71e-7,6.38e-13]` | complete smoke |
| `18667792` | `[2.25,.004075,4.87e-5,6.57e-14]`; `[.07,.005161,3.92e-6,8.96e-14]`; `[2.13,.004718,3.27e-6,5.66e-14]`; `[1.60,.005070,3.30e-6,1.20e-13]` | complete smoke |
| `18667941` | `[1.83,.000375,9.36e-6,8.52e-12]`; `[.07,.000618,8.73e-6,1.14e-12]`; `[1.80,.000779,2.13e-6,5.21e-14]`; `[4.50,.008115,4.48e-5,9.23e-12]` | complete smoke |
| `18668461` | `[2.04,.000190,1.37e-6,7.54e-13]`; `[.01,.000082,8.89e-7,2.53e-13]`; `[1.90,.000074,1.02e-6,7.77e-14]`; `[2.50,.001699,8.84e-6,2.94e-12]` | complete smoke |
| `18669377` | `[1.60,.000349,2.28e-6,1.63e-12]`; `[.03,.000255,1.28e-6,2.14e-13]`; `[1.80,.000094,1.13e-6,6.65e-14]`; `[2.80,.212908,8.03e-4,4.53e-12]` | complete; CoinRun high-KL concern |
| `18669429` | `[1.44,.000035,3.79e-7,3.04e-12]`; `[0,.000015,1.12e-7,1.22e-12]`; `[2.70,.001008,9.10e-6,5.69e-13]`; `[3.60,.005619,4.90e-5,6.71e-12]` | complete smoke |
| `18669454` | `[1.57,.000453,1.48e-4,1.75e-14]`; `[.03,.000263,3.48e-5,2.56e-14]`; `[1.70,.000411,2.43e-5,2.28e-14]`; `[2.80,.000552,7.83e-4,2.22e-14]` | complete smoke |
| `18669530` | `[2.04,.000190,1.37e-6,6.40e-13]`; `[.01,.000082,8.89e-7,3.27e-13]`; `[1.90,.000074,1.02e-6,6.81e-14]`; `[2.50,.001699,8.84e-6,2.31e-12]` | complete smoke |
| `18669613` | `[1.43,1.47e-5,1.73e-7,5.22e-13]`; `[.02,1.06e-5,1.02e-7,2.24e-13]`; `[1.40,1.36e-5,1.68e-7,2.25e-13]`; `[2.40,3.24e-5,4.63e-7,2.38e-13]` | complete smoke |
| `18669615` | `[2.00,.000192,.000303,2.07e-14]`; `[.03,.000252,.000737,1.45e-14]`; `[1.50,.000924,2.81e-5,1.92e-14]`; `[3.10,.000743,.001386,2.55e-14]` | complete smoke |
| `18669725` | `[2.73,.001244,1.45e-5,7.50e-13]`; `[.09,.004758,3.65e-5,6.64e-13]`; `[2.60,.008928,7.66e-5,2.48e-13]`; `[6.10,.003305,4.03e-5,4.48e-13]` | complete 250k gate |
| `18670437` | `[2.97,.003419,4.75e-5,6.83e-13]`; `[1.10,.006855,1.12e-4,5.73e-13]`; `[4.60,.011694,6.98e-5,3.07e-13]`; `[6.80,.005695,4.13e-5,4.91e-13]` | complete 500k gate |
| `18670696` | `[3.68,.003682,4.23e-5,3.36e-13]`; `[.36,.002664,2.12e-5,3.71e-13]`; `[2.78,.008445,2.76e-5,5.90e-13]`; `[6.90,.006004,4.77e-5,5.05e-13]` | complete 1M target |
| `18672560` | `[3.68,.003682,4.23e-5,3.36e-13]`; `[.36,.002664,2.12e-5,3.71e-13]`; `[2.06,.008051,5.72e-5,5.36e-13]`; `[6.90,.006004,4.77e-5,5.05e-13]` | complete strict causal control |

## Bede scheduler and scientific matrix

Artifact families are `procgen_joint2b_dualdamping_schur_20260816_v1`,
`procgen_jointb_rhsaligned_20260816_v1`,
`procgen_rat_expectedcritic_20260816_v1`, and
`procgen_joint2b_dualdamping_20260816_v1` under
`/nobackup/projects/bdman37/yihe/`. Bede used V100 32GB nodes and therefore is
not an exact hardware parent for the gpuA target.

| Job | Cells, command and scheduler result | Artifact/scientific result | Judgment vs 1M target |
|---|---|---|---|
| `1072327` | Schur 100k launcher `cd86dac8...`; raw `1072328` task0 gpu006 9:06 COMPLETE/0; tasks1-3 cancelled zero-time | only BigFish PASS/rc0, 106496: reward 1.66, KL .000246, cKL 1.27e-6, residual 3.22e-13; trainer `5b5c3078...`, config `e3905718...` | `not-strict-match`: budget, solver, host; incomplete four cells |
| `1072329` | original RHS 100k launcher `e2a40344...`; task0 gpu006 7s FAILED/1, tasks1-3 cancelled | rc2 before trace: `ModuleNotFoundError: utils` | `not-strict-match`; infrastructure failure |
| `1072331` | retry1 launcher `f281ff86...`; task0 gpu006 19s FAILED/1, tasks1-3 cancelled | rc2, empty trace: CUDA OOM allocating 5.59 GiB | `not-strict-match`; infrastructure failure |
| `1072333` | memory-efficient retry2 `703e54f9...`; raws `1072334/35/36/33`, gpu006, 9:51-9:53, all COMPLETE/0 | four PASS/rc0 at 106496; rewards `[1.37,.03,2.43,3.80]`; KL `[.000495,.000605,.006709,.009332]`; trainer `ff987e0d...`, config `0b7a67aa...` | `not-strict-match`: 100k, Bede config/host |
| `1072337` | expected RAT 100k `27d10c97...`; raws `1072339/40/41/37`, gpu008/006, 6:32-8:14, all COMPLETE/0 | four PASS/rc0; rewards `[2.01,.08,1.80,3.70]`; expected Gaussian trainer `0514703d...`, config `5b7a02ba...`, no cross | `not-strict-match`: method, 100k, host |
| `1072338` | direct Joint-2B 500k `8dd4f8fd...`; tasks0-3 cancelled zero-time/no node | no artifact | `not-strict-match`: direct method; incomplete |
| `1072342` | expected RAT 500k `80db6407...`; tasks0-3 cancelled zero-time/no node | no artifact | `not-strict-match`: expected method; incomplete |
| `1072343` | task0 direct 250k, gpu006 19:32 COMPLETE/0 | BigFish PASS, reward 2.38, KL .000348, residual 6.07e-12 | `not-strict-match`: direct 2B, 250k, host |
| `1072344` | task0 expected 250k, gpu008 15:23 COMPLETE/0 | BigFish PASS, reward 1.87, KL .000244 | `not-strict-match` |
| `1072345` | task1 direct 250k, gpu006 19:58 COMPLETE/0 | BossFight PASS, reward .05, KL .000306 | `not-strict-match` |
| `1072346` | task1 expected 250k, gpu008 15:48 COMPLETE/0 | BossFight PASS, reward .10, KL .000812 | `not-strict-match` |
| `1072347` | query resolves raw ID `1072347` as child `1072326_0` of an unrelated out-of-scope job; no Procgen parent record | no bounded Procgen command/root recovered; unrelated job excluded | `insufficient-evidence` for requested Procgen slot |
| `1072348` | task2 direct 250k, gpu006 19:39 COMPLETE/0 | CaveFlyer PASS, reward 2.40, KL .017087 | `not-strict-match` |
| `1072349` | task2 expected 250k, gpu008 15:30 COMPLETE/0 | CaveFlyer PASS, reward 2.19, KL .001306 | `not-strict-match` |
| `1072350` | task3 expected 250k, gpu008 15:31 COMPLETE/0 | CoinRun PASS, reward 5.00, KL .002495 | `not-strict-match` |
| `1072351` | task3 direct 250k, gpu006 19:37 COMPLETE/0 | CoinRun PASS, reward 2.90, KL .000381 | `not-strict-match` |

Direct 250k has trainer `5b5c3078...`, config `4d004c80...`, 1024-row
direct-2B/full-cross semantics. Expected 250k has trainer `0514703d...`, config
`94c44aa...`, 512-row analytic expected-Gaussian B and zero expected cross.
Both use rollout 4096, minibatch 512, four epochs, float64, momentum 0 and
Kaczmarz false. Their method and budget differences are scientific.

## Strict-match table against 1M RHS-aligned Joint-B target

| Candidate | Env/seed and data | Budget/stop | Curvature/RHS/solver | Source identity | Judgment |
|---|---|---|---|---|---|
| CSF3 RHS 250k `18669725` | same | **250k vs 1M** | same | same trainer; budget config/launcher differ | `not-strict-match` |
| CSF3 RHS 500k `18670437` | same | **500k vs 1M** | same | same trainer; budget config/launcher differ | `not-strict-match` |
| CSF3 RHS guard 1M `18672560` | same four envs/seed0, network, rollout 4096, minibatch 512, 4 epochs, reward/eval/data protocol | same 1M and 1,007,616 stop | same actor Fisher, clean all-param critic GGN, full cross, transformed RHS, rank-1 B, float64, clip, momentum0/Kaczmarz false; only low-Fisher floor interpolation added | exact source/config diff confined to declared guard and telemetry | `strict-match`; complete |
| CSF3 100k PAP/FADP/Joint-2B/Schur/RAT set | same env/seed/PPO geometry | **100k vs 1M** | PAP, direct, Schur, expected or block-trace differs | distinct trainer/config/launcher | `not-strict-match` |
| Bede RHS retry2 `1072333` | same env/seed/core network; V100 protocol differs | **100k vs 1M** | same family, memory-efficient retry | config/launcher/host differ | `not-strict-match` |
| Bede direct/expected 250k | same env/seed; V100 | **250k vs 1M** | direct Joint-2B or expected RAT, not RHS Joint-B | source/config/launcher/host differ | `not-strict-match` |
| Cancelled arrays | intended subsets known | differing 1M/3M/500k | differing method | unstarted/no artifacts | `not-strict-match`; incomplete |
| Bede slot `1072347` | unknown | unknown | unknown | no Procgen parent record | `insufficient-evidence` |

The exact config diff between target and strict control adds only
`joint_low_fisher_actor_critic_guard=true`, high `.50`, low `.20`, max `.05`.
The trainer diff only reads/validates those fields, raises the actor-from-critic
damping floor as categorical Fisher falls, and emits guard telemetry. It does
not change architecture, data, rollout, minibatch, epochs, RHS, solver,
cross-block, reward, evaluation, clip, or termination.

## Same-budget strict control metrics

| Env | Unguarded reward/KL | Guard reward/KL | Terminal Fisher / guard fraction / floor | Interpretation |
|---|---|---|---|---|
| BigFish | `3.68 / .003682` | `3.68 / .003682` | `.799117 / 0 / .01` | inactive; bit-identical terminal row |
| BossFight | `.36 / .002664` | `.36 / .002664` | `.796902 / 0 / .01` | inactive; bit-identical terminal row |
| CaveFlyer | `2.78 / .008445` | `2.06 / .008051` | `.321666 / .594445 / .033778` | activated; lower terminal reward in seed0 |
| CoinRun | `6.90 / .006004` | `6.90 / .006004` | `.540985 / 0 / .01` | inactive; bit-identical terminal row |

## Failure and health ledger

| Evidence | Classification | Preserved reason |
|---|---|---|
| ACTOR_J BossFight seed0 | `algorithm-failure/EARLY_STOPPED_FAILED` | 5.7933 versus strict E-v2 10.60, ratio .5465 |
| Original ACTOR_J BigFish/CaveFlyer/CoinRun | `infrastructure-failure` | original attempts interrupted; recovery does not overwrite them |
| P1 four seed1 roots | `infrastructure-failure` | host interruption; no strict completed identity |
| `18642230`, `18624888`, `18666591` | `cancelled-obsolete-unstarted` | Start=None, no node, zero elapsed/artifacts |
| Bede `1072329_0` | `infrastructure-failure` | missing `utils`; no trace |
| Bede `1072331_0` | `infrastructure-failure` | V100 CUDA OOM; empty trace |
| Bede `1072338`, `1072342` | `cancelled-unstarted` | zero elapsed, no node/artifact |
| Bede `1072347` slot | `unknown/insufficient-evidence` | raw child collision; no Procgen parent record |
| `18669377` CoinRun | complete with health concern | PASS/rc0 but behavior KL .212908 |
| `18670696`, `18672560` | scientifically complete | all four PASS/rc0, finite metrics/residuals |

No new early-stop evaluation was executed; the 3/5 rule is inapplicable to
these seed0-only gates.

## Promotion evidence boundary and next authority

- `18670696` and `18672560` provide complete four-environment seed0 1M gate
  integrity and a strict causal-control comparison. They do not provide 6M,
  seeds 0/1/2 evidence.
- `18670437` is only a seed0 500k gate. PAP/FADP/Schur/RAT/direct-2B are only
  100k smoke evidence; RAT block-trace CoinRun also has a high-KL concern.
- Bede direct/expected families provide at most 250k seed0 evidence, with
  separate failed/cancelled provenance preserved.
- No enumerated configuration meets formal four-environment x 6M x seeds
  0,1,2 success evidence. Promotion is exclusively a Planner decision.

The Executor does not choose or submit a follow-on experiment. The same
ChatGPT Planner must provide exactly one next bounded Procgen task, preserving
strict identities, cancellation/failure history, no-Jupyter, nonoverwriting
roots, quarantine, and the formal evidence boundary.
