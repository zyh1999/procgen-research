# Executor Report

## Metadata

- Task-ID: `PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-6M-S0-SCIENCE-20260826-37`
- Method: `DET_STANDARD_MSE_GGN_HEAD_CVLM_V1`
- Assignment/control: `71f9e17e2fd8411faf34e4c2530800d66301e377`
- Deployment freeze: `4be726357752b197d2c2fabf0d29500b193e8beb`
- Repository target: `origin/agent-work`

## Result

Unique Task37 conclusion: `CANDIDATE_REJECT`.

Task36's four environment `PRECHECK_PASS/rc0` evidence was reused without
rerunning preflight. Frozen trainer/config/bundle/manifest/science identities
were verified, gpuH was preferred, and exactly four fresh seed0 intended-6M
cells were submitted once: BigFish `19397520`, BossFight `19397521`,
CaveFlyer `19397522`, and CoinRun `19397523`.

All four reached the exact first common Paper stage at 2,007,040 transitions
and failed the strict 0.60 ratio threshold: BigFish `3.96/9.28 =
0.4267241379`; BossFight `0/2.92 = 0`; CaveFlyer `1.27/4.45 =
0.2853932584`; CoinRun `0/3.70 = 0`. The frozen stage monitor wrote one
`EARLY_STOPPED_ALGORITHM` ledger and rc3 per cell. Scheduler accounting now
shows all four `CANCELLED by 778916`, with elapsed/node pairs 00:37:13/node820,
00:37:15/node821, 00:28:51/node822 and 00:27:22/node820.

Root `RUNNING` markers and absent trainer rc files are stale after Slurm killed
the trainers. Scheduler accounting and exact monitor ledgers are authoritative.
Each root retains command, provenance, progress, full metric trace, logs,
scientific-start marker, hashes and `early_stop_2007040`; no checkpoint exists.
Strict hard-error scans are zero.

Numerically, every acted-stage solve has Cholesky info zero and relative
residual between 2.978e-17 and 3.745e-16. CVLM nevertheless does not identify
a competitive regime: BigFish and CoinRun reject to zero head delta,
BossFight accepts a negligible delta, and CaveFlyer accepts a substantial
held-out-positive delta but reaches only 28.5% of Paper reward. The four-cell
acted-stage mean ratio is 0.1780293491; zero cells reach the endpoint.

The complete remote model-free archive is content-addressed by SHA256
`14e3cca153da5a90c9463cc7f64c440d9f9688f14b30309d1ad74bf228853e4c`.
The Git-ready compact archive and exact tables are under
`remote_launch_staging/procgen_standard_mse_ggn_head_cvlm_6m_s0_science_20260826_37/evidence_terminal/`.
Models and checkpoints are excluded.

No retry, requeue, resubmission, successor, sweep, seed expansion, Paper
rerun, second monitor, Jupyter use, quarantined access or Task32/33 mutation
occurred. The Executor awaits exactly one Planner READY or NEED_DECISION task.

CANDIDATE_REJECT
