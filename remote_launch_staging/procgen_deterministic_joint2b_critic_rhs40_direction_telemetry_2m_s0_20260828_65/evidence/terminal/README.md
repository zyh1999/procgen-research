# Task65 terminal model-free evidence

Task `PROCGEN-DETERMINISTIC-JOINT2B-CRITIC-RHS40-DIRECTION-TELEMETRY-2M-S0-20260828-65`
completed naturally on Bede. This directory contains only model-free terminal
evidence: progress, frozen all-record aggregates, command/identity records,
root status/rc and checkpoint `stat` metadata. No checkpoint/model bytes or
content hashes were read, copied or committed.

## Scheduler and root reconciliation

| Environment | Job | Scheduler | Exit | Elapsed | Node | Root | Endpoint |
|---|---:|---|---|---|---|---|---:|
| BigFish | 1078983 | COMPLETED | 0:0 | 02:44:19 | gpu017 | PASS/rc0 | 2,007,040 |
| BossFight | 1078984 | COMPLETED | 0:0 | 02:46:49 | gpu017 | PASS/rc0 | 2,007,040 |
| CaveFlyer | 1078985 | COMPLETED | 0:0 | 02:45:53 | gpu022 | PASS/rc0 | 2,007,040 |
| CoinRun | 1078986 | COMPLETED | 0:0 | 02:45:53 | gpu022 | PASS/rc0 | 2,007,040 |

All four roots contain 49 progress rows and 15,680 complete telemetry records.
Every frozen aggregate reports `TASK65_AGGREGATION_PASS`. Each checkpoint is a
regular non-symlink file of 3,766,013 bytes and mode 0664; only the already
generated remote `checkpoint.stat` metadata is retained here.

## Exact endpoint and reward sanity

| Environment | Task65 | immutable Paper | Task65/Paper | Task63 parent |
|---|---:|---:|---:|---:|
| BigFish | 2.11 | 9.28 | 0.2274 | 5.08 |
| BossFight | 0.00 | 2.92 | 0.0000 | 0.04 |
| CaveFlyer | 3.20 | 4.45 | 0.7191 | 0.00 |
| CoinRun | 8.20 | 3.70 | 2.2162 | 10.00 |

This telemetry task never reward-stopped. RHS40 is not a general reward rescue:
it improved CaveFlyer relative to Task63 and remained above Paper on CoinRun,
but BigFish/BossFight remained poor and CoinRun was below Task63.

## Complete-record post-inverse aggregates

All entries are overall medians across 15,680 complete minibatches per root.

| Environment | actor metric energy share | full actor norm | full actor projection | shared actor norm | shared actor projection | KL | LR |
|---|---:|---:|---:|---:|---:|---:|---:|
| BigFish | 0.96743 | 0.03791 | 0.001724 | 0.07586 | 0.003951 | 0.007296 | 0.5 |
| BossFight | 0.96203 | 0.03309 | 0.001057 | 0.07729 | 0.004156 | 0.002171 | 0.5 |
| CaveFlyer | 0.98083 | 0.03775 | 0.001517 | 0.07731 | 0.003553 | 0.011669 | 0.5 |
| CoinRun | 0.97166 | 0.03369 | 0.002361 | 0.04657 | 0.003873 | 0.013153 | 0.5 |

The raw metric remains actor-heavy, but after inversion and the exact 40x
critic RHS the installed coupled direction is overwhelmingly critic-dominant:
median full actor norm share is only 3.31--3.79%, median actor signed projection
share only 0.106--0.236%, and median shared actor signed projection share only
0.355--0.416%. This is the clean causal effect requested by Task65 and is
distinct from Task64's curvature change.

## Numerical and identity evidence

- Every endpoint retains curvature `0.1`, objective `40`, critic RHS weight
  `126.49110640673517`, strict 1024 rows, natural nonzero cross blocks,
  history correction and adaptive KL/LR.
- Endpoint Cholesky info is zero in all four cells; FP64 relative residuals are
  `2.887e-15`, `6.848e-16`, `3.976e-15`, and `1.854e-15`.
- RHS reconstruction is exactly zero; alpha reconstruction is
  `1.214e-15`--`1.506e-15`; direction reconstruction is
  `1.520e-8`--`9.465e-8`; finite scans pass.
- Scoped log scans found zero Traceback, OOM, CUDA, NCCL, disk/quota,
  nonfinite, Cholesky, reconstruction, identity or structural-zero failures.

Final classification: `TERMINAL_TELEMETRY_COMPLETE_TASK65_AGGREGATION_PASS`.
