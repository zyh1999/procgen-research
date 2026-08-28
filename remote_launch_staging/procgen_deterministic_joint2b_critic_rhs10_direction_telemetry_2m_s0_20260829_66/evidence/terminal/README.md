# Task66 terminal model-free archive

All four Bede jobs completed naturally and scheduler-authoritatively with exit
`0:0`. No cancellation, retry, requeue, resubmission, tuning, or unrelated
job/root mutation occurred.

| Environment | Job | State | Elapsed | Node | Root | Transition | Reward | Paper | Ratio |
|---|---:|---|---|---|---|---:|---:|---:|---:|
| BigFish | 1084427 | COMPLETED/0:0 | 02:43:50 | gpu018 | PASS/0 | 2,007,040 | 7.31 | 9.28 | 0.7877155 |
| BossFight | 1084428 | COMPLETED/0:0 | 02:46:18 | gpu018 | PASS/0 | 2,007,040 | 0.00 | 2.92 | 0.0 |
| CaveFlyer | 1084429 | COMPLETED/0:0 | 02:45:56 | gpu017 | PASS/0 | 2,007,040 | 1.50 | 4.45 | 0.3370787 |
| CoinRun | 1084430 | COMPLETED/0:0 | 02:45:55 | gpu008 | PASS/0 | 2,007,040 | 0.00 | 3.70 | 0.0 |

Each environment archive contains the complete progress table, frozen
all-record aggregate, command/identity, GPU/node/job metadata, status/rc, and
checkpoint `stat` output only. Remote checkpoints are regular non-symlink
mode `0664` files of 3,766,013 bytes. No checkpoint/model bytes or content
hashes were copied or committed.

Every aggregate has 15,680 complete records and reports
`TASK66_AGGREGATION_PASS`. Final records preserve curvature `0.1`, objective
`10`, critic RHS weight `31.622776601683793`, strict 1024 rows, natural cross,
Cholesky info `0`, exact RHS reconstruction, finite reconstruction/structural
zeros, and finite scans. Scoped hard-error searches found zero matches.
