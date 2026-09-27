# Curated historical shared Procgen training logs

124 completed runs from the user's experiments, exported on 2026-09-27.
These are **shared actor–critic** results. They are selected historical
cohorts, **not new training runs from this cleaned GitHub package** and not
a claim that one method is best on every environment.

| Family | Exact experiment | Batch | Environments | Seeds | Runs | Endpoint |
|---|---|---:|---:|---|---:|---:|
| Fixed-LR PPO | Task278 shared PPO | 4096 / 8192 | 4 per batch | 0,1,2 | 24 | 15,007,744 |
| Full RAT | Task229 Full512, zero history/no Kaczmarz | 512 | 8 | 0–4 | 40 | 6,004,736 |
| 127+1 | Task253 uniform shared Dual127+1 | 512 | 4 available complete cohorts | 0,1,2 | 12 | 6,004,736 |
| 127+1 | Task281 shared B/B normal-aligned | 4096 / 8192 | 4 per batch | 0,1,2 | 24 | 15,007,744 |
| 128 | Task280 fixed-tail sample coefficients + residual ray | 512 | 8 | 0,1,2 | 24 | 6,004,736 |

Eight environments: BigFish, BossFight, StarPilot, CaveFlyer, CoinRun, Maze,
Jumper, Miner. Large-batch cohorts cover BigFish/BossFight/StarPilot/CaveFlyer.
The normal Task253 subset covers StarPilot/CaveFlyer/CoinRun/Maze only.
Missing exact Task253 cohorts are **not** filled with relabeled Task232 logs.

## Files

- `manifest.json`: exact task/environment/seed/batch-cohort, hardware label,
  PASS/rc0 evidence, endpoint, row count, column names and scalar-file hashes.
- `<cohort>/config.json`: documented scientific settings for that identity.
- `<cohort>/<environment>/seedN/progress.csv.gz`: every original numeric
  rollout row and column, with original scalar string values. CSV formatting
  is normalized to LF and deterministic gzip; no smoothing/downsampling.
- `per_seed_summary.csv`: early/middle/late thirds, overall reward, and
  last-1M-window mean per seed. Normal windows end at 6M, large at 15M;
  the final rounding-over-endpoint rollout remains in the raw log.
- `cohort_summary.csv`: mean and sample standard deviation across those
  per-seed last-1M means. Window membership is `(start, end]`.
- `verify_logs.py`: standalone standard-library integrity/coverage checker.

```bash
python logs/verify_logs.py
python - <<'PY'
import csv, gzip
path = 'logs/task280_coeff128_ray_B512/bigfish/seed0/progress.csv.gz'
with gzip.open(path, 'rt') as f:
    rows = list(csv.DictReader(f))
print(rows[-1])
PY
```

`eprewmean` is the training return statistic logged by the original runner,
not a held-out evaluation score. The summaries average these logged values;
they are not reconstructed episode-level returns. `kl`, `entropy`, and `lr`
retain each runner's original progress-row semantics; in particular a row
can represent the last minibatch rather than a rollout-wide distribution.
Use elapsed time/FPS only within hardware and concurrency-matched cohorts.

## Selection and interpretation

The goal is a useful, comparatively complete recent reference collection,
not an exhaustive experiment archive or an unbiased method search. Method
selection was made after results were available. **Within every included
environment/method cohort, all declared seeds are retained**, including weak
runs. No individual seed was chosen because it scored well.

- Task280 is the recent, fully audited normal coefficient128 identity.
- Task281 is the completed normal-aligned large-batch Dual127+1 reference;
  Task278 PPO is included at both exact matching batch sizes as its baseline.
- Task229 is the complete eight-environment/five-seed full-curvature RAT
  reference. It is older than the newer reduced-curvature experiments.
- Task253 provides only the complete exact-identity three-seed normal
  reference cohorts retrieved for this release, not full campaign coverage.

Task281 versus PPO has mixed outcomes, not uniform dominance. Normal and
large batches have different rollout sizes and budgets. PPO also differs
from RAT in optimizer, value loss and PopArt. Hardware varies across seeds.
Do not interpret these comparisons as normalization-only causality,
statistical significance, or held-out generalization.

## Important method/code boundaries

- Full RAT here means **all 512 curvature rows**, not reduced128. There is
  no standalone Full512 trainer entry in the current cleaned package.
- Dual127+1 means 127 real uniform anchors plus one exact aggregate, not
  128 independent anchors. Task253 and Task281 use system/B and final/B.
- Task280 uses **SAMPLE coefficient-space** fixed-tail solves and sample
  residual-ray scaling. It is not the historical PARAMETER-space FullRHS128
  Woodbury update, Task262 parameter-ray, or Task283 q/B method.
- The packaged `train_large8192_128.py` is a new mathematical port:
  **there are no full-training results for that profile in this release**.
- Packaged fixed-LR PPO uses the Task266/278 recipe. Included PPO logs are
  Task278 B4096/B8192, **not a completed B512 reproduction** of the new entry.
- Task278 q/q EF, Task279 critic-scale variants, Task283 q/B results, partial
  and failed experiments are outside this curated selection. Their omission
  must not be interpreted as a statement that they did not exist.
- No no-shared results are relabeled as shared. The new no-shared PPO
  campaign is not included as completed evidence.

## Provenance and validation scope

Task229 status/rc and scalar rows were read directly from the existing
training roots for this release: all 40 PASS/rc0/full endpoint. Task280 uses
the 2026-09-26 22:52 terminal scalar snapshot and its complete trace audit
(1,125,888 events, zero checked identity/numerical/controller violations).
Task253 uses the exact-identity 2026-09-27 11:40 scalar/status/rc snapshot.
Task278/281 use terminal scalar curves collected at 03:14 on 2026-09-27,
with PASS/rc0 cross-checked against the 12:46 terminal fleet snapshot.
Task281's completed trace audit checked 263,808 events with zero violations.
These trace counts are audit context, **not raw event traces included here**;
this release verifies the exported scalar records, not a new full audit of
every historical event. Original task numbers remain explicit throughout.

Every exported run has a complete, unique, increasing rollout grid ending
at its declared endpoint, PASS/rc0 evidence, and finite numeric scalar
values. The scalar export includes no checkpoint/model contents, raw
stdout/stderr, tokens, SSH commands, hostnames/IPs, usernames, absolute
machine paths or W&B account links. Checksums cover scalar CSV/gzip data
only; no checkpoint was read, copied or hashed.
