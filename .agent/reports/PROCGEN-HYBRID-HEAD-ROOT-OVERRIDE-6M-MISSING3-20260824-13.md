# PROCGEN-HYBRID-HEAD-ROOT-OVERRIDE-6M-MISSING3-20260824-13

## Unique conclusion

`CANDIDATE_REJECT`.

The authorized root-only launcher recovery passed its equivalence gate and the
three missing seed0 cells each started exactly once in new roots. BossFight and
CoinRun then triggered the frozen exact-2M algorithm early-stop rule. CaveFlyer
passed 2M and 4M and completed the 5,980,160 endpoint. Combined with immutable
Task11 BigFish's exact-4M algorithm early stop, three of four environments fail
the candidate; no infrastructure ambiguity remains for this decision.

## Frozen identities and launcher-only recovery

| Artifact | Identity | Result |
|---|---|---|
| Assignment | `6f7032a8fe3f3350efd7d2df7e68b597f8384332` | immutable |
| Launcher freeze | `c64040672893a2048953b94d5b6be1dc6366d3d0` | pushed before science |
| Original launcher | `ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e` | unchanged |
| Root-override launcher | `26f06ec93f84277c7e8d099b75f4e7d053cc74850926e0f04417813133cb07dd` | artifact routing only |
| Trainer | `7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54` | unchanged |
| Config | `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda` | unchanged |
| Stage monitor | `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e` | unchanged |
| Corrected preflight | `704278e8b5802498b8e065b9f12945e2cb72a665cdd28845b2401091b2e993ea` | unchanged |
| Structural manifest | `3f91f5c313480c089d300a7dd5aff4a664f28ff9d9f4718bbab430200298d623` | identical in all roots |

The complete line diff and dry-run audit are in
`ROOT_OVERRIDE_LAUNCHER_AUDIT_TASK13.md`. Removing only the marked root and
provenance blocks makes every remaining launcher line byte-identical. The
normalized command/preflight SHA is
`33f3fb9ed0485b6eda031bd22711f76c434b4744b5320f42c0d7ed06d3e57b4a`.
The variant requires an absolute campaign, rejects the Task11 campaign and all
descendants, retains collision exit90, and adds only base-launcher, Task-ID and
campaign provenance.

Task12's four accepted no-training validations were reused exactly; no GPU
scientific preflight was rerun. Pre-launch and post-deployment fingerprints of
the four Task11 root metadata sets remained identical: BigFish `eff6ccb1...`,
BossFight `c96733a7...`, CaveFlyer `7e88b612...`, CoinRun `9f8115c4...`.
BigFish job `19228676` and every Task11 root were not rerun or modified.

## New-root scientific execution

Campaign:
`/scratch/h99859yz/procgen_paper_hybrid_head_detggn_6m_missing3_20260824_13`.

| Environment | Job / node | Root | Terminal classification |
|---|---|---|---|
| BossFight | `19233036`, node822 | `runs/PAPER_HYBRID_SHARED_PAPER_HEAD_DETGGN_V1/bossfight-easy-0-10/seed0/6m` | `EARLY_STOPPED_ALGORITHM`; Slurm CANCELLED by778916, 25:38 |
| CaveFlyer | `19233037`, node823 | `runs/PAPER_HYBRID_SHARED_PAPER_HEAD_DETGGN_V1/caveflyer-easy-0-10/seed0/6m` | scientific PASS/rc0; Slurm COMPLETED/0:0, 1:04:25 |
| CoinRun | `19233038`, node820 | `runs/PAPER_HYBRID_SHARED_PAPER_HEAD_DETGGN_V1/coinrun-easy-0-10/seed0/6m` | `EARLY_STOPPED_ALGORITHM`; Slurm CANCELLED by778916, 23:39 |

All three roots were absent before submission, unique, non-overwriting, and
created only by their one authorized job. Each passed the corrected structural
and connectivity checks, records `GPUH_HYBRID_HEAD_COMPATIBILITY_PASS`, and
contains `scientific_started.marker`. No job was retried, requeued, resubmitted
or duplicated. Scheduler state is authoritative over the stale RUNNING/absent
rc markers left by the two scientific cancellations.

## Exact same-transition Paper comparisons

The immutable original Paper seed0 sources and header-only
`misc/total_timesteps` to `transitions_so_far` adapters are hash-recorded in
each stage directory. No terminal Paper row was substituted for an
intermediate target.

| Environment | Transition | Target | Paper | Ratio | Decision | KL | LR | Entropy | Head rel./solve residual | Cholesky info |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|
| BossFight | 2,007,040 | 1.24 | 2.92 | `.4246575342` | `EARLY_STOPPED_ALGORITHM` | `.0480683` | `.0003375` | `1.70434` | `6.028e-16 / 4.096e-15` | 0 |
| CaveFlyer | 2,007,040 | 5.20 | 4.45 | `1.168539326` | PASS | `.0581414` | `.0001` | `1.56761` | `6.324e-16 / 8.084e-15` | 0 |
| CaveFlyer | 4,014,080 | 5.50 | 5.85 | `.9401709402` | PASS | `.0285457` | `.0437894` | `.924291` | `9.163e-16 / 1.061e-14` | 0 |
| CaveFlyer | 5,980,160 | 6.60 | 6.62 | `.9969788520` | PASS | `.0576269` | `.00050625` | `.280721` | `5.530e-15 / 3.418e-14` | 0 |
| CoinRun | 2,007,040 | .10 | 3.70 | `.0270270270` | `EARLY_STOPPED_ALGORITHM` | `.00663787` | `.5` | `2.43014` | `5.890e-15 / 1.381e-13` | 0 |

The frozen monitor returned rc3 and cancelled only the corresponding
BossFight/CoinRun cell. It returned rc0 for every CaveFlyer stage. Paper shared
critic residual telemetry and all deterministic-head solves remained finite;
post-head policy KL is exactly zero in every listed row. The complete hard
error scans are empty: no Traceback, NaN/Inf, OOM, CUDA/NCCL, disk/quota or
stall signature. CaveFlyer's checkpoint exists remotely and is recorded by
size/SHA only; it is not committed.

## Combined Task11 + Task13 decision

Immutable Task11 BigFish passed exact2M at ratio `.7036637931`, then failed
exact4M at `6.23/13.28=.4691265060` and remains
`EARLY_STOPPED_ALGORITHM`. Task13 adds strict algorithm failures for BossFight
and CoinRun, while CaveFlyer completes at `.9969788520`. Since at least one of
the three new environments triggers `<.60`, and the cumulative matrix contains
at least two environment-level algorithm failures, Task13's acceptance rule
requires `CANDIDATE_REJECT`.

This is an algorithm result, not a solver or infrastructure failure. The old
Task11 per-job manifest-SHA preflight failures remain separate immutable
infrastructure provenance; Task12 corrected them without rewriting history.

## Evidence and restrictions

Complete model-free evidence is tracked under
`remote_launch_staging/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/evidence_task13/`.
It includes final scheduler accounting, commands/config identities, structural
and connectivity manifests, progress, compressed traces/logs, exact-stage
source/adapter/hash/ledger/scheduler evidence, artifact inventories,
checkpoint metadata/hash, and zero-length hard-error scans. `model.ckpt` is
excluded.

No model/checkpoint push, BigFish rerun, Paper rerun, scientific retry,
requeue, resubmit, second candidate, sweep, Jupyter, quarantined host access or
unrelated mutation occurred.

`CANDIDATE_REJECT`
