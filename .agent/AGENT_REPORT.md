# Executor Report

## Metadata

- Task-ID: `PROCGEN-HYBRID-HEAD-ROOT-OVERRIDE-6M-MISSING3-20260824-13`
- Assignment: `6f7032a8fe3f3350efd7d2df7e68b597f8384332`
- Launcher freeze: `c64040672893a2048953b94d5b6be1dc6366d3d0`
- Repository target: `origin/agent-work`

## Result

The root-only launcher equivalence audit passed. Original launcher
`ae7104e7...` remains unchanged; root-override launcher
`26f06ec9...` differs only in validated artifact routing and provenance.
Trainer/config/monitor/preflight/structural hashes stayed frozen. Task11 roots
and BigFish `19228676` were not touched.

Exactly three new seed0 intended-6M cells were submitted once. BossFight
`19233036` triggered the exact2M rule at `1.24/2.92=.4246575342`; CoinRun
`19233038` triggered it at `.10/3.70=.0270270270`. The frozen monitor returned
rc3 and cancelled only each failing cell; both are
`EARLY_STOPPED_ALGORITHM`. CaveFlyer `19233037` passed exact2M
`5.20/4.45=1.168539326`, exact4M `5.50/5.85=.9401709402`, and completed the
5,980,160 endpoint `6.60/6.62=.9969788520`, Slurm COMPLETED/0:0 and root
PASS/rc0.

All listed solver telemetry is finite, Cholesky info is zero, and hard-error
scans are empty. Combined with immutable Task11 BigFish's exact4M algorithm
early stop, BossFight and CoinRun make rejection unambiguous. Complete
model-free scheduler, command, progress, compressed trace/log, stage-ledger,
hash, artifact and checkpoint-metadata evidence is tracked in
`evidence_task13/`; no model/checkpoint is included.

No retry, requeue, resubmit, duplicate, second candidate, sweep, Paper rerun,
Jupyter, quarantined access or unrelated mutation occurred.

CANDIDATE_REJECT
