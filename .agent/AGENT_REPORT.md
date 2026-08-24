# Executor Report

## Metadata

- Task-ID: `PROCGEN-PAPER-SEPARATEB-DETGGN-6M-S0-20260824-07`
- Method: `PAPER_MATCHED_SEPARATE_B_DET_GGN_V1`
- Planner: ChatGPT thread `6a8309ee-bb34-83eb-9512-72acc5913334`
- Assigned task commit: `39f434a`
- Frozen implementation commit: `8a956130fe661aa41286a9b36ffe10965c223082`
- Repository target: `origin/agent-work`

## Result

Identity, historical-distinctness, regression, and H200 compatibility prechecks
passed. At exact transition 2,007,040, Target/Paper reward ratios were
`.3469827586`, `.0171232877`, and `.1640449438` for BigFish, BossFight, and
CaveFlyer. Frozen monitor ledgers early-stopped those three cells as
`EARLY_STOPPED_ALGORITHM`. Their scheduler cancellations override stale root
RUNNING markers; artifacts, logs, hashes, and zero-hit error scans are
preserved.

CoinRun passed at exact 2M and 4M, then completed scheduler/artifact rc0 at
5,980,160 with reward `6.40` versus Paper `9.40`, ratio `.6808510638`. Its
solver telemetry remained finite and its error scan was clean. Three of four
environments failed the predeclared `.60` threshold, so this candidate cannot
be promoted.

Full method hashes/diff, actor-equivalence evidence, historical provenance,
immutable Paper baseline hashes, exact stage rows, reward/KL/LR/entropy/solver
telemetry, scheduler/status/rc/progress/trace/checkpoint/error evidence, stage
ledgers, and preserved prior failures are recorded in
`.agent/reports/PROCGEN-PAPER-SEPARATEB-DETGGN-6M-S0-20260824-07.md` and the
model-free evidence export under the frozen staging directory.

No new method, sweep, retry, Jupyter session, or unrelated job mutation was
created. The same ChatGPT Planner must explain this result and return exactly
one next bounded scientific/code READY task; live hardware placement remains
the Executor's responsibility.

## Delivery

- Evidence/report commit: `eda4a2a6019245e919d152724c989d7e9e4939be` on
  `origin/agent-work`.
- Push target: `origin/agent-work`.

CANDIDATE_NOT_READY
