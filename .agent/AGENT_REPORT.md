# Executor Report

## Metadata

- Task-ID: `PROCGEN-READONLY-REFRESH-20260817-01`
- Inspection window: `2026-08-17T13:25:18Z` to `2026-08-17T13:37:08Z`
- Executor: Codex, bounded by `.agent/TASK.md`
- Branch: `agent-work`
- Inspected local/remote HEAD: `0e2c4b01ea4aafa95631b21076d73716da34b34d`
- Starting worktree: clean
- Control plane: CSF3 `login1.csf3.man.alces.network`
- Inspected hosts: CSF3; dual-5060 `47.114.81.212:60023`
- Unavailable: Bede noninteractive authentication; `procgen-3090` direct channel
- Explicitly excluded: `.54`, `10.49.7.54`, `ws4090-31`

## Agent-file reconciliation

All four control files were read before the live audit.

| File | SHA256 before report update | Reconciled finding |
|---|---|---|
| `.agent/GOAL.md` | `d0fb2942ace1b0736902011f3b836a1a4790c11ba7af70e3432e6efd7aa4fe9d` | Preserve matched pure PPO versus PPG/curvature evidence |
| `.agent/STATE.md` | `85496971599d0f2e5c50635a8bec0739c58e7fe1a5ea9caebe641744fa412232` | Its 1M gate claim was stale; all four cells are now terminal |
| `.agent/TASK.md` | `cfd19511d3f4efa73dbd67957579c3aef8acd513306577d8353289e004621ec6` | `READY`; read-only refresh only |
| `.agent/AGENT_REPORT.md` | `9da64662aa33af61187abdde591eadff1b059467f1f094ed55616fd0daf8ccbf` | Prior report covered infrastructure setup only |

## Research-line definitions

| Line | Objective and boundary | Highest recorded strict baseline | Match fields required |
|---|---|---|---|
| Pure-PPO DMLP1024 | IMPALA/ResNet + shared `256-1024-256` decision trunk + PopArt; no PPG auxiliary | Bede `1070573-1070576`, four environments x seeds 0-2 | environment, source/config, architecture, rollout geometry, 6M budget, seed and reward convention |
| PPG/curvature | E_v2/ACTOR_G-H-J-K actor ablations, shared Exact-GGN/RAT, strict Joint-2B/Joint-B diagnostics | `E_v2` for actor ablations; causal gates require their own matched parent | same checkpoint/data where directional, same trainer/config, environment, architecture, rollout/minibatch/epochs, budget, seed, KL/reward semantics |

The repository does not yet contain enough exact source/config mapping to call
cross-host results strictly matched merely from job names. Such cells are
reported as unverifiable rather than silently counted.

## Fresh scheduler, GPU and process snapshot

Commands were read-only: `squeue`, `sacct`, `screen -ls`, process trees,
`nvidia-smi`, and targeted log/artifact inspection.

| Host | Resource/job | Fresh state | Evidence |
|---|---|---|---|
| CSF3 | `18642230_0-3` `pg-j2b-acguardA` | PENDING, `JobHeldUser`, zero runtime | fresh `squeue` |
| CSF3 | `18624888_0-3` `pg-j2b-block05` | PENDING, `JobHeldUser`, zero runtime | fresh `squeue` |
| CSF3 | `18485353` `vscode-tunnel` | RUNNING on node1232; not a GPU experiment | fresh `squeue` |
| CSF3 | Procgen GPU work | none running | queue and owned-process scan |
| dual-5060 | GPUs 0/1 RTX 5060 Ti | 33/16311 MiB, 0%; 15/16311 MiB, 0%; no Procgen PID | `nvidia-smi`, `ps` at `13:32:42Z` |
| Bede | `1070573-1070576` | recorded complete, not live-refreshed | noninteractive SSH denied |

The persistent `codex-three-domain` screen is alive and cycling. Its current
catalog is header-only and its launcher registry empty, so it is observational
and proposal-only. The old Bede channel reports authentication failure and the
old 5060 endpoint in that controller is obsolete. No new Jupyter session was
started. The four 1M jobs did start per-job Jupyter processes, but the launcher
had an EXIT trap and their worker logs are terminal; no surviving job/session
was found.

## Recent completed run inventory

The 1M array used:

- Workdir: `/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1`
- Launcher: `jupyter_jointb_rhsaligned_actorrelative_criticfloor05_gate1m_gpua.sbatch`
- Trainer SHA256: `ff987e0dd5ca1f4c1bb9a91e3794991f5a848bdbfdadc0425d935a72acf3b501`
- Config SHA256: `d87a8f648c1c91ee0d260c64ab7dd59d12bb7f9e6b67b0ee0a135389e697fb40`
- Geometry: rollout 4096, minibatch 512, four epochs, seed 0, nominal 1M
- Method: strict full-joint, clean critic score, all critic parameters,
  RHS-aligned rank-1-B reduction, 1024 parent rows -> 512 reduced rows,
  actor damping 0.003, critic damping 0.5, no momentum/Kaczmarz.

| Array cell | Environment | Scheduler | Exit/status | Final transitions | reward | behavior KL | current-step KL | solve residual | class |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `18670696_0` / raw `18670697` | BigFish | COMPLETED 00:57:45 node847 | 0:0 / PASS | 1,007,616 | 3.68 | 0.003682 | 4.23e-05 | 3.36e-13 | completed-valid |
| `18670696_1` / raw `18670698` | BossFight | COMPLETED 00:57:34 node847 | 0:0 / PASS | 1,007,616 | 0.36 | 0.002664 | 2.12e-05 | 3.71e-13 | completed-valid |
| `18670696_2` / raw `18671119` | CaveFlyer | COMPLETED 00:57:47 node847 | 0:0 / PASS | 1,007,616 | 2.78 | 0.008445 | 2.76e-05 | 5.90e-13 | completed-valid |
| `18670696_3` / raw `18670696` | CoinRun | COMPLETED 00:57:41 node847 | 0:0 / PASS | 1,007,616 | 6.90 | 0.006004 | 4.77e-05 | 5.05e-13 | completed-valid |

Per-run evidence is under
`gate_1m_seed0_jupyter_rhsaligned_actorrelative_criticfloor05_v1/rhs_aligned_rank1_b/<env>/seed0/`.
Each directory contains `preflight`, `command.txt`, `metric_trace.jsonl`,
`stdout`, `stderr`, `rc`, and `status`. Trace files are about 46 MiB and were
updated at the scheduler end time. This launcher did not produce a checkpoint;
that absence is not evidence of a silent crash because terminal status, trace,
and rc agree.

## Error, failure and early-stop accounting

- Hard-error scan (`Traceback`, OOM, nonfinite, RuntimeError, AssertionError,
  segmentation fault): zero hits in all four 1M stdout/stderr pairs.
- No NaN/Inf was observed in the reported reward/KL/solve fields.
- ACTOR_J BossFight seed 0 remains historical `EARLY_STOPPED_FAILED` at
  4,096,000 transitions, robust ratio 0.5465.
- Three original ACTOR_J attempts and the P1 seed-1 shared Exact-GGN/RAT roots
  remain infrastructure failures; clean recoveries must not erase them.
- Held arrays are scheduler/user-hold state, not algorithm failures.
- No run was cancelled, restarted or early-stopped in this task.

The 3/5 rule is not applicable to the 1M gate because only seed 0 exists and
no fully proven five-seed strict baseline is mapped. No `early-stop-candidate`
is asserted from this gate.

## Contradictions and Planner decision inputs

- Prior STATE said `18670696_0-3` was running/queued; fresh evidence says all
  four are completed-valid.
- The legacy controller board calls recent studies scientifically unknown;
  direct run-directory inspection resolves this specific array, but not the
  other unmapped PAP/FADP/J2B/RAT/Schur arrays.
- Idle-looking dual-5060 capacity is real telemetry, but there is no Procgen
  launcher authorization in the legacy controller.
- Evidence is sufficient for the Planner to choose one bounded evidence-mapping
  or matched-seed task. It is not sufficient to authorize autonomous submission
  without a new `.agent/TASK.md`.

## Changes and delivery

- Changed only `.agent/STATE.md` and `.agent/AGENT_REPORT.md`.
- No code, config, dependency, checkpoint, artifact or scheduler state changed.
- Evidence commit: `TO_BE_FILLED_AFTER_COMMIT`
- Push target: `origin/agent-work`
- Final worktree/push verification: `TO_BE_FILLED_AFTER_PUSH`

TASK_COMPLETE
