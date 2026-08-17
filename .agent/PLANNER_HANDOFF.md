# Procgen Planner Handoff

## Delivery identity and audit boundary

- Task: `PROCGEN-PLANNER-HANDOFF-20260817-02`
- Evidence commit verified: `c9099117a1f62af35dc7ff430c9908503a849491`
- Prior Delivery HEAD verified: `62371cb789c98e814b767c0f2188155df7eaa433`
- Assigned-task commit inspected: `be356556d7d273253f831197af2fafb7f5244404`
- Incremental inspection window: `2026-08-17T13:52:54Z` to `2026-08-17T13:56:41Z`
- Scope: Procgen only, read-only experiment inspection. No experiment was
  selected, launched, resumed, restarted, cancelled, requeued, or early-stopped.
  No Jupyter session was started or used. `.54` and quarantined
  `ws4090-31` / `10.49.7.54` were not accessed.

The prior evidence commit and delivery commit are consecutive ancestors of the
assigned-task commit. The delivery commit changes only the prior report's
commit/push metadata; it does not change scientific evidence.

## The two established Procgen research lines

### Line 1: Pure-PPO DMLP1024 architecture control

Objective: determine whether a wider shared decision trunk improves the clean
PPO control before attributing gains to PPG or curvature. Boundary: unchanged
IMPALA/ResNet visual encoder, shared `256 -> 1024 -> 256` ReLU decision MLP,
linear policy head and PopArt critic head; pure PPO only, with no PPG auxiliary
head or auxiliary phase.

Highest strict baseline: Bede jobs `1070573-1070576`, BigFish, BossFight,
CaveFlyer and CoinRun, seeds `0,1,2`, 6M nominal transitions per child. All 12
children reached the last complete update at `5,980,160` transitions and are
freshly verified `PASS`, `rc=0`, with terminal checkpoints. This is the strict
baseline for the pure-PPO DMLP1024 line; older PPO results with another network,
seed set, budget, or reward convention must not be mixed into it.

### Line 2: PPG/curvature line

Objective: compare the matched official-schedule PPG actor ablations, shared
Exact-GGN/RAT, and strict joint actor-critic curvature while preserving causal
identity. Boundary: this line is not pure PPO, and its subfamilies are not
interchangeable:

- `E_v2`: official-schedule matched PPG baseline.
- `ACTOR_G`: entropy natural-gradient ablation.
- `ACTOR_H`: policy-KL/Fisher-clip ablation.
- `ACTOR_I`: one-actor-epoch ablation in the historical completed matrix.
- `ACTOR_J`: entropy-NG plus one actor epoch plus policy Fisher/KL clip.
- `ACTOR_K`: separate Exact-RAT adaptive-KL method; it must not be renamed as
  or substituted for `ACTOR_I`.
- P1 shared Exact-GGN/shared-RAT: deterministic critic GGN with the recorded
  symmetric-FP64/Jacobi semantics.
- Joint-2B/Joint-B: actor Fisher, critic GGN, and actor-critic cross blocks in
  one strict parent system; Joint-B is a defined reduction of that parent, not
  a policy-only substitute.

Highest strict baselines: `E_v2` is the actor-ablation baseline. The widened
DMLP1024 study has same-architecture `E_v2` seed 0 in all four environments.
Historical BigFish official-schedule E/G/H/I/J/K seed matrices are complete but
must retain their original architecture and schedule identity. P1 and each
Joint causal gate require their own exactly matched parent/control; neither the
pure-PPO baseline nor an unmatched job-name similarity proves a strict match.

## Strict match fields and metric semantics

A strict comparison must match or explicitly preserve:

| Field | Required identity |
|---|---|
| Environment | Exact Procgen ID, currently `bigfish-easy-0-10`, `bossfight-easy-0-10`, `caveflyer-easy-0-10`, or `coinrun-easy-0-10`; rewards never compare across environments |
| Algorithm | Pure PPO, E-v2, each actor ablation, shared Exact-GGN/RAT, strict Joint-2B, or the named Joint-B reduction; no silent family substitution |
| Network | IMPALA/ResNet encoder; exact decision trunk and head/PopArt/auxiliary identities; DMLP1024 PPO has 1,464,547 active parameters, while the recorded PPG DMLP1024 implementation has 1,464,804 |
| Training | Exact source/trainer/config hashes; rollout 4096 where specified; minibatch 512 and four epochs for the current Joint gate; formal budget and terminal-update convention |
| Seed | Same seed; no mixed-seed aggregate presented as a causal pair |
| Evaluation | Same online episodic-return logging/window, progress point, environment distribution and checkpoint/evaluation protocol |
| Reward | `eprewmean` is the terminal logged episodic-return mean at the last complete update, not a separate fixed-policy evaluation unless explicitly stated |
| KL | Pure PPO table uses the terminal `kl` column from `progress.csv`; Joint table separately reports `behavior_kl_after_step` and `current_step_kl`; these are not interchangeable |
| Curvature | Actor/critic blocks, critic score and RHS, cross blocks, damping, solver/reduction, clipping, precision, momentum and Kaczmarz semantics |
| PPG health | Reward plus auxiliary EV/MSE, clone KL, entropy, clip scale and solver residual; finite solve alone does not excuse a collapsed auxiliary head |

## Fresh scheduler, GPU, process and log state

### CSF3 control plane

Snapshot: `2026-08-17T13:52:54Z`,
`login1.csf3.man.alces.network`, read-only `squeue`, `sacct`, owned-process,
GPU, log and artifact queries.

| Resource | State, node, runtime, exit |
|---|---|
| `18642230_0-3` / raw cells `18642271,18642272,18642445,18642230`, `pg-j2b-acguardA` | `PENDING (JobHeldUser)`, no node, `00:00:00`, unknown start/end; scheduler displays `0:0` although the cells never ran; classify `queued/quota-waiting`, not scientific success or algorithm failure |
| `18624888_0-3`, `pg-j2b-block05` | `PENDING (JobHeldUser)`, no node, `00:00:00`, unknown start/end; scheduler displays `0:0` although unstarted; classify `queued/quota-waiting` |
| `18670696_0` / raw `18670697` | `COMPLETED`, node847, `00:57:45`, `0:0` |
| `18670696_1` / raw `18670698` | `COMPLETED`, node847, `00:57:34`, `0:0` |
| `18670696_2` / raw `18671119` | `COMPLETED`, node847, `00:57:47`, `0:0` |
| `18670696_3` / raw `18670696` | `COMPLETED`, node847, `00:57:41`, `0:0` |

No Procgen Slurm GPU job is running. The login host exposes an NVIDIA A2 at
`116/15356 MiB`, 0% utilization, but it is control-host telemetry and is not
launch capacity. No owned Procgen trainer exists. The owned-process scan only
found the audit command and an old controller-side Jupyter watchdog; it is not a
Procgen trainer and this audit did not attach to or use it. No scheduler PID can
therefore be mapped to a live Procgen GPU process.

The four 1M metric traces and logs have not changed since their terminal times
on 2026-08-16. Each has `status=PASS`, `rc=0`; the traces are 47.55-47.61 MB.
The most recent valid progress is 1,007,616 transitions. Targeted scans found
zero Traceback, OOM, CUDA error, RuntimeError, assertion, segmentation, disk
quota, NCCL/communication, configuration or nonfinite hits, and zero explicit
NaN/Inf metric tokens. No checkpoint exists for this launcher; terminal status,
return code, trace and scheduler accounting agree, so this is a documented
launcher property rather than evidence of a silent crash.

### Bede baseline

Snapshot: `2026-08-17T13:56:21Z-13:56:41Z`,
`login1.bede.dur.ac.uk`, read-only scheduler and fixed-root inspection.

| Job | Environment | Scheduler evidence |
|---|---|---|
| `1070573` `pg-pd-bf` | BigFish | `COMPLETED`, gpu031, `01:34:15`, `0:0` |
| `1070574` `pg-pd-bo` | BossFight | `COMPLETED`, gpu031, `01:37:48`, `0:0` |
| `1070575` `pg-pd-cf` | CaveFlyer | `COMPLETED`, gpu029, `01:36:56`, `0:0` |
| `1070576` `pg-pd-cr` | CoinRun | `COMPLETED`, gpu031, `01:32:39`, `0:0` |

No current queue row exists for these terminal jobs. Every child status is
`PASS`, every `rc` is zero, and every progress file ends at 5,980,160. All 12
checkpoint files exist, each 5,869,545 bytes, with terminal mtimes aligned to
the job end. Targeted stdout/stderr scans found no hard-error files; progress
files contain no explicit NaN/Inf tokens. Historical GPU utilization and PIDs
cannot be refreshed for terminal jobs and are therefore
`unknown/insufficient-evidence`, not inferred from scheduler completion.

### Registered dual-5060 host

Snapshot: `2026-08-17T13:55:15Z`, `lab-ubuntu`, obtained through the fixed
forced-command read-only interface from CSF3.

| GPU | Model | Memory | Utilization | Procgen ownership |
|---:|---|---:|---:|---|
| 0 | NVIDIA GeForce RTX 5060 Ti | 33/16311 MiB | 0% | former worker 216986 is terminal, queue empty; no active owned PID exposed |
| 1 | NVIDIA GeForce RTX 5060 Ti | 15/16311 MiB | 0% | former worker 216989 is terminal, queue empty; no active owned PID exposed |

The interface reconfirmed the three recovery status files as `COMPLETED`,
return code 0. It exposes their result paths and terminal progress rows but not
fresh filesystem mtimes or an unrestricted process list; those fields are
`unknown/insufficient-evidence`. The host is launch-disabled through this
interface, so idle-looking telemetry is not authorization or usable capacity.

`procgen-3090` has no fresh direct channel; capacity, GPU processes and current
filesystem integrity there remain `unknown/insufficient-evidence`.
`ws4090-31` remains quarantined and was not accessed.

## Completed baseline metrics: pure PPO DMLP1024

All rows are Bede terminal `progress.csv` values at 5,980,160 transitions.
`KL` is the file's terminal PPO `kl` column. The fixed artifact root is
`/nobackup/projects/bdman37/yihe/procgen_ppo_dmlp1024_bede_20260812_v1/formal_4env_x3seed_6m_20260812_v1/<env>/seed<seed>/`;
under each child, `code/logs/shared.ppo.resnet.dmlp1024.../<run>/` contains
`progress.csv` and `model.ckpt`, while the child root contains status, rc,
stdout and stderr.

| Environment | Seed | Reward `eprewmean` | KL | Status/artifact |
|---|---:|---:|---:|---|
| BigFish | 0 | 1.53 | 0.0111638 | PASS, rc0, checkpoint present |
| BigFish | 1 | 2.29 | 0.00687489 | PASS, rc0, checkpoint present |
| BigFish | 2 | 2.74 | 0.00274392 | PASS, rc0, checkpoint present |
| BossFight | 0 | 1.62 | 0.0801854 | PASS, rc0, checkpoint present |
| BossFight | 1 | 1.12 | 8.01637e-06 | PASS, rc0, checkpoint present |
| BossFight | 2 | 0.11 | 0.0432441 | PASS, rc0, checkpoint present |
| CaveFlyer | 0 | 3.17 | 0.0406683 | PASS, rc0, checkpoint present |
| CaveFlyer | 1 | 3.60 | 0.0423678 | PASS, rc0, checkpoint present |
| CaveFlyer | 2 | 3.80 | 0.0385867 | PASS, rc0, checkpoint present |
| CoinRun | 0 | 7.50 | 0.0849548 | PASS, rc0, checkpoint present |
| CoinRun | 1 | 3.90 | 0.0213999 | PASS, rc0, checkpoint present |
| CoinRun | 2 | 7.20 | 0.154330 | PASS, rc0, checkpoint present |

Strict-baseline count: 3/3 seeds complete for each of four environments, 12/12
overall. This states completeness, not that every seed has equal quality.

## PPG/curvature configuration and seed inventory

### Actor-ablation and P1 inventory

| Configuration | Environment/seed | State | Reward, KL and core metrics | Evidence/classification |
|---|---|---|---|---|
| Historical official-schedule `E_v2`, G/H/I/J/K BigFish matrices | Recorded seed matrices | completed | Terminal per-seed metric table is not carried by the bounded evidence; `unknown/insufficient-evidence` | Preserve original artifacts and identities; do not relaunch as missing |
| DMLP1024 `E_v2` | four environments, seed0 | completed | Per-environment terminal reward/KL/aux metrics unavailable through the current bounded channel, except BossFight matched robust score 10.60; otherwise `unknown/insufficient-evidence` | Former procgen-3090 root `/root/procgen_ppg_dmlp1024_20260810_v1`; same-architecture baseline count is 1 seed/environment |
| DMLP1024 ACTOR_J original attempts | BigFish s0 at 4,505,600; CaveFlyer s0 at 4,423,680; CoinRun s0 at 4,464,640 | failed | Terminal reward/KL incomplete and not safely mapped | `infrastructure-failure`: host died, no resumable checkpoint; clean recoveries do not erase these roots |
| DMLP1024 ACTOR_J recovery | BigFish s0 | completed at 5,980,160 | Fixed interface exposes raw terminal row but the complete header/evaluation mapping is not exposed; reward/KL therefore `unknown/insufficient-evidence` | return code0; progress path `.../phasic_ef_ggn.ACTOR_J_DMLP1024_bigfish.../bigfish-easy-0-10.20260811-151143_0/progress.csv` |
| DMLP1024 ACTOR_J | BossFight s0 | `EARLY_STOPPED_FAILED` at 4,096,000 | robust sampled score 5.7933 versus matched E-v2 10.60; ratio 0.5465; terminal KL/aux table unavailable | value/algorithm candidate failure preserved; ledger-confirmed early stop, not an infrastructure failure and not a recovery target |
| DMLP1024 ACTOR_J recovery | CaveFlyer s0 | completed at 5,980,160 | raw terminal row exposed but semantic header unavailable; reward/KL `unknown/insufficient-evidence` | return code0; fixed recovery path retained |
| DMLP1024 ACTOR_J recovery | CoinRun s0 | completed at 5,980,160 | raw terminal row exposed but semantic header unavailable; reward/KL `unknown/insufficient-evidence` | return code0; fixed recovery path retained |
| P1 shared Exact-GGN/RAT symmetric-FP64/Jacobi | four environments, seed0 | completed | Per-seed reward/KL/solver table not present in bounded evidence: `unknown/insufficient-evidence` | Preserve exact config `adv_resnet_shared_exact_deterministic_ggn_symfp64.yaml` and provenance |
| P1 shared Exact-GGN/RAT symmetric-FP64/Jacobi | CoinRun, BigFish, BossFight seed1 near 5,529,600; CaveFlyer seed1 near 2,048,000 | failed | terminal metrics incomplete; no checkpoint | `infrastructure-failure`: procgen-3090 shutdown; stale RUNNING files are not live evidence; roots explicitly retired from automatic relaunch |
| E-v2 and ACTOR_J DMLP1024 | seeds1-2 | not started in the recorded formal extension | no metrics | No approved catalog/launcher; a candidate is not a submitted task |

The completed DMLP1024 ACTOR_J recoveries have one seed in three environments,
but zero seeds are proven by the bounded evidence to exceed their highest strict
matched E-v2 baseline because the matched terminal metric table is missing.
BossFight is explicitly below the 3/5 threshold at 0.5465 and remains the only
recorded `early-stop-candidate`/executed value stop in this inventory. No new
early stop was performed.

### Joint-B gate metrics

The current 1M identity is:

- Root: `/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1`
- Run root:
  `gate_1m_seed0_jupyter_rhsaligned_actorrelative_criticfloor05_v1/rhs_aligned_rank1_b/<env>/seed0/`
- Trainer: `train_shared_jointb_rhsaligned_deterministic.py`, SHA256
  `ff987e0dd5ca1f4c1bb9a91e3794991f5a848bdbfdadc0425d935a72acf3b501`
- Config: `adv_resnet_shared_jointb_rhsaligned_actorrelative_criticfloor05_1m.yaml`,
  SHA256 `d87a8f648c1c91ee0d260c64ab7dd59d12bb7f9e6b67b0ee0a135389e697fb40`
- Launcher SHA256:
  `78f253f240d4b6c7ee5586d8a1c51bfa5e1f8fa0487a6c3db4637da4644a0f01`
- Geometry: rollout 4096, minibatch 512, four epochs, seed0, nominal 1M;
  strict full joint, clean critic score, all critic parameters, full compressed
  cross terms, paired score-residual RHS, 1024 parent rows reduced to 512 by
  deterministic RHS-aligned rank-1 Galerkin, actor damping .003, critic damping
  .5, FP64, no momentum or Kaczmarz.

| Environment | Job/raw ID | Transitions | Reward | Behavior KL | Current-step KL | Joint solve residual | State |
|---|---|---:|---:|---:|---:|---:|---|
| BigFish | `18670696_0` / `18670697` | 1,007,616 | 3.68 | 0.00368239 | 4.22625e-05 | 3.36381e-13 | completed-valid |
| BossFight | `18670696_1` / `18670698` | 1,007,616 | 0.36 | 0.00266445 | 2.11767e-05 | 3.71371e-13 | completed-valid |
| CaveFlyer | `18670696_2` / `18671119` | 1,007,616 | 2.78 | 0.00844484 | 2.75867e-05 | 5.90343e-13 | completed-valid |
| CoinRun | `18670696_3` / `18670696` | 1,007,616 | 6.90 | 0.00600422 | 4.77418e-05 | 5.05300e-13 | completed-valid |

Each run directory contains `preflight`, `command.txt`, `metric_trace.jsonl`,
`stdout`, `stderr`, `rc`, and `status`; none contains a checkpoint. Exact trace
sizes and mtimes are recorded in the refresh evidence, and no artifact changed
after scheduler end.

The preceding 500k seed0 gate `18670437_0-3` is scheduler- and artifact-complete
at 507,904 transitions: terminal rewards were BigFish 2.97, BossFight 1.10,
CaveFlyer 4.60, and CoinRun approximately 6.3; reduced residuals were finite
and no fatal error was recorded. Its exact terminal KL table is not in the
bounded handoff evidence and remains `unknown/insufficient-evidence`. A gate
PASS is a structural/execution result, not a 6M performance win.

Strict-baseline count for both 500k and 1M gates: zero seeds proven to exceed a
fully mapped highest strict matched baseline. Only seed0 exists, budgets are
gates rather than 6M formal runs, and exact matched parent results are not
mapped. The 3/5 value rule therefore cannot be applied.

### Other queued, terminal and unmapped Joint studies

- Held arrays `18642230_0-3` and `18624888_0-3`: queued/unstarted duplicate
  guards; exact environment/config/seed/root mapping is missing.
- Recent CSF3 arrays `18666610`, `18667225`, `18667467`, `18667627`,
  `18667792`, `18667941`, `18668461`, `18669377`, `18669429`, `18669454`,
  `18669530`, `18669613`, `18669615`, `18669725`, `18670437`, `18670696`,
  and `18672560` have terminal scheduler rows under PAP/FADP/J2B/RAT/Schur/RHS
  names. Except the directly inspected 500k/1M gates above, exact cells,
  seeds, source/config and terminal scientific artifacts are
  `unknown/insufficient-evidence`; scheduler completion alone is not promoted
  to scientific completion.
- `18666591` `pg-j2b-papklg` is scheduler-cancelled at zero runtime; preserve it
  separately from related `18666610` cells. Reason/identity remains unknown.
- Bede `1072327`, `1072329`, `1072331`, `1072333`, `1072337`, `1072338`, and
  `1072342-1072351` are recorded Schur/RHS/RAT/J2B attempts with mixed
  COMPLETE/FAILED/CANCELLED scheduler states. Exact cells and scientific
  artifacts are not mapped, so every such attempt remains
  `unknown/insufficient-evidence` rather than being merged into a valid result.
- No strict five-seed Joint formal configuration is currently evidenced as
  running or queued. It is `not-started`, not missing completion.

## Failure ledger and classifications

| Evidence | Classification | Reason that must be preserved |
|---|---|---|
| ACTOR_J DMLP1024 BossFight seed0 at 4,096,000 | `algorithm-failure` / `EARLY_STOPPED_FAILED` | robust 5.7933 versus strict E-v2 10.60, ratio 0.5465 below 3/5; ledger-confirmed historical early stop |
| Original ACTOR_J BigFish/CaveFlyer/CoinRun seed0 attempts | `infrastructure-failure` | procgen-3090 died before budget; no checkpoint; later clean recoveries do not overwrite provenance |
| P1 shared Exact-GGN/RAT four seed1 roots | `infrastructure-failure` | host shutdown; three near 5.53M and CaveFlyer near 2.05M; no resumable checkpoints |
| Held arrays | `queued/quota-waiting` | `JobHeldUser`, zero runtime and no node; not algorithm or numerical evidence |
| `18666591` and unmapped recent CSF3/Bede studies | `unknown/insufficient-evidence` | cancelled/terminal scheduler rows lack exact scientific identity/artifacts |
| Current 1M Joint-B gate | no numerical failure | finite reward/KLs and residuals, PASS/rc0, zero targeted hard-error or nonfinite hits |

No new `numerical-failure` is evidenced in this refresh. Historical failures
were not deleted, overwritten, weakened, or relabeled as successful recoveries.

## Artifact integrity and provenance summary

- Repository control inputs were read completely. SHA256 values recorded by
  the prior audit remain: GOAL
  `d0fb2942ace1b0736902011f3b836a1a4790c11ba7af70e3432e6efd7aa4fe9d`,
  pre-assignment STATE
  `85496971599d0f2e5c50635a8bec0739c58e7fe1a5ea9caebe641744fa412232`,
  prior TASK `cfd19511d3f4efa73dbd67957579c3aef8acd513306577d8353289e004621ec6`,
  and pre-audit AGENT_REPORT
  `9da64662aa33af61187abdde591eadff1b059467f1f094ed55616fd0daf8ccbf`.
- Pure-PPO trainer/config/launcher/submitter SHA256 identities are
  `989ea7f7607261872f753a8b4630eeb24b436ca01b668ee57f7e69e18ced90e5`,
  `35a7ac93189f7174b317040746556f3e3689e1c666527bfd062650fa1240a26b`,
  `65947f5fd90e8f91fb7d5897309f375b9214b4ff8b8b94aefdf76da76e0ae0be`,
  and `d7bc93b636536a1043c923cd2172faa8adbaffdb5f0defc2452f26979f9a0ccd`.
- Dual-5060 complete source-bundle SHA256 is
  `cb9d908997aadfa5d98b5bc7a14808b27f925407704416c784ee06e539ad578b`;
  it supersedes the initial incomplete archive
  `7c4b0f196268d5c55db766fc9137315ba4c76c3b56148d7bfc09802881b9562a`.
- No newly missing or damaged artifact was found in the directly inspected
  Bede or 1M Joint-B roots. Unreachable procgen-3090 files and unmapped recent
  studies remain unknown rather than presumed intact.

## Changes since evidence commit `c9099117...`

1. No scheduler state change for the two held arrays or the four completed 1M
   cells. No new Procgen GPU training job appeared on CSF3.
2. Bede changed from “not live-refreshed/authentication unavailable” to fresh
   verified evidence: all four jobs remain `COMPLETED/0:0`; all 12 children are
   PASS/rc0 at 5,980,160 with complete 5.87 MB checkpoints and recoverable
   terminal reward/KL rows; no targeted hard errors or NaN/Inf tokens.
3. Dual-5060 state is unchanged: three ACTOR_J recoveries and both workers are
   terminal, queues empty, GPUs 0% at 33/15 MiB, no active owned Procgen PID
   exposed.
4. The 1M Joint-B artifact roots remain unchanged and internally consistent.
5. No experiment, artifact, scheduler state, code, config, dependency or
   checkpoint was modified.

## Missing evidence, contradictions and blockers

- Exact per-cell mapping for the held arrays and most recent PAP/FADP/J2B/RAT/
  Schur studies is absent. Scheduler names cannot supply source/config/seed or
  scientific status.
- DMLP1024 E-v2 and ACTOR_J terminal metric tables, auxiliary diagnostics,
  precise log mtimes and checkpoint integrity on procgen-3090 are unavailable.
- Historical BigFish G/H/I/J/K and P1 seed0 per-seed terminal tables are not in
  the bounded evidence package.
- The dual-5060 forced interface exposes terminal status and raw progress tails
  but not a complete semantic header, fresh log mtimes, unrestricted process
  list, or checkpoint inventory.
- The older overview said the 1M gate was running/pending; evidence commit
  `c9099117...` and this refresh resolve it as four completed-valid cells.
- The controller's generic board still calls most recent scheduler-complete
  studies scientifically unknown. Direct root inspection resolves only the
  named 500k/1M gates and does not justify upgrading the others.
- Approved-task catalog is header-only and approved-launcher registry empty.
  No experiment is executable from this handoff.
- A candidate next direction is not authority. Only the Planner may issue one
  new bounded Procgen task with `Status: READY`.

## Planner request

Please provide exactly one next bounded Procgen task. It must preserve the two
research-line boundaries, strict match fields, historical failures,
nonoverwriting roots, exact source/config/artifact provenance, no-Jupyter and
quarantine restrictions, and must not treat idle capacity or scheduler
completion as scientific proof.
