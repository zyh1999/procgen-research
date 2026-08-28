# Task63 deterministic Joint-2B post-inverse direction telemetry

## Status

`TERMINAL_TRAINING_COMPLETE_TELEMETRY_AGGREGATION_PARTIAL_FAILURE`

## Frozen parent and only diff

- Task06 commit: `da34ce7c7d964765f336ac02111c9fde95aed1ec`
- parent trainer: `41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a`
- parent config: `69d12937debb8ef8b4531e79b8f9613185b26e4e9056a51e67754129b269391d`
- Paper control commit: `2b5affd64cbb3c624b4bc1f4767f449df231ffb2`

Task63 retains Task06's complete 1024-row full-cross deterministic Joint-2B,
critic curvature coefficient `.1`, objective coefficient `1`, damping `.5`,
FP64 Jacobi/Cholesky, Paper per-minibatch KL LR controller, SGD momentum
`1e-6`, history correction, global clip `.5`, PopArt/GAE and update order.

The only runtime addition partitions the already-effective RHS by actor and
critic rows, solves the two telemetry RHS columns with the exact existing
factorization, reconstructs `d_A` and `d_C`, and records role-wise norm,
signed projection, cosine and cancellation statistics. The original
single-RHS `flat_dir` alone remains installed. Raw metric shares are labeled
separately from post-inverse direction shares.

## Frozen Task63 files

- trainer: `067a3c9fc6f309aa13cba9d54ab6c29252ed318c4398df8de01256e5c439415a`
- science config: `4e2d18ae97084cf3ffaabb601ee63472408b77bd63fbc82d76fe1c385701db77`
- gate config: `492aea6d416db1f8937dd586af2a5122b23c9ecd7efc6889b5b6fdfac9b946ff`
- aggregator: `f7ffb9a8c885f738acadd4421a6333b1baa8d2dae023ef2b5c00693476bd2f78`
- Bede gate wrapper: `53b2ff2b0c4a2a40f559c13d9856220f4f9e0ab304d5d09037caa0d950cfa81c`
- Bede science wrapper: `c8f2bdbbd89f874914768ec0e948fb8b3585e5e302b11e25a58e3b8a5f56b917`

Local Python compile, shell syntax and frozen scalar checks pass. The science
config value `2,000,000` intentionally follows the unchanged parent loop
convention and terminates at exact progress transition `2,007,040`.

## Placement precheck

Bede account `bdman37g`, user `yihe` and PPC64LE Procgen runtime remain
available. Nodes gpu015--gpu020 were idle at the bounded refresh. The intended
campaign
`/nobackup/projects/bdman37/yihe/procgen_deterministic_joint2b_actor_critic_direction_telemetry_2m_s0_20260828_63`
was absent, with no Task63 job/process/duplicate. Task62 jobs1078176--1078179
were RUNNING and are excluded from every Task63 action.

## Sole production gate

- Bede job: `1078180`
- scheduler: `COMPLETED/0:0`, elapsed `00:01:57`, node `gpu025`
- root: `gate/production`, `PRECHECK_PASS`, rc `0`
- validation: `TASK63_GATE_PASS`, one complete real minibatch record
- the gate verified 512 actor plus 512 critic rows, the original single-RHS
  installed update, RHS/alpha/direction reconstruction, exclusive structural
  zeros, finite FP64 solve, Cholesky info `0`, and first-update parameter
  identity.
- the only stderr text was the environment's benign Gym deprecation notice;
  no Traceback, OOM, CUDA, NCCL, disk/quota, nonfinite or solver hard error.

## Science launch

All four cells were submitted in one bounded action without dependency, hold,
throttle, retry, requeue or resubmit:

| Environment | Job | Initial state | Node | Root |
|---|---:|---|---|---|
| BigFish | `1078181` | RUNNING | `gpu025` | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_ACTOR_CRITIC_DIRECTION_TELEMETRY_ONLY_V1/bigfish-easy-0-10/seed0/2m` |
| BossFight | `1078182` | RUNNING | `gpu025` | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_ACTOR_CRITIC_DIRECTION_TELEMETRY_ONLY_V1/bossfight-easy-0-10/seed0/2m` |
| CaveFlyer | `1078183` | RUNNING | `gpu006` | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_ACTOR_CRITIC_DIRECTION_TELEMETRY_ONLY_V1/caveflyer-easy-0-10/seed0/2m` |
| CoinRun | `1078184` | RUNNING | `gpu007` | `runs/PAPER_MATCHED_DETERMINISTIC_GGN_ACTOR_CRITIC_DIRECTION_TELEMETRY_ONLY_V1/coinrun-easy-0-10/seed0/2m` |

Every root is fresh and has its own scheduler job, hostname, PID, GPU identity,
status, command and logs. At the initial verification BigFish and BossFight had
32 complete telemetry records each. Their latest records retained strict
1024-row deterministic Joint-2B, Cholesky info `0`, and finite applied-solve
residuals `7.576e-14` and `9.259e-14`. CaveFlyer and CoinRun had started and
formed roots/PIDs but had not yet completed their first synchronized artifact
copy. Focused hard-error scans were clean for all four; the word `inf` inside
the benign Gym migration URL is not a numerical match.

Task62 jobs `1078176`--`1078179` remained RUNNING and untouched throughout.
Current bounded conclusion: `SCIENCE_RUNNING`. Terminal aggregation and
read-only Paper/Task62 reward sanity comparison remain pending exact
`2,007,040` endpoints; Task63 never reward-early-stops.

## Partial terminal archive: BigFish and CoinRun

The bounded read-only refresh reconciled two natural scheduler-authoritative
terminals. BigFish `1078181` is `COMPLETED/0:0` after `02:44:51` on `gpu025`;
CoinRun `1078184` is `COMPLETED/0:0` after `02:45:51` on `gpu007`. Both roots
are `PASS/rc0`, contain exact transition `2,007,040`, 49 progress rows and
15,680 complete records. Endpoint rewards are BigFish `5.08` versus immutable
Paper `9.28`, and CoinRun `10.00` versus Paper `3.70`. This telemetry campaign
never reward-stops.

BigFish passed the frozen complete Early/Middle/Late/overall aggregation.
Overall post-inverse full actor norm/projection medians are `.44048/.37842`;
shared medians are `.44970/.39641`. The result is critic-dominant after the
inverse despite actor-heavy raw metric rows.

CoinRun training and solver evidence are clean, but the frozen aggregator
stopped at immutable record 3808 with `policy projection drift`. At that
record the policy total norm is only `5.153e-17`, actor raw scale is
`7.786e-32`, entropy is `3.681e-18`, and the policy signed-projection sum is
`.002648`; full/shared/value sums remain 1, Cholesky is 0, residual is
`8.811e-14`, reconstruction and structural-zero checks pass. This is a
telemetry-aggregation validator failure caused by an actor-saturated near-zero
policy subspace, not a training/solver/GPU/infrastructure failure. It was not
repaired or rerun.

Each terminal source log has one regular non-symlink checkpoint of 3,766,013
bytes, mode 664. Only stat metadata is archived. Complete model-free hashes
and failure evidence are in
`evidence/partial_terminal_bigfish_coinrun_20260828.md`. BossFight `1078182`
and CaveFlyer `1078183` remain RUNNING and untouched. Task63 therefore remains
`SCIENCE_RUNNING_PARTIAL_TERMINAL`.

## Final BossFight and CaveFlyer terminals

BossFight1078182 and CaveFlyer1078183 naturally completed `COMPLETED/0:0` on
gpu025/gpu006 after02:51:51/02:49:28. Both roots are PASS/rc0 at exact2,007,040
with15,680 records and endpoint rewards `.04/0` versus Paper `2.92/4.45`.
CaveFlyer passes the frozen aggregate; overall full actor norm/projection is
`.62032/.73689` and shared is `.63194/.75562`, rising further actor-heavy in
the late third (`.70749/.86634` full).

BossFight hits the same frozen validator edge class as CoinRun: at record5952
the policy direction is exactly zero after entropy falls to `3.39e-39`, so the
policy projection sum is0. Full/shared/value sums, Cholesky, residual,
reconstruction, structural zeros and finite scan remain healthy. It is not a
training or solver failure and was not repaired or rerun.

Task63 is fully scheduler-terminal, but only BigFish and CaveFlyer have valid
frozen complete aggregates. BossFight and CoinRun have clean scientific runs
with aggregation-validator failures in zero/near-zero policy subspaces. Final
classification is
`TERMINAL_TRAINING_COMPLETE_TELEMETRY_AGGREGATION_PARTIAL_FAILURE`; model-free
details are in `evidence/final_terminal_boss_cave_campaign_20260828.md`.
