# Current Project State

Updated: 2026-08-24T23:04:27+01:00

## Current structural-manifest recovery for the three missing hybrid-head cells

- Task `PROCGEN-HYBRID-HEAD-STRUCTURAL-MANIFEST-RECOVERY-6M-MISSING3-20260824-12`
  terminates with unique conclusion `PRECHECK_BLOCKED`; no scientific cell was
  submitted and no Task 11 root was changed.
- Assignment `05fe72ba8d13217217a3039990cdba2ec5432279` was safely synchronized.
  The authorized preflight-only freeze
  `570cca72136a8a8dc1972d0eadee7167d236f93a` splits deterministic structure
  from environment-specific connectivity. Trainer/config/scientific-launcher/
  stage-monitor SHA256 values remain exactly
  `7bcf9bb6...`, `9497be42...`, `ae7104e7...`, and `536b8720...`.
- Exactly four no-training gpuH validations were submitted once:
  BigFish `19232320`, BossFight `19232321`, CaveFlyer `19232322`, CoinRun
  `19232323`. All completed `0:0` on nodes820/822/820/823. Each independently
  passed the canonical production model, optimizer/PopArt, actor/shared-critic,
  one-step policy/logit/shared-delta, head-only difference, memory, hard-error,
  and FP64/Jacobi/Cholesky checks.
- All four `structural_manifest.json` files are byte-identical with SHA256
  `3f91f5c313480c089d300a7dd5aff4a664f28ff9d9f4718bbab430200298d623`.
  Counts are total29 tensors/938,979 elements, trainable26/938,976,
  policy2/3,855, shared22/934,864, critic2/257, and PopArt state3/3. Critic
  names are exactly `last_v_layer.weight` and `last_v_layer.bias`.
- Connectivity probes independently pass with SHA256 BigFish `7e475693...`,
  BossFight `f54549e6...`, CaveFlyer `76759122...`, CoinRun `2558c07c...`.
  Every critic-head policy Jacobian is disconnected/exact-zero, every value
  path is connected/finite, partitions match structure, and no NaN/Inf or
  fallback is present.
- The final mandatory non-overwrite/launchability gate is blocked by a control
  contradiction: the frozen scientific launcher SHA `ae7104e7...` hard-codes
  `CAMPAIGN=/scratch/..._20260824_08` and
  `ROOT=$CAMPAIGN/runs/$METHOD/$ENV_NAME/seed0/6m`, exposes no root override,
  and all four such Task 11 roots exist. Task 12 simultaneously forbids
  changing that launcher and forbids overwriting/moving any Task 11 root.
  Launching the requested missing cells would require an unauthorized launcher
  change or mutation of immutable provenance, so none was submitted.
- Model-free validation evidence is under
  `remote_launch_staging/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/evidence_task12_preflight/`.
  No model/checkpoint, retry, requeue, Jupyter, quarantined access, duplicate,
  sweep, Paper rerun, or unrelated mutation occurred.

## Current hybrid-head trainable-gradient recovery and 6M seed0 matrix

- Task `PROCGEN-HYBRID-HEAD-TRAINABLE-GRAD-PREFLIGHT-AND-6M-S0-20260824-11`
  terminates with unique conclusion `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`.
- Allowed harness-only fix commit
  `26b2252527076df4bfe537a8612446317cbdcf3a` uses the ordered 26-tensor,
  938,976-element `requires_grad=True` parameter set for gradients, directions
  and one-step updates. It proves item-by-item production-update identity and
  separately preserves/audits non-trainable PopArt state. Frozen scientific
  trainer/config/launcher/monitor hashes remained unchanged.
- The only authorized full preflight, job `19227905`, completed `0:0` in
  `00:02:02` on node822 with durable `PRECHECK_PASS`. Canonical config/model,
  exact partition/manifest, policy-Jacobian zero, actor/shared-critic and
  one-step equivalences, H200 memory, finite FP64/Jacobi/Cholesky and hard-error
  checks all passed. Evidence was pushed before launch at
  `dcfd7b08e1827de1cb23dec0241149dd30632d79`.
- BigFish job `19228676` passed its frozen per-job compatibility gate and
  started science on node820. At exact2,007,040 its reward was6.53 versus
  Paper9.28, ratio `.7036637931`, so the frozen monitor recorded PASS. At
  exact4,014,080 it was6.23 versus Paper13.28, ratio `.4691265060`; frozen
  monitor rc3 applied the required scientific early stop. Scheduler state
  `CANCELLED by 778916`, elapsed `00:44:52`, is authoritative over stale root
  RUNNING/absent-rc markers. Exact-row telemetry remained finite and hard-error
  scan count was zero.
- BossFight `19228677`, CaveFlyer `19228678`, and CoinRun `19228679` never
  reached `scientific_started.marker`. They are terminal `FAILED/70:0` after
  24/31/31 seconds because the immutable per-job preflight asserted the
  BigFish full-file partition-manifest SHA against environment-sensitive
  connectivity serialization. Preserve these as
  `infrastructure-failure/per-job-preflight-design`; they contain no scientific
  transition, progress, trace, checkpoint or model and were not retried.
- With three cells lacking scientific evidence, the candidate cannot satisfy
  promotion. One BigFish algorithm early stop alone does not satisfy the
  two-environment rejection criterion, so the bounded result is infrastructure
  inconclusive rather than candidate rejection.
- Complete model-free evidence is under
  `remote_launch_staging/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/evidence_task11/`.
  It includes scheduler accounting, exact stage ledgers/adapters/hashes,
  immutable Paper provenance, progress, compressed traces/logs and all three
  pre-training failures; no checkpoint/model is included. No retry, requeue,
  Jupyter, quarantine access, duplicate, sweep, or unrelated mutation occurred.

## Current final hybrid-head assertion-fix recovery

- Task `PROCGEN-HYBRID-HEAD-ASSERTION-FIX-AND-6M-S0-20260824-10`
  terminates with unique conclusion `PRECHECK_BLOCKED`; no scientific cell was
  launched.
- Assertion-fix commit `a22f1a51bbcc953881e780f4dc00da16b2fc317f`
  changed only the canonical preflight harness and static invariant test. The
  exact frozen scientific trainer/config/launcher/monitor hashes did not
  change.
- Final authorized preflight `19225707` passed the complete exact production
  invariant, three-way resolved config identity, real production construction,
  exact manifest SHA, parameter partition, and critic-head zero policy
  Jacobian.
- It then FAILED/1:0 after17s on node820 in the next actual-network one-step
  test because the harness included non-trainable PopArt state in the
  `autograd.grad` input list. This is
  `infrastructure-failure/preflight-design`, not scientific, numerical,
  solver, config, partition/Jacobian, or H200 incompatibility evidence.
- One-step equality, production-scale memory, final head solve and complete
  error scan remained unreached. Task10 forbids field repair/retry after this
  one run, so no scientific job/root/process/artifact exists.

## Current canonical hybrid-head preflight recovery

- Task `PROCGEN-HYBRID-HEAD-CANONICAL-PREFLIGHT-AND-6M-S0-20260824-09`
  terminates with unique conclusion `PRECHECK_BLOCKED`; no scientific cell was
  launched.
- Canonical recovery commit `9a56a6aba6d70ce7a16b9e81cf105dccbd43d638`
  changes only the preflight harness/launcher and its test. Frozen scientific
  trainer/config/launcher/monitor hashes remain `7bcf9bb6...`, `9497be42...`,
  `ae7104e7...`, `536b8720...`.
- Recovery preflight `19225085` used the trainer's real `main()` config path,
  original `train_fn()`, and production `SharedActorCritic`. Three resolved
  configs were byte-identical. Actual partition proved 3,855 policy-exclusive,
  934,864 shared, and 257 critic-exclusive parameters; the value head was
  autograd-disconnected from policy logits with exact zero Jacobian probe.
- The job then FAILED/1:0 after 20s on node820 because the recovery harness
  carried a stale assertion that shared numel exceed 1,000,000. The frozen
  production network reports 934,864 shared and 938,979 total. This is an
  immutable `infrastructure-failure/preflight-design`, not scientific,
  numerical, partition, config, solver, or H200-incompatibility evidence.
- Actual-network one-step equivalence and production-scale memory/final solve
  checks remained unreached. The single Planner-authorized recovery was
  consumed, so no retry or four-cell launch occurred. Final queue/root/process
  reconciliation found no target job, run root, trainer, transition, artifact,
  checkpoint, or model.

## Current hybrid-head deterministic-GGN candidate

- Task `PROCGEN-PAPER-HYBRID-HEAD-DETGGN-6M-S0-20260824-08` terminates at its
  mandatory preflight gate with unique conclusion `PRECHECK_BLOCKED`.
- Frozen method `PAPER_HYBRID_SHARED_PAPER_HEAD_DETGGN_V1` starts from exact
  Paper RAT. It retains Paper actor and sampled critic direction on all shared
  trunk parameters, and changes only the 257-parameter critic-exclusive value
  head to deterministic normalized-residual J_v GGN lambda `.1`, independent
  head-only BxB symmetric FP64/Jacobi/Cholesky.
- Frozen commit `fe4b8a58812e80689705abec11364457cae31e26`; preserved
  infrastructure correction/evidence commit
  `896f54459b53f9f489951fb3c9f9ed5fec32c11e`. Trainer/config hashes are
  `7bcf9bb6...` / `9497be42...`.
- Static audit and CSF3 numerical regression passed: exhaustive disjoint
  parameter groups, exact zero-disconnected critic-head policy Jacobian,
  bit-identical actor matrix/RHS/direction, sampled shared-trunk critic,
  one-step policy parameters/logits, and only the value-head delta differs.
  Head solver relative residual was `2.616e-16`; historical formula
  distinctness passed.
- Persistent gpuH preflight `19220448` failed before compatibility testing
  because the staged `utils` package was absent from Python's import path.
  Corrected preflight `19220752` imported the target but failed before the
  actual-network partition/memory proof because its constructor namespace
  omitted required `norm_obs`. Both are immutable
  `infrastructure-failure/preflight-design`, scheduler FAILED/1:0 in 15s on
  node820, not scientific or hardware-incompatibility results.
- At final reconciliation there was no live target job, scientific root,
  trainer process, transition, progress, trace, checkpoint, or model. The
  four-environment 6M matrix was not submitted. No further retry, Jupyter,
  quarantined access, duplicate, sweep, Paper rerun, or unrelated job mutation
  occurred.
- The next action requires a Planner-authored READY task or explicit decision;
  the Executor must not silently repair/relaunch this blocked candidate.

## Current separate-B deterministic-GGN 6M candidate

- Task `PROCGEN-PAPER-SEPARATEB-DETGGN-6M-S0-20260824-07` has the unique
  conclusion `CANDIDATE_NOT_READY`.
- Frozen method `PAPER_MATCHED_SEPARATE_B_DET_GGN_V1` passed strict Paper
  actor-equivalence, independent deterministic critic-B identity, historical
  formula distinctness, numerical regression, and H200 compatibility. Frozen
  commit is `8a956130fe661aa41286a9b36ffe10965c223082`; trainer/config hashes are
  `b0dad110...` / `9497be42...`.
- Exact 2,007,040 Target/Paper seed0 reward ratios are BigFish `.3469827586`,
  BossFight `.0171232877`, and CaveFlyer `.1640449438`. Frozen monitor ledgers
  classified each `EARLY_STOPPED_ALGORITHM`; Slurm cancellations on node822
  are authoritative over stale root RUNNING markers and absent rc files.
- CoinRun passed exact stages at 2,007,040 (`1.8648648649`) and 4,014,080
  (`.8875`), then completed rc0 at 5,980,160 with target/Paper `6.40/9.40 =
  .6808510638`. It has terminal PASS, trace/progress/checkpoint, finite solver
  telemetry, and zero hard-error hits.
- Jobs were BigFish `19210448`, BossFight `19210449`, CaveFlyer `19210450`,
  CoinRun `19210451`, all one H200 on node822. There was no retry, requeue,
  resubmission, Jupyter use, quarantined access, duplicate root, or unrelated
  job mutation.
- The complete evidence package, including immutable Paper seed0 baseline
  provenance/hashes, exact rows, telemetry, compressed logs/traces, stage
  ledgers, scheduler state, and error scans, is in
  `remote_launch_staging/procgen_paper_separateb_detggn_6m_s0_20260824_07/evidence_logs`.
  No checkpoint/model is included.
- The Executor must push this evidence and callback the same ChatGPT Planner.
  It must not invent a next algorithm or launch a sweep while awaiting exactly
  one Planner-authored READY task. Planner owns algorithm/code direction;
  Executor owns live resource placement and monitoring.

## Current Paper-matched deterministic-GGN gate

- Task `PROCGEN-PAPER-MATCHED-DETERMINISTIC-GGN-1M-GATE-20260824-06` has the
  unique conclusion `GATE_FAIL`.
- Machine audit and regression tests passed for the single frozen method
  `PAPER_MATCHED_DETERMINISTIC_GGN_V1`: Paper actor/network/schedule semantics
  are retained, while only deterministic critic J_v/residual, lambda `.1`,
  joint-2B symmetric FP64/Jacobi/Cholesky and telemetry are migrated.
- The explicit user resource-race override expanded the scientific execution
  to seeds0--7 on gpuH. BigFish `19203172`, BossFight `19203173`, and CaveFlyer
  `19203174` completed all 24 children PASS/rc0 at 1,007,616 with checkpoints.
  CoinRun `19203175` was later user-authorized early-stopped for scientific
  futility; scheduler state is CANCELLED after 58:17 on node821, while its
  eight stale RUNNING child markers and absent rc/copies are preserved.
- At the exact 983,040 progress row, seed0 Target/Paper reward ratios are
  BigFish `.2583`, BossFight `0`, and CaveFlyer `.2188`. All are below `.60`,
  so the required 3-of-4 gate is mathematically impossible regardless of
  CoinRun. Residuals remain finite near `1e-14--1e-12`, hard-error scans are
  clean, and the failure is algorithmic/step-calibration rather than solver or
  infrastructure failure.
- gpuL race loser array `19203054` remains
  `cancelled-race-loser-unstarted`: Start=None, elapsed0, no node or root.
  The gpuA and gpuL preflight infrastructure failures remain immutable.

## Research lines

1. Pure-PPO DMLP1024 remains a separate control line and was not changed or
   reinterpreted by this task.
2. The PPG/curvature line now includes the completed strict five-seed
   CaveFlyer 1M low-Fisher guard gate
   `PROCGEN-JOINT-LOWFISHER-CAVE-5SEED-GATE-20260818-04`, while preserving the
   complete provenance map from `PROCGEN-JOINT-PROVENANCE-MAP-20260817-03`.

## Current formal-comparison precheck

- Task `PROCGEN-BXB-GGN-VS-PAPER-RAT-FORMAL-6M-X3-20260824-05` stopped at its
  mandatory identity gate. Unique status: `PRECHECK_BLOCKED`.
- Original Paper RAT was recovered exactly as commit `2b5affd...`, trainer
  `cbcd6811...`, config `1ed4eab5...`, Bede array `1063880`. The requested
  four environments x seeds0--2 are all strict reusable PASS/rc0 completions
  at 5,980,160 with terminal checkpoints.
- Historical P1 candidate is trainer `2b50f8cc...`, config `c177ac09...`,
  wrapper `9c7806fc...`, deterministic critic GGN 2B with symmetric FP64/
  Jacobi. It differs from Paper RAT outside critic curvature/solver telemetry:
  initial LR `.004` vs `.5`, rollout-level vs minibatch-level adaptive KL,
  and momentum/history `0/disabled` vs `1e-6/enabled`.
- `procgen-3090` is currently unresolvable, so historical P1 seed0 artifacts
  cannot be freshly upgraded to strict reuse. Seed1 failures remain
  infrastructure-interrupted and seed2 is absent.
- No formal cell was launched, and no new root, checkpoint, scheduler row, or
  Jupyter allocation was created.

## Current bounded conclusion

- Unique conclusion: `GUARD_NOT_HELPFUL`.
- Frozen arrays `18833574` (parent seeds1--4) and `18833575` (guard
  seeds1--4) completed all eight cells on gpuA with scheduler `COMPLETED/0:0`,
  artifact PASS/rc0, exact frozen hashes, clean error scans, and 1,007,616
  transitions.
- With historical seed0, guard reward wins/ties/losses are `1/3/1` and paired
  guard-minus-parent reward has mean `-0.0900`, median `0`, sample SD
  `0.3711`. Guard is below parent in only `1/5` seeds, so the `3/5`
  early-stop-candidate condition is not met.
- The guard strongly activates at the seed0 terminal row (`.594445`) and
  transiently in seeds2/3, but does not yield a reproducible benefit. This is
  only a 1M causal gate and authorizes no 6M extension.

## Preserved provenance conclusion

- Unique conclusion: `STRICT_PARENT_COMPLETE`.
- Target `18670696` is the seed-0, four-environment, 1M RHS-aligned Joint-B
  gate. All cells are scheduler-complete and scientifically complete at
  1,007,616 transitions with PASS/rc0.
- Completed successor/control `18672560` is a strict single-causal-ablation
  match. Environment, seed, architecture, rollout 4096, minibatch 512, four
  epochs, 1M budget/termination, data/reward/evaluation protocol, full Joint-B
  actor-Fisher/critic-GGN/cross/RHS semantics, float64 solver, momentum=0 and
  Kaczmarz=false are unchanged. The only scientific change is the predeclared
  low-Fisher actor-from-critic damping guard (high 0.50, low 0.20, max 0.05),
  plus its validation and telemetry.
- The guard was inactive in BigFish, BossFight and CoinRun, which reproduced
  target terminal metrics bit-for-bit. It activated in CaveFlyer (terminal
  fraction 0.594445; actor-from-critic floor 0.033778), where terminal reward
  was 2.06 versus 2.78 in the unguarded target. This is a completed causal
  control, not evidence of a performance improvement.
- The 250k/500k/1M gates are gates only. None is a 6M, multi-seed performance
  result, and no candidate is authorized for formal expansion by this state.

## Fresh live state

- CSF3 control plane `login2.csf3.man.alces.network`, refreshed at
  `2026-08-24T11:05:32Z`: no target-array queue row and no live target Procgen
  trainer. Unrelated owned multicore job `19051570` is running and was not
  changed.
- Bede refreshed at `2026-08-24T11:05:50Z`: owned queue empty and most
  V100-32GB nodes idle. Capacity was not used because the identity gate failed.
- Authorized `ws4090-92`, `ws4090-76`, and `procgen-3090` names were not DNS
  resolvable from this Executor; their current state remains unknown. No
  quarantined host was queried.
- Arrays `18833574/18833575` have eight terminal gpuA accounting rows. They
  ran on nodes852/854/855/863 for 56:59--1:00:33 and all report
  `COMPLETED/0:0`.
- Old arrays `18642230` and `18624888` were user-authorized cancellations at
  CSF3 local `2026-08-18 14:08`; every cell has Start=None, no node, elapsed
  00:00:00 and no scientific artifact. They are
  `cancelled-obsolete-unstarted`.
- `18666591` is likewise cancelled/unstarted at zero runtime and was replaced
  by completed gpuA array `18666610`.
- Bede accounting was refreshed at `2026-08-18T13:19:36Z`. Bounded jobs have
  been mapped to scientific artifacts or an explicit failure/cancellation.
  Numeric ID `1072347` resolves only to raw child `1072326_0` of an unrelated,
  out-of-scope job; no Procgen parent job `1072347` is evidenced.
- This bounded task submitted only its eight frozen CaveFlyer cells. During
  reconciliation no experiment was resumed, resubmitted, cancelled, released,
  requeued, or early-stopped. No Jupyter service was used. Quarantined `.54`,
  `ws4090-31`, and `10.49.7.54` were not accessed.

## Failure and cancellation preservation

- ACTOR_J BossFight seed0 remains `algorithm-failure/EARLY_STOPPED_FAILED`
  (5.7933 versus strict E-v2 10.60; ratio 0.5465).
- Original ACTOR_J BigFish/CaveFlyer/CoinRun attempts and P1 seed1 roots remain
  `infrastructure-failure`.
- Bede `1072329_0` failed before a trace because `utils` was absent;
  `1072331_0` failed before a trace with a V100 CUDA OOM. Retry `1072333`
  completed all four cells, but does not erase either failure.
- CSF3 PAP full-column job `18667792`, all other mapped recent CSF3 smoke
  arrays, and gates `18669725/18670437/18670696/18672560` are scientifically
  complete. RAT block-trace CoinRun has terminal behavior KL 0.212908 and is a
  health concern, not a strict-match control or a formal-performance result.

## Planner boundary

- The current precheck evidence package is
  `.agent/reports/PROCGEN-BXB-GGN-VS-PAPER-RAT-FORMAL-6M-X3-20260824-05.md`.
- The current task evidence package is
  `.agent/reports/PROCGEN-JOINT-LOWFISHER-CAVE-5SEED-GATE-20260818-04.md`.
- The complete evidence package is
  `.agent/reports/PROCGEN-JOINT-PROVENANCE-MAP-20260817-03.md`.
- Scientific evidence is sufficient to identify a completed strict causal
  control, but insufficient for any four-environment 6M x seeds 0,1,2
  promotion: all mapped candidates lack that formal budget/seed evidence.
- Only the ChatGPT Planner may issue the next bounded Procgen task.
