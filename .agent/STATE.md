# Current Project State

Updated: 2026-08-24T10:12:20Z

## Research lines

1. Pure-PPO DMLP1024 remains a separate control line and was not changed or
   reinterpreted by this task.
2. The PPG/curvature line now includes the completed strict five-seed
   CaveFlyer 1M low-Fisher guard gate
   `PROCGEN-JOINT-LOWFISHER-CAVE-5SEED-GATE-20260818-04`, while preserving the
   complete provenance map from `PROCGEN-JOINT-PROVENANCE-MAP-20260817-03`.

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
  `2026-08-24T10:12:20Z`: no target-array queue row and no live target Procgen
  trainer. Unrelated owned multicore job `19051570` is running and was not
  changed.
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

- The current task evidence package is
  `.agent/reports/PROCGEN-JOINT-LOWFISHER-CAVE-5SEED-GATE-20260818-04.md`.
- The complete evidence package is
  `.agent/reports/PROCGEN-JOINT-PROVENANCE-MAP-20260817-03.md`.
- Scientific evidence is sufficient to identify a completed strict causal
  control, but insufficient for any four-environment 6M x seeds 0,1,2
  promotion: all mapped candidates lack that formal budget/seed evidence.
- Only the ChatGPT Planner may issue the next bounded Procgen task.
