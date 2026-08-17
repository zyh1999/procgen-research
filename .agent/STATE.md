# Current Project State

Updated: 2026-08-17T13:37:08Z

## Research lines

1. Pure-PPO DMLP1024: IMPALA/ResNet encoder, shared `256 -> 1024 -> 256`
   decision trunk, PopArt critic, no PPG auxiliary phase. Bede jobs
   `1070573-1070576` are the recorded four-environment, three-seed control.
2. PPG/curvature line: matched `E_v2` and ACTOR_G/H/J/K actor ablations,
   followed by shared Exact-GGN/RAT and strict Joint-2B/Joint-B causal
   diagnostics. `E_v2` is the recorded actor-ablation baseline; exact
   cross-host source/config hashes remain a proof requirement.

## Fresh live state

- Control plane: CSF3 `login1.csf3.man.alces.network`, sampled
  `2026-08-17T13:25Z-13:37Z`.
- No Procgen GPU training job is running on CSF3.
- Arrays `18642230_0-3` (`pg-j2b-acguardA`) and `18624888_0-3`
  (`pg-j2b-block05`) are unstarted `PENDING (JobHeldUser)` duplicate guards.
- The former 1M Joint-B RHS-aligned array `18670696_0-3` is complete at both
  scheduler and scientific-log levels: all four cells exited `0:0`, contain
  `PASS`, and reached 1,007,616 transitions.
- Terminal seed-0 rewards for that 1M gate are BigFish 3.68, BossFight 0.36,
  CaveFlyer 2.78, and CoinRun 6.90. Terminal behavior KL values are 0.00368,
  0.00266, 0.00844, and 0.00600 respectively; joint solve residuals are
  `3.36e-13` to `5.90e-13`.
- Dual-5060 host `47.114.81.212:60023` is reachable. Both RTX 5060 Ti GPUs
  sampled idle (33/16311 MiB and 15/16311 MiB, 0%); no Procgen process exists.
- Bede is not directly reachable noninteractively in this cycle. Its recorded
  pure-PPO results are historical verified evidence, not a fresh resource
  sample.
- `procgen-3090` has no fresh direct telemetry. Do not infer capacity.
- `ws4090-31` / `10.49.7.54` remains quarantined and is zero capacity.

## Failure and early-stop preservation

- ACTOR_J BossFight seed 0 remains `EARLY_STOPPED_FAILED` at 4,096,000
  transitions (recorded robust ratio 0.5465). Do not relaunch automatically.
- The three original ACTOR_J infrastructure-interrupted attempts and the
  P1 shared Exact-GGN/RAT seed-1 interrupted roots remain failed provenance.
- No new early stop was performed. The current 1M gate has only seed 0 and
  cannot support a strict five-seed 3/5 decision.

## Planner blockers

- Exact per-cell identities for the two held arrays and several historical
  recent CSF3 studies are still unmapped.
- The approved-task catalog is header-only and the launcher registry is empty;
  therefore the old CSF3 controller can propose work but cannot submit it.
- A new task must preserve nonoverwriting roots, exact trainer/config hashes,
  environment, architecture, rollout 4096, minibatch 512, four epochs, budget,
  seed and evaluation semantics.
