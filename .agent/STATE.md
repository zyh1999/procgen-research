# Current Project State

Updated: 2026-08-16

## Established lines

- Pure PPO DMLP1024: IMPALA/ResNet encoder, shared `256 -> 1024 -> 256`
  decision trunk, PopArt critic, no PPG auxiliary phase.
- PPG/EF actor ablations: matched `E_v2` baseline and ACTOR_G/H/J/K identities.
- Shared Exact-GGN/RAT and strict Joint-2B/Joint-B causal diagnostics.

## Verified evidence

- Bede jobs `1070573-1070576` completed four environments by three seeds for
  the pure-PPO DMLP1024 control.
- Dual-5060 clean recoveries completed ACTOR_J seed 0 for BigFish, CaveFlyer,
  and CoinRun. BossFight remains an `EARLY_STOPPED_FAILED` record.
- CSF3 completed the four-environment 500k RHS-aligned Joint-B gate and had the
  matching 1M gate `18670696_0-3` running/queued at the latest handoff snapshot.

## Infrastructure

- CSF3 is the control plane; the persistent controller runs on login2 under
  screen `codex-three-domain`.
- Bede is ppc64le and Slurm-only.
- `procgen-3090` contains historical PPG/RAT roots; current capacity requires
  fresh verification.
- `ws4090-31` (`10.49.7.54`) is quarantined and is not capacity.

Live scheduler, process, log, and artifact state must be refreshed before any
operational conclusion. See `THREE_DOMAIN_EXPERIMENT_OVERVIEW_20260816.md` for
the dated cross-domain handoff.
