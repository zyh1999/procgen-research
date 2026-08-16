# Verified Supervisor Coverage

Updated by operator handoff on 2026-08-11. Future child work is an active duplicate only while its exact parent/root remains evidenced live; every controller cycle must revalidate this against `state/live_snapshot.txt`.

## Procgen

- The former root `/root/procgen_ppg_dmlp1024_20260810_v1` no longer owns live work. E_BIGFISH/E_BOSSFIGHT/E_CAVEFLYER/E_COINRUN are terminal complete. J_BIGFISH/J_CAVEFLYER/J_COINRUN are dead incomplete infrastructure failures; J_BOSSFIGHT is ledger-confirmed `EARLY_STOPPED_FAILED`. Preserve all eight records.
- The fixed recovery supervisor on `procgen-5060` is `/home/zzz/rlstack5060/campaigns/procgen_actorj_dmlp1024_recovery_20260811_v1`. GPU0 owns `J_DMLP1024_BIGFISH_S0` followed by `J_DMLP1024_COINRUN_S0`; GPU1 owns `J_DMLP1024_CAVEFLYER_S0`. Treat each queued child as an active duplicate while its exact worker/status/container evidence remains live. The status snapshot exposes the current Docker client as `owned_child`; the bounded stop gate accepts only that exact child PID, its registered worker parent, the fixed Procgen workspace, and one of these three container names. Stopping BigFish does not kill GPU0's serial worker, so CoinRun remains scheduled. This queue does not authorize any other seed, environment, or method.
- The root-local `launchers/watch_sweep.sh` only aggregates RUNNING/COMPLETED/FAILED state; it does not make research decisions, submit new work, or perform value early stopping. CSF3 remains the research controller.
- Original E_v2, ACTOR_G/H/I/J, and separate ACTOR_K BigFish seeds0-2 are terminal complete. Do not describe the actor-ablation matrix as G/H/J/K.
- P1 symmetric-FP64/Jacobi seed0 is terminal complete. P1 seed1 CoinRun/BigFish/BossFight/CaveFlyer are absent and infrastructure-interrupted, not complete; stale RUNNING files must not be treated as live coverage. They have no resumable checkpoints and are retired from automatic launch.

## MuJoCo

- CSF3 array `18302268` is terminal: elements0-9 and11-41 are complete, including element39, and element10 failed. It provides no future live coverage. Do not submit a duplicate array; identify failed element10 exactly before any replacement.
- ws4090-92 worker `1068627` owns the curv256 study root, a remaining bounded serial trial budget, and live Humanoid ktrue exact-kernel L2 clip0.5 actor/critic momentum0.5 trainers `1830478/1830479`, seeds0/1, on command device0; both are at update149 and 39,321,600 transitions. Its future children count as active duplicates. Former exact-FVP-Fisher clip0.5 momentum1e-6 seed0 trainer `1764438` is terminal at update499; seed1 trainer `1764439` is absent and permanently failed incomplete at update429. Former worker `1068628` is absent, so its future coverage is withdrawn. Ant ktrue FVP-Fisher clip1.0 momentum0.5 trainers `1720163/1720164`, Ant ktrue L2 momentum0.9 trainers `1700694/1700695`, Walker2d ktrue FVP-Fisher clip1.0 momentum0.5 trainers `1713894/1713895`, and Humanoid L2 clip0.5 momentum1e-6 trainers `1743692/1743693` are terminal at update499. Prior incomplete Walker2d trainers `1406135/1406136` remain absent. Former full-EF/full-GGN worker `1030969` is absent, so its future coverage remains withdrawn. The full-EF and curv256 identities remain separate.
- ws4090-76 former parent `1347159` is absent. Its coverage is withdrawn; a continuation requires a newly approved guarded launcher and terminal-root skip proof.

## Isaac

- Former CSF3 owners `18299181_0/_2`, `18300853_0/_1`, and `18302792_1` are cancelled and no longer provide live coverage. Their partial and terminal trial artifacts remain preserved.
- CSF3 true-FVP array `18303444` has completed elements0-3,9-10,14-26 and no longer owns live elements. Former exact srun roots `669595-669598` for retry controls are absent and their incomplete failures remain preserved; reconcile their identities against the terminal grid before replacement.
- Jupyter allocation `18229504` is terminal by time limit, while its exact Isaac Ant root `1397957` was previously verified complete. Do not extend it; the independent watchdog remains the sole reclamation authority.
- Jobs `18304359` and `18304360` are complete and no longer provide live coverage.
- ws4090-92 parent `1300544` and trial008 trainer `1410726` are absent. Trial008 is terminal at update500 with reward39.5, so preserve it as COMPLETE and withdraw all former future-trial coverage. Trials are not multiple seeds; any next worker requires fresh remaining-budget and non-duplication proof plus an approved launcher.
