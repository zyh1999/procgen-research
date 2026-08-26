# Task-ID: PROCGEN-FULL-SHARED-JOINT2B-PPO500K-RAT-SCHEDULER-6M-S0-20260826-50

Status: RUNNING_ON_BEDE

Method: `FULL_SHARED_JOINT2B_PPO500K_RAT_ROLLOUT_SCHED_V1`.

Parent Task49 implementation is
`e0dc2e5ca4efd85419e974e42561eea11145c96f`; frozen parent trainer/config are
`4403ef006f53e8647adbcdb829a442037384f623e66eb69573843f21064db28a` and
`e26f66a616b1d0314561a645ef26111da1b15988aad1391d1ef64b6a146d8135`.

Preserve standard Procgen PPO with independent Adam through transition
`503,808`, then the same full-shared strict deterministic Joint-2B network,
actor empirical-Fisher rows, full-network critic Jacobian, every natural cross
block, damping/FP64/RHS/reconstruction, PopArt/GAE, rollout 4096, minibatch
512, four epochs, momentum/history, global clip, reward/evaluation/checkpoint,
four environments, seed0 and intended 6M horizon.

The only scientific difference is rollout-level Joint LR scheduling. At the
single phase switch create a clean Joint SGD path at LR `.004`. Freeze behavior
at each Joint rollout start; hold one LR constant through every minibatch of
all four epochs; then compute exact full-class categorical mean
`KL(pi_behavior || pi_final)` on the frozen rollout observations and update LR
once for the next rollout: divide by 1.5 above `.04`, multiply by 1.5 below
`.005`, otherwise unchanged, bounded to `[1e-4,.5]`. No other signal, rollback,
line search, warmup, sweep or scientific change is permitted.

The sole Bede gate `1075026` passed model/device, one PPO
update, one switch, Joint LR `.004`, one complete constant-LR Joint rollout,
post-rollout scheduler updates and finite strict cross-preserving solves.
Exactly four fresh Bede seed0 intended-6M cells were submitted once, one V100
each: BigFish `1075028`, BossFight `1075029`, CaveFlyer `1075030`, CoinRun
`1075031`. Task49 jobs `1074926-1074929` and every historical/unrelated job/root
remain untouched. Never retry, requeue or resubmit.

Update the existing sole automation `monitor-procgen-task49-ppo-warmup` in
place after Task50 IDs exist to monitor both frozen Task49 and Task50 sets at
20-minute cadence. Exact Paper comparison is only same env/seed0/evaluation
at first common >=2M, >=4M and 5,980,160; cancel only an individual exact
Target/Paper ratio below `.60`. Git model-free evidence only.
