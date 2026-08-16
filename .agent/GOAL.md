# Long-Term Research Goal

Establish a reproducible, causally matched Procgen comparison of pure PPO,
PPG actor ablations, shared Exact-GGN/RAT, and joint actor-critic curvature
methods on BigFish, BossFight, CaveFlyer, and CoinRun.

The program should determine whether the DMLP1024 decision trunk and structured
actor/critic curvature improve sample efficiency or final return without
sacrificing numerical stability. Conclusions must use matched architecture,
rollout geometry, training budget, environment, and seeds, and must retain
failed and early-stopped configurations in the evidence record.

This file defines the long-term destination. Only the Planner may turn it into
a new high-level research objective in `TASK.md`.
