# Task33 user matrix override: two seeds

- Original Planner matrix: four environments, seed0 only (four scientific cells).
- Explicit user override received 2026-08-25: run two seeds.
- Executed matrix: the same four environments, seeds 0 and 1 (eight scientific cells).
- Scientific method, trainer, config, preflight, stage monitor, intended 6M horizon, and Paper-matched early-stop semantics remain unchanged.
- Seed1 is an independent replicate. For seed1, Paper comparison is permitted only if an immutable original Paper RAT seed1 artifact exists for the same environment, evaluation semantics, and exact transition. Absence of that exact baseline means no performance cancellation for seed1.

The versioned multiseed launcher differs from the frozen seed0 launcher only by:

1. requiring `PROCGEN_SEED` and accepting exactly `0` or `1`;
2. routing artifacts to `seed${SEED}/6m` and recording `seed.txt`;
3. passing the selected seed to the unchanged trainer; and
4. selecting the matching seed-specific training log directory.

It does not change the trainer, algorithm, configuration, per-job compatibility preflight, optimizer, schedule, horizon, GPU request, or failure/requeue policy.

Versioned multiseed launcher SHA256: `15c067938d2d4d947f5bf3464f9a4c36d67308594d0e8b81b54e7fbfe0b6b8eb`.
