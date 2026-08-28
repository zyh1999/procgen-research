# Task64 CoinRun and campaign-final terminal evidence

Bounded read-only refresh: 2026-08-28 12:53Z. No scheduling or scientific mutation occurred.

## CoinRun terminal

- job `19531932`: scheduler-authoritative `COMPLETED/0:0`, elapsed `00:46:17`, node `node821`.
- root `PASS`, rc `0`, exact transition `2,007,040`, reward `0.00`, 49 data rows plus header, 15,680 complete trace records.
- checkpoint: regular non-symlink file, 3,766,013 bytes, mode 0644. Bytes and content hash were not read or committed.
- final solver: curvature `4`, objective `1`, strict rows `1024`, Cholesky info `0`, relative residual `2.222e-15`, RHS reconstruction `0`, alpha reconstruction `2.933e-20`, direction reconstruction `1.351e-8`, finite PASS.
- final direction is fully critic-dominant after actor saturation: full/shared/value critic norm and signed projection shares are `1`; actor raw scale is `3.607e-25` versus critic raw scale `153,972.125`; entropy is `2.767e-14`.
- hard-error scan: zero.

Model-free SHA256: progress `8ed74cdbef8b8cfa6ce03f9198a4ee0308c19dddb66d215920f0b1d9676d09b4`; trace `1e9e58ec32e84d0a90422b1a8b6ab30e6a0caf2c29f486527433748290107787`; stdout `2ba7d05a1f872ebfe99fee12172133779953708dd0ded3d4a9622484b80c7ac8`; stderr `0334fa063a9f46c8ac2a150361b321857138c3a69602c6a876bb895db42bf752`.

The frozen all-record aggregator stops at record `2272`, transition `294,912`, with `policy projection drift`: actor raw scale `2.751e-27`, policy total norm `7.960e-15`, actor signed projection `.984464`, critic policy contribution zero, and projection sum `.984464`. The same record has Cholesky `0`, residual `6.472e-13`, exact RHS reconstruction, alpha reconstruction `8.291e-16`, direction reconstruction `1.663e-11`, and finite PASS. This is an immutable telemetry-validator edge under near-zero policy direction, not a training, solver, GPU, or infrastructure failure. It was not repaired or rerun.

All four Task64 jobs completed naturally with PASS/rc0 exact-2M training roots: BigFish `1.97`, BossFight `.01`, CaveFlyer `1.49`, CoinRun `0.00`. BigFish/Boss/Cave have valid frozen complete-trace aggregates; CoinRun has the immutable validator failure above. Fixed critic curvature `4` made the metric critic-heavy, but valid aggregates remained strongly actor-dominant after the coupled inverse. It did not provide a general post-inverse critic-contribution or reward rescue. Final classification: `TERMINAL_TRAINING_COMPLETE_TELEMETRY_AGGREGATION_PARTIAL_FAILURE`.
