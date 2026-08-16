# Procgen PPO vs Exact RAT curve validation

> **Version warning:** These curves come from runs derived from desktop HEAD `7a0698e`, not the requested commit `2b5affd64cbb3c624b4bc1f4767f449df231ffb2`. They must not be reported as results for the requested commit.

- Extracted from completed 4090 stdout logs on 2026-07-22.
- Metric: `eprewmean` (trainer's rolling episodic-return statistic).
- Aggregation: arithmetic mean with 95% normal-approximation CI across 5 seeds.
- No additional temporal smoothing.
- All 80 seed logs have return code 0.

| Environment | PPO seeds | RAT seeds | PPO points/seed | RAT points/seed | PPO final mean | RAT final mean |
|---|---:|---:|---:|---:|---:|---:|
| BigFish | 5 | 5 | 146-146 | 146-146 | 11 | 16.3 |
| BossFight | 5 | 5 | 146-146 | 146-146 | 1.59 | 1.74 |
| CaveFlyer | 5 | 5 | 146-146 | 146-146 | 5 | 4.66 |
| CoinRun | 5 | 5 | 146-146 | 146-146 | 9.94 | 9.56 |
| Jumper | 5 | 5 | 146-146 | 146-146 | 8.9 | 8.12 |
| Maze | 5 | 5 | 146-146 | 146-146 | 8.32 | 7.48 |
| Miner | 5 | 5 | 146-146 | 146-146 | 5.57 | 4.7 |
| StarPilot | 5 | 5 | 146-146 | 146-146 | 20.9 | 24.3 |
