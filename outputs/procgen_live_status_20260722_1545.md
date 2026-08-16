# Procgen 最终状态：2026-07-22 15:45 BST

## Bede shared Exact RAT

- Slurm array：`1062382_[0-7]`
- 8/8 array members：`COMPLETED`
- elapsed：约 7:09–7:14
- Slurm exit code：全部 `0:0`
- 8/8 environments：`PASS`
- 40/40 seeds：return code 0
- 最终日志点：每 seed 约 5.98M steps，对应配置的 6M horizon 与 rollout 日志间隔

环境：BigFish、BossFight、CaveFlyer、CoinRun、Jumper、Maze、Miner、StarPilot。

## 4090 shared PPO

- `.54` 分片：20/20 完成，status `PASS`，overall rc 0，无失败 seed
- `.92` 分片：20/20 完成，status `PASS`，overall rc 0，无失败 seed
- 合计：40/40 seeds 完成

## 4090 shared Exact RAT

- `.92` GPU 1 分片：20/20 完成，status `PASS`，overall rc 0
- `.54` GPU 0 分片：10/10 完成，status `PASS`，overall rc 0
- `.92` GPU 0 分片：10/10 完成，status `PASS`，overall rc 0
- 合计：40/40 seeds 完成

## CSF3

- 没有 Procgen 作业提交或运行；2026-07-21 的去重核查发现没有未覆盖任务，因此保持零提交。

## 本次核查边界

本次确认了 Slurm/launcher 状态、每个环境和 seed 的 return code、最终训练步数及当前无残留 Procgen 进程。尚未重新提取或聚合学习曲线，因此这里不对 PPO 与 Exact RAT 的性能排序作结论。
