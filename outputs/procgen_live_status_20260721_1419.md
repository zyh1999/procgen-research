# Procgen 实时状态：2026-07-21 14:19 BST

## 已完成

- 4090 shared PPO：40/40 seed runs 完成；`.54` 20/20 PASS，`.92` 20/20 PASS
- 4090 Exact RAT：40/40 seed runs 完成；三个分片分别 20/20、10/10、10/10 PASS

## Bede 正在运行

正式数组：`1062382`，job name `pg_rat8_pkl04`。

- 6/8 环境已 PASS：BigFish、BossFight、CaveFlyer、CoinRun、Jumper、Maze
- Miner：5 seeds 正在运行，约 5.32–5.37M / 6M steps
- StarPilot：5 seeds 正在运行，约 3.36M / 6M steps
- 当前没有 PENDING 的 Procgen array member

## CSF3 gpuH 核查

- partition：`gpuH`
- hardware：H200，节点 `node820..823`，每节点 8 GPUs
- user association：account `gpu-h200-fse-pgdr`，QOS `gpu-h200-fse`，MaxJobs `4`
- 用户要求的目标布局可实现：2 张 GPU，每卡一个环境、并行 5 seeds
- 本次提交数：0

未提交原因：所有 4090 Procgen PPO/Exact RAT 已完成；Bede 剩余 Miner/StarPilot 已经运行，不是等待任务。在 CSF3 再提交会产生重复数据，破坏受控实验记录。

若出现新的未覆盖 Procgen 环境或现有运行失败，可按 2 个单 GPU array/task、每卡 5 seeds 的方式补跑，并使用 Bede 正式 trainer/config 快照和独立 run root。
