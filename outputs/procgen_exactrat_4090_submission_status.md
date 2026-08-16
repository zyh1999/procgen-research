# Procgen Exact RAT 4090 正式批次状态

- 日期：2026-07-20
- 目标：与 Bede 正式数组 `1062382` 完全对应的 shared Exact RAT，8 环境 × 5 seeds = 40 runs
- 当前状态：三个分片均已完成并 `PASS`，40/40 Exact RAT seed runs 完成
- 并发安全：每张 GPU 同一时刻最多运行 1 个 RAT seed；等待器同时检查基线显存低于 6000 MiB、GPU utilization 低于 85% 才启动
- Bede：`1062382_[0-7%8]` 截至 19:45 BST 仍为 `PENDING (Resources)`；未取消、修改或重复提交 Bede 作业

## Bede 正式版本校验

来源目录：

`/nobackup/projects/bdman37/yihe/procgen_author_sharedrat_20260720/formal/author_sharedrat_maincfg_procgenkl_u004_6m_pack5_20260720_1740/snapshot`

- trainer：`train_shared_procgen_maincfg_pklbranch.py`
- trainer SHA-256：`f4cfcd3a5dd9ea84e9d7533a5f17c2d897db545a49d352850df89bdc69142369`
- config：`adv_resnet_shared_procgen_maincfg_pklbranch.yaml`
- config SHA-256：`476b210d9da6e1dc973cf293d812a5c1e2f3c6f20654736a9687e397131da1ca`
- 本地 4090 bundle SHA-256：`7f1040117069572c8acd55a1071b3d539f1cb480a0b626f0bfd60238c4251a46`
- 4090 launcher SHA-256：`136c8f692e57673c47463025fda1b7389f2ef8632f8fdd28d4d52b819323362b`
- 等待器 SHA-256：`f5674da4ceb26f026c96af2a53be927132b88aaa3dba2ac663a532ee36d48203`

trainer 与 config 从 Bede 正式 run snapshot 逐字节复制，没有改算法代码或超参数；x86 bundle 新增的只有串行 launcher、PPO 等待器和 GPU 监控。

## 方法与公共实验条件

- environments：BigFish、BossFight、CaveFlyer、CoinRun、Jumper、Maze、Miner、StarPilot
- levels：`easy-0-10`
- seeds：`0..4`
- horizon：每 seed `6,000,000` environment steps
- `num_envs=16`，`nsteps=256`，rollout batch `4096`
- epochs `4`，minibatches `8`，minibatch size `512`
- optimizer：SGD，momentum `0.1`
- initial LR：`0.5`
- damping：`0.5`
- Kaczmarz：`false`
- Procgen 专用 KL controller：upper `0.04`，lower `0.005`，LR 降/升因子 `1.5`，范围 `[1e-4, 0.5]`
- shared IMPALA ResNet `[8,16]`，embedding `256`，PopArt on，BN/dropout off
- Procgen/Gym3：`0.10.7` / `0.3.3`

## 三路任务分片

### `.92` GPU 1：立即运行

- scheduler/launcher PID：`6964`
- task range：`0..19`
- environments：BigFish、BossFight、CaveFlyer、CoinRun；各 seeds `0..4`
- run root：`/home/yihe/procgen_exactrat_4090_20260720_1950/formal/exactrat_bede1062382_split92_gpu1_task00_19_pack1_20260720_2000`
- runtime：Python 3.10，Torch `2.9.0+cu128`
- 启动后早期状态：`RUNNING`；最终状态：`PASS`，20/20 完成

### `.54` GPU 0：等待 PPO 后启动

- scheduler PID：`1636324`
- wait PID：PPO launcher `1631662`
- task range：`20..29`
- environments：Jumper、Maze；各 seeds `0..4`
- run root：`/home/yihe/procgen_exactrat_4090_20260720_1950/formal/exactrat_bede1062382_split54_gpu0_task20_29_pack1_20260720_2000`
- runtime：Python 3.10，Torch `2.12.1+cu130`
- 启动后早期状态：`QUEUED_WAIT_PPO`；最终状态：`PASS`，10/10 完成

### `.92` GPU 0：等待 PPO 后启动

- scheduler PID：`6919`
- wait PID：PPO launcher `5147`
- task range：`30..39`
- environments：Miner、StarPilot；各 seeds `0..4`
- run root：`/home/yihe/procgen_exactrat_4090_20260720_1950/formal/exactrat_bede1062382_split92_gpu0_task30_39_pack1_20260720_2000`
- runtime：Python 3.10，Torch `2.9.0+cu128`
- 启动后早期状态：`QUEUED_WAIT_PPO`；最终状态：`PASS`，10/10 完成

三个 task range 无重叠且并集恰为 `0..39`。

## Smoke 与早期健康检查

`.92` GPU 1 使用正式 Bede trainer/config，仅 CLI 覆盖 horizon 为 40,960 steps：

- status：PASS
- total steps：40,960
- elapsed：17 秒
- GPU 显存峰值：5769 MiB / 24564 MiB
- 温度峰值：65 C
- 功率峰值：372.53 W
- OOM / NaN / traceback / CUDA error：0

正式 task 0 在 19:53 BST 的检查：

- launcher 已运行约 153 秒
- progress：约 369K / 6M steps
- `misc/time_elapsed`：148 秒
- 正式吞吐：约 2490 environment steps/s
- GPU 显存峰值：5751 MiB
- 温度峰值：72 C
- 功率峰值：312.18 W
- error scan：0
- `.92` boot time 仍为 `2026-07-20 18:56:03`，未再次重启

## 完成时间估计

按正式首段吞吐计算，单 seed 约 40 分钟：

- `.92` GPU 1 的 20 seeds：约 13.4 小时，预计 2026-07-21 09:15 BST 左右完成
- `.54` GPU 0 和 `.92` GPU 0 各 10 seeds：PPO 预计约 01:00–02:30 结束，之后各需约 6.7 小时，预计约 08:00–09:30 完成
- 整批预计约 09:15–10:00 完成；即使吞吐慢约 20%，仍预计接近或早于 13:00

该估计以当前 BigFish 吞吐为基准；不同 Procgen 环境速度可能变化，因此 13:00 是目标窗口而非完成保证。

## 2026-07-21 完成核查

2026-07-21 14:18 BST 实时核查：

- `.92` GPU 1 task `0..19`：status `PASS`，20/20 returncode 均为 0
- `.54` GPU 0 task `20..29`：status `PASS`，10/10 returncode 均为 0
- `.92` GPU 0 task `30..39`：status `PASS`，10/10 returncode 均为 0
- 三个分片并集为 `0..39`，合计 40/40 Exact RAT seed runs 完成
