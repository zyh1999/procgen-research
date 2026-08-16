# Procgen shared PPO 4090 正式批次：对齐与启动状态

- 日期：2026-07-20
- 目标：8 环境 × 5 seeds = 40 seed runs
- 当前状态：40/40 个正式 PPO seed runs 已全部完成，两个分片均为 `PASS`
- 安全状态：全程 `PACK_PER_GPU=1`；没有 OOM 或重复 run

## 与新版 Bede RAT 的公共实验条件

PPO 与当前 Bede RAT 正式批次保持以下条件一致：

- 环境：BigFish、BossFight、CaveFlyer、CoinRun、Jumper、Maze、Miner、StarPilot
- Procgen level：`easy-0-10`
- seeds：`0,1,2,3,4`
- 每 seed：`6,000,000` environment steps
- `num_envs=16`
- `nsteps=256`
- rollout batch：`4096`
- inner epochs：`4`
- minibatches：`8`
- shared IMPALA ResNet，embedding `256`
- PopArt：开启
- observation normalization：关闭
- reward normalization：关闭
- entropy coefficient：`0`
- global gradient norm clip：`0.5`
- Procgen：要求 `0.10.7`

## PPO 方法自身配置

直接使用最新下载 `trust-region-main (3).zip` 中未经改写的
`configs/ppo_resnet_shared.yaml` 和 `train_shared.py`：

- algorithm：shared PPO
- optimizer：Adam
- learning rate：`0.001`
- PPO clip range：`0.2`
- KL-adaptive LR：关闭
- Kaczmarz：关闭
- value coefficient：`1.0`
- 无额外 actor auxiliary loss

这些是 PPO 方法定义本身与 RAT 的必要差异，不为表面对齐而改成 RAT 的 SGD、LR、damping 或 KL controller。

版本校验：

- source archive：`/Users/user/Downloads/trust-region-main (3).zip`
- archive SHA-256：`6ea93e1285b6fd84c4b74239a18533e2ccd410c4313bb2a333aae4187e473167`
- exact archive trainer SHA-256：`1d20658b154022450b8598949f693b3c04a9bd34eb22ad2f002d59f9573b74d1`
- exact PPO config SHA-256：`fdf1538ef199a222ea2caafe9264c5db00319a6f1882d7d86b04506522601807`

## 已准备 bundle

- Bundle：`/Users/user/Documents/procgen/work/procgen_ppo_4090_bundle.tgz`
- Bundle SHA-256：`51460d36b16fee194bd396c173a827b3efe43f9ed35ccebcf18d73ccacd34aa8`
- Split bundle：`/Users/user/Documents/procgen/work/procgen_ppo_4090_split_bundle_20260720_1902.tgz`
- Split bundle SHA-256：`166ecfd281940f17f1c99b634e1e7b29e2e0eac18a5f136a55c1fd74eb379c29`
- Bash syntax：通过
- Trainer AST：通过
- Formal/smoke YAML parse：通过

Bundle 包含：

- 最新 archive 中 PPO 所需的精简源码与原始 formal config
- 一个同方法、短 horizon 的 smoke config
- preflight/smoke 脚本
- 8×5 formal launcher
- 防重复保护：formal `RUN_ROOT` 已存在时拒绝启动
- 每个环境/seed 独立 stdout、stderr、return code
- run_info 与 trainer/config/launcher snapshot

正式 launcher 增加 `TASK_START` / `TASK_END` 参数，使两台机器可以运行互不重叠的任务区间。launcher SHA-256：`03be5e4302b87198c01858c746f159ae2c7a30626820548db54cc82afd7fffd4`。

## 实时资源检查

用户配置公钥后，三台机器均可无交互 SSH。2026-07-20 18:59 BST 的实时检查：

- `.92`（`mingfei-workstation-92`）：18:56:03 刚重启；两张 4090 均为 0%、15 MiB；根盘余 155GB；当前没有用户训练进程
- `.15`（`mingfei-workstation-76`）：两张 4090 均约 99%；根盘只余 11GB（99% 使用）；不适合本批次
- `.54`（`mingfei-workstation-31`）：两张 4090 均约 39%，已有 `yingxiao` 的 Atari/Procgen 进程；根盘余 38GB。随后按用户明确要求，只在显存占用较低的 GPU 0 增加一个 PPO seed 进程

## `.92` 部署与验证

- 部署根目录：`/home/yihe/procgen_ppo_shared_4090_20260720_1850`
- Python：`3.10.12`
- PyTorch：`2.9.0+cu128`
- Procgen：`0.10.7`
- Gym3：`0.3.3`
- 依赖 overlay：`/home/yihe/procgen_ppo_overlay_20260720`
- 实际 Procgen VecEnv reset/step：通过
- 单 seed smoke：通过，约 4.8k FPS
- 5 seeds 同卡 smoke：5/5 通过，无 OOM、NaN 或 traceback；总显存峰值约 8.8GB（包含当时已有的 Humanoid 进程）

## Formal 启动事故

- 启动时间：约 2026-07-20 18:54:47 BST
- 目标布局：GPU 1，pack5，8 个环境依次执行，每环境 seeds 0..4
- formal root：`/home/yihe/procgen_ppo_shared_4090_20260720_1850/formal/shared_ppo_maincfg_6m_pack5_20260720_1855`
- launcher log：`/home/yihe/procgen_ppo_shared_4090_20260720_1850/formal_launcher_20260720_1855.out`
- 约 69 秒后 `.92` 整机硬重启；`uptime -s` 为 `2026-07-20 18:56:03`，`last -x` 将旧会话标为 `crash`
- formal root 中日志/状态文件均为 0 字节，说明重启前缓冲尚未落盘；不能作为有效训练数据
- 当前无该批次进程；没有重复启动，也没有取消或修改 Bede 作业

用户权限无法读取上一 boot 的 kernel/system journal，因此不能确认是 GPU Xid、温度、电源还是其他系统故障。启动前 GPU 0 已被其他任务满载，GPU 1 也有既有 Humanoid 进程；pack5 formal 增加整机持续负载后发生重启，这一关联只能作为推断，不能当成已确认根因。

失败现场被保留，没有复用该 run root，也没有再次启动 pack5。

## 当前正式 20+20 split

启动时间：约 2026-07-20 19:22 BST。两边均只使用 GPU 0，且 `PACK_PER_GPU=1`，所以每张卡同一时刻只有一个 PPO seed 进程。

`.54`：

- host：`mingfei-workstation-31`
- launcher PID：`1631662`
- task range：`0..19`
- 环境：BigFish、BossFight、CaveFlyer、CoinRun；各 seeds `0..4`
- Python：`/home/yihe/rat_procgen_env/bin/python`
- Torch：`2.12.1+cu130`
- Procgen/Gym3：`0.10.7` / `0.3.3`
- run root：`/home/yihe/procgen_ppo_shared_4090_20260720_1920/formal/shared_ppo_maincfg_6m_split54_gpu0_task00_19_pack1_20260720_1922`
- `.54` 单-seed smoke：PASS；总 GPU 显存峰值 `2892 MiB`（包括原有进程），功率峰值 `181.2 W`

`.92`：

- host：`mingfei-workstation-92`
- launcher PID：`5147`
- task range：`20..39`
- 环境：Jumper、Maze、Miner、StarPilot；各 seeds `0..4`
- Python：`/home/yihe/.venv/bin/python`
- Torch：`2.9.0+cu128`
- Procgen/Gym3：`0.10.7` / `0.3.3`
- run root：`/home/yihe/procgen_ppo_shared_4090_20260720_1850/formal/shared_ppo_maincfg_6m_split92_gpu0_task20_39_pack1_20260720_1922`

两边 trainer/config SHA 与最新 archive 一致。19:24 BST 的早期健康检查：

- `.54`：`RUNNING`，约 `0.78M / 6M` steps（首个 BigFish seed 0），GPU 监控峰值 `2934 MiB`、`66 C`、`172.38 W`
- `.92`：`RUNNING`，约 `0.74M / 6M` steps（首个 Jumper seed 0），GPU 监控峰值 `957 MiB`、`63 C`、`164.17 W`
- 两边 `OOM` / `NaN` / `traceback` / `CUDA error` 扫描均为 0
- `.92` boot time 仍为 `2026-07-20 18:56:03`，已超过此前 pack5 的约 69 秒故障窗口，没有再次重启

上述 19:24 检查是启动后的早期快照；最终完成状态见下节。

## 2026-07-21 完成核查

2026-07-21 14:18 BST 实时核查：

- `.54` task `0..19`：status `PASS`，20/20 returncode 文件存在
- `.92` task `20..39`：status `PASS`，20/20 returncode 文件存在
- 合计：40/40 PPO seed runs 完成
