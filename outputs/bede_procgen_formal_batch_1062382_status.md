# Bede Procgen 正式批次状态：1062382

- 最近核对时间：2026-07-22 15:44:53 BST
- 当前正式数组：`1062382_[0-7%8]`
- Job name：`pg_rat8_pkl04`
- 当前状态：8/8 环境全部 `PASS`；40/40 seeds return code 0
- 旧数组：`1062354_[0-7]` 在 `PENDING (Resources)`、运行 0 秒时已取消并由本批次替换
- 操作边界：本次只读核对并记录；没有重复提交、取消或修改任何作业

## 布局

Slurm 数组范围为 `0-7%8`，共 8 个单 GPU array task。每个 array task 对应一个环境，并在同一张 GPU 上并行启动 seeds `0,1,2,3,4`，因此计划总量为：

- 8 个环境
- 每环境 5 seeds
- 40 个 seed runs
- 最多同时请求 8 张 V100

环境映射：

1. `bigfish-easy-0-10`
2. `bossfight-easy-0-10`
3. `caveflyer-easy-0-10`
4. `coinrun-easy-0-10`
5. `jumper-easy-0-10`
6. `maze-easy-0-10`
7. `miner-easy-0-10`
8. `starpilot-easy-0-10`

2026-07-22 15:44 BST 的最终环境级状态：

- BigFish、BossFight、CaveFlyer、CoinRun、Jumper、Maze、Miner、StarPilot：全部 `PASS`
- 每个环境 5/5 seeds return code 0
- 每个 seed 最终记录约 5.98M steps；这是 6M horizon 下按 4096-step rollout 记录间隔得到的最后日志点
- `sacct` 显示 8/8 array members 全部 `COMPLETED`，elapsed 约 7:09–7:14，exit code `0:0`

## 正式配置

- 来源：最新下载的 `trust-region-main (3).zip`
- 方法：shared RAT
- 与来源 `configs/adv_resnet_shared.yaml` 一致，仅新增 `use_procgen_kl_thresholds: true`
- easy horizon：每 seed `6,000,000` environment steps
- `num_envs=16`
- `nsteps=256`
- rollout batch：每 seed `4096`
- `epochs=4`
- `minibatches=8`，minibatch size `512`
- optimizer：SGD，momentum `0.1`
- initial LR：`0.5`
- damping：`0.5`
- Kaczmarz：`false`

## KL 自适应学习率

Bede trainer 快照保留真实 KL 计算：控制器使用
`KL(current rollout behavior || policy after current minibatch)`，并记录相对 rollout old policy 的 real KL。

- Procgen 专用 upper：`0.02 * 2 = 0.04`
- 其他配置默认 upper：`0.01 * 2 = 0.02`
- lower：`0.01 / 2 = 0.005`
- 降 LR：除以 `1.5`，下限 `1e-4`
- 升 LR：乘以 `1.5`，上限 `0.5`

## 归档与校验

Run directory：

`/nobackup/projects/bdman37/yihe/procgen_author_sharedrat_20260720/formal/author_sharedrat_maincfg_procgenkl_u004_6m_pack5_20260720_1740`

该目录已保存 `run_info.txt` 和提交快照，包括 trainer、config 与 sbatch。

- trainer SHA-256：`f4cfcd3a5dd9ea84e9d7533a5f17c2d897db545a49d352850df89bdc69142369`
- config SHA-256：`476b210d9da6e1dc973cf293d812a5c1e2f3c6f20654736a9687e397131da1ca`
- sbatch SHA-256：`8f180ae17a87d6c83dc014926a58408c38bd228f5607e786b348c8f3a9c5e9d6`

相关已通过 smoke：`1062352`，pack5 的 5/5 seeds 均返回 `rc=0`。
