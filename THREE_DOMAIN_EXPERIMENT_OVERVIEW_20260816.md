# 三大强化学习实验总览：Procgen、MuJoCo 与 Isaac Lab

> 版本：2026-08-16  
> 实时状态截点：2026-08-16 09:00 UTC（CSF3 与双 5060）  
> Bede 最近可信快照：2026-08-16 08:18 UTC  
> 目的：统一说明三大实验到底在研究什么、各条算法线如何区分、当前跑到哪里，以及之后应按什么证据继续或停止。

## 1. 一页式总结

这三大实验共享一个研究主题：**在 PPO/RAT 类策略优化中，用更有结构的曲率近似、样本空间线性系统和自适应 KL 控制，换取比普通 PPO 更好的样本效率或最终回报，同时保证训练稳定、计算可控且比较公平。**

| 板块 | 核心问题 | 当前主要比较 | 当前状态摘要 | 下一关键问题 |
|---|---|---|---|---|
| Procgen | 视觉强化学习中，曲率信息应只作用于 actor，还是联合 actor–critic；更大的决策层是否比原小头更匹配 IMPALA 编码器 | 纯 PPO DMLP1024、PPG/EF actor ablation、shared Exact-GGN/RAT、严格 Joint-2B/压缩 Joint-B | Bede 纯 PPO 4 环境 × 3 seed 已完成；ACTOR_J seed0 有 3 个完成、BossFight 被记录为早停失败；CSF3 的 500k Joint-B 四环境 gate 已完成，1M gate 正在运行/排队 | Joint-B/Joint-2B 的结构改进能否在 4 环境一致超过 matched PPO/PPG，而不是只在单环境数值稳定 |
| MuJoCo | 在连续控制中，256 行近似、255+1 energy-free、完整 EF/GGN 与 Kaczmarz/动量怎样权衡性能、稳定性和成本 | 大批量 Curv256/K-opt；小批量 no-shared Transformer 的 PPO、K-FAC、Emp256、EF255+1、FullEmp | Bede `1072326` 正在跑/排队 18 个单元；双 5060 上有 8 个 MuJoCo 容器；历史 no-shared MLP 五 seed 表明 RAT/EF 系在多个环境明显强于固定 LR PPO | 完成七环境、同 seed、同 10M 预算的 Transformer 全局比较，并修齐 Swimmer-v3 与 Humanoid 缺口 |
| Isaac Lab | 大规模并行机器人任务中，SB3 PPO、K-FAC、Emp256 与 EF255+1 是否在相同 rollout/网络/seed 下真正可比 | Ant、Unitree A1、ANYmal-C；PPO、K-FAC、Emp256、EF255+1；rolling-old、自适应 KL、line-search/no-rescale 因果对照 | 旧的 3 seed × 3 task matched 表已完成；当前 CSF3 在跑 Ant DirectionObserver v2 u500，约 35.0M transitions、u356 | 用 matched seed 跑到 u500，再以 trailing-10、AUC、KL 和求解残差作结论，不能用 seed42 或单个 Optuna trial 代替多 seed 证据 |

### 当前最重要的管理事实

- CSF3 训练没有整体停止：09:00 UTC 时有两个 Procgen 1M gate array elements 和一个 Isaac L40S 作业运行，另外两个 Procgen elements 等待 `AssocGrpGRES`。
- 常驻控制器位于 CSF3 `login2`，`screen` 名为 `codex-three-domain`；已确认 `controller_loop.sh` 存活并每 1200 秒运行一个周期。SSH 若落到 `login3`，当地 `screen -ls` 看不到 `login2` 的 socket，这不能解释为控制器停止。
- 控制器实际调用 `gpt-5.6-sol`，推理强度 `medium`。它先生成只读快照和候选，再交给 deterministic validator；现有 catalog/launcher 安全门仍限制任意新实验的自动提交。
- Bede 当前免交互 SSH 通道失效；本文的 Bede 运行状态来自 08:18 UTC 的最后可信控制器采样，不冒充 09:00 UTC 实时状态。
- `ws4090-31` / `10.49.7.54` 继续整机隔离，不承担新任务。
- 训练尽量不用 Jupyter；任何完全空闲的 Jupyter GPU allocation 必须在一小时内取消。当前 Procgen sbatch 仍附带本地 Jupyter sidecar，这是需要从后续 launcher 中移除的技术债。

## 2. 统一术语与比较原则

### 2.1 方法名

| 名称 | 本文含义 |
|---|---|
| PPO | clipped policy objective；控制组必须固定网络、rollout、minibatch、epoch、seed 和训练预算 |
| K-FAC | 用 Kronecker-factored 近似构造自然梯度方向；是结构化参数空间近似 |
| RAT / Exact-RAT | 当前代码族中的曲率感知、样本空间求解策略优化；“Exact”指当前定义下保留完整或精确构造的目标几何，不等于全参数 Hessian |
| Emp256 | 使用 256 个经验 score/curvature 行构造 actor 几何，再求样本空间方向 |
| Curv256 | actor 和/或 critic 各抽取 256 行曲率信息；具体 actor Fisher 与 critic GGN 语义必须随配置记录 |
| EF255+1 / EnergyFree255+1 | 255 个普通曲率行加 1 个特殊 anchor/free-rho 行，用低成本补偿全批量方向/能量约束 |
| FullEmp / FullEF | 不把 actor curvature 限制在 256 行的完整经验 Fisher/score 构造 |
| GGN | critic 侧的 generalized Gauss–Newton 曲率近似；与 actor Fisher 不是同一个矩阵 |
| Joint-2B | 将 actor 与 critic 的样本空间块堆叠成严格 `2B × 2B` 系统，并保留交叉块 |
| Joint-B | 从父 `2B` 系统做有定义的约化，例如当前 1024 行父系统压缩为 512 行 RHS-aligned Galerkin 系统 |

### 2.2 K-opt 到底是什么

`K-opt` 不是一套独立于 RAT/Curv256 的新优化器。当前 MuJoCo 代码中，它是 **Curv256 + critic GGN256 实验外层的 Optuna 搜索**，搜索：

- actor clip 类型：普通 L2 或 Fisher/FVP clip；
- clip radius：`0.125, 0.25, 0.5, 1, 2, 4`；
- actor/critic SGD momentum：近零、`0.5`、`0.9`；
- 是否启用 Kaczmarz 历史投影。

每个 trial 在一个环境上跑两个 seed，以末尾 10 个 reward 的均值作为 Optuna objective。`K-true` 通常指 `is_karzmarz=True` 的明确 Kaczmarz 基线；`K-opt` 指同时搜索 Kaczmarz 开关和其他相关超参数后的最佳组合。两者都建立在同一个 Curv256/GGN256 训练器上。

### 2.3 结果状态

| 状态 | 判定要求 |
|---|---|
| `COMPLETE` | 计划预算或认可的最后完整 update 到达，return code 正常，终点 artifact 存在，无 fatal error |
| `RUNNING` | scheduler/process 存活，日志或 artifact 在增长；仅看到队列名不够 |
| `PENDING` | 已提交但未获得资源；`Resources`、`Priority`、`AssocGrpGRES`、`QOSMaxGRESPerUser` 不是算法失败 |
| `FAILED` | OOM、CUDA error、NaN/nonfinite、Traceback、非零退出，或进程死亡且输出不完整 |
| `EARLY_STOPPED_FAILED` | 经价值早停规则停止；必须保留配置、曲线、停止点和原因 |
| `CANCELLED` | 被用户/系统取消；除非日志证明，否则不能自动解释成数值失败 |
| `INCOMPLETE` | 证据、seed、环境或预算未齐；不能写成方法优劣结论 |

### 2.4 统一价值早停规则

一个 run 只有同时满足以下条件才可因“没有价值”早停：

1. 有严格 matched、正值且可信的最高 baseline；
2. 已超过该板块的最小观察进度；
3. 至少有三个对齐进度点；
4. robust score 持续低于最高 matched baseline 的 `3/5`；
5. 连续两个控制周期得出相同结论。

单个低 reward、单个 seed、GPU 利用率低或 scheduler pending 都不是价值早停证据。硬故障可以立即停止，但只能停止精确 job element 或明确归属的进程树。

## 3. Procgen

### 3.1 研究目标

Procgen 研究的不是简单“哪条曲线最高”，而是三个相互关联的问题：

1. IMPALA/ResNet 视觉编码器之后，原决策头是否太小；把共享决策 MLP 扩为 `256 → 1024 → 256` 后，纯 PPO 是否先获得稳定增益；
2. actor 的 entropy/Fisher/EF 处理是否应保留 PPG 的辅助阶段，还是纯 PPO 已足够；
3. critic 曲率和 actor–critic 交叉块是否真能帮助 policy update，还是只增加数值复杂度。

当前正式环境主要是：BigFish、BossFight、CaveFlyer、CoinRun。核心比较要求同环境、同 seed、同 6M 预算和同 DMLP1024 架构。

### 3.2 网络架构：DMLP1024

当前 Procgen 主架构是：

```text
RGB observation
    → IMPALA/ResNet visual encoder
    → shared decision MLP: 256 → 1024 → 256, ReLU
    → linear policy head
    → linear/PopArt critic head
```

- active parameters：约 `1,464,547`；Joint-B 日志中去除最后少量头参数后记录约 `1,464,544` 个参数列。
- 决策 MLP 对 actor 和 critic 共享，因此**决策层变大时 critic 的共享表征也一起变大**。
- critic 最后的 scalar value head 本身仍很小；扩大的是它前面的共享决策 trunk，而不是单独堆一个巨大的 critic-only head。
- 该架构仍属于 IMPALA/ResNet encoder，不是把整个模型换成普通 MLP。

### 3.3 线 A：纯 PPO DMLP1024 控制组

这是“不使用 PPG auxiliary phase、不使用 RAT 曲率”的干净控制：

| 项目 | 设置 |
|---|---|
| 环境 × seed | 4 环境 × seeds `0,1,2`，共 12 根 |
| rollout | 16 env × 256 steps = 4096 transitions |
| PPO update | 4 epochs × 8 minibatches |
| optimizer | Adam，LR `1e-3` |
| clip | PPO ratio clip `0.2`，max grad norm `0.5` |
| critic | PopArt |
| 预算 | 6M transitions/child |
| 平台 | Bede；每个环境一个一 GPU sbatch，每个 sbatch 同时跑三个独立 seed |

已验证结果：Bede jobs `1070573–1070576` 的 12 个 child 全部 `PASS`，均到约 6M transitions。它是后续 PPG/RAT/Joint 方法的必要控制，但不能与不同网络或不同 seed 的旧 PPO 混合。

### 3.4 线 B：PPG/EF actor ablation

该线保留 phasic/auxiliary 语义，主要比较：

| 标签 | 作用 |
|---|---|
| `E_v2` | 官方 schedule 的 matched PPG baseline |
| `ACTOR_G` | entropy natural-gradient 线 |
| `ACTOR_H` | policy KL / Fisher clip 线 |
| `ACTOR_J` | entropy-NG + policy clip 的组合 |
| `ACTOR_K` | Exact-RAT + adaptive-KL 线；不能与旧的 `ACTOR_I` 混名 |

除了 reward，还必须看 auxiliary explained variance/MSE、clone KL、clip scale、entropy、solver residual。数值求解有限但 auxiliary head 崩溃，仍算方法失败。

当前已建立的 seed0 证据：

- `E_v2` 的四环境 baseline 已存在；
- ACTOR_J 的 BigFish、CaveFlyer、CoinRun clean recovery 均完成到 `5,980,160` transitions；
- ACTOR_J BossFight 在 `4,096,000` 时 robust ratio `0.5465`，已登记为 `EARLY_STOPPED_FAILED`，不会因为其他环境成功而被抹掉；
- 下一项最有价值的 matched 扩展，是 BigFish 的 `E_v2 seed1` 与 `ACTOR_J seed1` 成对运行，而不是再造一个新的 J 变体。

### 3.5 线 C：shared Exact-GGN/RAT 与严格 Joint 系统

这部分要严格分两种身份：

1. **P1 shared Exact-GGN/shared-RAT**：确定性 critic GGN、symmetric FP64/Jacobi 等固定语义；seed1 四环境曾因基础设施中断且没有 checkpoint，现保留失败记录、暂不自动重跑。
2. **Joint-2B/Joint-B 因果线**：actor Fisher、critic GGN 与 actor–critic cross block 放进同一系统，专门检验“critic 信息是否真正改善 actor 方向”。

严格 Joint-2B 的典型设置是：rollout `4096`、minibatch `512`、4 epochs、父系统 `1024 × 1024`、FP64 solve、paired critic RHS，并记录：

- actor/critic block Frobenius norm；
- normalized cross-block；
- joint solve residual；
- actor-only 与 full-joint 方向余弦/范数比；
- critic-induced actor quadratic；
- KL、entropy、ratio range、reward；
- critic score/RHS 模式以及 actor/critic damping。

### 3.6 2026-08-16 当前 Procgen 运行

CSF3 当前 array `18670437_0-3` 是一个 500k、seed0、四环境的 Joint-B gate：

- 父系统 1024 行，RHS-aligned rank-1 Galerkin 压缩为 512 行；
- actor absolute damping `0.003`；critic absolute damping `0.5`；
- block-relative damping floor `0.10`；actor-from-critic floor `0.01`；
- full compressed cross terms；clean critic score；momentum `0`；非 Kaczmarz；FP64 solve。

500k gate 的最终状态：

| 环境 | array element | 状态 | 最新进度 | 最新 eprewmean | 数值健康 |
|---|---:|---|---:|---:|---|
| BigFish | `_0` | `PASS` | 507,904 | 2.97 | reduced residual 约 `1e-12`，无 fatal error |
| BossFight | `_1` | `PASS` | 507,904 | 1.10 | 无 fatal error；reward 仍需 matched baseline 判定 |
| CaveFlyer | `_2` | `PASS` | 507,904 | 4.60 | 无 fatal error |
| CoinRun | `_3` | `PASS` | 507,904 | 末段约 6.3 | 无 fatal error |

这只是 500k gate，不是 6M 性能结论。四个 `PASS` 表示执行和结构门通过，不代表已经超过 PPO/PPG baseline。

随后同一算法身份已经扩到 1M gate，CSF3 array 为 `18670696_0-3`：

| element | 环境 | 09:00 UTC 状态 | 节点/原因 |
|---:|---|---|---|
| `_0` | BigFish | `RUNNING` | `node847`，A100 80GB |
| `_1` | BossFight | `RUNNING` | `node847`，A100 80GB；与 `_0` 共享该节点的不同 GPU allocation |
| `_2` | CaveFlyer | `PENDING` | `AssocGrpGRES` |
| `_3` | CoinRun | `PENDING` | `AssocGrpGRES` |

1M root 为 `/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/gate_1m_seed0_jupyter_rhsaligned_actorrelative_criticfloor05_v1/rhs_aligned_rank1_b`；配置与 500k gate 保持父 1024 行、约化 512 行、actor damping .003、critic damping .5、block floor .10、actor-from-critic floor .01。该 sbatch 仍启动 Jupyter sidecar，应在后续无 Jupyter launcher 中修正。

另有 `18642230_[0-3]` 和 `18624888_[0-3%4]` 被 `JobHeldUser` 持有。它们不耗 GPU，也不是代码失败；在解除 hold 前必须先核对是否和现有/已完成 Joint 单元重复。

## 4. MuJoCo

### 4.1 研究目标

MuJoCo 用连续控制验证曲率近似的通用性，主要环境为：

`Ant`、`HalfCheetah`、`Hopper`、`Walker2d`、`Humanoid`、`HumanoidStandup`、`Swimmer`。

核心要求是：不能只根据 Hopper 选全局超参数。需要在方法、damping、环境、seed 和预算对齐后，以归一化得分或跨环境 rank 聚合做选择。Swimmer-v3 的版本/路径缺口单独修，不与 v4 或其他环境静默混合。

### 4.2 M1：大批量 Curv256 / K-opt

大批量线主要使用：

- rollout：`1024 env × 256 = 262,144` samples；
- 4 minibatches、4 epochs；
- 约 500 个完整更新；
- nominal 约 130M transitions；
- actor Curv256 + critic GGN256；
- damping 常以 `0.03` 为主，比较 Kaczmarz、momentum、L2/Fisher clip。

这里需要特别说明终点：当最后一个完整 update 到达 `128,450,560` 且 terminal artifacts 完整时，应算 `COMPLETE`。它相对 nominal 130M 少约 1.2%，是 rollout/update 离散化造成的正常边界，不是失败、不是补跑理由，也不应阻塞下一批 K-opt。

M1 中必须分开的方法身份：

- ordinary `curv256 + critic GGN256`；
- `EF255+1 + critic GGN256`；
- `FullEmp/FullEF + FullGGN`；
- Kaczmarz on/off 及 K-opt 外层选择。

### 4.3 M2：小批量 no-shared Transformer

当前新线把 actor 和 critic 设为不共享的 body-token Transformer：

| 项目 | 设置 |
|---|---|
| 网络 | actor/critic 各自 hidden 64、1 层 Transformer、4 heads、FF multiplier 2 |
| rollout | 32 env × 256 = 8192 transitions |
| update | 4 epochs × 8 minibatches |
| 预算 | 10M transitions |
| PPO | fixed LR `3e-4`、clip `0.2`、L2 clip `0.5` |
| Emp256 / EF255+1 / FullEmp | actor SGD LR `0.05`、critic LR `0.1`、adaptive KL target `0.008`、damping `0.03`、L2 clip `0.5` |
| critic | GGN；Emp/EF 用 256 rows，FullEmp 用完整 rows |

这一线从旧 FVP clip 转向原 RAT 风格 L2 clip，是因为 `KL(old || current)` 对非线性参数做完整二阶微分得到的是 full parameter Hessian，并不自动等于 PSD empirical Fisher；因此观察到负二次型时，不能用 `clamp_min(0)` 假装理论问题消失。要使用 Fisher norm clip，必须是 score-Fisher 或 detached-current local-KL 定义；否则 controlled continuation 使用 L2 clip。

### 4.4 当前 MuJoCo 运行

截至 08:18 UTC 的最后可信 Bede 快照：

| 作业 | 状态 | 说明 |
|---|---|---|
| `1072326_0-1` | `COMPLETE` | `tf-g7-10m` 的两个单元结束；精确方法/环境/seed 尚未从 Bede artifact 映射回来 |
| `1072326_2-4` | `RUNNING` | 位于 `gpu002/gpu014/gpu018`；继续运行，不重提重复项 |
| `1072326_5-17` | `PENDING (Resources)` | 已在队列中等待；不是失败，也不应重复提交 |

双 5060 主机在 09:00 UTC 有 8 个 `rlstack5060/mujoco-rat:cu128` 容器，GPU0/GPU1 分别约 96%/84%。当前 batch root 是 `/home/zzz/rlstack5060/workspaces/perf_runs/global7env_missing52_5060_20260816`，补的是 Transformer 全局表缺失 cell，统一为 3M gate、momentum `.5`、`fisher_l2`、target KL `.008`：

| GPU | 方法 | 环境 | seed | damping |
|---:|---|---|---:|---:|
| 0 | Emp256 | HalfCheetah | 6 | .03 |
| 0 | Emp256 | HalfCheetah | 6 | .05 |
| 0 | EF255+1 | HalfCheetah | 6 | .03 |
| 0 | Emp256 | HalfCheetah | 5 | .10 |
| 1 | Emp256 | Walker2d | 2 | .03 |
| 1 | Emp256 | Ant | 5 | .05 |
| 1 | Emp256 | HalfCheetah | 1 | .05 |
| 1 | Emp256 | Hopper | 6 | .03 |

八个 persistent workers 为 `missing52_w0` 到 `missing52_w7`，任务清单位于 `.../global7env_missing52_5060_20260816/tasks/worker0.tsv` 到 `worker7.tsv`。这些是当前明确的 active duplicates，另一个 AI 在任一 cell 结束并登记前都不应重复启动。

当前 M2 最有价值的明确缺口是 Humanoid `EnergyFree255p1` independent-anchor seed4：旧运行仅到 `737,280/10,000,000` 后死亡。是否续跑取决于 checkpoint 是否包含 policy、optimizer、normalizer、critic 和统计状态；否则应创建不覆盖旧 root 的 clean replacement。

### 4.5 已有 MuJoCo 证据应怎样使用

旧 no-shared MLP 五 seed、约 10M、last-10 汇总中，RAT EF+GGN 或 EF255+1 在多数环境显著高于固定 LR PPO；例如：

| 环境 | PPO fixed LR | RAT EF+GGN tuned | 最佳 EF255+1（d=.03/.1） |
|---|---:|---:|---:|
| Ant | 4565 | 5551 | 5204 |
| HalfCheetah | 5859 | 6646 | 6783 |
| Hopper | 2295 | 3439 | 3642 |
| Walker2d | 4658 | 5175 | 4789 |
| Humanoid | 3188 | 6953 | 6599 |
| HumanoidStandup | 138235 | 195154 | 163660 |
| Swimmer | 149 | 339 | 336 |

这些数据说明曲率线值得继续，但它们是**旧 MLP 架构的历史参考**，不能直接当作当前 Transformer 的完成结果。Transformer 全局表仍要求同架构、三 seed、约 10M、相同 last-10 口径；Hopper 的两个已知失败组合也要保留为失败，而不是从表中消失。

## 5. Isaac Lab

### 5.1 研究目标与任务

Isaac Lab 把同一类策略优化方法放到大规模并行机器人任务：

- `Isaac-Ant-v0`；
- `Isaac-Velocity-Flat-Unitree-A1-v0`；
- `Isaac-Velocity-Flat-Anymal-C-v0`。

核心几何通常为 `4096 env × 24 steps = 98,304 transitions/update`，4 minibatches、5 epochs，目标约 u500 / 50M transitions。每个 one-GPU sbatch 可并行三个独立 task child，但每个 child 必须有独立 root/stdout/stderr/rc；一个 child 失败应使 bundle 返回非零，却不能因此删除另外两个 child 的有效结果。

### 5.2 主要方法矩阵

| 方法 | 角色 |
|---|---|
| Official PPO | SB3 joint Adam 控制；历史 matched 版本 LR `0.001`、entropy `0.005` |
| PPO entropy=0 | 去掉 entropy 的因果控制，排除 entropy bonus 带来的差异 |
| K-FAC | Kronecker 结构自然梯度控制；需保持相同三任务、seed 和 u500 |
| Emp256 | 256 score anchors 的样本空间直接解；actor/critic 网络与 PPO 对齐 |
| EF255+1 | 255 普通行 + 1 anchor/free-rho；需检查 `K D_rho + μI` 的非对称/能量语义 |
| DirectionObserver | 不只是追 reward；在 u100/u200/u300/u400/u500 抽取方向、KL、残差、optimizer displacement 等诊断 |

### 5.3 当前主线超参数语义

当前保守主线曾选择：momentum `0.8`、target KL `0.012`、adaptive LR `[1e-4, 3e-3]`、PPO warmup 100、rolling pre-step reference、无 line search、critic 不变。它是为了减少多 minibatch 累积 behavior-KL 漂移。

另有两类因果变体必须单独标记：

- **line-search + momentum rescale**：接受步长为 `αΔ` 后，同步缩放 momentum state；
- **no-rescale**：参数仍走 `αΔ`，但 momentum buffer 保留完整 `Δ`。它不是“无 line search”，而是专门测试 state/parameter 不一致是否影响下一步。

不要把 no-rescale 的结果合并进保守 no-line-search 主线。

### 5.4 已完成的 matched 基线

旧的 SB3 大批量 3 seed × 3 task 对照已完成到 update 509：

| Task | PPO official trailing-10 | PPO entropy=0 trailing-10 | Emp256 trailing-10 |
|---|---:|---:|---:|
| Ant | 112.878 | 138.304 | 93.569 |
| Unitree A1 | 37.123 | 38.327 | 33.503 |
| ANYmal-C | 22.350 | 23.954 | 23.068 |

这组证据说明：该早期 Emp256 配置在 Ant/Unitree 落后，但 ANYmal 与 official PPO 接近或略高；Unitree seed44 还出现明显方差。它说明“直接解数值正常”不等于“策略性能一定更好”，因此后续才引入 rolling reference、warmup、自适应 KL、K-FAC 对齐和 EF255+1。

### 5.5 2026-08-16 当前 Isaac 运行

CSF3 job `18670421`：

- 名称：`ilab-dir-v2-u500`；
- 平台：gpuL / L40S；
- task：Ant；seed `42`；
- 方法：Emp256 DirectionObserver v2；
- 当前日志约 iteration/update `356`、`34,996,224` transitions、`n_updates=1775`（SB3 每 iteration 记录 5 个 optimizer updates）；
- 最新可见 `ep_rew_mean ≈ 103`；
- `scheduler_kl_mean ≈ 0.00948`，solve residual mean 约 `3.1e-7`；
- line search 接受率 1，当前 momentum `0.7`；
- L40S 采样约 24% utilization、6.2 GiB。对 MLP policy 来说 GPU utilization 不高，但日志高速增长，因此不是空闲作业。

stderr 有 Warp `cuDeviceGetUuid`/driver compatibility warning 和 SB3 的 MLP-on-GPU 性能提示；训练仍持续推进，不能据此判为 CUDA hard failure。日志里的 `rollout_scheduler_kl=nan` 是未填充的诊断 sentinel；实际 scheduler KL 与 residual 有限。

近期还有若干已完成的 DirectionObserver/audit，以及被取消或失败的 Emp256/K-FAC/EF255+1 尝试。它们必须保留各自状态，不能因为当前 u500 运行健康就覆盖：

- `18669563`：audit 到 u200，完成；
- `18668390/18668391`：Emp256 seed43/44 warmup 中断，取消；
- `18669209`：K-FAC seed42 三任务 bundle 中断；
- `18669306`：EF255+1 damping .03 seed42 bundle 中断；
- `18642877`、`18649409`：startup CUDA failure，已记为早停失败。

## 6. 计算资源与调度分工

| 平台 | 主要角色 | 约束与注意事项 |
|---|---|---|
| CSF3 gpuA | Procgen A100 80GB，尤其 Joint-2B/Joint-B | 适合大样本系统；当前有 held arrays；Jupyter sidecar 应从后续训练脚本移除 |
| CSF3 gpuH | Isaac H200 多 child bundle、部分 Procgen/MuJoCo probe | `JobHeldUser` 与 GPU/算法故障分开解释 |
| CSF3 gpuL | Isaac L40S、部分 MuJoCo | 对 SB3 MLP，低瞬时 utilization 不等于无进展 |
| Bede | Procgen 纯 PPO、MuJoCo 大批量/Transformer 队列 | ppc64le + Slurm；一个 sbatch 可并行 3 child；可提交约 1–2 个 GPU wave 等待，不能制造大规模陈旧 backlog |
| 3×双 4090 主机 | 直接运行 Procgen/MuJoCo/Isaac 的独立 anchor、补洞和矩阵 | 每个 PID 必须有 host/GPU/root/config/seed 映射；`ws4090-31` 整机隔离 |
| 双 5060 | MuJoCo Transformer 与 Procgen recovery | 容器并发根据显存/CPU/进度动态定；当前 8 个 MuJoCo trainer 已映射，均视为 active duplicate |

### 6.1 远端登录拓扑

以下只列登录入口，不记录密码或私钥内容。通常由本地先登录 CSF3，再由 CSF3 使用 alias 访问各工作站：

| 逻辑名 | 登录入口 | GPU | 当前可用性 | 主要数据位置 |
|---|---|---|---|---|
| `csf3` | 本地执行 `ssh csf3`；用户 `h99859yz` | Slurm 的 A100 80GB、H200、L40S | 可用，主控制面 | `/scratch/h99859yz/` |
| `bede` | 本地或 CSF3 执行 `ssh bede`；`yihe@bede.dur.ac.uk` | V100 32GB Slurm 节点 | 集群可用，但 CSF3 的免交互通道当前失效 | `/nobackup/projects/bdman37/yihe/` |
| `procgen-3090` | 从 CSF3 执行 `ssh procgen-3090`；`root@162.14.139.38 -p 32546` | 8×RTX 3090 | 当前遥测/容量未知 | `/root/procgen_*` |
| `ws4090-92` | 从 CSF3 执行 `ssh ws4090-92`；`yihe@130.88.192.92` | 2×RTX 4090 24GB | 09:00 UTC 两卡空闲采样；无批准 launcher | `/home/yihe/` 下各 MuJoCo/Isaac root |
| `ws4090-76` | 从 CSF3 执行 `ssh ws4090-76`；`yihe@10.49.7.15` | 2×RTX 4090 24GB | 09:00 UTC 两卡空闲采样；无批准 launcher | `/home/yihe/rat_dualenergyfree255p1_20260806/` 等 |
| `ws4090-31` | 从 CSF3 执行 `ssh ws4090-31`；`yihe@10.49.7.54` | 2×RTX 4090 24GB | **整机隔离，计为 0 卡** | 只保留旧日志，不启动新任务 |
| `procgen-5060` | `ssh -p 60023 zzz@47.114.81.212` | 2×RTX 5060 Ti 16GB | 可用，当前两卡均忙 | `/home/zzz/rlstack5060/` |

CSF3 登录节点是负载均衡入口。控制器固定运行在 `login2.csf3.man.alces.network`；若需要附着监管 screen，应先确保位于 login2，再执行 `screen -r codex-three-domain`。Bede 的 GPU 工作必须经 Slurm；CSF3 的 GPU 工作也必须经 Slurm，不能在登录节点直接训练。

### 6.2 按实验板块划分远端

#### Procgen 在哪里跑

| 远端 | 承担的 Procgen 工作 | 当前/历史状态 |
|---|---|---|
| CSF3 gpuA / A100 80GB | Joint-2B、Joint-B、Schur/RHS/阻尼/交叉块诊断和正式 gate | 当前 `18670696_0-1` 跑 1M BigFish/BossFight，`_2-3` 排队；root `/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1` |
| Bede / V100 | 纯 PPO DMLP1024 控制 | `1070573–1070576`：4 环境 × 3 seed 共 12 根完成；formal root `/nobackup/projects/bdman37/yihe/procgen_ppo_dmlp1024_bede_20260812_v1/formal_4env_x3seed_6m_20260812_v1` |
| `procgen-3090` / 8×3090 | 原 PPG E-v2、ACTOR_G/H/J/K，以及 P1 shared Exact-GGN/RAT | 原始根主要为 `/root/procgen_ppg_dmlp1024_20260810_v1`、`/root/procgen_ef_actor_ablation_20260806_v1`、`/root/procgen_ef_adaptivekl_exactrat_20260806_v1`；当前容量未知，旧失败/中断必须保留 |
| 双 5060 | ACTOR_J DMLP1024 seed0 clean recovery | BigFish、CaveFlyer、CoinRun 已完成；campaign `/home/zzz/rlstack5060/campaigns/procgen_actorj_dmlp1024_recovery_20260811_v1`；当前 GPU 已转跑 MuJoCo |
| `ws4090-92` | 一部分 Joint-2B/Schur/mujoco-ported 行归一化诊断 | 当前两卡空闲，无 Procgen trainer；仅在批准 exact launcher 后才可复用 |

#### MuJoCo 在哪里跑

| 远端 | 承担的 MuJoCo 工作 | 当前/历史状态 |
|---|---|---|
| Bede / V100 | Transformer no-shared 七环境全局表、大批量 Curv256/K-opt | 最近可信状态 `1072326_0-1` 完成、`_2-4` 运行、`_5-17` 资源排队；当前免交互遥测失效，必须先刷新 |
| 双 5060 | Transformer no-shared Emp256/EF255+1 全局缺失 cell 和 Hopper tuning | 当前 8 个 3M trainer，精确 GPU/环境/seed/damping 见 4.4；root `/home/zzz/rlstack5060/workspaces/perf_runs/global7env_missing52_5060_20260816` |
| `ws4090-92` | M1 Curv256/K-true/K-opt、FullEF/FullGGN、Humanoid 诊断 | 多个 u499/500 terminal run 已完成；一个 exact-FVP seed1 在 u429 失败保留；当前两卡空闲 |
| `ws4090-76` | M2 EF255+1 independent-anchor 七环境五 seed | 原 parent 已退出；Humanoid seed4 只到 737,280/10M，当前两卡空闲但 continuation launcher 未批准 |
| CSF3 | 历史 M2 七环境矩阵和部分 Transformer probe | `18302268` 多数 element 已完成、`_10` 失败待映射；当前 CSF3 没有 MuJoCo GPU job |
| `procgen-3090` | 曾承担部分 MuJoCo/Procgen 混合直接任务 | 当前遥测未知；在任务身份和容量恢复前不投新工作 |

#### Isaac 在哪里跑

| 远端 | 承担的 Isaac 工作 | 当前/历史状态 |
|---|---|---|
| CSF3 gpuL / L40S | SB3 PPO/Emp256/EF255+1、DirectionObserver 与 no-rescale 因果对照 | 当前 `18670421` 在 `node878` 跑 Ant seed42 u500；root `/scratch/h99859yz/isaaclab_sb3_matched` |
| CSF3 gpuH / H200 | K-FAC、三任务 bundled formal seeds、Emp256/EF255+1 | 近期多个完成/取消尝试；当前无 gpuH running job，但仍需按 catalog/launcher 调度 |
| `ws4090-92` | 早期 matched seed42、Unitree study、Optuna trial008 | trial008 u500 完成；09:00 UTC 两卡空闲，不等于可直接新开重复 trial |
| `ws4090-31` | 早期 matched seeds43/44 和 direct Isaac | 后续出现 CUDA 初始化失败，现整机隔离；旧 manifest 仍用于结果 provenance，不能继续调度 |
| 双 5060 | 已配置 Isaac/SB3 与非 SB3 基础环境 | 当前没有 Isaac trainer，GPU 正全部用于 MuJoCo |
| Bede | 可承接经过 ppc64le 验证的 Isaac/MuJoCo bundle | 当前没有可信实时 Isaac running 证据；不能仅因队列空白推断可启动 |

### 6.3 CSF3 控制器应承担的闭环

理想闭环是：

1. 刷新 scheduler、进程、GPU、日志、artifact 和错误；
2. 只比较 matched runs；
3. 登记完成、失败、取消和早停；
4. 按“缺 seed → 缺环境 → 必要控制 → 已定义超参数点”排序下一项；
5. 证明无重复、root 不覆盖、launcher 已审核、GPU 可用后才提交；
6. 在新结果出来后重新判断扩 seed、补环境、换控制或淘汰配置。

当前这个闭环正在 `login2` 常驻执行：

- `screen`：`codex-three-domain`，detached；
- loop：`/scratch/h99859yz/codex_three_domain_controller_20260807/controller_loop.sh`，周期 1200 秒；
- planner：`gpt-5.6-sol`，`model_reasoning_effort="medium"`；
- `APPROVED_TASK_CATALOG.tsv` 只有表头；
- `actions/approved_launchers.tsv` 为空；
- 因此 controller 可以持续审计和排序，但任意新 launcher 仍应被 deterministic executor 拒绝。当前新出现的 `18670696` 需要在 provenance 中确认是谁提交以及为何通过/绕过了旧白名单状态，不能仅凭作业出现就假定安全门已经更新。

本文更新只进行了只读远端核查和本地 Markdown 修改，没有提交、取消或修改远端训练。

## 7. 三大板块的统一评价表

| 维度 | Procgen | MuJoCo | Isaac |
|---|---|---|---|
| 主要性能量 | episodic reward，环境内 matched | last-10 `eprewmean`，跨环境 rank/normalized score | trailing-10 reward、AUC/learning efficiency |
| 稳定性 | entropy、behavior KL、aux EV/MSE、collapse window | seed stderr、KL、clip、Fisher/L2 norm | matched seeds、KL drift、regression count |
| 线性系统 | residual、block/cross norm、方向余弦 | residual、Fisher/GGN identity、Kaczmarz | direct/K-FAC residual、direction norm、optimizer displacement |
| 完成门槛 | 先 500k/1M/2M gate，再 6M × 3 seed | 10M 小批量；约 128.45M 大批量完整终点 | u500/约 50M，三任务 matched seeds |
| 不可混合 | PPO vs PPG vs Joint；不同 seed/架构 | shared vs no-shared；MLP vs Transformer；v3 vs v4 | exact-path vs no-rescale；不同 warmup/reference/seed |

## 8. 当前结论：哪些可以说，哪些还不能说

### 可以说

- Procgen DMLP1024 纯 PPO 在 Bede 的 4 环境 × 3 seed 已完成，证明扩大的决策 trunk 能稳定训练。
- ACTOR_J seed0 在三个环境完成，但 BossFight 低于价值阈值并已作为失败保留，说明 PPG/EF 组合并非跨环境自动稳胜。
- Joint-B 500k 四环境 gate 已全部完成且求解数值基本健康；同配置 1M gate 已开始运行/排队。
- MuJoCo 旧 MLP 五 seed 表明 RAT EF+GGN/EF255+1 在多个环境优于固定 LR PPO，足以支持继续做 Transformer 和大批量验证。
- Isaac 早期 matched Emp256 的直接解稳定，但总体没有普遍超过 PPO；新 rolling/adaptive/direction-observer 线是在修正优化语义，而不是重复旧实验。

### 还不能说

- 不能说 Joint-B/Joint-2B 已经优于 Procgen PPO/PPG；当前 500k 只是 gate。
- 不能由 seed0 或单个环境宣布 ACTOR_J/Exact-RAT 稳健胜出。
- 不能把 MuJoCo 旧 MLP 表当成当前 Transformer 的结果，也不能从 Hopper 一项选全局 damping。
- 不能把 Bede pending、`JobHeldUser`、wall-time 或缺 `mujoco.py` 当作算法失败。
- 不能用 Isaac seed42、trial008 或 u300 临时曲线替代 u500 matched multi-seed 结论。
- 不能因为当前 GPU utilization 低就取消 Isaac；进度与数值日志都在增长。

## 9. 下一步优先级

### 9.1 立即的管理优先级

1. 保持 CSF3 `login2` 上的常驻 controller，修复 Bede 免交互监控通道；不要再用 `login3` 的空 screen 列表判断 controller 状态。
2. 把 `18670696`、`18670421`、Bede `1072326` 持续映射到 exact task/config/seed/root；双 5060 当前八个 cell 已完成映射，应纳入 duplicate registry。
3. 清理后续 launcher 中不必要的 Jupyter sidecar；当前已在跑任务不因 sidecar 名称粗暴取消。
4. 给经过审核的 exact launcher 建立窄白名单；在此之前继续 `Executable Now = None`。

### 9.2 科学优先级

1. **Procgen**：500k 四环境 gate 已完成；继续当前 1M gate 的 BigFish/BossFight，并让 CaveFlyer/CoinRun 排队项自然启动。完成后做四环境 matched gate 汇总；PPG 线优先补 BigFish `E_v2/J seed1` 成对证据。
2. **MuJoCo**：让 Bede `1072326` 既有队列自然排空并恢复 cell 映射；让双 5060 当前 8 个已映射的 missing52 workers 先完成，再做跨主机去重并决定是否补 Humanoid EF255+1 seed4；Swimmer-v3 缺口单列修复。
3. **Isaac**：让 `18670421` 到 u500，导出 u100/200/300/400/500 的 reward、KL、direction/residual；随后只补同 identity 的 matched seeds，不再插入新的 line-search factorial。

## 10. 给下一个 AI 的接手说明

接手时把本文视为 2026-08-16 09:00 UTC 的快照，而不是永久实时状态。第一轮只读刷新顺序：

1. `ssh csf3` 后运行 `hostname` 和 `squeue -u "$USER"`；如果需要看 controller，连接 `login2.csf3.man.alces.network` 再检查 `screen -ls`，不要在随机落到的 `login3` 上误判。
2. 读取 `/scratch/h99859yz/codex_three_domain_controller_20260807/state/RESEARCH_INTENT.md`、`BOARD.md`、`RESOURCE_MAP.md`、`CYCLE_SUMMARY.md` 和 `live_snapshot.txt`。
3. 刷新 CSF3 job `18670696`、`18670421` 的 `scontrol`、`sacct`、trainer logs、artifact 和 allocation-scoped GPU telemetry。
4. 恢复/刷新 Bede 后再解释 `1072326`；当前没有权限把 08:18 UTC 状态说成实时状态。
5. 只读检查双 5060 的八个 workers 和 tasks TSV；结束的 cell 先登记结果，再让对应 worker 启动队列下一项。
6. 对 `ws4090-92`、`ws4090-76` 和 `procgen-3090`，先核对 process command、GPU、root 和 status，再判断空闲；不要碰隔离的 `ws4090-31`。

未经新的明确授权，不应：取消整个 heterogeneous array、覆盖旧 root、删除失败记录、把空闲 GPU 自动解释为应提交新变体，或绕过 approved launcher/catalog。任何状态汇报都应区分当前直接验证、控制器最近快照和历史 artifact。

## 11. 证据与文件位置

### 本地

- Procgen pure-PPO Bede campaign：`/Users/user/Documents/procgen/remote_launch_staging/procgen_ppo_dmlp1024_bede_20260812_v1/README.md`
- Procgen strict Joint-2B 说明：`/Users/user/Documents/procgen/work/procgen_joint2b_dmlp1024_3090_20260811_v1/README.md`
- MuJoCo Transformer configs：`/Users/user/Documents/mujoco/transformer_noshared_stage/configs/`
- MuJoCo K-opt worker：`/Users/user/Documents/mujoco/run_curv256_ktrue_clip_momentum_optuna_worker.py`
- MuJoCo 历史 last-10 表：`/Users/user/Documents/mujoco/analysis_outputs/mujoco_method_curves_20260719/final_last10_summary.csv`
- Isaac matched 结果：`/Users/user/Documents/issac/reports/isaaclab_sb3_emp256_matched_20260805/summary.md`
- Isaac K-FAC aligned scripts：`/Users/user/Documents/issac/experiments/isaaclab_sb3_kfac_emp256_aligned_20260816/`

### CSF3

- 三域控制器：`/scratch/h99859yz/codex_three_domain_controller_20260807/`
- Procgen Joint-B：`/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/`
- Isaac matched root：`/scratch/h99859yz/isaaclab_sb3_matched/`

---

本文是“研究身份 + 当前状态”的主说明，不替代逐 run 的 artifact 表。以后更新时，应保留本版中的失败/早停记录，只追加新的 matched 证据，并同步修改顶部的实时状态截点。
