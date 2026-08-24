# Procgen Task

Status: READY
Task-ID: PROCGEN-PAPER-MATCHED-DETERMINISTIC-GGN-1M-GATE-20260824-06

## 唯一科学目标

定义并验证唯一的新方法：

`PAPER_MATCHED_DETERMINISTIC_GGN_V1`

它必须完整保留 original Paper RAT 的网络、actor优化器和训练schedule，仅将
Paper RAT的critic-curvature/solve替换为P1的deterministic critic-GGN构造、
symmetric-FP64/Jacobi求解及必要遥测。

完成四环境、seed0、1M的严格因果执行门控。此任务不启动6M正式矩阵；通过后，
下一轮才可将该唯一配置推进至四环境 × 6M × seeds0,1,2。

## 重新定义的冻结身份

必须从已验证的original Paper RAT trainer/config出发进行最小修改，不得从历史
P1代码出发再反向修补actor字段。

### 必须保持与Paper RAT完全一致

- shared ResNet hidden256网络及全部heads；
- rollout `4096`、minibatch `512`、epochs `4`；
- PopArt、GAE、entropy、ratio范围及数据/evaluation语义；
- initial LR `.5`；
- adaptive-KL在每个minibatch后执行；
- momentum `1e-6`；
- original history correction启用；
- KL thresholds `.005/.04`；
- damping/global clip `.5/.5`；
- seed传播、停止协议、reward和checkpoint语义。

### 唯一允许改变的科学部分

从P1准确移植并冻结：

- deterministic `J_v`/critic residual构造；
- critic lambda `.1`；
- 已验证的joint-2B deterministic critic-GGN系统；
- symmetric FP64；
- Jacobi；
- Cholesky/直接求解语义；
- residual、Jacobi、GGN健康遥测。

不得带入P1的initial LR `.004`、rollout-level adaptive-KL、momentum `0`或
disabled history correction。

原Paper RAT和历史P1文件不得原位修改；必须新增独立trainer/config/manifest。

## 严格因果审计门

启动前必须生成机器可审计diff，逐项证明：

1. Paper actor/network/schedule代码路径保持不变；
2. adaptive-KL调用频率仍为每minibatch；
3. momentum/history状态创建、更新和修正与Paper RAT一致；
4. 唯一执行差异属于critic curvature、joint solve及必要telemetry；
5. 没有隐含的LR、clip、epoch、loss weighting或optimizer差异；
6. 新方法的source/trainer/config/launcher SHA256已冻结；
7. original Paper RAT的12个6M cells及artifact未被修改。

同时增加最小回归测试，验证：

- initial LR、KL更新时间、momentum和history correction；
- deterministic critic score/RHS、lambda和joint-2B维度；
- FP64/Jacobi/solver路径；
- 非法P1 actor字段不能通过配置验证。

若严格diff或测试失败，任务以`REDEFINITION_BLOCKED`结束，不得启动训练。

## 1M执行门控

仅运行新Target：

| Environment | Seed | Budget |
|---|---:|---:|
| `bigfish-easy-0-10` | 0 | 1M |
| `bossfight-easy-0-10` | 0 | 1M |
| `caveflyer-easy-0-10` | 0 | 1M |
| `coinrun-easy-0-10` | 0 | 1M |

使用既定完整更新语义，预期终点约`1,007,616` transitions。

Baseline不重跑：从已验证original Paper RAT seed0进度文件中抽取完全相同
transition点的在线指标。不得用6M terminal值与1M Target直接比较。

所有Target使用全新、不可碰撞、包含方法/environment/seed/budget的root。

## 计算要求与调度边界

- 计算需求：四个独立1M Procgen训练cell，需支持FP64 joint solve、完整trace和
  checkpoint；
- Executor先刷新全部授权资源的scheduler、GPU、进程、所有权、容量、依赖、
  重复任务和artifact，再自主决定主机、partition、GPU数、并发和队列；
- 资源安排不得改变冻结算法、四环境、seed0、预算或评估语义；
- 不得使用Jupyter；
- `.54`、`ws4090-31`和`10.49.7.54`继续隔离。

## 允许动作

- 新增独立trainer、config、manifest、测试和launcher；
- 运行非训练式import/config/unit/regression preflight；
- 严格审计通过后提交四个1M Target cells；
- 监控至终态并解析现有Paper RAT的匹配1M行；
- 更新`.agent/STATE.md`、`.agent/AGENT_REPORT.md`及专属报告；
- 提交并推送`agent-work`。

基础设施失败只记录，不得自动修改配置或覆盖root重跑。

## 必需证据

### 身份证据

- original Paper RAT、historical P1及新Target的source/config SHA256；
- Paper→Target逐字段和执行路径diff；
- 回归测试命令、输出及覆盖字段；
- 新Target完整命令和冻结依赖。

### 每个Target cell

- environment、seed、预算和实际transitions；
- job/raw ID及Executor记录的调度证据；
- 唯一root、status、rc、progress、trace、stdout/stderr和checkpoint；
- reward、准确KL字段、terminal LR；
- critic loss/EV；
- GGN/Jacobi/solve residual、condition/health及clip telemetry；
- actor momentum/history和adaptive-KL执行计数；
- Traceback、NaN/Inf、OOM、通信、磁盘、依赖、配置和停滞扫描。

### 严格1M比较

对每个环境报告相同transition点的：

- Target reward与Paper reward；
- Target/Paper reward比例；
- KL及actor-update健康；
- critic/solver健康；
- 不得跨环境聚合reward后再比较。

失败分类必须区分：

- `algorithm-failure`
- `numerical-failure`
- `infrastructure-failure`
- `queued/quota-waiting`
- `unknown/insufficient-evidence`

## 门控结论

报告只能给出一个：

- `GATE_PASS`
  - 严格diff和测试通过；
  - 四个Target均科学完成、PASS/rc0；
  - 无数值/solver/actor-schedule异常；
  - 至少3/4环境的Target/Paper reward比例不低于`3/5`。
- `GATE_FAIL`
  - 严格身份成立，但出现算法/数值崩溃，或至少2/4环境低于`3/5`。
- `GATE_INCONCLUSIVE_INFRASTRUCTURE`
  - 身份成立，但基础设施使科学结果不完整。
- `REDEFINITION_BLOCKED`
  - 无法实现唯一critic-side差异，或actor/schedule严格匹配失败。

`3/5`仅用于核算和下一轮决策；本任务不得据此执行early stop。

## Required Outputs

生成：

`.agent/reports/PROCGEN-PAPER-MATCHED-DETERMINISTIC-GGN-1M-GATE-20260824-06.md`

报告必须包含：

1. 新方法精确定义；
2. Paper→Target严格diff；
3. 回归测试结果；
4. 四环境Target与同进度Paper RAT表；
5. reward比例、KL及critic/solver/actor健康；
6. 失败分类和唯一门控结论；
7. 完整历史失败/取消账本，包括：
   - 历史P1 seed0不可新鲜复用；
   - P1 seed1四个基础设施失败；
   - ACTOR_J历史失败；
   - 已取消的旧arrays；
   - low-Fisher `GUARD_NOT_HELPFUL`；
8. Delivery HEAD、evidence/report commit、push验证及最终工作树状态。

Executor callback必须直接粘贴严格diff、测试结果、四环境对照表、失败账本和唯一结论。

## Acceptance Criteria

- 只定义一个新的deterministic critic-GGN配置；
- Paper actor optimizer和schedule严格保留；
- P1仅贡献允许的critic/solve/telemetry部分；
- 原Paper RAT和历史P1源码/artifact未被修改；
- 四个1M Target roots唯一且非覆盖；
- Baseline来自同seed、同环境、同transition点的original Paper RAT；
- 未把1M gate表述为6M正式结论；
- 历史失败和取消provenance完整保留；
- Planner未指定具体资源放置；
- 报告已提交并推送至`origin/agent-work`。

## Prohibited Actions

- 不得恢复或推进历史P1原配置；
- 不得测试第二种deterministic-GGN候选；
- 不得改变Paper initial LR、adaptive-KL timing、momentum或history correction；
- 不得运行low-Fisher guard、Joint-B、Joint-2B变体或其他actor ablation；
- 不得启动6M或seeds1/2；
- 不得重跑Paper RAT baseline；
- 不得覆盖、删除或弱化历史root和失败记录；
- 不得自动重试失败cell或执行early stop；
- 不得使用Jupyter或隔离资源；
- 不得规划MuJoCo或Isaac；
- 不得提交无关文件。

## 提交与推送

训练前先提交冻结的新trainer/config/manifest/tests/launcher，提交信息包含：

`PROCGEN-PAPER-MATCHED-DETERMINISTIC-GGN-1M-GATE-20260824-06`

终态后提交报告和状态，推送至`origin/agent-work`，验证远端HEAD并在callback中报告。
