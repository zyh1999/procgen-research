# Procgen Task

Status: READY
Task-ID: PROCGEN-JOINT-LOWFISHER-CAVE-5SEED-GATE-20260818-04

## 唯一目标

完成 CaveFlyer 上 RHS-aligned Joint-B 与其 low-Fisher guard 严格配对的
5-seed、1M 因果门控，判断 seed0 中 guard 被实际触发且回报下降的现象是否可重复。

只扩展现有 seed0 严格配对证据；不得扩展到6M或其他环境。

## 已验证输入

- Unguarded parent：CSF3 `18670696_2`，CaveFlyer seed0，1,007,616；
  reward `2.78`，behavior KL `0.00844484`，current-step KL
  `2.75867e-05`，residual `5.90343e-13`。
- Guard control：CSF3 `18672560_2`，CaveFlyer seed0，1,007,616；
  reward `2.06`，behavior KL `0.008051`，current-step KL
  `5.72e-05`，residual `5.36e-13`；guard fraction `0.594445`，
  damping floor `0.033778`。
- 两者唯一预声明因果差异是：
  `joint_low_fisher_actor_critic_guard=true`,
  `high=.50`, `low=.20`, `max=.05`，以及读取、插值和遥测这些字段所需代码。
- Seed0不得重跑或覆盖。

## 范围与固定设计

仅运行 `caveflyer-easy-0-10`，新增 seeds `1,2,3,4`：

- Unguarded RHS-aligned Joint-B：4 runs；
- Low-Fisher guard严格control：4 runs；
- 合并既有seed0后，每种方法共5 seeds。

固定保持：

- rollout `4096`、minibatch `512`、epochs `4`；
- nominal 1M，按现有协议终止于约 `1,007,616` transitions；
- 相同IMPALA/ResNet、decision trunk、heads、数据与评估语义；
- actor Fisher、clean all-parameter critic GGN、full compressed cross terms；
- transformed RHS、`rhs_aligned_rank1_b`、FP64；
- actor damping `.003`、critic damping `.5`、原clip语义；
- momentum `0`、Kaczmarz `false`；
- 除guard预声明差异和seed外不得改变任何字段。

## 用户指定的执行角色边界

- ChatGPT Planner只负责代码、算法、实验设计和本任务的科学边界；不负责判断或指定实时GPU、主机、partition、并发和排队布局。
- Codex Executor必须自行刷新已授权资源的scheduler、GPU、进程、ownership和容量，并在不改变上述科学身份的前提下选择具体运行位置与并发方案。
- Planner文本中的CSF3仅表示控制面和可用执行选项，不构成固定卡位；Executor可在Bede、CSF3 gpuA/gpuH、ws4090-92、ws4090-76及其他已登记授权资源间作非重复调度。`.54/ws4090-31/10.49.7.54`继续隔离。

## 启动前必做

1. 完整读取最新 `.agent/GOAL.md`、`STATE.md`、`TASK.md`、
   `AGENT_REPORT.md`及本任务直接引用报告。
2. 以CSF3为控制面刷新scheduler、GPU、进程、日志、artifact和错误扫描。
3. 确认无相同method/seed的活跃或已完成run。
4. 创建两个不可变launcher/config清单并记录完整SHA256；逐字段证明两组仅有
   guard差异。
5. 使用全新、非碰撞、包含method和seed的输出根；验证所有8个目标目录不存在。
6. 做静态import/config/command preflight，不启动额外smoke训练。
7. 若严格diff失败、目标目录碰撞或源码身份无法冻结，禁止提交，报告
   `PRECHECK_BLOCKED`。

## 允许动作

- 为本门控新增最小launcher、manifest和scheduler脚本；
- 由Codex Executor按实时容量，在已授权资源上通过非Jupyter batch作业提交上述8个run；
- 合理使用可用GPU数量，不要求固定卡数；
- 只读监控至每个cell终态并采集结果；
- 更新`.agent/STATE.md`、`.agent/AGENT_REPORT.md`及专属报告；
- 提交并推送`agent-work`。

不得因单个run早期表现取消其他run。基础设施失败只记录，不自动改变配置重跑。

## 必需证据

每个run记录：

- commit、trainer/config/launcher SHA256及完整命令；
- job/raw ID、partition、node、时间、状态、exit code；
- environment、method、seed、预算及实际transitions；
- 输出根、status、rc、trace、stdout/stderr及checkpoint策略；
- terminal reward、behavior KL、current-step KL、solve residual；
- Fisher/critic damping、guard fraction、floor及clip telemetry；
- critic/auxiliary健康指标中现有实现可提供的全部字段；
- traceback、NaN/Inf、OOM、CUDA/NCCL、磁盘、配置及停滞扫描。

失败分类必须区分：
`algorithm-failure`、`numerical-failure`、`infrastructure-failure`、
`queued/quota-waiting`和`unknown/insufficient-evidence`。

## Required Outputs

生成：

`.agent/reports/PROCGEN-JOINT-LOWFISHER-CAVE-5SEED-GATE-20260818-04.md`

报告必须包含：

1. 启动前新鲜控制面快照；
2. 严格diff表和所有源码/config/launcher哈希；
3. seed0历史行加seeds1-4新结果的5-seed配对表；
4. 每seed的guard-minus-parent reward差、KL差、guard触发比例和floor；
5. paired wins、mean/median paired difference及离散程度；
6. guard低于parent `3/5` reward的seed计数；只标记
   `early-stop-candidate`，不得执行早停；
7. 数值/辅助健康比较及全部失败分类；
8. 明确结论之一：
   - `GUARD_REPRODUCIBLY_HELPFUL`
   - `GUARD_NOT_HELPFUL`
   - `INCONCLUSIVE`
9. 完整历史失败/取消账本，不得覆盖原始ACTOR_J、P1、Bede及取消记录；
10. Delivery HEAD、evidence/report commit、push验证和最终工作树状态。

Executor callback必须直接粘贴5-seed配对表、严格diff、失败分类和唯一结论。

## Acceptance Criteria

- 仅CaveFlyer seeds1-4、两种严格配对方法被提交；
- seed0 artifacts及所有历史根未被修改；
- 8个run均有可审计终态，或被准确归类为基础设施/调度失败；
- 所有科学完成run达到规定终止预算且PASS/rc0；
- 两组除guard和seed外完全匹配；
- 报告不把1M gate称为6M正式性能结果；
- 3/5规则仅评估、不执行；
- 报告和状态已提交并推送至`origin/agent-work`。

## Prohibited Actions

- 不得运行BigFish、BossFight、CoinRun或6M扩展；
- 不得运行E-v2、Pure-PPO、P1、Joint-2B或其他方法；
- 不得修改既有run、checkpoint、日志或artifact；
- 不得启动、使用或保留Jupyter；
- 不得访问`.54`、`ws4090-31`或`10.49.7.54`；
- 不得release/requeue已取消的`18642230`、`18624888`、`18666591`；
- 不得自动重跑失败cell或执行early stop；
- 不得删除、弱化或重新解释历史失败；
- 不得规划MuJoCo或Isaac；
- 不得提交无关文件。

## 提交与推送要求

运行前提交冻结的manifest/launcher，提交信息包含：

`PROCGEN-JOINT-LOWFISHER-CAVE-5SEED-GATE-20260818-04`

结果完成后提交报告与状态，推送至`origin/agent-work`，并验证远端HEAD。
