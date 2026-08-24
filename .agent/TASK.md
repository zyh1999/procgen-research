# Procgen Task

Status: READY
Task-ID: PROCGEN-BXB-GGN-VS-PAPER-RAT-FORMAL-6M-X3-20260824-05

## 唯一科学目标

建立一个正式、严格配对的四环境 × 6M × seeds `0,1,2` 比较：

- Target：P1 B×B deterministic critic-GGN，symmetric-FP64/Jacobi；
- Baseline：仓库中定义的 original Paper RAT。

最终形成每种方法12个逻辑cell、共24个逻辑cell的严格矩阵。优先恢复并复用已有严格完成终点，只运行缺失cell。

Low-Fisher guard结论已为`GUARD_NOT_HELPFUL`，不得进入正式矩阵。

## 冻结方法身份

### Target

历史P1 shared Exact-GGN/shared-RAT中的B×B deterministic critic-GGN配置。候选配置为：

`adv_resnet_shared_exact_deterministic_ggn_symfp64.yaml`

必须通过历史命令、源码、配置、日志和artifact确认其唯一身份，包括：

- deterministic critic GGN；
- B×B系统；
- symmetric FP64；
- Jacobi语义；
- actor更新、damping、clip、solver及共享网络语义。

### Baseline

必须使用仓库原始Paper RAT的准确source/config/algorithm身份。不得根据名称将
`ACTOR_K`、ACTOR_I、P1、Joint-B、Joint-2B或其他RAT变体替代为Paper RAT。

Target相对Baseline只允许存在预先定义的critic-curvature构造及其直接求解/遥测差异。网络、actor、训练schedule、数据、预算、seed和评估协议必须相同。

若任一方法的唯一身份无法恢复，结论为`PRECHECK_BLOCKED`，不得启动实验。

## 实验矩阵

环境：

- `bigfish-easy-0-10`
- `bossfight-easy-0-10`
- `caveflyer-easy-0-10`
- `coinrun-easy-0-10`

Seeds：`0,1,2`

预算：每cell 6M nominal transitions，并使用仓库正式终止更新语义；若既定协议的最后完整更新为`5,980,160`，必须保持该语义。

每个方法共12个逻辑cell。历史cell仅在完整满足严格匹配、预算、终点指标和artifact完整性时复用。

## 启动前证据门

1. 完整读取最新`.agent/GOAL.md`、`STATE.md`、`TASK.md`、
   `AGENT_REPORT.md`及相关报告。
2. Executor刷新全部授权资源的scheduler、GPU、进程、所有权、容量、日志、
   artifact和错误状态，自主决定具体调度方案。
3. 恢复Target和Paper RAT的：
   - source/trainer/config/launcher路径与SHA256；
   - 完整命令、依赖和训练环境；
   - 网络、参数、actor/critic、schedule及评估语义；
   - 历史job、root、status、rc、progress、checkpoint和终点指标。
4. 对24个逻辑cell分别标记：
   - `REUSE_STRICT_COMPLETE`
   - `LAUNCH_MISSING`
   - `NOT_STRICT`
   - `IDENTITY_BLOCKED`
5. `REUSE_STRICT_COMPLETE`必须同时满足准确方法身份、相同环境/seed、正式6M终点、
   PASS/rc0、完整日志和可验证artifact。
6. 历史P1 seed1基础设施失败不得复用或覆盖；若需重新运行，必须使用新root。
7. 冻结唯一manifest和两方法逐字段diff；验证所有新root不存在且无重复活跃任务。
8. 做非科学训练式import、依赖、命令和配置preflight。不得增加新的短预算算法gate。

任一方法存在`IDENTITY_BLOCKED`，或两方法除预定critic-curvature差异外不能严格匹配时，不得启动任何cell。

## 科学严格匹配

两组必须匹配：

- exact Procgen环境分布与seed；
- IMPALA/ResNet encoder、decision trunk及全部heads；
- actor目标、advantage、entropy、KL、clip和更新schedule；
- rollout、minibatch、epochs及正式训练预算；
- PopArt、auxiliary phase和value目标语义；
- reward logging、evaluation window及checkpoint协议；
- damping、precision及除目标critic差异外的solver设置；
- source bundle、依赖版本和随机种子传播。

不得跨环境比较reward，不得混用不同KL字段，不得将scheduler completion视为科学完成。

## 计算要求与调度边界

- 计算需求为最多24个6M逻辑cell，仅提交未被严格复用的cell；
- 需要支持冻结软件环境、FP64曲率求解、完整日志和终点checkpoint；
- Executor在实时检查后自主决定主机、partition、GPU数量、并发度和队列安排；
- 调度选择不得改变算法身份、实验矩阵、seed、预算或评估语义；
- `.54`、`ws4090-31`及`10.49.7.54`继续隔离；
- 不得使用Jupyter。

## 允许动作

- 恢复和验证历史Target/Paper RAT证据；
- 新增最小manifest、不可变配置、launcher和batch入口；
- 提交严格缺失的正式cells；
- 监控全部已提交cell至终态；
- 更新`.agent/STATE.md`、`.agent/AGENT_REPORT.md`和专属报告；
- 提交并推送`agent-work`。

允许为可移植性或结构化遥测作最小代码调整，但必须对两组一致应用并证明不改变科学语义。

## 必需证据

每个逻辑cell记录：

- reuse/launch决定及证据；
- method、environment、seed、预算和实际transitions；
- git commit、source/trainer/config/launcher SHA256；
- 完整命令及冻结依赖；
- Executor选择的调度位置、job/raw ID、时间、状态和exit code；
- 唯一且非碰撞的artifact root；
- status、rc、progress、trace、stdout/stderr及checkpoint完整性；
- terminal reward和准确KL字段；
- critic loss/EV、GGN/RAT solver residual、damping、clip及数值健康；
- Traceback、NaN/Inf、OOM、通信、磁盘、依赖、配置和停滞扫描。

失败分类：

- `algorithm-failure`
- `numerical-failure`
- `infrastructure-failure`
- `queued/quota-waiting`
- `unknown/insufficient-evidence`

## 3/5核算

对每个环境和seed计算：

`Target terminal reward / strict Paper RAT terminal reward`

- 比例低于`3/5`或前期明确崩溃，只标记`early-stop-candidate`；
- 本任务不得因此取消或删除run；
- Paper RAT cell不严格或不完整时标记`not-evaluable`；
- 必须保留所有历史算法、数值、基础设施失败和取消记录。

## Required Outputs

生成：

`.agent/reports/PROCGEN-BXB-GGN-VS-PAPER-RAT-FORMAL-6M-X3-20260824-05.md`

报告必须包含：

1. Target与original Paper RAT身份证明及逐字段diff；
2. 24-cell manifest与reuse/launch表；
3. 历史终点恢复结果；
4. 两方法四环境×三seed终点结果表；
5. 每环境mean、std、median和paired seed差值；
6. reward、KL、critic/solver及健康指标分析；
7. 3/5 early-stop accounting；
8. 完整、不可变的历史失败与取消账本；
9. 唯一状态：
   - `FORMAL_COMPARISON_COMPLETE`
   - `FORMAL_COMPARISON_PARTIAL_INFRASTRUCTURE`
   - `PRECHECK_BLOCKED`
10. Delivery HEAD、evidence/report commit、push验证和最终工作树状态。

Executor callback必须直接粘贴身份diff、24-cell状态表、正式结果表、3/5核算、
失败账本及唯一状态。

## Acceptance Criteria

- 只推进一个B×B deterministic critic-GGN target；
- Baseline是准确的original Paper RAT，不是相近变体；
- 所有可复用历史cell均通过完整严格匹配；
- 所有新cell使用非覆盖root且无重复提交；
- 科学完成cell达到正式6M终点、PASS/rc0并具有完整artifact；
- 两方法除预定critic-curvature差异外严格一致；
- low-Fisher guard未被包含；
- 历史失败和取消provenance完整保留；
- Planner未指定具体资源放置，Executor记录其调度依据；
- 报告已提交并推送至`origin/agent-work`。

## Prohibited Actions

- 不得测试第二个deterministic-GGN候选；
- 不得用ACTOR_K、P1、Joint-B等名称相似性替代Paper RAT身份验证；
- 不得重跑严格完成cell；
- 不得覆盖或删除任何历史root、checkpoint、日志或失败记录；
- 不得自动重试基础设施失败或执行early stop；
- 不得修改seed集合、环境集合、6M预算或评估语义；
- 不得启动low-Fisher guard；
- 不得使用Jupyter或隔离资源；
- 不得规划MuJoCo或Isaac；
- 不得提交无关文件。

## 提交与推送

启动前提交冻结的identity manifest、配置和launcher；提交信息包含：

`PROCGEN-BXB-GGN-VS-PAPER-RAT-FORMAL-6M-X3-20260824-05`

终态后提交报告和状态，推送至`origin/agent-work`，验证远端HEAD并在callback中报告。
