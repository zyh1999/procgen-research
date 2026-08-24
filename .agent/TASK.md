# Procgen Task

Status: READY
Task-ID: PROCGEN-PAPER-SEPARATEB-DETGGN-6M-S0-20260824-07

## 唯一科学目标

构造并评估一个新的、唯一的 deterministic critic-GGN 候选：

`PAPER_MATCHED_SEPARATE_B_DET_GGN_V1`

它必须从 exact original Paper RAT 出发，完整保留 Paper RAT actor 系统，只将
Paper RAT 的 sampled critic curvature 替换为 deterministic critic GGN。
四个环境的 seed0 均以 6M 为预定训练终点，并按照严格同阶段 Paper RAT 指标执行
不早于 2M 的 3/5 early-stop。

上一候选的 joint-2B coupling 导致高KL和LR快速降至`.0001`，结论为
`GATE_FAIL`；本候选通过取消joint coupling来隔离critic-GGN本身，不得调整LR、
KL阈值或其他actor超参数补偿失败。

## 冻结算法身份

### 必须与 original Paper RAT 完全一致

- shared IMPALA/ResNet hidden256 网络及全部 heads；
- actor sampled score、actor RHS及actor B×B系统；
- actor inverse/solve、update composition和更新顺序；
- initial LR `.5`；
- adaptive-KL在每个minibatch后执行；
- KL thresholds `.005/.04`；
- SGD momentum `1e-6`；
- original `rhs - H @ momentum_buffer` history correction；
- rollout `4096`、minibatch `512`、epochs `4`；
- damping/global clip `.5/.5`；
- PopArt、GAE、entropy、ratio、reward、evaluation及checkpoint语义；
- seed传播和6M正式停止协议。

### 唯一允许的科学替换

只替换critic分支：

- sampled value score/unit pseudo-advantage critic system
  → deterministic `J_v`/critic residual GGN；
- critic lambda `.1`、objective coefficient `1`；
- critic使用独立B×B系统；
- critic solve采用symmetric FP64、Jacobi和Cholesky；
- 增加必要的critic GGN、Jacobi和relative-residual telemetry。

明确禁止：

- joint-2B stacking；
- actor-critic cross blocks；
- 从critic系统改变actor matrix、RHS或actor direction；
- P1 LR `.004`、rollout-level KL、momentum `0`或disabled history；
- low-Fisher guard或任何新的step-calibration超参数。

## 实现与身份Preflight

必须从original Paper RAT trainer/config复制出独立文件，不原位修改Paper RAT、
P1或上一失败候选。

启动前生成机器可审计diff及回归测试，证明：

1. actor matrix、RHS、solve、momentum/history和adaptive-KL路径与Paper RAT一致；
2. 在固定synthetic batch且禁用critic贡献时，新Target与Paper RAT的actor
   direction、LR/KL状态更新达到bit-identical或明确容差内一致；
3. critic matrix维度为B×B，而不是2B×2B；
4. deterministic `J_v`、residual、lambda `.1`及critic RHS正确；
5. actor/critic cross blocks不存在；
6. FP64/Jacobi/Cholesky solve有限且relative residual满足现有严格容差；
7. 非法P1字段、joint-2B和low-Fisher字段会被配置校验拒绝；
8. 新trainer/config/launcher/manifest SHA256冻结；
9. 四个新root不存在，且无相同method/env/seed/budget任务。

若任何条件失败，以`PRECHECK_BLOCKED`结束，不得启动训练。

## 6M候选矩阵

仅运行：

| Environment | Seed | Intended horizon |
|---|---:|---:|
| `bigfish-easy-0-10` | 0 | 6M |
| `bossfight-easy-0-10` | 0 | 6M |
| `caveflyer-easy-0-10` | 0 | 6M |
| `coinrun-easy-0-10` | 0 | 6M |

正式完整终点使用仓库既定语义，预期最后完整更新约`5,980,160` transitions。

Original Paper RAT不重跑；使用现有strict-complete相同环境、seed0进度文件。

## Stage-matched Early-stop协议

Executor必须实现持久化monitor，但不得在Target低于2M transitions时进行
scientific-futility取消。

预声明检查阶段：

- 首个双方共有且不低于2M的完整logged transition；
- 首个双方共有且不低于4M的完整logged transition；
- 6M正式终点。

每次检查必须使用：

- 同一环境；
- 同一seed；
- 完全相同transition；
- 相同`eprewmean`/evaluation window语义；
- 对应Paper RAT中间行，而非Paper 6M终值。

若在任一不早于2M的检查点：

`Target reward < 0.60 × Paper RAT reward`

则允许并要求取消该Target cell，保存：

- method/config/seed/environment；
- 检查transition；
- Target与Paper reward；
- ratio；
- scheduler、日志、trace、checkpoint和取消时间；
- `EARLY_STOPPED_ALGORITHM`分类。

基础设施失败、用户取消、race取消和算法early-stop必须分开记录。不得删除或把
stale RUNNING marker解释为仍在运行。

## 计算要求与调度边界

- 需求：最多四个独立、6M intended-horizon Procgen cells；
- 需要支持critic FP64 B×B直接求解、完整trace、stage monitor及checkpoint；
- Executor刷新全部授权资源的scheduler、GPU、进程、所有权、容量、依赖、
  artifact和重复任务后，自主决定主机、partition、GPU数量、并发和队列；
- 资源选择不得改变算法、环境、seed、预算、检查阶段或evaluation语义；
- 不得使用Jupyter；
- `.54`、`ws4090-31`和`10.49.7.54`继续隔离。

## 允许动作

- 新增此单一候选的trainer、config、manifest、测试、launcher和monitor；
- 运行非科学训练式import/config/regression preflight；
- preflight通过后提交四个6M-horizon cells；
- 执行上述stage-matched monitor和授权的逐cell early-stop；
- 只读解析original Paper RAT匹配行；
- 更新`.agent/STATE.md`、`.agent/AGENT_REPORT.md`及专属报告；
- 提交并推送`agent-work`。

除预声明3/5 early-stop外，不得自动改变配置或重启失败cell。

## 必需证据

### 方法身份

- Paper、donor、上一失败候选及新Target的source/config SHA256；
- Paper→Target逐字段和执行路径diff；
- actor-equivalence及critic-GGN回归测试输出；
- 完整冻结命令和依赖。

### 每个Target cell

- environment、seed、intended budget和实际transitions；
- Executor记录的job及调度证据；
- 唯一root、status、rc、progress、trace、stdout/stderr和checkpoint；
- 每个stage的Target/Paper reward、ratio、KL及LR；
- actor direction/clip、momentum/history和adaptive-KL telemetry；
- critic loss/EV、GGN/Jacobi/Cholesky residual及数值健康；
- Traceback、NaN/Inf、OOM、通信、磁盘、依赖、配置和停滞扫描；
- terminal或early-stop的准确原因和时间。

失败分类：

- `algorithm-failure`
- `EARLY_STOPPED_ALGORITHM`
- `numerical-failure`
- `infrastructure-failure`
- `queued/quota-waiting`
- `cancelled-nonscientific`
- `unknown/insufficient-evidence`

## 唯一候选结论

报告只能给出一个：

- `CANDIDATE_PROMOTE_TO_3SEED`：
  4/4 cells达到6M正式终点、PASS/rc0，无算法/数值异常，且各环境所有检查点均
  不低于strict Paper RAT的3/5。
- `CANDIDATE_NOT_READY`：
  任一cell触发scientific early-stop或出现算法/数值失败。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：
  方法身份成立，但仅因基础设施导致矩阵不完整。
- `PRECHECK_BLOCKED`：
  无法证明Paper actor等价或独立B×B critic-GGN身份。

不得将部分环境成功表述为四环境正式成功。

## Required Outputs

生成：

`.agent/reports/PROCGEN-PAPER-SEPARATEB-DETGGN-6M-S0-20260824-07.md`

必须包含：

1. 新候选精确定义和全部SHA256；
2. Paper→Target严格diff；
3. actor-equivalence和critic-GGN测试；
4. 四环境各stage的严格对照表；
5. 终点/early-stop状态、reward、KL、LR和solver健康；
6. 所有失败及取消分类；
7. 唯一候选结论；
8. 不可变历史账本，包括：
   - joint-2B Paper-matched candidate `GATE_FAIL`；
   - CoinRun用户授权futility early-stop；
   - gpuL race-loser cancellation；
   - gpuA/gpuL preflight基础设施失败；
   - P1、ACTOR_J、旧cancelled arrays及Bede失败；
   - low-Fisher `GUARD_NOT_HELPFUL`；
9. Delivery HEAD、evidence/report commit、push验证和最终工作树状态。

Executor callback必须直接粘贴严格diff、测试结果、stage表、失败账本和唯一结论。

## Acceptance Criteria

- 只定义一个separate-B deterministic critic-GGN候选；
- actor路径经测试与Paper RAT严格一致；
- critic GGN为独立B×B且无cross blocks；
- 四个任务均以6M为intended horizon；
- scientific early-stop只在>=2M、严格同阶段比较后执行；
- 未使用Paper 6M终值检查中间Target；
- 所有root非碰撞，历史artifact和失败记录未被覆盖；
- Planner未指定具体资源放置；
- 报告已提交并推送至`origin/agent-work`。

## Prohibited Actions

- 不得重新测试上一joint-2B失败候选；
- 不得定义第二个新候选或进行超参数sweep；
- 不得改变Paper actor LR、KL timing、momentum/history或网络schedule；
- 不得加入cross blocks、joint-2B、low-Fisher或Kaczmarz；
- 不得增加seeds1/2；
- 不得重跑Paper RAT；
- 不得在2M前按reward执行early-stop；
- 不得自动重试基础设施失败；
- 不得覆盖、删除或弱化历史root、日志和失败；
- 不得使用Jupyter或隔离资源；
- 不得规划MuJoCo或Isaac；
- 不得提交无关文件。

## 提交与推送

训练前提交冻结的trainer/config/manifest/tests/launcher/monitor，提交信息包含：

`PROCGEN-PAPER-SEPARATEB-DETGGN-6M-S0-20260824-07`

终态后提交报告和状态，推送至`origin/agent-work`，验证远端HEAD并在callback中报告。
