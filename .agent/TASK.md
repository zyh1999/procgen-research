# Procgen Task

Status: READY
Planner-Kind: ChatGPT
Planner-Thread-ID: 6a8309ee-bb34-83eb-9512-72acc5913334
Executor-Callback: Wake this same Planner after AGENT_REPORT is pushed.
Task-ID: PROCGEN-READONLY-REFRESH-20260817-01

## 唯一目标

建立 Procgen `agent-work` 分支及相关计算节点的最新、可审计状态快照，恢复仓库既定的两条研究线与严格匹配 baseline 语义，为 Planner 下一轮选择唯一实验任务提供证据。

本任务只采集和报告状态，不启动、停止或修改实验。

## 范围

仅限：

`https://github.com/zyh1999/procgen-research/tree/agent-work`

首先确认本地对应远端 `agent-work` 最新提交，然后完整阅读：

- `.agent/GOAL.md`
- `.agent/STATE.md`
- `.agent/TASK.md`
- `.agent/AGENT_REPORT.md`
- 四个文件直接引用的报告、配置、代码、日志和结果表

必须从仓库原文恢复，不得凭记忆猜测：

1. 两条 Procgen 研究线的名称、目标和边界；
2. 各研究线最高严格匹配 baseline；
3. 严格匹配所要求的环境、算法、网络、训练预算、seed、评估协议和指标语义；
4. 已完成、运行中、排队、失败及未开始的配置。

## 允许动作

- 同步并只读检查 `agent-work`、提交历史、代码、配置、日志和 artifact；
- 以 CSF3 为决策控制面刷新 scheduler、GPU、进程及日志状态；
- 对 Bede、双 5060 和其他已登记远端进行必要的只读核验；
- 运行不会启动训练、不会改变 checkpoint 的只读解析/汇总；
- 更新 `.agent/AGENT_REPORT.md`；
- 仅用本轮已验证事实更新 `.agent/STATE.md`；
- 提交并推送上述控制面报告到 `agent-work`。

报告文件及其 git commit/push 是唯一允许的持久修改。

## 必需证据

所有易过期状态必须在本轮重新采集，并记录采集时间、时区、主机和命令：

- 当前 HEAD、远端 `agent-work` HEAD、工作树状态；
- scheduler：job ID、状态、partition、节点、运行时间和退出码；
- GPU：主机、型号、显存、利用率、PID/job 对应关系；
- 相关进程：PID、启动时间、命令、所属 job、存活状态；
- 每个活跃或最近结束任务的日志路径、mtime、最后有效进度；
- 最新 reward、KL 及仓库定义的核心 Procgen 指标；
- checkpoint、评估结果、汇总表等 artifact 的路径、大小、mtime和完整性；
- traceback、NaN/Inf、OOM、磁盘、通信、被杀任务、配置错误和静默停滞扫描。

旧的 STATE、TASK 或 AGENT_REPORT 只能用作历史线索，不能代替刷新。

## 状态与失败分类

分别为两条研究线建立配置表。每项标记为：

- completed
- running
- queued/quota-waiting
- algorithm-failure
- numerical-failure
- infrastructure-failure
- unknown/insufficient-evidence
- not-started

所有失败配置必须保留，并记录完整配置、seed、证据路径和分类理由。

按照仓库原有严格匹配语义，统计候选相对最高严格匹配 baseline 达标的 seed 数。低于 3/5 或前期明显崩溃时，只能标记 `early-stop-candidate`；不得执行停止、取消或重启。

## Required Outputs

`.agent/AGENT_REPORT.md` 必须包含：

1. Task-ID、执行时间窗、Executor、检查的 commit SHA；
2. 四个控制文件的读取确认及 SHA256；
3. 两条研究线及其严格匹配定义；
4. scheduler、GPU、进程的新鲜快照；
5. running、queued、completed 和 failed 作业清单；
6. 每配置、每 seed 的 reward、KL、核心指标及 baseline 对照；
7. artifact 清单与完整性判断；
8. 错误扫描结果；
9. 历史失败和 early-stop-candidate 表；
10. 每项失败的算法/数值/基础设施/调度配额分类；
11. 控制文件、日志和实际运行状态之间的矛盾；
12. 缺失证据与下一决策阻塞项；
13. 是否已具备规划下一项单一实验的充分证据；
14. 报告 commit SHA、push 结果和最终工作树状态。

## Acceptance Criteria

- 已读取最新 `agent-work` 的四个指定控制文件；
- 两条研究线分别汇报，未合并或改变研究目标；
- 各线最高严格匹配 baseline 及全部匹配字段可复核；
- scheduler、GPU、进程、日志、reward/KL、artifact 和错误均已刷新；
- 每个作业可追溯到配置、job ID、日志和 artifact；
- 失败类型与排队/配额等待明确区分；
- 历史失败配置完整保留；
- 3/5 规则只用于评估，未执行 early stop；
- 未启动、续跑、重启、取消或重新排队实验；
- 仅控制面报告发生预期变更；
- 报告已提交并成功推送到远端 `agent-work`。

## Prohibited Actions

- 不得启动、续跑、重启、取消或重新排队训练/评估；
- 不得修改算法、训练代码、配置、依赖或长期研究方向；
- 不得合并两条 Procgen 研究线；
- 不得放宽或重新解释严格匹配语义；
- 不得使用 `.54` 或 `ws4090-31`；
- 不得启动或使用 Jupyter；
- 不得执行 early stop；
- 不得删除失败配置、日志、checkpoint 或 artifact；
- 不得把基础设施失败或调度等待记为算法失败；
- 不得检查、规划或执行 MuJoCo、Isaac；
- 不得提交无关文件。

## 提交与推送要求

1. 更新 `.agent/AGENT_REPORT.md`，必要时更新 `.agent/STATE.md`；
2. 检查 diff，确认不存在代码、配置或实验状态修改；
3. 提交信息必须包含 `PROCGEN-READONLY-REFRESH-20260817-01`；
4. 推送至远端 `agent-work`；
5. 在报告末尾记录 commit SHA、push 成功证据和最终工作树状态。
