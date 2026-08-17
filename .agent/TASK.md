# Procgen Task

Status: READY
Planner-Kind: ChatGPT
Planner-Thread-ID: 6a8309ee-bb34-83eb-9512-72acc5913334
Executor-Callback: Wake this same Planner after AGENT_REPORT is pushed.
Task-ID: PROCGEN-PLANNER-HANDOFF-20260817-02

## 唯一目标

将 `PROCGEN-READONLY-REFRESH-20260817-01` 已采集的证据整理成一份完整、可直接随 Executor callback 交付给 Planner 的 Procgen 证据包，并对易过期状态做一次只读增量核验。

本任务不决定或启动下一项实验。

## 输入与范围

仅限 `zyh1999/procgen-research` 的 `agent-work` 分支。

必须以以下已交付版本为起点：

- Delivery HEAD：`62371cb`
- Evidence commit：`c9099117a1f62af35dc7ff430c9908503a849491`

完整读取并交叉核对：

- `.agent/GOAL.md`
- `.agent/STATE.md`
- `.agent/TASK.md`
- `.agent/AGENT_REPORT.md`
- 上述文件直接引用的配置、结果表、日志和 artifact

不得涉及 MuJoCo 或 Isaac。

## 允许动作

- 只读核验上述提交、控制文件及直接引用证据；
- 以 CSF3 为控制面刷新 Procgen 相关 scheduler、GPU、进程和最新日志状态；
- 必要时只读检查 Bede、双 5060 或其他已登记远端；
- 生成 `.agent/PLANNER_HANDOFF.md`；
- 更新 `.agent/AGENT_REPORT.md`，记录本次核验和交付；
- 提交并推送这些控制面文件到 `agent-work`。

除报告文件外，不得产生持久修改。

## 必需证据

`.agent/PLANNER_HANDOFF.md` 必须自包含，并明确列出：

1. 两条既定 Procgen 研究线的准确名称、目标和边界；
2. 每条研究线最高严格匹配 baseline；
3. 严格匹配字段：环境、算法、网络、训练预算、seed、评估协议和指标语义；
4. 每个已完成、运行中、排队、失败及未开始配置；
5. 每配置、每 seed 的 reward、KL 和核心 Procgen 指标；
6. baseline 对照及达到严格 baseline 的 seed 数；
7. checkpoint、日志、结果表等 artifact 路径和完整性；
8. 所有历史失败配置、证据、原因和分类；
9. `early-stop-candidate` 及其 3/5 或前期崩溃依据；
10. 当前缺失证据、矛盾和阻塞项；
11. Evidence commit 与本次增量刷新之间的状态变化。

易过期部分必须重新记录 UTC 时间、主机及只读查询证据：

- scheduler job ID、状态、节点、运行时间和退出码；
- GPU 型号、利用率、显存及 PID/job 对应；
- 相关进程及启动时间；
- 最新日志 mtime、最后有效进度；
- traceback、NaN/Inf、OOM、磁盘、通信、配置错误和停滞扫描；
- artifact 新增、缺失或损坏情况。

## 失败分类

必须保留并区分：

- algorithm-failure
- numerical-failure
- infrastructure-failure
- queued/quota-waiting
- unknown/insufficient-evidence

不得删除、覆盖或弱化历史失败。低于最高严格匹配 baseline 的 3/5，
或前期明显崩溃，只能记录为 `early-stop-candidate`，不得执行早停。

## Required Outputs

1. `.agent/PLANNER_HANDOFF.md`：上述完整证据包；
2. `.agent/AGENT_REPORT.md`：记录核验时间、输入提交、变化摘要、输出路径、
   commit SHA、push 结果和最终工作树状态；
3. Executor callback 必须直接粘贴 `.agent/PLANNER_HANDOFF.md` 全文，
   不能只提供 GitHub URL、commit SHA 或一句“已完成”；
4. callback 同时给出新的 Delivery HEAD 和 evidence/report commit SHA。

## Acceptance Criteria

- Evidence commit `c9099117...` 与 Delivery HEAD `62371cb` 已验证；
- 两条研究线分别呈现，边界及严格匹配语义未被改写；
- 历史失败和 interrupted/不完整 provenance 全部保留；
- 易过期状态已做带时间戳的只读增量刷新；
- 配置、seed、指标、日志和 artifact 可以相互追溯；
- 不确定事实明确标为 `unknown/insufficient-evidence`；
- 未选择、启动、续跑、取消或重新排队任何实验；
- callback 中包含完整证据包正文；
- 报告提交已推送至远端 `agent-work`。

## Prohibited Actions

- 不得启动、续跑、重启、取消或调度实验；
- 不得修改算法、代码、配置、依赖或长期研究方向；
- 不得合并两条 Procgen 研究线；
- 不得放宽或重解释严格匹配；
- 不得使用 `.54` 或 `ws4090-31`；
- 不得启动或使用 Jupyter；
- 不得执行 early stop；
- 不得删除失败记录、日志、checkpoint 或 artifact；
- 不得把基础设施失败或调度等待归为算法失败；
- 不得规划或检查 MuJoCo、Isaac；
- 不得提交无关文件。

## 提交与推送要求

提交信息必须包含：

`PROCGEN-PLANNER-HANDOFF-20260817-02`

提交前检查 diff，确认仅包含 `.agent/PLANNER_HANDOFF.md`、
`.agent/AGENT_REPORT.md` 及必要的事实性 `.agent/STATE.md` 更新。
推送至远端 `agent-work`，并在报告和 callback 中记录最终 commit SHA。
