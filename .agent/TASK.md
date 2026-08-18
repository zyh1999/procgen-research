# Procgen Task

Status: READY
Planner-Kind: ChatGPT
Planner-Thread-ID: 6a8309ee-bb34-83eb-9512-72acc5913334
Executor-Callback: Wake this same Planner after AGENT_REPORT is pushed.
Task-ID: PROCGEN-JOINT-PROVENANCE-MAP-20260817-03

## 唯一目标

只读重建近期 Joint/PAP/FADP/RAT/Schur/RHS 作业的完整科学 provenance，确定已完成的 seed0 500k/1M Joint-B gate 是否存在严格匹配的 parent/control；在此结论明确前不得规划或启动正式扩展实验。

本任务仅属于 PPG/curvature 研究线，不改变 Pure-PPO DMLP1024 控制线。完成后必须把结论回传同一 ChatGPT Planner，由 Planner 决定哪个有希望的 deterministic critic-GGN 候选晋级正式 6M × 3 seeds。

## 范围

以 Delivery HEAD `f850e1d439642108763c630f137a9e97ebf07e76` 及 evidence/report commit `18c69bae5c47b9b0b7b5b708522f6866229d700d` 为输入，完整读取 `.agent/GOAL.md`、`STATE.md`、`TASK.md`、`AGENT_REPORT.md` 和 `PLANNER_HANDOFF.md`。

必须核验以下有限作业集合：

- CSF3 cancelled：`18642230`, `18624888`, `18666591`
- CSF3 recent：`18666610`, `18667225`, `18667467`, `18667627`, `18667792`, `18667941`, `18668461`, `18669377`, `18669429`, `18669454`, `18669530`, `18669613`, `18669615`, `18669725`, `18670437`, `18670696`, `18672560`
- Bede：`1072327`, `1072329`, `1072331`, `1072333`, `1072337`, `1072338`, `1072342-1072351`

其中 `18642230` 与 `18624888` 已于 2026-08-18 按用户授权取消，均为零运行时间、无科学产物的旧 Jupyter 数组；必须记录其精确身份、被替代关系和取消证据，不得删除历史记录。

不得扩大到无关作业或其他研究仓库。

## 严格匹配要求

不得仅凭 job name、scheduler completion 或相似算法名称认定匹配。每个 parent/control 必须与目标 Joint-B gate 核对：

- 相同 Procgen environment 和 seed；
- 相同网络、encoder、decision trunk、heads/PopArt/auxiliary 身份；
- 相同 rollout、minibatch、epoch、预算和终止更新语义；
- 相同数据、reward、evaluation 和 checkpoint 协议；
- 明确 actor Fisher、critic GGN、cross blocks、RHS、damping、clip、precision、solver/reduction、momentum/Kaczmarz 身份；
- exact source、trainer、config、launcher SHA256；
- 非目标差异只能是预先定义的单一 causal ablation。

Pure-PPO baseline、E-v2、P1、Joint-2B 和 Joint-B 不得互相替代。

## 允许动作

- 只读查询 scheduler/accounting、日志、命令、环境快照和 artifact；
- 只读解析现有 `progress.csv`、JSONL、stdout/stderr、status、rc、checkpoint 及提交历史；
- 以 CSF3 为控制面，只读核验 Bede、gpuA、gpuH、三台已授权 4090 和其他已登记远端；`.54/ws4090-31` 仍隔离；
- 更新 `.agent/AGENT_REPORT.md`、`.agent/STATE.md`；
- 新建 `.agent/reports/PROCGEN-JOINT-PROVENANCE-MAP-20260817-03.md`；
- 提交并推送控制面报告到 `agent-work`。

## 必需证据

先刷新并记录 UTC 时间、主机和查询命令：

- 当前 Procgen scheduler、GPU、进程及最新日志状态；
- 两组旧 held 数组的取消状态和零运行时间；
- 是否出现新的 Procgen trainer 或 artifact 变化；
- `.54`、`ws4090-31` 不得访问。

对范围内每个 job 及其可枚举 cell，报告：

1. job/raw ID、环境、seed、状态、节点、runtime、exit code；
2. 完整启动命令、工作目录、artifact root；
3. trainer/config/launcher 路径及 SHA256；
4. 方法的准确身份，不用简称猜测；
5. rollout、minibatch、epochs、预算和实际 transitions；
6. actor/critic/cross/RHS/damping/clip/precision/solver 语义；
7. reward、behavior KL、current-step KL、solver residual 及辅助健康指标；
8. status、rc、日志、trace、checkpoint 完整性；
9. traceback、NaN/Inf、OOM、CUDA、NCCL、磁盘、配置和停滞扫描；
10. 与 1M Joint-B gate 的逐字段匹配表；
11. 科学状态和失败分类。

无法恢复的字段必须写成 `unknown/insufficient-evidence`，并记录尝试读取的确切路径或查询；不得从 scheduler 名称推断。

## 历史失败保护

- ACTOR_J BossFight seed0：`algorithm-failure/EARLY_STOPPED_FAILED`，5.7933 对严格 E-v2 10.60，比例 0.5465；
- ACTOR_J BigFish/CaveFlyer/CoinRun 原始尝试：`infrastructure-failure`；
- P1 四个 seed1 根目录：`infrastructure-failure`；
- `18642230`、`18624888`：`cancelled-obsolete-unstarted`，零运行时间；
- `18666591` 及尚未映射作业：证据充分前保持 `unknown/insufficient-evidence`。

3/5 规则只评估，不执行 early stop。

## Required Outputs

1. 全部目标 job/cell 的 provenance matrix；
2. 1M Joint-B 与每个候选 parent/control 的逐字段差异表；
3. 每个候选判定：`strict-match`、`not-strict-match` 或 `insufficient-evidence`；
4. 唯一匹配结论：`STRICT_PARENT_COMPLETE`、`STRICT_PARENT_DEFINED_BUT_INCOMPLETE` 或 `NO_STRICT_PARENT_FOUND`；
5. 若存在严格 parent，提供同环境同 seed 同预算指标对照；不存在则明确缺少的最小因果对照，但不得自行提交实验；
6. 更新后的失败账本、artifact 清单、取消账本和当前运行状态；
7. 向 Planner 明确列出哪些配置满足或不满足晋级正式四环境 × 6M × seeds 0,1,2 的证据条件；晋级决策仍由 Planner 作出；
8. commit SHA、push 证据和最终工作树状态。

Executor callback 必须直接粘贴结论、provenance matrix、严格匹配差异表及失败账本，不能只返回文件链接或 commit SHA。

## Acceptance Criteria

- 所有枚举 job 均被逐项核验，没有静默遗漏；
- 500k/1M Joint-B 保持原身份，不被宣传为 6M 性能胜利；
- scheduler 完成与科学完成明确区分；
- 每个候选有可复核的严格匹配判定；
- 历史失败和取消 provenance 完整保留；
- 易过期状态已刷新；
- 本任务未启动、恢复、取消、重排或 early-stop 任何额外实验；
- 仅报告文件发生变更；
- 报告已提交并推送到 `origin/agent-work`。

## Prohibited Actions

- 不得启动、续跑、重启、取消、release hold 或重新排队任何额外实验；
- 不得修改训练代码、配置、依赖、checkpoint 或 artifact；
- 不得使用 Jupyter；
- 不得访问 `.54`、`ws4090-31` 或 `10.49.7.54`；
- 不得将 idle GPU 视为授权容量；
- 不得混合两条 Procgen 研究线或改变长期目标；
- 不得删除、弱化或重分类无充分证据的历史失败；
- 不得规划 MuJoCo 或 Isaac；
- 不得提交无关文件。

## 提交与推送要求

提交信息包含 `PROCGEN-JOINT-PROVENANCE-MAP-20260817-03`。推送至 `origin/agent-work`，并在 AGENT_REPORT 及 callback 中记录 Delivery HEAD、evidence/report commit、push 验证和最终工作树状态。
