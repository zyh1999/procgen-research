Status: READY

# TASK.md

Task-ID: `PROCGEN-NORMMATCH-V2-MP-MAIN-INPATH-CAPTURE-READONLY-20260825-31R`

## 唯一目标

进行最后一次有界、只读取证：利用冻结 closure/origin-scan 路径中已经存在的 Python 模块遍历和记录对象，捕获自然 `__main__`/`__mp_main__` 状态，不新增 observer import 或审计框架；证明 capture-on 与 capture-off 对照的 import order 和规范化运行时科学证据一致。

本任务只解决 Task30 的 observer 扰动缺口，不批准任何模块，不启动 NormMatch V2 科学实验。

## 范围与冻结边界

保持字节不变：

- NormMatch V2 trainer、config、regression、monitor；
- bundle、manifest、science/preflight launcher；
- Task23 hook、Task25 classifier、Task27 semantic binding；
- Task28R exact-probe validator；
- 所有 origin acceptance、allowlist 和 scientific identity。

允许版本化一个仅用于取证的 closure-probe 副本，但不得改变原冻结探针。

## 唯一允许的实现

在现有 origin scan 已经取得 loaded-module record 的位置，对其内存记录进行序列化：

- 不得增加任何 import；
- 不得提前 import `multiprocessing`；
- 不得注册新的 audit hook、trace、profile或 import hook；
- 不得读取会触发 lazy import 的属性；
- capture-on 与 capture-off 必须执行同一版本化代码；唯一差异是是否把已经计算出的记录写入证据文件；
- 写出只能使用该路径在 capture 分支之前已加载并实际使用的 Python 对象。

## 必需捕获字段

在以下自然里程碑保留 `__main__`/`__mp_main__` 证据：

1. child entry；
2. closure-probe start；
3. trainer import 前后；
4. production model construction 后；
5. origin scan 前。

每个里程碑记录：

- presence、对象ID及对象同一性；
- type/MRO；
- name、file、spec、loader、package、origin；
- module-dict身份、键集合及规范化摘要；
- backing raw/resolved path；
- `lstat/stat`、device/inode、UID/GID、mode、size；
- fd identity、SHA256；
- code-object filename/name/firstlineno及稳定摘要；
- backing 是否为精确 Task23 probe、部署版 Task27 preflight、bundle manifest文件或其他来源。

不得把对象ID、临时路径、时间戳或随机初始化值纳入跨进程稳定hash。

## 对照设计

在同一有界取证活动中运行：

- 至少两个 capture-on clean process；
- 至少两个 capture-off clean process。

四者必须使用相同冻结环境、入口和自然 multiprocessing 时序。不得为了观察而预先导入模块。

必须比较：

- 完整 import order；
- origin-scan module cardinality与规范化module集合；
- resolved config、结构manifest、connectivity和AST证据；
- Task27 wrapped/unwrapped telemetry；
- RNG状态摘要；
- critical stdout；
- Task28R最终拒绝点；
- 非观察性运行时证据hash。

## 验收标准

只有同时满足以下条件才可得出 `NATURAL_MP_MAIN_RELATIONSHIP_PROVEN`：

- capture-on 两次完整关系一致；
- capture-off 两次完整关系一致；
- capture-on/off import order与规范化module集合完全一致；
- Task27、RNG、config、network、AST、connectivity及critical stdout一致；
- 写证据是唯一可解释差异；
- 完整证明自然终态中：

  - `__main__` 与 `__mp_main__` 是不同对象；
  - `__main__` 精确背靠冻结Task23 probe；
  - `__mp_main__` 精确背靠部署版Task27 preflight；
  - 状态转换符合冻结CPython spawn源码链。

否则只能选择：

- `OBSERVER_PERTURBED`
- `INSUFFICIENT_EVIDENCE`
- `NO_SAFE_ALIAS_RELATION`

任一结论后立即停止。

## 硬停止与禁止事项

- 仅允许一次有界取证活动；失败后不得现场修复或重跑。
- 不得实现或批准 `__mp_main__` classifier。
- 不得修改origin policy、allowlist、manifest或bundle。
- 不得运行closure acceptance、formal audit、四环境preflight或科学训练。
- 不得创建科学root、transition、checkpoint、model或monitor。
- 不得修改或重做Task28R。
- 不得创建第二候选或泛化审计框架。
- 不得使用Jupyter。
- 不得访问`.54`、`ws4090-31`或`10.49.7.54`。
- 不得规划MuJoCo或Isaac。
- Planner不指定主机、GPU、partition、卡数、并发或队列；Executor负责所有实时资源判断。

## 必需证据与报告字段

必须报告：

- assignment、implementation、evidence、Delivery commits；
- origin/agent-work远端验证；
- 所有冻结SHA；
- capture-on/off逐进程字段与规范化hash；
- import-order和module-set精确diff；
- 非扰动判定及任何矛盾；
- CPython `multiprocessing/__init__.py` 与 `spawn.py` 行号映射；
- Task29/30失败账本完整保留；
- scheduler、process、artifact及hard-error终态；
- 明确声明没有classifier、policy、manifest、preflight、science、root、model或monitor；
- 唯一允许结论。

## 提交、推送与回调

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-MP-MAIN-INPATH-CAPTURE-READONLY-20260825-31R.md`

提交模型无关证据和报告，不提交模型/checkpoint。推送至`origin/agent-work`并验证远端HEAD后回调Planner。
