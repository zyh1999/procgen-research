Status: READY

# TASK.md

Task-ID: `PROCGEN-NORMMATCH-V2-MP-MAIN-NATURAL-STATE-READONLY-20260825-30`

## 唯一目标

在不改变任何接受策略、manifest、冻结探针或科学代码的前提下，只读证明自然 Python 3.9 multiprocessing 启动过程中 `__main__` 与 `__mp_main__` 的实际来源、状态转换和 backing-source 关系，为下一轮决定能否建立严格的非对象同一性分类器提供充分证据。

本任务不得假设或要求 `sys.modules["__main__"] is sys.modules["__mp_main__"]`；Task29 已在自然时序中反驳该关系。

## 范围

仅限 NormMatch V2 当前冻结执行链中的 multiprocessing 模块语义取证。

保持以下内容字节不变：

- NormMatch V2 trainer、config、regression、monitor。
- Hermetic bundle、manifest、科学与 preflight launcher。
- Task23 non-reentrant hook。
- Task25 Torch pseudo-origin classifier。
- Task27 runtime semantic binding。
- Task28R exact frozen-probe alias validator。
- 所有既有报告、失败账本和实验根目录。

不得修改 origin acceptance、bundle manifest 或 `sys.modules`。

## 允许动作

1. 实现一个纯观察、无训练、无接受决策的取证 harness。
2. 在至少三个独立 clean process 中复现自然启动时序。
3. 在以下里程碑采集快照：

   - child-process entry；
   - closure probe 开始；
   - trainer import 前后；
   - production model construction 后；
   - origin scan 前。

4. 每个快照记录：

   - `__main__`、`__mp_main__` 是否存在；
   - 两者的对象 ID 与对象同一性；
   - module type、MRO、`__name__`、`__file__`、`__spec__`、loader、package、origin；
   - module dictionary 的对象 ID、键集合和规范化内容差异；
   - backing file 的 raw/resolved 路径、`lstat/stat`、device/inode、UID/GID、mode、size、fd identity 与 SHA256；
   - 顶层 code object 的 filename、name、firstlineno 和可稳定摘要；
   - backing source 是否严格对应 Task28R frozen probe、bundle 中已列明文件或其他来源。

5. 将观察到的状态转换逐项映射到冻结 CPython 文件及行号：

   - `multiprocessing/__init__.py` SHA `a5a42976033c7d63ee2740acceef949a3582dcb0e0442845f9717e1be771c68b`；
   - `multiprocessing/spawn.py` SHA `16ce6d81f8b5ef7228e5500bff04b37bdceb3d7dfc8d6de3ad523598798c43f4`；
   - `spawn_main -> _main -> prepare -> _fixup_main_from_path`。

6. 设置无 observer 的对照进程，证明 observer：

   - 不提前导入 `multiprocessing`；
   - 不创建、替换或重绑定 `__main__`/`__mp_main__`；
   - 不改变 module cardinality、import order、RNG、resolved config、模型参数或 Task27 telemetry；
   - 不触发额外文件加载或 origin-scan 差异。

## 必需证据

- 至少三个独立自然时序复现及一个无 observer 对照。
- 每个里程碑的完整规范化快照和 SHA256。
- `__main__` 与 `__mp_main__` 的逐字段差异表。
- backing-file/source/code-object 等价性或非等价性的精确证明。
- CPython 状态转换与冻结源码行号的对应表。
- observer 非扰动证明。
- Task29 两次失败继续保留：

  - premature-import observer：`infrastructure-failure/proof-observer-import-timing`；
  - natural non-alias：`precheck-failure/task29-natural-mp-main-not-exact-main-object-alias`。

- 冻结科学哈希、Task27 telemetry 与历史失败账本未改变的证明。

## 验收标准

仅允许以下一个终局结论：

- `NATURAL_MP_MAIN_RELATIONSHIP_PROVEN`：独立复现一致，observer 非扰动，并精确证明自然 `__mp_main__` 的创建来源、backing source、状态转换以及它与 `__main__` 的非对象同一关系；
- `NO_SAFE_ALIAS_RELATION`：证据显示不存在足够严格、稳定且不可伪造的来源关系；
- `OBSERVER_PERTURBED`：无法证明观察器不改变自然状态；
- `INSUFFICIENT_EVIDENCE`：所需字段或复现完整性不足。

本任务即使得到 `NATURAL_MP_MAIN_RELATIONSHIP_PROVEN`，也不得直接批准该模块；分类器设计由下一轮 Planner 决定。

## 禁止事项

- 不得创建或修改 `__mp_main__` classifier、allowlist、manifest 或 origin policy。
- 不得按模块名 `__mp_main__`、文件名或目录进行宽泛放行。
- 不得重新要求或声称 `__main__ is __mp_main__`。
- 不得修改冻结科学代码、算法、bundle、launcher、monitor 或 Task28R 修复。
- 不得启动 formal audit、四环境 preflight 或科学训练。
- 不得创建实验根、checkpoint、模型或科学 monitor。
- 不得重试既有任务、覆盖旧根或删除失败记录。
- 不得使用 Jupyter。
- 不得访问 `.54`、`ws4090-31` 或 `10.49.7.54`。
- 不得规划 MuJoCo 或 Isaac。
- Planner 不指定主机、GPU、partition、卡数、并发或队列位置；所有实时资源判断归 Executor。

## 报告字段

Executor 必须报告：

- Task-ID 与唯一终局结论；
- assignment、implementation/evidence/delivery commits；
- origin/agent-work 推送验证；
- 所有冻结文件 SHA256；
- 各独立进程和对照的启动方式与非扰动证明；
- 每个里程碑的模块状态表；
- backing-file、fd、源码和 code-object 身份表；
- CPython 源码状态转换映射；
- 复现一致性及所有矛盾；
- scheduler/process/artifact/error 的终态扫描；
- 明确声明没有 classifier、policy、manifest、preflight、science、root、checkpoint 或 monitor；
- 完整保留的失败与取消账本；
- 下一轮若要设计严格分类器，仍缺少的精确证据。

## 提交、推送与回调

- 更新 `.agent/STATE.md` 和 `.agent/AGENT_REPORT.md`。
- 写入 `.agent/reports/PROCGEN-NORMMATCH-V2-MP-MAIN-NATURAL-STATE-READONLY-20260825-30.md`。
- 提交全部只读证据与报告，不提交模型或 checkpoint。
- 推送到 `origin/agent-work`，并验证远端 HEAD。
- 回调 Planner 时提供唯一终局结论、Delivery HEAD、evidence commit、报告路径及关键状态转换证据。
