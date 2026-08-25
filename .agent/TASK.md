Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-MP-MAIN-EXACT-ALIAS-AND-6M-S0-20260825-29

## 唯一目标

仅证明并规范化冻结Python 3.9 multiprocessing中的`__mp_main__`执行别名，使其引用已批准的exact `__main__` backing module/file identity，而不是被误判为manifest外的新bundle模块。

不得重做Task 28R、扩展bundle manifest、建立通用allowlist或改变NormMatch V2。

## 证据判断

Task 28R exact-probe storage-alias修正已经PASS。唯一新失败是：

```text
bundle module absent from manifest or hash mismatch: __mp_main__
```

该异常出现在closure scan，不是trainer、GGN、NormMatch、数值或科学失败。`__mp_main__`通常由multiprocessing spawn为main module建立执行别名，但必须用实际Python 3.9证据证明其对象、source、loader和process-start关系，不能只按名称放行。

## 第一阶段：只读实际alias证明

在实际Python 3.9.25 multiprocessing child中记录：

- `sys.modules["__main__"]`和`sys.modules["__mp_main__"]`；
- 二者的Python object identity；
- module name、`__file__`、`__spec__`、loader、package和origin；
- module dictionary的逐字段差异；
- backing file raw/resolved path、samefile、device/inode、UID/GID、mode、size及SHA；
- 创建alias的stdlib multiprocessing函数、调用栈、source path及SHA；
- process start method；
- backing module究竟是Task 28R exact frozen probe还是bundle中的exact main module。

若不能证明严格alias关系，直接`PRECHECK_BLOCKED`，不得实施放行。

## 唯一允许的代码修改

证据通过后新增窄类别：

```text
APPROVED_CPYTHON39_MULTIPROCESSING_MAIN_ALIAS
```

批准条件：

1. Key严格为`__mp_main__`。
2. Alias由冻结Python 3.9 multiprocessing start语义创建。
3. `__mp_main__`与`__main__`满足实际证据确定的exact object关系。
4. Backing file已被以下之一精确批准：

   - Task 28R frozen closure probe；或
   - bundle manifest中的exact main module。

5. Alias与backing module共享相同source identity：

   - raw/resolved samefile；
   - device/inode一致；
   - regular nonsymlink file；
   - loader/spec/importer符合实际CPython语义；
   - size和SHA完全一致。

6. Alias不得引入新文件、新代码、不同origin或不同inode。
7. Module dictionary只能包含实际CPython multiprocessing明确造成的差异；逐字段记录。
8. Generic bundle scan应引用backing approved entry，而不是要求manifest新增名为`__mp_main__`的文件。
9. Formal science进程不得因此批准手工或无关的`__mp_main__`。

不得按名称、basename、相同SHA、multiprocessing已导入或整个Python版本宽泛放行。

## 必需负向测试

必须拒绝：

- 手工注入`__mp_main__`；
- 不同object或不符合实际alias语义；
- backing file未批准；
- 不同inode、SHA、origin、loader、spec或importer；
- 非multiprocessing child或错误start method；
- module dictionary异常变化；
- `__main__`/`__mp_main__`被替换；
- 任意trainer/bundle外文件伪装；
- Python或stdlib multiprocessing source identity不匹配。

Task 16–28R全部回归必须继续通过，Task 28R validator不得修改。

## 证据持久化

Task 28R probe ledger和本任务alias ledger必须在后续module scan前原子写入。即使下游失败，也必须保留：

- module object关系；
- backing approved entry；
- raw/resolved/fd identity；
- loader/spec/importer/start-method；
- stdlib alias provenance；
- pre/post SHA。

这只改善审计可追溯性，不得改变其他origin acceptance。

## 有界执行

1. 完成实际alias只读证明。
2. 证明失败即`PRECHECK_BLOCKED`。
3. 证明通过后，仅提交exact alias classifier、原子ledger和负向测试。
4. Python 3.9环境通过后，只执行一次closure job。
5. 两个独立clean process必须产生一致closure；失败即`PRECHECK_BLOCKED`，不得修补或重试。
6. Closure通过后立即执行既定必要formal audit，不得新增audit层。
7. Formal audit或任一四环境preflight失败即`PRECHECK_BLOCKED`。
8. 全部通过后，以全新非覆盖root运行四环境seed 0，每格intended horizon 6M、终点`5,980,160`、最多一次科学提交。
9. Executor独立负责全部资源和placement决定。

## 科学要求

保留Task 27冻结遥测：

- det/Paper/target norm：`.6050832272/.9192549586/.9192548990`
- scale：`1.519220710`
- cosine：`.8612535000`
- residual：`8.627e-16`
- Cholesky `info=0`

科学运行继续记录proposal norm/cosine、value prediction、value/advantage/PopArt及solver telemetry。解释必须保持：deterministic Gaussian GGN可以理论正确；NormMatch检验其与Paper finite-sample damped update的scale alignment。

## 严格早停

仅比较Original Paper RAT同环境、seed 0、同evaluation semantics及同transition：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当`Target/Paper < 0.60`时取消对应cell。不得使用Paper终点比较中间Target。

## 验收标准

唯一结论：

- `CANDIDATE_PROMOTE_TO_3SEED`
- `CANDIDATE_REJECT`
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`
- `PRECHECK_BLOCKED`

Promotion要求至少3/4环境达到终点且终点ratio均不低于0.60；rejection要求至少2个环境严格早停或完整终点证据明确否定候选。

## 禁止事项

不得修改算法、trainer、config、preflight、bundle/manifest、launchers、monitor或Task 28R；不得建立通用alias/control manifest；不得引入第二候选、覆盖旧root、重跑Paper、retry科学cell、使用Jupyter、访问`.54`/`ws4090-31`/`10.49.7.54`、指定资源或触碰无关任务。历史失败必须全部保留。

## 报告与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-MP-MAIN-EXACT-ALIAS-AND-6M-S0-20260825-29.md`

提交alias证明、classifier、负向测试、atomic ledger及model-free证据，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、commit身份、CPython alias provenance、module/file identity ledger、closure/formal audit/preflight结果、四环境科学终态、严格阶段比率、proposal norm/cosine及failure-ledger增量。
