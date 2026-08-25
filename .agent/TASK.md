Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-TORCH-PSEUDO-ORIGIN-AND-NONREENTRANT-CLOSURE-20260825-23

## 唯一目标

仅完成两项audit修正：

1. 将`torch.classes.__file__ == "_classes.py"`严格识别为installed PyTorch定义的synthetic module metadata，而非物理文件origin。
2. 将closure provenance hook改为非递归实现，同时保留所有一级filesystem/import事件。

不得改变NormMatch V2、bundle、launchers或任何科学身份。

## 证据判断

Task 22已经证明：

- `torch.classes`类型严格为`torch._classes._Classes`；
- `__file__`为相对伪字符串`"_classes.py"`；
- spec、loader、package和origin均为`None`；
- designated目录始终为空；
- 没有物理`_classes.py`；
- frozen Torch源码明确赋值该伪字符串。

因此这是synthetic metadata，不应解析为cwd文件。

完整closure失败则是audit hook调用`traceback.extract_stack`，后者通过`linecache/tokenize`触发新的`open`事件并递归重入。这属于`infrastructure-failure/closure-provenance-audit-hook-recursion`，没有科学证据。

## 唯一允许的修改

### Synthetic metadata类别

新增：

```text
APPROVED_INSTALLED_DISTRIBUTION_PSEUDO_ORIGIN
```

仅当全部满足时批准`torch.classes`：

- `sys.modules`键严格为`torch.classes`；
- 类型严格为`torch._classes._Classes`；
- `__file__`严格为相对字符串`"_classes.py"`；
- `__spec__`、loader、package和origin均为`None`；
- designated目录及批准源码root不存在对应物理文件；
- installed Torch版本严格为`2.5.1+cu121`；
- `torch/_classes.py` SHA256为
  `2a3dd93d72e9f19450670b89f3a57b5b5adf245709f6a49a551bbfad33c434bf`；
- size为`1721`，RECORD为
  `sha256=Kj3ZPXLp8ZRQZwuJ86V7W1rfJFcJ9qSaVRu_rTPENL8`；
- 源码中`_Classes`定义及`__file__`赋值被静态定位；
- module对象和属性在审计期间未被替换。

不得推广至其他相对`__file__`、无spec对象或Torch namespace模块。

### Non-reentrant hook

Audit callback内部禁止：

- `traceback`、`inspect.stack`、`linecache`、`tokenize`；
- 源码读取；
- import；
- 可能触发audit event的复杂`repr`或序列化。

允许使用预先导入对象、thread-local reentrancy guard和`sys._getframe()`，但只能记录`co_filename`、`co_name`、line、PID/TID、事件名及安全标量。符号解析和序列化必须在hook外完成。

一级事件必须完整保留。重入事件必须计数；不得静默丢弃与文件创建、写入、rename、删除或import相关的事件。正常复现中reentrant计数应为零，否则必须判定是否影响closure完整性。

## 必需负向测试

必须拒绝：

- 相同伪文件字符串但module key或类型不同；
- spec/loader/package/origin任一非预期；
- 存在真实`_classes.py`；
- Torch版本、源码、RECORD或赋值位置不匹配；
- module对象或属性被替换；
- hook内诱发递归；
- 丢失一级filesystem事件；
- 未登记物理generated artifact或bundle外repository origin。

Task 16–22全部回归必须继续通过，并使用实际Python 3.9环境。

## 有界执行

1. 仅提交pseudo-origin分类、non-reentrant hook及测试。
2. 在两个独立clean process中完成trainer import和production model construction。
3. 两次规范化closure必须一致，所有physical artifacts和synthetic origins逐项批准；否则`PRECHECK_BLOCKED`。
4. Closure通过后仅执行一次formal clean-room audit。
5. Formal audit失败即`PRECHECK_BLOCKED`，不得修补或重试。
6. Audit通过后，对四环境各执行一次真实网络preflight。
7. 任一preflight失败即`PRECHECK_BLOCKED`，不得启动科学cell。
8. 全部通过后，以全新非覆盖root运行四环境seed 0；每格intended horizon 6M、终点`5,980,160`、最多一次提交。
9. Executor独立负责全部实时资源及placement。

## 严格早停

仅比较原始Paper RAT同环境、seed 0、同evaluation semantics及同transition：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当`Target/Paper < 0.60`时取消对应cell。不得用Paper终点比较中间Target。

## 验收标准

唯一结论：

- `CANDIDATE_PROMOTE_TO_3SEED`
- `CANDIDATE_REJECT`
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`
- `PRECHECK_BLOCKED`

Promotion要求至少3/4环境达到终点且终点ratio均不低于0.60；rejection要求至少2个环境严格早停或终点证据明确否定候选。

## 禁止事项

不得修改算法、bundle、科学文件、launchers、monitor或既有provenance；不得建立文件名、相对路径或Torch namespace白名单；不得引入第二候选、覆盖旧root、重跑Paper、retry科学cell、使用Jupyter、访问`.54`/`ws4090-31`/`10.49.7.54`、指定资源或触碰无关任务。历史失败必须全部保留。

## 报告与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-TORCH-PSEUDO-ORIGIN-AND-NONREENTRANT-CLOSURE-20260825-23.md`

提交分类修正、hook修正、Python 3.9回归、closure证据和报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、commit身份、pseudo-origin proof、hook/reentrancy ledger、完整closure、formal audit/preflight/科学终态、严格阶段比率及failure-ledger增量。
