Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-TORCH-CLASS-ATTRIBUTE-PSEUDO-ORIGIN-AND-6M-S0-20260825-25

## 唯一目标

纠正Task 24中错误的dynamic-attribute假设，按冻结PyTorch真实实现将`torch.classes.__file__="_classes.py"`严格识别为`torch._classes._Classes`类级静态synthetic metadata。复用Task 23已验证的non-reentrant hook，完成closure、formal audit、四环境preflight，并仅在全部通过后运行四环境seed 0预定6M实验。

## 证据判断

实际Python 3.9.25 / Torch 2.5.1+cu121已证明：

- instance `vars(torch.classes)`不含`__file__`；
- `inspect.getattr_static(torch.classes, "__file__") == "_classes.py"`；
- `getattr(torch.classes, "__file__") == "_classes.py"`；
- `type(torch.classes).__dict__["__file__"] == "_classes.py"`；
- `__getattr__`不参与该属性解析；
- spec、loader、package、origin均为`None`；
- 没有物理`_classes.py`文件。

因此Task 24的static-sentinel及dynamic-provider要求与冻结实现矛盾。该结果属于`precheck-failure/task-spec-static-vs-dynamic-provider-contradiction`，不是代码、算法、数值或科学失败。

## 冻结身份

Trainer、config、scientific preflight、regression、monitor、bundle/manifest、science/preflight launchers、Task 18 provenance、Task 21 file identity及Task 23 non-reentrant hook必须全部字节不变。

固定Torch证据：

- Torch：`2.5.1+cu121`
- Source：installed `torch/_classes.py`
- SHA256：`2a3dd93d72e9f19450670b89f3a57b5b5adf245709f6a49a551bbfad33c434bf`
- Size：`1721`
- RECORD：`sha256=Kj3ZPXLp8ZRQZwuJ86V7W1rfJFcJ9qSaVRu_rTPENL8`
- `_Classes`定义：line 19
- class-level `__file__="_classes.py"`：line 20
- `__getattr__`：lines 25–28，仅处理其他缺失属性

## 唯一允许的修改

仅修改pseudo-origin classifier，新增窄类别：

```text
APPROVED_INSTALLED_DISTRIBUTION_CLASS_ATTRIBUTE_PSEUDO_ORIGIN
```

`torch.classes`只有同时满足以下条件时可批准：

1. `sys.modules`键和module name严格为`torch.classes`。
2. 实例类型严格为冻结`torch._classes._Classes`。
3. 类型MRO及`types.ModuleType`继承关系匹配冻结实现。
4. Instance dictionary不含`__file__`：

```python
"__file__" not in vars(module)
```

5. Static lookup严格返回：

```python
inspect.getattr_static(module, "__file__") == "_classes.py"
```

6. Class dictionary严格包含：

```python
type(module).__dict__["__file__"] == "_classes.py"
```

7. Public lookup严格返回相同字符串：

```python
getattr(module, "__file__") == "_classes.py"
```

8. `type(module).__dict__["__getattr__"]`的identity/source保持冻结值，但必须证明本次`__file__`访问不调用它。
9. `__spec__`、loader、package、origin均为`None`。
10. Installed source path、SHA、size、RECORD及class-level赋值位置全部匹配。
11. Designated目录、bundle root及其他批准源码root不存在由`"_classes.py"`解析出的物理文件。
12. 审计前后module object、type、class dictionary、instance dictionary和关键属性未被替换。
13. Ledger必须分别记录instance、static、class-level和public lookup结果，并明确provider为class attribute。

不得对其他模块、类型、相对`__file__`、class attribute或Torch namespace推广该规则。

## 必需负向测试

必须拒绝：

- instance dictionary注入`__file__`；
- class-level值缺失或改变；
- static/public/class-level结果不一致；
- 不同module key、type、MRO或source；
- monkeypatch class、`__getattr__`或module object；
- spec/loader/package/origin非None；
- 出现物理`_classes.py`；
- Torch版本、source SHA、size、RECORD或赋值位置不匹配；
- lookup产生文件、网络或repo读取副作用。

必须使用实际Python 3.9.25 / Torch 2.5.1+cu121真实对象作为正向测试，并保留Task 16–23全部安全回归。

## 有界执行

1. 仅提交class-attribute classifier及测试。
2. 本地真实环境通过后，只执行一次closure provenance job。
3. 两个独立clean process必须完成trainer import和production model construction，且规范化closure一致。
4. Closure失败即`PRECHECK_BLOCKED`，不得修补或重试。
5. Closure通过后只执行一次formal clean-room audit。
6. Formal audit失败即`PRECHECK_BLOCKED`。
7. Audit通过后，对四环境各执行一次真实网络preflight；任一失败即`PRECHECK_BLOCKED`。
8. 全部通过后，以全新非覆盖root运行四环境seed 0，每格intended horizon 6M、终点`5,980,160`、最多一次提交。
9. Executor独立负责全部实时资源及placement。

## 严格比较与早停

仅比较原始Paper RAT同环境、seed 0、同evaluation semantics和同transition：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当`Target/Paper < 0.60`时取消对应cell。无精确共同点不得操作；不得用Paper终点比较中间Target。

## 验收标准

唯一结论：

- `CANDIDATE_PROMOTE_TO_3SEED`
- `CANDIDATE_REJECT`
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`
- `PRECHECK_BLOCKED`

Promotion要求至少3/4环境达到终点且终点ratio均不低于0.60；rejection要求至少2个环境严格早停或完整终点证据明确否定候选。

## 禁止事项

不得修改算法、bundle、科学文件、launchers、monitor、non-reentrant hook或既有provenance；不得建立模块名、相对路径、class attribute或Torch namespace白名单；不得引入第二候选、覆盖旧root、重跑Paper、retry科学cell、使用Jupyter、访问`.54`/`ws4090-31`/`10.49.7.54`、指定资源或触碰无关任务。历史失败必须全部保留。

## 报告与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-TORCH-CLASS-ATTRIBUTE-PSEUDO-ORIGIN-AND-6M-S0-20260825-25.md`

提交classifier、Python 3.9真实对象测试、closure证据和报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、commit身份、class-attribute provider ledger、完整closure、formal audit/preflight/科学终态、严格阶段比率及failure-ledger增量。
