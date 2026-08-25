Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-TORCH-DYNAMIC-ATTRIBUTE-CLASSIFIER-AND-6M-S0-20260825-24

## 唯一目标

仅修正`torch.classes` synthetic pseudo-origin classifier，使其正确区分module实例字典中不存在的`__file__`与`torch._classes._Classes.__getattr__`动态提供的公开`"_classes.py"`属性。不得改变NormMatch V2、bundle、audit hook或任何科学身份。

## 证据判断

Task 23真实Python 3.9正向对象满足：

- module key/name：`torch.classes`
- type：`torch._classes._Classes`
- `module.__dict__.get("__file__") is None`
- spec、loader、package、origin均为`None`

但classifier错误地要求实例字典直接包含`"_classes.py"`。实际公开属性由`_Classes.__getattr__`动态返回。该失败属于`precheck-failure/pseudo-origin-positive-classifier-dict-vs-synthetic-attribute`，没有算法、数值、GPU或科学证据。

## 冻结身份

Trainer、config、preflight、regression、monitor、bundle/manifest、science/preflight launchers、Task 18 provenance、Task 21 file identity及Task 23 non-reentrant hook必须全部字节不变。

固定Torch证据继续为：

- Torch：`2.5.1+cu121`
- Installed `torch/_classes.py`
- SHA256：`2a3dd93d72e9f19450670b89f3a57b5b5adf245709f6a49a551bbfad33c434bf`
- Size：`1721`
- RECORD：`sha256=Kj3ZPXLp8ZRQZwuJ86V7W1rfJFcJ9qSaVRu_rTPENL8`

## 唯一允许的修改

仅修改pseudo-origin positive classifier：

1. 使用`vars(module)`或`module.__dict__`证明实例字典不含物理`__file__`：

```python
"__file__" not in vars(module)
```

2. 使用不会触发动态fallback的static inspection证明没有静态实例origin，例如：

```python
inspect.getattr_static(module, "__file__", sentinel) is sentinel
```

3. 单独调用公开属性协议：

```python
public_file = getattr(module, "__file__")
```

并要求其严格等于相对伪字符串`"_classes.py"`。

4. 证明该返回值来自精确类型`torch._classes._Classes.__getattr__`：

   - 类型及方法identity匹配installed Torch；
   - 方法源码路径、SHA和RECORD匹配冻结证据；
   - 静态定位其`__file__`返回语义；
   - 调用前后module字典、类型及关键属性未被修改。

5. `__spec__`、loader、package、origin仍必须为`None`。
6. designated目录及批准源码root不得存在物理`_classes.py`。
7. Ledger必须分别记录：

   - `dict_file`
   - `static_file`
   - `public_dynamic_file`
   - dynamic provider type/method及源码hash

8. 禁止把动态`getattr`结果当作普通文件路径进行resolve/stat。

不得对其他模块、类型、相对字符串或任意`__getattr__`对象推广该规则。

## 必需负向测试

必须拒绝：

- 实例字典实际包含`__file__`；
- static attribute存在；
- public dynamic值不是精确`"_classes.py"`；
- dynamic provider不是冻结`_Classes.__getattr__`；
- monkeypatch type、method或module；
- spec/loader/package/origin非预期；
- designated目录存在物理`_classes.py`；
- Torch版本、source SHA、size或RECORD不匹配。

必须使用实际Python 3.9/PyTorch环境运行真实`torch.classes`正向测试，并保留Task 16–23全部回归。

## 有界执行

1. 仅提交classifier修正及测试。
2. 本地实际环境通过后，只执行一次closure provenance job。
3. 在两个独立clean process中完成trainer import和production model construction；规范化closure必须一致。
4. Closure失败即`PRECHECK_BLOCKED`，不得修补或重试。
5. Closure通过后只执行一次formal clean-room audit。
6. Formal audit失败即`PRECHECK_BLOCKED`。
7. Audit通过后，对四环境各执行一次真实网络preflight；任一失败即`PRECHECK_BLOCKED`且不得启动科学cell。
8. 全部通过后，以全新非覆盖root运行四环境seed 0，每格intended horizon 6M、终点`5,980,160`、最多一次提交。
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

不得修改算法、bundle、科学文件、launchers、monitor、non-reentrant hook或既有provenance；不得建立模块名、相对路径、动态属性或Torch namespace白名单；不得引入第二候选、覆盖旧root、重跑Paper、retry科学cell、使用Jupyter、访问`.54`/`ws4090-31`/`10.49.7.54`、指定资源或触碰无关任务。全部历史失败必须保留。

## 报告与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-TORCH-DYNAMIC-ATTRIBUTE-CLASSIFIER-AND-6M-S0-20260825-24.md`

提交classifier修正、Python 3.9回归、closure证据和报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、commit身份、dynamic-attribute ledger、完整closure、formal audit/preflight/科学终态、严格阶段比率及failure-ledger增量。
