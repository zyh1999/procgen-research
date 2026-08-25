Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-AST-CALL-AUDIT-AND-6M-S0-20260825-26

## 唯一目标

仅将冻结preflight中对以下单行源码字符串的脆弱断言：

```text
match_head_proposal_norm(head_direction, paper_head_proposal)
```

替换为line-wrap无关的AST及运行时语义验证。保持V2 trainer、算法、bundle、class-attribute classifier、non-reentrant hook及其他科学身份不变。

## 证据判断

Task 25已通过：

- 真实`torch.classes` class-level pseudo-origin分类；
- 全部负向测试；
- hermetic bundle验证；
- production model构造，参数数目938,979。

唯一失败是trainer中的合法调用跨557–558两行，而preflight要求完全相同的单行substring。调用语义没有失败。这属于`precheck-failure/frozen-preflight-source-text-assertion-linewrap-mismatch`，不是算法、数值、求解器、GPU或科学证据。

## 冻结身份

必须保持字节不变：

- V2 trainer SHA256：`0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b`
- Config：`9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Regression：`f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c`
- Monitor：`536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`
- Bundle/manifest及science/preflight deployment launchers
- Task 23 hook SHA：`8d9206a6defc4525114398a952d29ffdd4872cd933dc5c9b96fc838bd1273dbe`
- Task 25 classifier SHA：`f80de2abbcbce29e7a57ef456156c86636798c4e1ea37171922b3b466b6790fc`

允许产生一个新的preflight SHA，但其唯一语义diff必须是本任务规定的assertion替换。

## 唯一允许的代码修改

删除单行substring断言，替换为以下检查。

### AST验证

对冻结trainer执行`ast.parse`，定位实际训练更新路径中的call：

```python
match_head_proposal_norm(
    head_direction,
    paper_head_proposal,
)
```

必须证明：

1. callee严格解析到预期的`match_head_proposal_norm`定义，而不是同名局部变量、attribute或shadow。
2. 位置参数恰为两个，顺序严格为：

   - `head_direction`
   - `paper_head_proposal`

3. 无额外 positional、keyword、`*args`或`**kwargs`。
4. Call处于实际minibatch update控制流中，不是字符串、注释、测试函数或明显dead branch。
5. 返回值流向冻结V2后续head update路径。
6. 记录call及函数定义的AST normalized dump、source span和SHA256。
7. 格式、空白和换行变化不得影响结果。

### 运行时语义验证

在既有one-step preflight中对目标函数做无副作用wrapper/spy，证明：

- 每个预期更新恰调用一次；
- 两个实参分别与当前`head_direction`和`paper_head_proposal`对象/张量identity一致；
- shape、dtype、device及有限性正确；
- 返回proposal满足既定norm matching；
- wrapper不改变RNG、tensor、optimizer或控制流；
- 移除wrapper后结果与wrapped路径bit-identical。

不得修改trainer来帮助preflight通过。

## 必需负向测试

必须拒绝：

- 仅在字符串或注释中出现调用；
- 参数反序、缺失、增加或改为错误名称；
- callee shadowing或attribute call；
- 调用位于dead/test-only路径；
- 返回值未用于head update；
- runtime未调用、重复调用或实参identity错误；
- wrapper改变RNG、参数或返回值。

必须保留Task 16–25全部回归和冻结身份检查。

## 有界执行

1. 仅修改preflight的源码审计断言及相应测试。
2. 本地实际Python 3.9环境通过后，仅执行一次closure provenance job。
3. 两个独立clean process必须完成trainer import及production model construction，规范化closure一致。
4. Closure失败即`PRECHECK_BLOCKED`，不得修补或重试。
5. Closure通过后仅执行一次formal clean-room audit。
6. Formal audit失败即`PRECHECK_BLOCKED`。
7. Audit通过后，对四环境各执行一次真实网络preflight；任一失败即`PRECHECK_BLOCKED`。
8. 全部通过后，以全新非覆盖root运行四环境seed 0，每格intended horizon 6M、终点`5,980,160`、最多一次提交。
9. Executor独立负责全部实时资源及placement。

## 严格比较与早停

仅比较原始Paper RAT同环境、seed 0、同evaluation semantics和同transition：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当`Target/Paper < 0.60`时取消对应cell。不得用Paper终点比较中间Target；无精确共同点不得操作。

## 验收标准

唯一结论：

- `CANDIDATE_PROMOTE_TO_3SEED`
- `CANDIDATE_REJECT`
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`
- `PRECHECK_BLOCKED`

Promotion要求至少3/4环境达到终点且终点ratio均不低于0.60；rejection要求至少2个环境严格早停或完整终点证据明确否定候选。

## 禁止事项

不得修改trainer、算法、config、bundle、launchers、monitor、hook、classifier或既有provenance；不得用更宽泛substring、regex或跳过调用检查；不得引入第二候选、覆盖旧root、重跑Paper、retry科学cell、使用Jupyter、访问`.54`/`ws4090-31`/`10.49.7.54`、指定资源或触碰无关任务。全部历史失败必须保留。

## 报告与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-AST-CALL-AUDIT-AND-6M-S0-20260825-26.md`

提交preflight修正、AST/runtime负向测试、closure证据和报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、commit身份、新preflight SHA、AST call ledger、runtime identity ledger、closure、formal audit/preflight/科学终态、严格阶段比率及failure-ledger增量。
