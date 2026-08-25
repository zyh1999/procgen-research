Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-RUNTIME-SPY-SEMANTIC-BINDING-AND-6M-S0-20260825-27

## 唯一目标

仅修正runtime spy的expected-object绑定：在冻结one-step preflight边界上，将实际deterministic proposal对象`det_proposal`对应到trainer AST角色`head_direction`，并将实际Paper proposal对象绑定到`paper_head_proposal`。不得按trainer词法变量名查找preflight局部变量。

修正通过后立即沿既定路径完成一次closure、复用已通过且身份仍有效的审计证据、执行四环境preflight和NormMatch V2 seed-0预定6M科学实验。不得建立新audit框架或第二算法。

## 证据判断

Task 26已经证明：

- 唯一AST调用正确；
- callee、参数顺序、scope、control flow和返回值流向正确；
- trainer、模型、config、structural manifest及connectivity checks正确。

唯一失败是测试harness把trainer源码名`head_direction`当成preflight局部变量名，而preflight中同一语义tensor名为`det_proposal`。这是`precheck-failure/runtime-spy-preflight-variable-identity-binding`，不是deterministic GGN、NormMatch、数值或科学失败。

## 唯一允许的修改

在preflight中定义显式、不可变的semantic-role mapping：

```text
deterministic_head_proposal:
  trainer_ast_name: head_direction
  preflight_object: det_proposal

counterfactual_paper_head_proposal:
  trainer_ast_name: paper_head_proposal
  preflight_object: paper_head_proposal
```

Wrapper必须直接捕获preflight实际对象：

```python
expected_det = det_proposal
expected_paper = paper_head_proposal

def spy(actual_det, actual_paper):
    assert actual_det is expected_det
    assert actual_paper is expected_paper
```

禁止通过字符串、`locals()`、trainer AST名称或模糊value equality寻找对象。

## 必需证明

1. AST/dataflow继续证明trainer `head_direction`来自冻结deterministic head solve并进入唯一norm-match call。
2. Preflight `det_proposal`由相同公式、输入、阻尼、precision及solver产生并位于相同call boundary。
3. `actual_det is det_proposal`。
4. `actual_paper is paper_head_proposal`。
5. 记录object identity、storage/data pointer、shape、stride、dtype、device、version counter、`requires_grad`及确定性value摘要。
6. 调用前后输入对象未被修改。
7. 每个预期update恰调用一次。
8. 返回值满足：

```text
||u_target||₂ = ||u_paper||₂
```

9. Wrapped与unwrapped路径的RNG、outputs、parameters、optimizer state及telemetry bit-identical。
10. 科学运行时不残留spy或测试hook。

若实际冻结代码在边界上创建clone、view、detach、cast或其他不同对象，不得放宽identity；必须`PRECHECK_BLOCKED`并报告真实dataflow。

## 必需负向测试

必须拒绝：

- 按`head_direction`字符串查询preflight变量；
- det/Paper反序；
- equal-value但不同object或storage；
- clone、detach、cast、view或重计算替代；
- 调用缺失或重复；
- wrapper修改input、output、RNG或optimizer；
- semantic mapping与AST/dataflow不一致。

保留Task 16–26全部已通过回归和冻结身份。

## 冻结身份

Trainer、config、AST contract、regression、bundle/manifest、science/preflight deployment launchers、monitor、Task 23 hook、Task 25 classifier及全部既有provenance必须字节不变。仅允许产生一个runtime-binding修正后的新preflight SHA。

## 有界执行

1. 仅提交runtime-spy binding修正、mapping ledger和测试。
2. 在实际Python 3.9/Torch环境本地通过后，仅执行一次closure job。
3. 两个独立clean process必须完成production model construction并产生一致closure；失败即`PRECHECK_BLOCKED`，不得修补或重试。
4. Closure通过后复用仍严格匹配的已通过bundle、path、pseudo-origin及hook证据，只执行必要的formal audit。
5. Formal audit或任一四环境preflight失败即`PRECHECK_BLOCKED`。
6. 全部通过后，以全新非覆盖root运行：

   - BigFish seed 0
   - BossFight seed 0
   - CaveFlyer seed 0
   - CoinRun seed 0

7. 每格intended horizon为6M，终点`5,980,160`，每格最多一次科学提交。
8. Executor独立负责全部实时资源和placement。

## 科学遥测要求

除既定reward/KL/LR/value/advantage/solver telemetry外，必须记录同minibatch：

- `||u_det||₂`
- `||u_paper||₂`
- NormMatch scale
- `||u_target||₂`
- det/Paper proposal cosine
- global pre/post-clip norm
- value prediction change

解释必须保持理论校正：deterministic Gaussian GGN可以理论正确；本实验检验的是其与Paper finite-sample sampled-score update之间的scale校准，而非“GGN公式是否正确”。

## 严格早停

只与原始Paper RAT同环境、seed 0、同evaluation semantics和同transition比较：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当`Target/Paper < 0.60`时取消该cell。不得用Paper终点比较中间Target。

## 验收标准

唯一结论：

- `CANDIDATE_PROMOTE_TO_3SEED`
- `CANDIDATE_REJECT`
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`
- `PRECHECK_BLOCKED`

Promotion要求至少3/4环境达到终点且终点ratio均不低于0.60；rejection要求至少2个环境严格早停或终点证据明确否定候选。

## 禁止事项

不得修改算法、trainer、AST contract、bundle、launchers、monitor、hook或classifier；不得创建第二候选或新通用audit框架；不得覆盖旧root、重跑Paper、retry科学cell、使用Jupyter、访问`.54`/`ws4090-31`/`10.49.7.54`、指定资源或触碰无关任务。历史失败必须保留。

## 报告与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-RUNTIME-SPY-SEMANTIC-BINDING-AND-6M-S0-20260825-27.md`

提交binding修正、identity/dataflow ledger、科学遥测和报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、commit身份、新preflight SHA、identity ledger、closure/audit/preflight结果、四环境科学终态、严格阶段比率、proposal norm/cosine及failure-ledger增量。
