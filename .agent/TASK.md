Status: READY

# Task-ID: PROCGEN-HYBRID-HEAD-TRAINABLE-GRAD-PREFLIGHT-AND-6M-S0-20260824-11

## 唯一目标

修正preflight one-step harness，使 `raw_grads` 只对生产优化器实际更新的 `requires_grad=True` 参数计算；PopArt非训练状态必须保留并单独审计。随后仅执行一次完整preflight；只有全部预检通过，才运行冻结Hybrid-Head候选四环境、seed 0、预定6M实验。

## 证据判断

`19225707` 的唯一阻塞是将 `requires_grad=False` 的PopArt `mean`、`mean_sq`、`debiasing`传给 `torch.autograd.grad`。错误发生在更新和训练之前，属于 `infrastructure-failure/preflight-design`，不是算法、数值、求解器、分区或硬件失败。

## 冻结科学身份

以下SHA256必须保持不变：

- Trainer：`7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54`
- Config：`9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Scientific launcher：`ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e`
- Monitor：`536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`

科学方法保持为：Paper actor及shared-trunk sampled Paper critic完全不变；deterministic normalized-residual `J_v` GGN、`lambda=0.1` 仅作用于257个critic-exclusive value-head参数；独立head-only `B×B` symmetric FP64/Jacobi/Cholesky。

## 唯一允许的修正

从 `a22f1a51bbcc953881e780f4dc00da16b2fc317f` 开始，仅修改preflight harness：

```python
trainable_named_params = [
    (name, parameter)
    for name, parameter in net.named_parameters()
    if parameter.requires_grad
]
```

要求：

1. `raw_grads`、方向向量及one-step更新仅使用上述有序集合。
2. 逐项证明该集合与production optimizer参数集合在名称、顺序、shape、dtype、device及object identity上完全一致。
3. PopArt `mean`、`mean_sq`、`debiasing`继续保留在model/state中，但不得进入optimizer、`autograd.grad`、方向向量或参数更新。
4. 单独记录这些PopArt状态在one-step前后的值及其原始Paper更新语义。
5. 禁止使用全局 `allow_unused=True`、异常吞掉、零梯度填充或删除检查来制造PASS。
6. 对预期连接的任一trainable参数，`None` gradient必须使预检失败。

不得修改trainer、科学config、scientific launcher、monitor或其他算法语义。

## 有界执行

1. 增加针对trainable/optimizer集合一致性和PopArt非训练状态的静态或CPU真实模型回归。
2. 执行且仅执行一次修正后的完整production preflight。
3. 若该preflight任一检查失败，立即结束为 `PRECHECK_BLOCKED`；不得再次现场修补、重试或启动科学单元。
4. 仅在preflight全部通过后，运行seed 0：

   - `bigfish-easy-0-10`
   - `bossfight-easy-0-10`
   - `caveflyer-easy-0-10`
   - `coinrun-easy-0-10`

5. 每格intended horizon为6M，终点为`5,980,160`。
6. 科学job/root实际存在前不得创建科学monitor。
7. Executor负责实时资源刷新及全部host、GPU、partition、卡数、并发和queue placement决定。

## 强制预检证据

必须证明：

- 三方resolved JSON继续字节一致，SHA256为  
  `61f8ebe38443acbdbf141981f4e9921435dccd5d4abb6a63959e3d4bdb9232ab`。
- 参数分区仍为total 938,979；policy-exclusive 2/3,855；shared 22/934,864；critic-exclusive 2/257。
- Partition manifest SHA256仍为  
  `b45298be8fc5bdccfa36ce653c7dbc0c41f2d013f23d4f4db0a4a580035f3087`。
- critic-exclusive仅为 `last_v_layer.weight/bias`，policy Jacobian为零或disconnected，value connected。
- trainable参数与production optimizer集合逐项相同。
- Paper actor matrix、RHS、direction bit-identical。
- shared-trunk sampled Paper critic direction bit-identical。
- one-step后policy参数、logits及shared delta bit-identical；仅value-head delta允许不同。
- PopArt非训练状态未被错误加入梯度或参数更新。
- head-only GGN公式、RHS、阻尼及维度正确。
- FP64/Jacobi/Cholesky `info=0`、残差有限且无fallback。
- production-scale内存检查和NaN/Inf、OOM、CUDA、hard-error扫描通过。
- 无重复root或active objective。

## 严格比较与早停

仅与不可变原始Paper RAT的同环境、seed 0、同evaluation semantics、同transition记录比较：

- 首个共同点 `>=2,000,000`
- 首个共同点 `>=4,000,000`
- 终点 `5,980,160`

仅当 `Target reward / Paper reward < 0.60` 时取消对应cell并记录 `EARLY_STOPPED_ALGORITHM`。不得使用Paper终点比较中间Target。

## 验收标准

唯一结论必须为：

- `CANDIDATE_PROMOTE_TO_3SEED`：预检全部通过，至少3/4环境达到终点，且这些终点ratio均不低于0.60。
- `CANDIDATE_REJECT`：至少两个环境触发严格早停，或终点证据明确否定候选。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：科学运行开始后因基础设施故障无法形成充分结论。
- `PRECHECK_BLOCKED`：本次唯一授权preflight未完整通过。

## 禁止事项

- 不得改变科学身份或引入第二候选、sweep、Paper重跑。
- 不得用 `allow_unused=True` 或零填充掩盖错误。
- 不得覆盖旧root或弱化四次preflight失败及其他历史记录。
- 不得使用Jupyter。
- 不得访问`.54`、`ws4090-31`或`10.49.7.54`。
- 不得指定计算资源或触碰无关任务。

## 报告、提交与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-HYBRID-HEAD-TRAINABLE-GRAD-PREFLIGHT-AND-6M-S0-20260824-11.md`

报告必须包含修正diff及哈希、科学哈希不变证明、trainable/optimizer/PopArt manifests、完整preflight证据、不可变failure ledger，以及若启动科学单元时的2M/4M/终点严格比较表。提交允许的model-free证据，保持worktree干净，推送并验证`origin/agent-work`，然后回调唯一结论及全部commit身份。
