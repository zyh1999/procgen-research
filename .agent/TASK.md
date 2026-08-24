Status: READY

# Task-ID: PROCGEN-PAPER-HYBRID-HEAD-NORMMATCH-DETGGN-6M-S0-20260825-14

## 唯一目标

构建并评估一个严格Paper匹配的单一新候选：保留Hybrid-Head V1的deterministic-GGN value-head方向，但将其每个minibatch的最终head update proposal范数精确匹配到同一步原始Paper sampled-critic head proposal的范数，以检验三环境失败是否源于head更新幅度校准，而非deterministic-GGN方向本身。

四个环境均使用seed 0、预定6M horizon；不得同时测试其他校准、系数或候选。

## 科学解释

Hybrid-Head V1排除了此前joint-2B和shared deterministic-GGN的主要问题：

- Paper actor、sampled shared critic及单步shared delta均bit-identical；
- deterministic GGN只作用于257个value-head参数；
- FP64求解稳定、残差约`1e-15`，无hard error。

但结果仍为：

- BigFish：2M通过，4M ratio `0.4691`，失败；
- BossFight：2M ratio `0.4247`，失败；
- CoinRun：2M ratio `0.0270`，失败；
- CaveFlyer：2M/4M/6M均通过，终点ratio `0.9970`。

Actor KL/LR不是统一失败机制：CoinRun失败时KL仅`.00664`且LR仍为`.5`，而BossFight LR已降低，BigFish则到4M才失败。共同的唯一科学差异是value-head update。即使shared/policy单步更新相同，value-head尺度变化也会改变value prediction、GAE、后续采样分布和长期策略轨迹。

CaveFlyer接近Paper说明deterministic head方向并非普遍无效；因此下一项单因果修正应只校准head proposal的尺度，而不得恢复joint/cross/shared-GGN、low-Fisher guard或actor调参。

## 候选身份

方法名：

`PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`

基线代码必须来自已验证的Hybrid-Head V1科学freeze：

`fe4b8a58812e80689705abec11364457cae31e26`

保持不变：

- 原始Paper actor路径；
- shared trunk上的sampled Paper critic路径；
- network、PopArt、GAE、reward及evaluation semantics；
- rollout 4096、minibatch 512、epochs 4；
- initial actor LR `.5`；
- per-minibatch adaptive-KL阈值`.005/.04`；
- momentum `1e-6`及原始history correction；
- damping `.5`及global parameter L2 clip `.5`；
- deterministic value-head `J_v`、normalized residual、`lambda=0.1`；
- head-only `B×B` symmetric FP64/Jacobi/Cholesky；
- seed、budget及terminal-update语义。

## 唯一允许的算法变化

在每个minibatch同一更新边界计算：

- `u_det`：Hybrid-Head V1产生的最终deterministic-GGN head proposal；
- `u_paper`：同一当前模型、同一minibatch和同一已有sampled Paper critic计算产生的counterfactual Paper head proposal。

二者必须处于相同参数坐标、相同优化器/history处理阶段，并位于global L2 clip和实际参数写入之前。

定义：

```text
s = ||u_paper||₂ / ||u_det||₂
u_target = s · u_det
```

边界规则：

- 两个范数均为零：`u_target=0`；
- `||u_det||`为零但`||u_paper||`非零：hard failure，不得fallback；
- `||u_paper||`为零：`u_target=0`；
- 除原始Paper global clip外，不允许额外cap、floor、EMA、guard或可调系数。

必须复用Hybrid V1/shared Paper critic计算中已经产生的Paper head信息；不得为获得`u_paper`额外采样、消耗RNG或改变数据顺序。

## 必需代码审计与回归

科学启动前必须证明：

1. 相对Hybrid-Head V1，唯一科学diff是上述head proposal norm matching及必要遥测。
2. 相对原始Paper，actor/shared路径仍严格一致；只有value-head方向为deterministic GGN。
3. `u_det`和`u_paper`来自相同minibatch、模型状态、PopArt状态和参数顺序。
4. norm matching位于相同的明确更新边界，不能比较不同阶段的raw gradient和applied delta。
5. 每步满足有限精度容差内：

   ```text
   ||u_target||₂ = ||u_paper||₂
   ```

6. 因actor/shared proposal不变且disjoint head范数相等，global L2 clip的总范数及clip scale与counterfactual Paper相同。
7. one-step regression必须证明：

   - actor参数和policy logits与Paper bit-identical；
   - shared参数delta与Paper bit-identical；
   - global clip scale与Paper bit-identical；
   - 只有value-head delta方向不同；
   - target head-delta norm与Paper head-delta norm一致；
   - critic-head policy Jacobian为零或disconnected。

8. trainable/optimizer集合及PopArt非训练状态继续通过Task 11审计。
9. 四环境structural manifest使用Task 12拆分语义并保持一致。
10. FP64/Jacobi/Cholesky `info=0`、残差有限，无NaN/Inf或fallback。
11. 静态拒绝joint/shared-GGN/cross、low-Fisher、projection、Kaczmarz、额外actor字段及任何自由scale参数。

若上述任一项失败，结论为`PRECHECK_BLOCKED`，不得启动科学cell或现场重新定义算法。

## 科学执行范围

预检通过后，运行以下四个seed-0 cell：

- `bigfish-easy-0-10`
- `bossfight-easy-0-10`
- `caveflyer-easy-0-10`
- `coinrun-easy-0-10`

每格：

- intended horizon：6M
- terminal transition：`5,980,160`
- 全新、非覆盖root
- 最多一次科学提交

这是新候选，四环境均需重新运行；不得复用Hybrid V1 reward作为V2结果。Executor在启动前刷新所有授权资源的scheduler、GPU、进程、ownership、capacity、roots和重复任务。具体资源及调度完全由Executor决定。

## 严格比较与早停

唯一早停基线是不可变原始Paper RAT的同环境、seed 0、同evaluation semantics和同transition记录：

- 第一个共同点`>=2,000,000`
- 第一个共同点`>=4,000,000`
- 终点`5,980,160`

仅当：

```text
Target reward / Paper reward < 0.60
```

时取消对应cell并记录`EARLY_STOPPED_ALGORITHM`。没有精确共同记录时不得操作；不得用Paper终点比较中间Target。

Hybrid V1只能作为机制对照报告，不得替代Paper早停基线。

## 必需遥测

每个评估点及适当的minibatch摘要必须记录：

- `||u_det||₂`
- `||u_paper||₂`
- scale `s`
- `||u_target||₂`
- deterministic/Paper head proposal cosine
- global pre-clip norm及clip scale
- actor KL、actor LR、entropy
- value loss/MSE、explained variance
- PopArt mean、variance及debiasing状态
- advantage mean、std及有限性
- head relative residual、solve residual、Cholesky info
- reward和transitions

报告必须检查失败环境是否存在head scale漂移、方向冲突、value/advantage退化或PopArt异常；求解器残差不得被当作性能成功。

## 验收标准

唯一结论必须为：

- `CANDIDATE_PROMOTE_TO_3SEED`：至少3/4环境达到终点，且所有达到终点的环境终点ratio均不低于0.60；无身份偏移、数值失败或隐藏fallback。
- `CANDIDATE_REJECT`：至少2个环境严格触发`<0.60`早停，或完整终点证据明确否定norm-matching机制。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：预检通过且科学运行开始，但基础设施故障阻止充分判定。
- `PRECHECK_BLOCKED`：代码审计、one-step等价、结构、内存或求解器预检未通过。

## 必需证据与历史保留

报告必须包含：

- Paper、Hybrid V1、V2 trainer/config/launcher/monitor完整SHA及逐行科学diff。
- norm-matching更新边界的代码定位和公式审计。
- RNG、数据顺序、optimizer/history及global-clip等价证明。
- 四环境scheduler、return code、transitions、root和artifact清单。
- 2M、4M及终点的Target/Paper严格比较表。
- V1/V2机制对照表，但不混作严格早停基线。
- OOM、CUDA、NCCL、disk、stall、Traceback、NaN/Inf扫描。
- joint-2B、separate-B、Hybrid V1、low-Fisher、P1、ACTOR_J及所有preflight/launcher基础设施失败和取消记录原样保留。

## 禁止事项

- 不得测试多个scale规则、cap、系数或候选。
- 不得改变actor、shared critic、network、schedule、momentum、KL timing或Paper语义。
- 不得复活joint/cross/shared deterministic-GGN或low-Fisher guard。
- 不得覆盖既有root、重跑Paper或删除失败历史。
- 不得retry、requeue或resubmit失败科学cell。
- 不得使用Jupyter。
- 不得访问`.54`、`ws4090-31`或`10.49.7.54`。
- Planner不指定host、GPU、partition、卡数、并发或queue placement。
- 不得触碰无关任务。

## 报告、提交与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-PAPER-HYBRID-HEAD-NORMMATCH-DETGGN-6M-S0-20260825-14.md`

提交冻结实现、model-free证据和报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、assignment/frozen/evidence/Delivery commits、V2科学身份、四格终态、严格阶段比率、norm/cosine/value遥测及failure-ledger增量。
