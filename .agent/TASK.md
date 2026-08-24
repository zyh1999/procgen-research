Status: READY

# Task-ID: PROCGEN-HYBRID-HEAD-ASSERTION-FIX-AND-6M-S0-20260824-10

## 唯一目标

在保持全部科学文件字节不变的前提下，以冻结生产网络的精确 partition invariant替换陈旧的 `SHARED.numel > 1,000,000` 断言，执行一次完整预检；仅当预检全部通过后，运行同一 Hybrid-Head deterministic-GGN候选四环境、seed 0、预定6M实验。

## 决策依据

`19225085` 已通过 canonical配置加载、真实模型构造、实际参数分区和critic-exclusive Jacobian检查。唯一失败是与科学语义无关的陈旧参数数量阈值。故明确授权修正该断言一次；其失败继续分类为 `infrastructure-failure/preflight-design`，不得解释为算法或硬件失败。

## 冻结科学身份

以下SHA256必须保持不变：

- Trainer：`7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54`
- Config：`9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Scientific launcher：`ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e`
- Monitor：`536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`

科学方法保持：

- shared trunk继续使用原始Paper actor和sampled Paper critic更新；
- deterministic normalized-residual `J_v` GGN、`lambda=0.1` 仅作用于critic-exclusive value head；
- 独立head-only `B×B` symmetric FP64/Jacobi/Cholesky；
- 不得引入shared GGN、joint/cross blocks、guard或actor修改。

## 唯一允许的代码修正

删除陈旧断言：

```text
SHARED.numel > 1,000,000
```

改为验证冻结生产网络的完整精确不变量：

- 总计：938,979 parameters
- `POLICY_EXCLUSIVE`：2 tensors / 3,855 parameters
- `SHARED`：22 tensors / 934,864 parameters
- `CRITIC_EXCLUSIVE`：2 tensors / 257 parameters
- critic-exclusive仅为：
  - `last_v_layer.weight`
  - `last_v_layer.bias`
- Partition manifest SHA256：
  `b45298be8fc5bdccfa36ce653c7dbc0c41f2d013f23d4f4db0a4a580035f3087`

不得仅删除检查；必须用上述完整invariant替换。除此之外不得修改harness逻辑、production config路径或科学文件。

## 执行范围

1. 从 canonical recovery freeze `9a56a6aba6d70ce7a16b9e81cf105dccbd43d638` 开始。
2. 提交上述单一assertion修正及针对该invariant的静态回归。
3. 执行且仅执行一次修正后的完整真实网络preflight。
4. 如果预检仍因任何原因失败，立即结束为 `PRECHECK_BLOCKED`；不得进行下一轮现场修补。
5. 只有预检全部通过，才可启动以下seed 0科学单元：

   - `bigfish-easy-0-10`
   - `bossfight-easy-0-10`
   - `caveflyer-easy-0-10`
   - `coinrun-easy-0-10`

6. 每格intended horizon为6M，终点为`5,980,160` transitions。
7. Executor在启动前刷新授权资源、ownership、进程、容量、scheduler和重复root；所有实际资源与调度选择均由Executor负责。
8. 科学job/root真实存在前，不得启动科学monitor。

## 强制预检证据

必须完成并保存：

- 三方resolved JSON继续字节一致，SHA256：
  `61f8ebe38443acbdbf141981f4e9921435dccd5d4abb6a63959e3d4bdb9232ab`
- 精确partition和manifest全部匹配。
- critic-exclusive policy Jacobian为零或disconnected，value Jacobian connected。
- Paper actor matrix、RHS和direction bit-identical。
- shared-trunk sampled Paper critic direction bit-identical。
- 一步更新后policy参数和logits bit-identical，只有value-head delta不同。
- head-only deterministic-GGN公式、RHS、阻尼和维度正确。
- FP64/Jacobi/Cholesky `info=0`，残差有限，无fallback。
- production-scale内存验证通过。
- OOM、CUDA、NaN/Inf及hard-error扫描为零。
- 目标root不存在且无重复active objective。

任一检查未通过都禁止科学启动。

## 严格比较和早停

仅与不可变原始Paper RAT的同环境、seed 0、同evaluation semantics、同transition记录比较：

- 第一个共同点 `>=2,000,000`
- 第一个共同点 `>=4,000,000`
- 终点 `5,980,160`

仅当：

```text
Target reward / Paper reward < 0.60
```

时取消对应cell并记为 `EARLY_STOPPED_ALGORITHM`。不得以Paper 6M终值比较中间Target。通过检查的cell继续到下一阶段或终点。

## 验收标准

唯一结论必须是：

- `CANDIDATE_PROMOTE_TO_3SEED`：全部预检通过，至少3/4环境达到终点且终点ratio均不低于0.60，无身份偏移或数值失败。
- `CANDIDATE_REJECT`：至少两个环境严格触发早停，或终点证据明确否定候选。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：科学运行开始后仅因基础设施故障无法形成充分结论。
- `PRECHECK_BLOCKED`：本次唯一授权的修正预检仍未全部通过。

## 必需报告字段

- assignment、assertion-fix、frozen-launch、evidence及Delivery commits。
- assertion修正的逐行diff、理由和SHA256。
- 四个科学文件哈希不变证明。
- `19220448`、`19220752`、`19225085`及本次预检的完整failure ledger。
- resolved-config、partition、Jacobian、一步等价、内存、求解器和错误扫描证据。
- 若启动科学单元：每格scheduler终态、return code、transitions、artifact，以及2M、4M、终点的reward、Paper reward、ratio、KL、LR、entropy和solver telemetry。
- 所有历史算法失败、基础设施失败和取消记录原样保留。

## 禁止事项

- 不得改变科学trainer、config、launcher、monitor或方法语义。
- 不得引入第二候选、sweep、Paper重跑或重复objective。
- 不得放宽或删除partition验证。
- 不得覆盖旧root或重写失败历史。
- 不得使用Jupyter。
- 不得访问`.54`、`ws4090-31`或`10.49.7.54`。
- 不得指定或预设host、GPU、partition、卡数、并发或queue placement。
- 不得触碰无关任务。

## 提交与推送要求

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-HYBRID-HEAD-ASSERTION-FIX-AND-6M-S0-20260824-10.md`

提交允许的preflight修正、model-free证据和报告，保持worktree干净，推送`origin/agent-work`并验证远端HEAD。回调必须报告唯一结论、全部commit、科学哈希、预检结果、四格终态、严格阶段比率及failure-ledger增量。
