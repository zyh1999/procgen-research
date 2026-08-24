Status: READY

# Task-ID: PROCGEN-HYBRID-HEAD-STRUCTURAL-MANIFEST-RECOVERY-6M-MISSING3-20260824-12

## 唯一目标

修正per-job preflight错误地将环境相关connectivity probe数值纳入结构manifest哈希的问题；保持科学候选完全不变，仅为BossFight、CaveFlyer和CoinRun补齐seed 0、预定6M的科学证据，并与既有BigFish结果合并判定候选。

## 结果解释

BigFish是有效算法失败：

- 2,007,040：`6.53/9.28=0.70366`，通过。
- 4,014,080：`6.23/13.28=0.46913`，严格早停。
- 求解器有限、Cholesky `info=0`、hard-error为零。

这说明即使单步policy、shared-trunk更新与Paper bit-identical，value-head变化仍可通过后续value estimate、GAE和训练轨迹产生长期影响。2M通过不能排除4M失败。

其余三格没有科学数据。它们的模型结构、参数名称和partition一致，但完整JSON包含环境观测产生的probe值，导致整文件SHA不同。因此属于`infrastructure-failure/per-job-preflight-design`，不能与BigFish一起构成两环境算法拒绝。

## 冻结科学身份

以下SHA256必须保持不变：

- Trainer：`7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54`
- Config：`9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Scientific launcher：`ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e`
- Stage monitor：`536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`

方法仍为原始Paper actor及sampled shared critic，加仅作用于257个critic-exclusive value-head参数的deterministic normalized-residual `J_v` GGN、`lambda=0.1`、独立head-only `B×B` symmetric FP64/Jacobi/Cholesky。

## 唯一允许的代码修正

将preflight输出拆分为：

### Structural manifest

只包含环境无关字段：

- 参数有序名称、partition标签、shape、dtype、`requires_grad`和numel；
- trainable/optimizer成员关系；
- tensor及参数计数；
- critic-exclusive精确名称。

必须严格满足：

- total：938,979
- policy-exclusive：2 tensors / 3,855
- shared：22 tensors / 934,864
- critic-exclusive：2 tensors / 257
- trainable：26 tensors / 938,976
- critic-exclusive仅为`last_v_layer.weight`、`last_v_layer.bias`

四环境structural manifest必须字节一致并产生相同的新SHA256。

### Connectivity evidence

环境/输入相关probe值写入独立的`connectivity_probe.json`。每个环境分别保存SHA256，不要求跨环境相等，但必须分别通过：

- critic-exclusive policy autograd disconnected或Jacobian L2严格为零；
- value路径connected且有限；
- partition与structural manifest一致；
- 无NaN、Inf或fallback。

禁止硬编码已观察到的环境SHA、建立白名单、跳过manifest检查或放宽结构不变量。

## 有界执行

1. 从Task 11 harness freeze `26b2252527076df4bfe537a8612446317cbdcf3a`开始，仅实现上述证据拆分。
2. 对四个环境各进行一次无训练compatibility validation。
3. 任一结构manifest不同或connectivity语义失败，立即结束为`PRECHECK_BLOCKED`；不得继续修补。
4. 全部通过后，只启动先前没有科学数据的三个seed-0 cell：

   - `bossfight-easy-0-10`
   - `caveflyer-easy-0-10`
   - `coinrun-easy-0-10`

5. 不得重跑BigFish。`19228676`的4M算法早停永久保留。
6. 三格分别使用全新、非覆盖root；intended horizon为6M，终点`5,980,160`。
7. 每个缺失环境仅允许一次科学提交；不得retry、requeue或resubmit。
8. Executor负责实时资源检查及全部host、GPU、partition、卡数、并发和queue placement决策。

## 强制预检

启动科学单元前必须证明：

- 全部科学文件哈希未变。
- 三方resolved configuration继续字节一致，SHA256为  
  `61f8ebe38443acbdbf141981f4e9921435dccd5d4abb6a63959e3d4bdb9232ab`。
- 四环境structural manifest字节一致。
- 四环境connectivity probe分别通过语义检查。
- trainable集合与production optimizer逐项一致。
- PopArt非训练状态保留且不进入optimizer、autograd或方向更新。
- Paper actor、sampled shared critic、one-step policy/logits/shared delta继续bit-identical。
- 仅value-head delta不同。
- FP64/Jacobi/Cholesky `info=0`且残差有限。
- 内存及OOM、CUDA、NaN/Inf、hard-error检查通过。
- 新root不存在且没有重复active objective。

## 严格比较与早停

三个新cell仅与不可变原始Paper RAT的同环境、seed 0、同evaluation semantics和同transition记录比较：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当`Target reward / Paper reward < 0.60`时取消对应cell并记录`EARLY_STOPPED_ALGORITHM`。不得使用Paper终点比较中间Target；没有精确共同记录时不得取消。

## 验收标准

合并BigFish既有结果后，唯一结论必须为：

- `CANDIDATE_PROMOTE_TO_3SEED`：三个新环境全部到达终点且终点ratio均不低于0.60；BigFish失败仍保留。
- `CANDIDATE_REJECT`：三个新环境中至少一个严格触发早停，从而形成至少两个环境算法失败；或完整终点证据明确否定候选。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：预检通过并开始科学运行，但基础设施故障仍阻止充分判定。
- `PRECHECK_BLOCKED`：cross-environment预检未全部通过。

## 必需证据

- 修正diff、旧/新harness SHA256及科学哈希不变证明。
- 四环境structural manifest、新SHA及逐字段一致性表。
- 四份connectivity probe、各自SHA和语义判定。
- Task 11全部结果及preflight failure ledger原样保留。
- 三个新cell的root、命令、scheduler、return code、transitions和artifacts。
- 2M、4M、终点的Target/Paper reward、ratio、KL、LR、entropy、head/solve residual和Cholesky info。
- checkpoint及OOM、CUDA、NCCL、disk、stall、Traceback、NaN/Inf扫描。
- 明确区分算法失败、preflight设计失败和其他基础设施失败。

## 禁止事项

- 不得改变算法、trainer、config、scientific launcher或monitor。
- 不得重跑BigFish或覆盖四个既有root。
- 不得引入第二候选、sweep或Paper重跑。
- 不得使用Jupyter。
- 不得访问`.54`、`ws4090-31`或`10.49.7.54`。
- 不得指定具体计算资源或触碰无关任务。

## 报告、提交与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-HYBRID-HEAD-STRUCTURAL-MANIFEST-RECOVERY-6M-MISSING3-20260824-12.md`

提交允许的preflight修正、model-free证据和报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、assignment/freeze/evidence/Delivery commits、四环境manifest/probe结果、三个新cell终态、严格阶段比率及failure-ledger增量。
