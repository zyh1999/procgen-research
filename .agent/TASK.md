Status: READY

# Task-ID: PROCGEN-HYBRID-HEAD-CANONICAL-PREFLIGHT-AND-6M-S0-20260824-09

## 唯一目标

仅修复 `PAPER_MATCHED_HYBRID_HEAD_DETGGN_V1` 的预检 harness，使其通过生产配置路径构造真实网络并完成实际网络兼容性证明；预检通过后，按冻结科学身份执行四个 Procgen 环境、seed 0、每格预定 6M transitions 的严格 Paper RAT 匹配实验。

## 证据判断

前两次失败均发生在训练和兼容性测试之前：

1. `19220448` 因预检 import path 错误无法导入 `utils`。
2. `19220752` 已修复导入，但手工构造的测试 namespace 缺少生产模型必需的 `norm_obs`。

因此它们属于 `infrastructure-failure/preflight-design`，不构成算法、数值、求解器或硬件不兼容证据。静态审计与回归已经证明冻结方法满足所需隔离；当前缺口仅是真实生产配置和真实网络上的验证。允许进行一次明确、可审计的预检 harness 恢复，但不得改变科学候选。

## 冻结科学身份

必须保持以下文件和语义不变：

- Trainer SHA256：`7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54`
- Config SHA256：`9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Scientific launcher SHA256：`ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e`
- Monitor SHA256：`536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`

方法必须继续是：

- 所有 shared-trunk 参数使用原始 Paper RAT 的 actor 更新和 sampled critic 更新。
- 仅对 257 个 critic-exclusive value-head 参数，以 normalized residual、`lambda=0.1` 的 deterministic `J_v` GGN 替换 Paper critic 更新。
- 使用独立 head-only `B×B`、symmetric FP64、Jacobi scaling、Cholesky direct solve。
- 禁止 joint-2B、shared deterministic GGN、cross blocks、low-Fisher guard、投影、Kaczmarz或任何 actor 调整。

## 范围与允许动作

1. 从已验证的 `fe4b8a58812e80689705abec11364457cae31e26` 冻结实现开始。
2. 只修改预检 harness、预检 launcher及其测试；不得修改 trainer、科学 config、科学 launcher或 monitor。
3. 删除手工拼装的不完整模型 namespace。预检必须调用与科学 trainer 相同的生产配置加载、默认值合并和 `SharedActorCritic` 构造路径。
4. 不得通过仅向旧 mock namespace 补加 `norm_obs` 来绕过配置一致性。
5. 将以下三处的 fully resolved configuration 确定性序列化并比较：
   - 修正后的预检；
   - scientific launcher dry-run；
   - trainer 入口。
6. 仅允许非科学性的兼容性适配，例如 import path、模块入口或预检输出位置；每项修改必须形成逐行 diff、理由和 SHA256。
7. 在启动科学单元前刷新所有授权资源的 scheduler、GPU、进程、所有权、容量、现有 Procgen roots和重复任务状态。具体资源与调度完全由 Executor 决定。
8. 预检通过后，执行四个环境的 seed 0：

   - `bigfish-easy-0-10`
   - `bossfight-easy-0-10`
   - `caveflyer-easy-0-10`
   - `coinrun-easy-0-10`

   每格 intended horizon 为 6M，终点为 `5,980,160` transitions。

## 强制预检

科学启动前必须在真实生产模型上同时证明：

- 生产配置加载和真实 `SharedActorCritic` 构造成功。
- resolved configuration 与科学 trainer/launcher 完全一致。
- 参数分区 exhaustive、mutually exclusive且顺序稳定。
- critic-exclusive 集合恰为 257 个 value-head 参数。
- 这些参数对 policy logits 的 Jacobian严格为零或 autograd-disconnected。
- Paper actor matrix、RHS和方向 bit-identical。
- shared-trunk sampled Paper critic方向 bit-identical。
- 单步更新后 policy参数和 logits bit-identical；只有 value-head delta允许不同。
- head-only deterministic GGN采用规定的 residual、`lambda=0.1`、`B×B` 几何。
- FP64/Jacobi/Cholesky有限，`info=0`，相对残差有记录。
- 实际网络的内存/运行兼容性通过，无 OOM、NaN、Inf、CUDA错误或 silent fallback。
- 非重复 root、命令、源码、配置和环境身份全部冻结。

任一项失败均不得启动科学单元；结论必须为 `PRECHECK_BLOCKED`，并准确分类失败。

## 严格比较与早停协议

只与不可变的原始 Paper RAT同环境、同 seed、同 evaluation semantics和同 transition记录比较：

- 第一个共同点 `>=2,000,000`；
- 第一个共同点 `>=4,000,000`；
- 精确终点 `5,980,160`。

在任一阶段，仅当：

`Target reward / Paper reward < 0.60`

时，才取消对应 cell并标记 `EARLY_STOPPED_ALGORITHM`。必须保存环境、seed、精确 transition、Target/Paper reward、ratio、KL、LR、entropy、求解器遥测、日志、scheduler证据和取消证据。

不得将中间 Target与 Paper 6M终值比较。通过阶段检查的 cell继续至下一阶段或6M终点。

## 必需证据

报告必须包含：

- assignment、修复提交、冻结启动提交、证据提交和 Delivery HEAD。
- 所有相关文件的完整 SHA256及逐行 diff分类。
- 两次既有预检失败和本次恢复的不可变 failure ledger。
- 三方 resolved-configuration比较。
- 真实网络分区、Jacobian、方向等价、单步等价和内存预检结果。
- 每格 scheduler、状态、返回码、节点运行事实、transitions和artifact清单。
- 2M、4M及终点的严格 Target/Paper表和比率。
- reward、KL、actor LR、entropy、critic residual、Cholesky info及非有限扫描。
- checkpoint存在性和完整性；早停前尚未产生 checkpoint时明确记录。
- hard-error、OOM、CUDA、NCCL、disk/quota、stall、NaN/Inf扫描。
- 所有历史失败、取消和负面结果原样保留，不得被成功恢复覆盖。

## 验收标准

仅允许以下唯一终局分类之一：

- `CANDIDATE_PROMOTE_TO_3SEED`：预检完整通过，至少三个环境到达6M，且到达终点的环境均保持 ratio `>=0.60`；任何早停环境及其失败仍完整保留。
- `CANDIDATE_REJECT`：预检通过，但至少两个环境在2M或4M触发严格 `<0.60` 早停，或终点证据明确不支持扩展。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：科学运行开始后仅因基础设施原因无法形成足够算法证据。
- `PRECHECK_BLOCKED`：真实生产网络的任一强制预检仍未通过。

不得宣称四环境×三种子正式完成；本任务仅为 seed-0 因果候选判定。

## 禁止事项

- 不得改变冻结科学身份或引入第二候选。
- 不得进行 sweep、超参数搜索或 Paper RAT重跑。
- 不得复活 joint/cross、low-Fisher或历史 expected/no-cross方法。
- 不得覆盖或重用旧 root；所有新 root必须唯一且先做非重复检查。
- 不得把预检设计失败归类为算法失败。
- 不得访问 `.54`、`ws4090-31` 或 `10.49.7.54`。
- 不得使用 Jupyter。
- Planner不指定主机、GPU、partition、卡数、并发或队列位置；这些全部由 Executor在实时刷新后决定。
- 不得触碰无关任务或取消无关作业。

## 报告与提交推送

完成后更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-HYBRID-HEAD-CANONICAL-PREFLIGHT-AND-6M-S0-20260824-09.md`

报告必须给出唯一结论、完整证据路径、失败分类、严格比较表和下一步阻塞项。提交全部允许的代码、配置外预检修改、model-free证据和报告，保持 worktree干净，推送 `origin/agent-work`，验证远端 HEAD，并回调：

- Task-ID
- assignment/frozen/evidence/Delivery commits
- 远端验证结果
- 唯一结论
- 四格终态及严格阶段比率
- 科学身份是否保持不变
- failure ledger增量
