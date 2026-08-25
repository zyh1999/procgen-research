Status: READY

# TASK.md

Task-ID: `PROCGEN-GAE-GGN-HEAD-WIDENTITY-6M-S0-20260825-33`

## 唯一目标

立即实现、冻结并提交独立方法：

`DET_GAE_GGN_HEAD_WIDENTITY_V1`

该候选用于检验Task32的actor weighting是否因BigFish出现`weight max=512`及effective-rank collapse而破坏GAE-GGN。新方法明确使用 \(W=I\)，可与Task32并存排队或运行，但不得取消、修改、重提、覆盖或等待Task32。

## 独立版本与非覆盖边界

必须创建独立的：

- trainer；
- config；
- method name；
- launcher/monitor；
- campaign/run root；
- source/config/launcher hashes。

Git继续使用`agent-work`，不得创建或推送`main/master`。这里的独立版本仅指独立代码身份和非覆盖root。

## 严格控制身份

与Task32/Paper control保持完全一致：

- actor及actor optimizer；
- shared-trunk sampled critic及其更新；
- 网络、257参数value head划分；
- rollout、return、GAE、done mask、bootstrap；
- PopArt；
- schedule、minibatch、epochs；
- momentum/history；
- adaptive-KL、global clip；
- seed、evaluation、reward/KL语义和6M停止规则。

只允许去除Task32的actor weighting。不得引入其他科学差异。

## 精确算法

在冻结PopArt标准化坐标中：

\[
e=V_\theta-\operatorname{stopgrad}(\mathrm{return}),
\qquad
q=D_{\gamma,\lambda,\mathrm{mask}}e.
\]

其中 \(D\) 必须严格复用Task32已验证的trajectory、terminal、truncation和bootstrap语义。

明确设：

\[
W=I.
\]

不得计算或使用actor score、policy概率权重、权重归一化、clip、floor或proposal norm matching。

目标：

\[
L_{\mathrm{GAE}}=\frac{1}{2B}\|q\|_2^2.
\]

仅对`last_v_layer.weight/bias`的257个参数构造：

\[
J_h=\frac{\partial V}{\partial\theta_h},
\qquad
K=D J_h,
\qquad
r=q.
\]

求解：

\[
\left(\frac{K^\top K}{B}+0.5I\right)u
=-\frac{K^\top r}{B}.
\]

使用symmetric FP64、Jacobi scaling、Cholesky及既有global clip `.5`。不得根据preflight或训练结果改变damping、目标、权重或其他超参数。

## 必需代码Diff与回归

科学提交前必须生成Task32→Task33逐字段、逐函数和AST diff，并证明唯一科学差异为：

- 删除 \(w_t\) 的构造；
- `diag(sqrt(w))DJ_h → DJ_h`；
- `diag(sqrt(w))q → q`；
- weighted GAE loss → unweighted GAE loss。

必须证明：

- trainer中不存在actor-score/actor-weight路径；
- 运行时无`weight max`、weight clipping/floor或weight-normalization；
- 所有样本的隐式权重严格为1；
- Task32 BigFish的`max=512`集中加权机制在本方法中不可发生；
- effective-rank必须由未加权 \(DJ_h\) 报告，不得用加权矩阵替代；
- actor/shared方向和一步delta与Task32及Paper control bit-identical；
- policy logits bit-identical；
- 仅257个value-head参数delta不同；
- \(D\) finite-difference、PopArt仿射不变性、小矩阵/autograd参考、Cholesky info0、finite residual及nonfinite扫描全部PASS。

若除 \(W=I\) 外存在任何科学差异，停止为`PRECHECK_BLOCKED`。

## 科学矩阵与提交

Preflight PASS后立即提交且仅提交：

- BigFish、BossFight、CaveFlyer、CoinRun；
- seed0；
- 每格intended horizon 6M；
- 四个独立Slurm job；
- 全新、预先验证不存在的非覆盖roots。

允许job进入`PENDING`并等待资源。Executor负责全部实时scheduler、ownership、GPU、partition、concurrency、capacity和queue placement判断。

不得等待Task32终态；也不得触碰Task32的job、root、monitor或artifact。

## 严格早停协议

仅在相同环境、seed0、evaluation语义和精确共同进度比较Original Paper RAT：

- first common `>=2M`
- first common `>=4M`
- `5,980,160`

只有：

\[
\mathrm{Target}/\mathrm{Paper}<0.60
\]

才可取消该单格并记录`EARLY_STOPPED_ALGORITHM`。没有精确共同row则不得操作；中间Target不得比较Paper terminal。

## 必需科学证据

每个stage记录：

- Target/Paper reward及ratio；
- KL、LR、entropy、value loss；
- unweighted \(L_{\mathrm{GAE}}\)；
- GAE mean/variance/RMS；
- TD residual及return error；
- \(DJ_h\) spectrum、effective rank、condition number；
- prediction/parameter/GAE change norm；
- predicted/realized GAE-loss change；
- damping、clip scale、residual、Cholesky info；
- hard-error及NaN/Inf扫描。

必须与Task32同阶段比较：

- effective rank；
- step/prediction/GAE change；
- reward ratio；
- Task32集中actor weighting是否解释BigFish异常。

Task32尚无对应stage时标记`TASK32_PENDING`，不得阻塞Task33或使用非同阶段数据代替。

## 唯一终局结论

仅允许：

- `PRECHECK_BLOCKED`
- `QUEUED_RESOURCE_WAIT`
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`
- `CANDIDATE_REJECT`
- `WIDENTITY_GAE_GGN_SEED0_PROMISING`
- `CANDIDATE_NOT_READY`

`WIDENTITY_GAE_GGN_SEED0_PROMISING`要求：至少三个环境到达5,980,160、最多一个算法早停、至少两个环境终点超过Paper、计入早停stage ratio后的四环境平均ratio大于1，且数值与GAE健康。

不得启动seeds1–2。

## 禁止事项

- 不得取消、修改、覆盖、重排或等待Task32。
- 不得加入actor weighting、norm matching、joint/cross、projection、low-Fisher、adaptive damping或第二候选。
- 不得进行sweep或按结果调参。
- 不得重跑或修改Paper baseline。
- 不得继续Task14–31 provenance/origin observer工作。
- 不得覆盖历史root或改写失败分类。
- 不得使用Jupyter。
- 不得访问`.54`、`ws4090-31`或`10.49.7.54`。
- 不得规划MuJoCo或Isaac。
- Planner不指定host、GPU、partition、卡数、并发或queue placement。

## 报告、提交与回调

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-GAE-GGN-HEAD-WIDENTITY-6M-S0-20260825-33.md`

报告必须包含冻结hash、Task32→Task33唯一科学diff、preflight、job/root映射、scheduler与artifact状态、stage表、失败账本及唯一结论。

提交代码和模型无关证据，不提交model/checkpoint。推送`origin/agent-work`并验证远端HEAD后回调Planner。
