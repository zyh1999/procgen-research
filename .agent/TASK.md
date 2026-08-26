Status: READY

# TASK.md

Task-ID: `PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-HERMETIC-PREFLIGHT-20260826-35R`

## 唯一目标

仅修复Task34R的hermetic packaging/import path，使冻结方法`DET_STANDARD_MSE_GGN_HEAD_CVLM_V1`完成四环境各一次actual-network preflight。

本任务不得启动科学训练。它取代此前未执行的Task35草案，并且是唯一Task34R恢复目标。

## 已确认分类

Task34R的四个旧job `19319418–19319421`均在模型构造前因：

`ModuleNotFoundError: No module named 'utils'`

失败，分类为`infrastructure-failure/deployment-package-import`，不是算法、数值、GPU或科学失败。不得重试这些旧job。

## 冻结科学身份

必须保持Task34R implementation `55984df39bf883685583f22894edd5eb615f95ea`的以下内容字节或语义完全不变：

- trainer及config；
- standard frozen-lambda-return MSE；
- \(D=I,\ W=I,\ K=J\)；
- \(G=J^\top J/B,\ g=J^\top e/B\)；
- Gaussian precision=1；
- CVLM train/calibration pairing、阈值、trial及回滚规则；
- 仅更新257个value-head参数；
- actor、shared sampled critic、PopArt；
- schedule、minibatch、epochs、momentum/history；
- adaptive-KL、global clip及evaluation语义。

保留历史缩放结论：Task13等效standard-coordinate damping为5、RHS multiplier为10，变换误差`1.1102230246251565e-16`。

## 唯一允许修改

仅可版本化：

- source bundle；
- bundle manifest/verifier；
- deployment/preflight launcher；
- import path及全新preflight root。

不得修改trainer源码来绕过import。

## Hermetic Bundle要求

1. Bundle只能由冻结Git对象构建。
2. 必须包含trainer/config及完整repository-local import closure，包括正确的：

   - `utils` package；
   - `utils.logger`；
   - trainer实际可达的其他local modules。

3. Manifest逐文件记录repo path、Git blob、SHA256、size和mode。
4. 两次独立构建必须产生字节相同的archive和manifest。
5. 在bundle外的空cwd中，仅以bundle root作为repository-local import root。
6. 禁止从ambient checkout、历史run root、未记录scratch目录或网络补文件。
7. 负测试必须拒绝缺失`utils`、错误hash、不同Git blob及ambient-path fallback。

## Launcher等价性

新launcher与Task34R原launcher的规范化scientific command必须完全一致。唯一允许差异：

- bundle验证与解包；
- import root；
- 新的非覆盖preflight root；
- deployment provenance字段。

env、seed、config、method、device参数和所有科学变量不得变化。

## Local Gates

提交远端preflight前必须PASS：

- compile/import smoke；
- 空cwd trainer import；
- `utils.logger`及所有local module origin均位于bundle manifest；
- resolved config一致；
- trainer/config hash一致；
- launcher normalized-command equality；
- Task34R历史scaling audit原样PASS；
- 四个新preflight roots均不存在；
- 没有Task34R science process/root或duplicate。

任一本地门失败即`PRECHECK_BLOCKED`，不得提交远端preflight。

## 四个一次性Actual-Network Preflight

Local Gates全部PASS后，允许对以下环境各提交恰好一次新preflight：

- BigFish seed0；
- BossFight seed0；
- CaveFlyer seed0；
- CoinRun seed0。

必须使用全新、互不覆盖的preflight roots。不得修复后重提任何失败格。

每格必须验证：

- hermetic bundle及module origins；
- production network构造与参数总数；
- 精确257参数value-head partition；
- standard MSE、\(G,g\)、sign、`1/B`及precision=1；
- 完整512-row train minibatch；
- validation block不参与当前\(G,g\)，但保留原训练schedule；
- `ared_T=pred_T`达到FP64容差；
- 非退化cross-minibatch`\rho_cv`回归；
- rejected trial对参数、optimizer、momentum、PopArt和RNG bitwise回滚；
- accepted delta仅由完整train rows构造；
- actor/shared方向、delta及policy logits与control bit-identical；
- PopArt affine-scale回归；
- Cholesky info0、finite residual和nonfinite/hard-error扫描。

## 验收标准

仅允许以下唯一结论之一：

- `PRECHECK_RECOVERED`：四环境preflight全部PASS，科学算法仍冻结；停止等待Planner。
- `PRECHECK_BLOCKED`：任一local或actual-network gate失败。
- `QUEUED_RESOURCE_WAIT`：提交后仅因队列/配额未运行。
- `RESOURCE_PLACEMENT_BLOCKED`：刷新后没有符合约束的可用放置证据。

不得把scheduler完成单独视为preflight成功。

## 资源边界

Executor负责刷新gpuH ownership、account、QOS、GRES、capacity、process和duplicate状态，并拥有全部live placement决定权。遵循用户偏好优先考虑gpuH；若不可用或不兼容，记录精确证据，不得静默换queue或改变科学身份。

Planner不指定GPU、partition、卡数或并发。

## 禁止事项

- 不得启动任何6M科学job、monitor、transition或checkpoint。
- 不得重试旧Task34R jobs。
- 不得修改CVLM、damping、objective、threshold或comparison protocol。
- 不得加入GAE算子、actor weighting、Paper matching、joint/cross或sweep。
- 不得触碰Task32/Task33及其roots/history。
- 不得继续Task14–31 provenance observer框架。
- 不得使用Jupyter。
- 不得访问`.54`、`ws4090-31`或`10.49.7.54`。
- 不得规划MuJoCo或Isaac。

## 报告与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-HERMETIC-PREFLIGHT-20260826-35R.md`

报告必须包含bundle/manifest hashes、module-origin表、launcher等价性、四环境preflight矩阵、scheduler/root/artifact状态、完整failure ledger及唯一结论。

提交代码和模型无关证据，不提交model/checkpoint。推送`origin/agent-work`并验证远端HEAD后回调Planner。

User placement preference: prioritize gpuH after refreshing live ownership/account/QOS/GRES/capacity/duplicate state. Begin this unique READY task now. Do not run 6M science in this task; callback with exact preflight job IDs/states or the bounded terminal conclusion.
