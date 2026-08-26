Status: READY

# TASK.md

Task-ID: `PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-6M-S0-SCIENCE-20260826-37`

## 唯一目标

运行冻结方法`DET_STANDARD_MSE_GGN_HEAD_CVLM_V1`的四环境seed0、intended-6M科学检验，判断标准per-sample head GGN与cross-minibatch CVLM能否改善Task13固定相对阻尼的跨环境失败。

不得修改算法或创建第二候选。

## 冻结身份

必须保持Task34R/36验证身份不变：

- Trainer：`ca53efd549ae58738c5c215c84dfe5f342c0f801d3eb0f7fb61a2507f31e69fc`
- Config：`52c133559825947cd233184f2468c4aa715c6c419274bb78f7f715d542718132`
- Preflight：`2baac759c26c4fb663ac24ff68a6de8b640bb9bd0c443ec8f65106de3d36759a`
- Historical audit：`9d4929685bd8368f861770f155c5fcc0b7f86b91e9cfc92481987b0dc8ec2723`
- Bundle：`3a9d9720ae7b3c9e6d13a2fd63521d51bba8cb62e7ae7ae2498553b57c00609f`
- Manifest：`287a744078b10054d107974125bac6b5fac43fd944b6200ec54720cd2695c9af`
- Science launcher：`6dffce265e55f87fa5be848ec5bd9940fcecb6203f621a1454fb4f6a9c74caca`
- Task36 adapter：`7b8cd684f448b730720e4acd1a9c6762faac95778339471770bd40b11f889dd4`

直接复用Task36四环境`PRECHECK_PASS`，不得重跑preflight。

## 科学定义

保持：

\[
e=V-\operatorname{stopgrad}(R_\lambda),\qquad D=I,\quad W=I,
\]

\[
G=J^\top J/B,\qquad g=J^\top e/B,\qquad
(G+\mu I)u=-g.
\]

仅更新257个value-head参数；Gaussian precision为1。Cross-minibatch CVLM的完整512-row train block、独立512-row calibration block、\(\rho_{\rm cv}\)阈值、trial上限、bitwise rollback、momentum/history、global clip及\(\alpha\)更新规则全部冻结。

Actor、shared sampled critic、PopArt、rollout、lambda-return、schedule、adaptive-KL和evaluation保持严格control身份。

## 启动前检查

Executor必须刷新并记录：

- scheduler、ownership、account、QOS、GRES和capacity；
- Procgen jobs、trainer processes及duplicate；
- 四个新science roots不存在；
- bundle、manifest、trainer、config、launcher hashes；
- hermetic import无ambient fallback；
- Task36四个PASS证据完整。

若存在identity漂移、duplicate或root碰撞，停止，不得提交。

Executor负责live placement，并按用户偏好优先考虑gpuH；若不可用或不兼容，报告精确证据，不得静默换queue或改变科学身份。

## 科学矩阵

仅允许各提交一次：

- `bigfish-easy-0-10`, seed0
- `bossfight-easy-0-10`, seed0
- `caveflyer-easy-0-10`, seed0
- `coinrun-easy-0-10`, seed0

每格：

- intended horizon 6M；
- terminal convention `5,980,160`；
- 全新且互不覆盖的root；
- 独立status、rc、stdout、stderr、progress、metric trace和checkpoint；
- 启动时仅复核冻结bundle身份，不重新运行Task36完整preflight。

禁止retry、requeue或resubmit。

## 唯一Monitor

创建一个只绑定本任务四个job IDs/roots的5分钟monitor，负责：

- scheduler优先于stale marker；
- 核对process、progress、trace、checkpoint、status及rc；
- 扫描Traceback、OOM、CUDA、NCCL、disk、stall及NaN/Inf；
- 仅在规定stage执行Paper比较；
- 四格终态后退役。

不得触碰Task32或重提Task33。

## 严格同阶段早停

仅比较相同环境、seed0、evaluation语义的immutable matching Paper RAT：

- first common `>=2M`
- first common `>=4M`
- `5,980,160`

只有：

\[
\mathrm{Target}/\mathrm{Paper}<0.60
\]

才可取消该单格并记录`EARLY_STOPPED_ALGORITHM`。无精确共同row不得操作；中间Target不得比较Paper terminal。

## 必需证据

每个stage记录：

- Target/Paper reward及ratio；
- KL、actor LR、entropy；
- MSE、TD error、GAE统计；
- PopArt mean/std；
- \(G\) trace、spectrum、condition、effective rank；
- \(\alpha,\mu\)、trial数和accept/reject率；
- `pred_T`、`ared_T`、`ared_C`、`\rho_cv`；
- MSE gradient、raw GGN及最终head delta的norm/cosine；
- momentum/global-clip前后变化；
- prediction change；
- residual、Cholesky info及hard-error/nonfinite扫描。

必须判断：

- CVLM实际相对阻尼与Task13固定等效damping 5的差异；
- BigFish是否再次2M通过后4M失败；
- CaveFlyer是否保持成功；
- held-out acceptance是否对应后续GAE/reward稳定性。

## 唯一结论

仅允许：

- `QUEUED_RESOURCE_WAIT`
- `RESOURCE_PLACEMENT_BLOCKED`
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`
- `CANDIDATE_REJECT`
- `STANDARD_GGN_CVLM_SEED0_PROMISING`
- `CANDIDATE_NOT_READY`

`STANDARD_GGN_CVLM_SEED0_PROMISING`要求：至少三个环境到达5,980,160、最多一个算法早停、至少两个环境终点超过Paper、计入早停stage ratio后的四环境平均ratio大于1，且CVLM和数值证据健康。

不得启动seeds1–2或正式x3扩展。

## 禁止事项

- 不得修改trainer、config、CVLM、threshold、damping定义或comparison protocol。
- 不得使用GAE时序算子、actor weighting、Paper matching、joint/cross或sweep。
- 不得重跑Paper或Task36 preflight。
- 不得重试、requeue或resubmit任何cell。
- 不得修改Task32或重提Task33。
- 不得覆盖历史root或改写失败分类。
- 不得使用Jupyter。
- 不得访问`.54`、`ws4090-31`或`10.49.7.54`。
- 不得规划MuJoCo或Isaac。
- Planner不指定具体GPU、partition、卡数或并发。

## 报告与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-6M-S0-SCIENCE-20260826-37.md`

报告必须包含冻结身份、启动检查、job/root映射、2M/4M/终点表、CVLM诊断、scheduler/artifact/error状态、历史失败账本及唯一结论。

提交代码和模型无关证据，不提交model/checkpoint。推送`origin/agent-work`并验证远端HEAD后回调Planner。

Begin this unique science READY now. Refresh live gpuH state and prefer gpuH. Reuse Task36 PRECHECK_PASS without rerunning it. Submit exactly the four authorized seed0 intended-6M cells once, with fresh roots and no retries. Callback immediately with frozen assignment/delivery, exact job IDs/roots and initial scheduler states so the existing automation procgen-3090 can be converted—not duplicated—into the sole 5-minute Task37 science monitor.
