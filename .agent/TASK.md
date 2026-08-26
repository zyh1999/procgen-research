Status: READY

# TASK.md

Task-ID: `PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-AUDIT-PATH-RECOVERY-20260826-36`

## 唯一目标

仅修复Task35R historical-scaling audit的trainer路径定位，使冻结audit读取hermetic manifest中位于`bundle/code/`的精确冻结trainer；local gate通过后，对四环境各运行一次actual-network preflight。

不得改变科学算法、bundle内容或启动6M训练。

## 冻结身份

保持不变：

- Task34R implementation：`55984df39bf883685583f22894edd5eb615f95ea`
- Trainer：`ca53efd549ae58738c5c215c84dfe5f342c0f801d3eb0f7fb61a2507f31e69fc`
- Config：`52c133559825947cd233184f2468c4aa715c6c419274bb78f7f715d542718132`
- Preflight：`2baac759c26c4fb663ac24ff68a6de8b640bb9bd0c443ec8f65106de3d36759a`
- Historical audit数值逻辑：`9d4929685bd8368f861770f155c5fcc0b7f86b91e9cfc92481987b0dc8ec2723`
- Bundle archive：`3a9d9720ae7b3c9e6d13a2fd63521d51bba8cb62e7ae7ae2498553b57c00609f`
- Manifest：`287a744078b10054d107974125bac6b5fac43fd944b6200ec54720cd2695c9af`
- Scientific launcher：`6dffce265e55f87fa5be848ec5bd9940fcecb6203f621a1454fb4f6a9c74caca`

## 唯一允许修改

版本化audit路径适配层，使audit通过显式参数获得trainer路径，而不是假设trainer与audit相邻。

该路径必须：

- 由manifest按精确repo path解析；
- 位于当前验证bundle的`bundle/code/`内；
- 匹配冻结Git blob、SHA256、size和mode；
- 是regular non-symlink file；
- 在audit执行前后保持相同identity/hash。

不得：

- 修改audit的任何数学断言；
- 修改或复制trainer；
- 在`bundle/frozen/`创建兼容文件或symlink；
- 重建或修改bundle/manifest；
- 使用ambient checkout或历史run root。

## 必需负测试

必须拒绝：

- 旧的`bundle/frozen/...trainer.py`假路径；
- symlink、路径逃逸；
- 同字节不同manifest身份；
- 错误blob/hash/size/mode；
- 缺失或重复manifest条目；
- ambient repository fallback；
- 对audit数值逻辑的任何修改。

## 单次Local Gate

只允许运行一次完整local gate：

1. 复核原bundle/archive/manifest hashes；
2. 解析并记录精确trainer identity；
3. empty-CWD import及module-origin ledger继续PASS；
4. 运行冻结historical audit全部断言；
5. 必须恢复以下结果：

   - \(\|V-\operatorname{stopgrad}(R_\lambda)\|^2/(2B)\)
   - \(G=J^\top J/B\)
   - \(g=J^\top e/B\)
   - Task13 effective standard-coordinate damping `5`
   - RHS multiplier `10`
   - transformed equality误差不超过既有FP64容差

任何失败即`PRECHECK_BLOCKED`，不得修复或重跑。

## 四环境一次性Preflight

Local Gate PASS后，对以下环境各提交恰好一次全新preflight：

- BigFish seed0
- BossFight seed0
- CaveFlyer seed0
- CoinRun seed0

使用互不覆盖且预先不存在的新roots。每格必须验证：

- hermetic imports及module origins；
- production network及257参数head partition；
- standard MSE、`D=I`、`W=I`、precision=1；
- 完整512-row train block；
- calibration rows不进入当前\(G,g\)，但保留原训练schedule；
- `ared_T=pred_T` FP64等式；
- 非退化cross-minibatch`\rho_cv`回归；
- rejected-trial完整bitwise rollback；
- accepted delta只由完整train rows构造；
- actor/shared方向、delta及policy logits与control bit-identical；
- PopArt affine-scale regression；
- Cholesky info0、finite residual、NaN/Inf及hard-error扫描。

任一失败不得现场修复、重提或切换科学身份。

## 唯一结论

仅允许：

- `PRECHECK_RECOVERED`：local gate与四环境preflight全部PASS；停止等待Planner。
- `PRECHECK_BLOCKED`
- `QUEUED_RESOURCE_WAIT`
- `RESOURCE_PLACEMENT_BLOCKED`

不得启动科学job、monitor、transition、checkpoint或model。

## 资源边界

Executor负责刷新ownership、account、QOS、GRES、capacity、process、duplicate及live placement，并按用户偏好优先考虑gpuH。若不可用或不兼容，报告精确证据；不得静默换queue。Planner不指定具体GPU、partition、卡数或并发。

## 禁止事项

- 不得修改Task34R算法、trainer/config、CVLM或comparison protocol。
- 不得重试Task34R/35R旧jobs。
- 不得触碰Task32/Task33。
- 不得加入GAE算子、actor weighting、Paper matching、joint/cross或sweep。
- 不得使用Jupyter。
- 不得访问`.54`、`ws4090-31`或`10.49.7.54`。
- 不得规划MuJoCo或Isaac。

## 报告与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-AUDIT-PATH-RECOVERY-20260826-36.md`

报告必须包含路径适配diff、trainer manifest identity、负测试、historical audit、四环境preflight矩阵、scheduler/root/artifact状态、失败账本及唯一结论。

提交代码和模型无关证据，不提交model/checkpoint。推送`origin/agent-work`并验证远端HEAD后回调Planner。

Begin this unique READY task now. Preserve user placement preference: after the single local gate passes, refresh live gpuH ownership/account/QOS/GRES/capacity/duplicate state and prefer gpuH. Do not start 6M science in Task36. Callback with exact preflight job IDs/states or the bounded terminal conclusion.
