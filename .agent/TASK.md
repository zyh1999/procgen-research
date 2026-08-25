Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-HERMETIC-BUNDLE-AND-6M-S0-20260825-15

## 唯一目标

为冻结的 `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2` 建立完整、可审计的hermetic生产源码bundle，修复缺失本地`utils`模块的部署问题；执行一次完整四环境真实网络preflight，并仅在全部通过后运行四环境、seed 0、预定6M科学实验。

不得改变算法或重新选择候选。

## 证据判断

Task 14的四个preflight均在导入冻结trainer时以相同错误终止：

```text
ModuleNotFoundError: No module named 'utils'
```

失败发生在production model构造、真实网络检查和训练之前。静态及allocation regression已经通过，且没有reward、transition或数值求解结果。因此该结果属于`infrastructure-failure/deployment-package-incomplete`，不是norm-matching方法的算法或数值证据。

## 冻结科学身份

必须保持字节不变：

- Method：`PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`
- V2 trainer：`0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b`
- Config：`9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Preflight：`b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc`
- Regression：`f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c`
- Monitor：`536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`

原Task 14 launcher `85e12886ce5cf81fd98647aa5163319a50174a39210cbeea1ccfde015aaf9d19`必须保留。允许创建一个deployment-only launcher变体，其科学命令必须与原launcher路径归一化后字节一致。

## 唯一允许的代码/部署变化

1. 从冻结V2源树构造content-addressed source bundle，包含trainer及其全部repository-local import closure，包括实际被导入的`utils.py`或`utils/` package。
2. bundle内每个源码文件必须来自冻结Git对象；记录：

   - repository-relative path；
   - blob SHA或commit provenance；
   - SHA256；
   - size；
   - import依赖关系。

3. 创建deployment-only launcher变体，仅允许：

   - 验证bundle manifest和文件SHA；
   - 将bundle解压至全新临时目录；
   - 将该目录设置为明确的`PYTHONPATH`/working directory；
   - 调用冻结preflight和trainer；
   - 记录bundle及部署provenance。

4. 禁止修改、内联、重写或以不同版本替代`utils`。
5. 禁止依赖远端已有checkout、用户site-packages、当前工作目录或隐式`PYTHONPATH`碰巧提供本地模块。
6. 不得通过捕获`ImportError`、fallback import或动态下载包来绕过失败。

## Hermetic bundle验收

远端preflight前必须在一个不包含仓库checkout路径的干净进程中证明：

- bundle manifest完整且SHA验证通过；
- `sys.path`仅包含标准/环境依赖及显式bundle路径，不包含源仓库；
- V2 trainer可导入；
- `utils`实际解析路径位于bundle内；
- 递归repository-local imports全部解析到bundle；
- 四环境resolved config均可生成；
- trainer/config/preflight哈希保持冻结值；
- 原launcher与deployment变体在归一化bundle/root路径后，trainer命令、参数顺序、环境、seed、budget和科学变量完全一致。

若clean-room import或等价审计失败，结论为`PRECHECK_BLOCKED`，不得提交远端preflight。

## 有界执行范围

1. 提交bundle manifest、构建/验证工具和deployment-only launcher变体。
2. 运行一次clean-room bundle审计。
3. 审计通过后，对四环境各执行一次完整真实网络preflight。
4. 任一环境preflight失败即结束为`PRECHECK_BLOCKED`；不得现场修补、retry或启动任何科学cell。
5. 四环境全部通过后，运行：

   - `bigfish-easy-0-10`，seed 0
   - `bossfight-easy-0-10`，seed 0
   - `caveflyer-easy-0-10`，seed 0
   - `coinrun-easy-0-10`，seed 0

6. 每格intended horizon为6M，终点`5,980,160`，使用全新非覆盖root，每格最多一次科学提交。
7. Executor负责实时scheduler、GPU、进程、ownership、capacity及nonduplicate检查，并独立决定全部资源和调度位置。

## 强制真实网络preflight

每个环境必须证明：

- canonical production model成功构造；
- structural manifest跨环境一致；
- trainable集合与production optimizer逐项一致；
- PopArt非训练状态正确保留和隔离；
- critic head policy Jacobian为零或disconnected，value connected；
- `u_det`与同一步`u_paper`处于相同更新边界；
- `||u_target||₂=||u_paper||₂`在规定容差内；
- actor/shared proposal、global clip scale、one-step policy参数、logits和shared delta与Paper bit-identical；
- 仅value-head方向不同；
- 无额外RNG、sample或data-order变化；
- FP64/Jacobi/Cholesky `info=0`、残差有限；
- 内存及hard-error、OOM、CUDA、NaN/Inf扫描通过；
- 禁止字段静态检查通过。

## 严格比较与早停

仅与不可变原始Paper RAT同环境、seed 0、同evaluation semantics和同transition记录比较：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当：

```text
Target reward / Paper reward < 0.60
```

时取消对应cell并记录`EARLY_STOPPED_ALGORITHM`。无精确共同记录时不得取消；不得用Paper终点比较中间Target。

## 必需遥测

记录并报告：

- reward、transitions、Paper reward和ratio；
- `||u_det||₂`、`||u_paper||₂`、scale及`||u_target||₂`；
- deterministic/Paper head-direction cosine；
- global pre-clip norm和clip scale；
- KL、actor LR、entropy；
- value MSE/loss、explained variance；
- PopArt及advantage统计；
- head/solve residual、Cholesky info；
- scheduler、return code、checkpoint和hard-error扫描。

## 验收标准

唯一结论必须为：

- `CANDIDATE_PROMOTE_TO_3SEED`：至少3/4环境到达终点，且到达终点的环境ratio均不低于0.60，无身份偏移或数值失败。
- `CANDIDATE_REJECT`：至少2个环境严格触发早停，或完整终点证据明确否定候选。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：preflight通过且科学运行开始，但基础设施故障阻止充分判定。
- `PRECHECK_BLOCKED`：bundle、等价审计或任一真实网络preflight未通过。

## 历史与禁止事项

必须原样保留Task 14四个missing-`utils`失败以及joint-2B、separate-B、Hybrid V1、low-Fisher、P1、ACTOR_J和所有基础设施/取消记录。

禁止：

- 改变算法、trainer、config、preflight、regression或monitor；
- 测试第二种scale、cap、floor、EMA、guard或候选；
- 覆盖旧root、重跑Paper或复用Task 14失败root；
- 科学cell retry、requeue或resubmit；
- 使用Jupyter；
- 访问`.54`、`ws4090-31`或`10.49.7.54`；
- 指定host、GPU、partition、卡数、并发或queue placement；
- 触碰无关任务。

## 报告、提交与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-HERMETIC-BUNDLE-AND-6M-S0-20260825-15.md`

提交bundle manifest、deployment-only变体、model-free证据和报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、全部commit、bundle manifest SHA、import closure、launcher等价审计、四环境preflight和科学终态、严格阶段比率及failure-ledger增量。
