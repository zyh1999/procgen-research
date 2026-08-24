Status: READY

# Task-ID: PROCGEN-HYBRID-HEAD-ROOT-OVERRIDE-6M-MISSING3-20260824-13

## 唯一目标

授权一个仅改变artifact destination的launcher变体，使已通过全部科学预检的Hybrid-Head候选能够在三个全新、非覆盖root中运行BossFight、CaveFlyer和CoinRun seed 0预定6M实验；不得改变训练命令或科学语义。

## 控制矛盾及决策

Task 12的四环境科学预检已经全部通过。当前阻塞仅来自：

- 原launcher `ae7104e7...` 将campaign/root硬编码到Task 11路径；
- 该launcher在root已存在时正确退出；
- Task 12同时禁止修改launcher和覆盖既有root。

因此现有约束无法同时满足。明确授权创建一个root-override launcher变体。其新SHA必须记录，原launcher及Task 11 roots保持不可变。该变化仅属于artifact routing/provenance，不是新算法或第二候选。

## 冻结科学身份

必须保持字节不变：

- Trainer：`7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54`
- Config：`9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Stage monitor：`536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`
- Corrected preflight harness：`704278e8b5802498b8e065b9f12945e2cb72a665cdd28845b2401091b2e993ea`
- Structural manifest：`3f91f5c313480c089d300a7dd5aff4a664f28ff9d9f4718bbab430200298d623`

原scientific launcher `ae7104e7...`必须保留，不能修改或重写。

## 唯一允许的代码变更

复制原scientific launcher，创建一个版本化的root-override变体。相对原launcher，只允许：

1. 将硬编码`CAMPAIGN`替换为必填的显式artifact-root参数或环境变量，例如：

```text
PROCGEN_CAMPAIGN_ROOT
```

2. 若该值为空、非绝对路径、解析到Task 11 campaign/root或目标root已经存在，必须在训练前失败。
3. 允许增加仅用于记录base-launcher SHA、override-root和Task-ID的provenance字段。
4. 必须继续生成：

```text
$CAMPAIGN_ROOT/runs/$METHOD/$ENV_NAME/seed0/6m
```

5. 除campaign/root解析与provenance外，launcher的trainer、config、参数、环境ID、seed、budget、Python入口、preflight、monitor和训练命令不得变化。

不得修改trainer来接受新的科学参数；root routing应完全位于launcher层。

## Launcher等价审计

科学提交前必须：

- 提供原launcher与变体的逐行diff。
- 证明所有非root/provenance行字节相同。
- 分别dry-run原launcher和变体，并将绝对artifact路径归一化为占位符。
- 证明归一化后的trainer命令、参数顺序、环境、seed、预算、配置和preflight调用字节一致。
- 证明变体绑定corrected harness `704278e8...`。
- 记录新launcher完整SHA256。
- 证明三个目标root均为全新且不存在。
- 证明Task 11四个root及其文件、mtime、状态未被修改。

若等价审计失败，结束为`PRECHECK_BLOCKED`，不得启动科学运行。

## 已接受且无需重复的预检证据

Task 12以下证据可严格复用，不得重新解释为缺失：

- 四环境structural manifest字节一致。
- 四环境connectivity probes均通过。
- 三方resolved config在各环境内部字节一致。
- trainable/optimizer集合一致，PopArt非训练状态正确隔离。
- Paper actor、sampled shared critic、one-step policy/logits/shared delta bit-identical。
- 仅value-head delta不同。
- H200内存检查、Cholesky `info=0`、FP64 residual和hard-error检查通过。

除launcher等价审计外，不需要再次运行GPU科学预检。

## 科学执行范围

launcher等价审计通过后，仅启动：

- `bossfight-easy-0-10`，seed 0
- `caveflyer-easy-0-10`，seed 0
- `coinrun-easy-0-10`，seed 0

每格：

- intended horizon：6M
- terminal transition：`5,980,160`
- 一个全新非覆盖root
- 最多一次科学提交

不得重跑BigFish。`19228676`在4,014,080的`EARLY_STOPPED_ALGORITHM`结果永久保留。

Executor必须在提交前刷新授权资源的scheduler、GPU、进程、ownership、capacity、已有root和重复任务状态。具体host、GPU、partition、卡数、并发和queue placement完全由Executor决定。

## 严格比较与早停

三个新cell只与不可变原始Paper RAT的同环境、seed 0、同evaluation semantics、同transition记录比较：

- 第一个共同点`>=2,000,000`
- 第一个共同点`>=4,000,000`
- 终点`5,980,160`

仅当：

```text
Target reward / Paper reward < 0.60
```

时取消对应cell并记录`EARLY_STOPPED_ALGORITHM`。没有精确共同记录时不得采取动作；不得用Paper终点比较中间Target。

## 合并验收标准

必须合并既有BigFish算法早停与三个新结果，输出以下唯一结论之一：

- `CANDIDATE_PROMOTE_TO_3SEED`：BossFight、CaveFlyer和CoinRun全部达到终点，且各自终点ratio均不低于0.60；BigFish失败继续保留。
- `CANDIDATE_REJECT`：三个新环境中至少一个严格触发`<0.60`，从而累计至少两个环境算法失败；或完整终点证据明确否定候选。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：科学运行开始后，基础设施故障仍阻止充分判定。
- `PRECHECK_BLOCKED`：root-only launcher等价审计未通过。
- `LAUNCH_BLOCKED_NONOVERWRITE`：无法建立三个经过验证的全新root，但没有科学身份问题。

## 必需证据

- 原launcher、新launcher及逐行diff和SHA256。
- 路径归一化后的命令等价证明。
- trainer/config/monitor/harness哈希不变证明。
- 新旧root存在性、inode/mtime或等价完整性证据。
- Task 11和Task 12全部failure ledger原样保留。
- 三个新cell的root、command、scheduler、return code、transitions和artifact清单。
- 2M、4M、终点的Target/Paper reward、ratio、KL、LR、entropy、head residual、solve residual和Cholesky info。
- checkpoint及OOM、CUDA、NCCL、disk/quota、stall、Traceback、NaN/Inf扫描。
- 明确区分BigFish算法失败、旧per-job preflight设计失败及任何新基础设施失败。

## 禁止事项

- 不得修改算法、trainer、config、monitor或corrected preflight harness。
- 不得修改或覆盖原launcher和Task 11 roots。
- 不得重跑BigFish、Paper RAT或启动第二候选。
- 不得retry、requeue或resubmit任何新科学cell。
- 不得使用Jupyter。
- 不得访问`.54`、`ws4090-31`或`10.49.7.54`。
- Planner不指定任何计算资源；不得触碰无关任务。

## 报告、提交与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-HYBRID-HEAD-ROOT-OVERRIDE-6M-MISSING3-20260824-13.md`

提交launcher变体、model-free证据和报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、assignment/launcher-freeze/evidence/Delivery commits、新launcher SHA、等价审计、三个新cell终态、严格阶段比率及failure-ledger增量。
