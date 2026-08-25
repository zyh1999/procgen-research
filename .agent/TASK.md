Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-SYSPATH-AUDIT-RECOVERY-AND-6M-S0-20260825-16

## 唯一目标

修正hermetic clean-room auditor将其自己创建的空工作目录误判为源码污染的问题；保持NormMatch V2算法、bundle和科学文件不变，完成一次clean-room审计、四环境真实网络preflight，并仅在全部通过后运行四环境seed 0预定6M实验。

## 证据判断

Task 15已证明：

- bundle可确定性重建；
- archive、manifest及全部文件哈希正确；
- repository-local import closure完整包含trainer、`utils`和`vec_env`；
- deployment launcher与原科学命令等价。

唯一失败是auditor拒绝了它明确创建的空工作目录：

```text
/mnt/.../tmp/procgen-nm2-empty-19241161.WFurfV
```

该目录出现在`sys.path`并不等同于发生非hermetic import。正确安全条件是：目录确实为空、不可提供模块，并且所有实际导入模块的origin均来自批准的bundle、标准库或固定环境依赖。故该失败属于`infrastructure-failure/clean-room-harness-design`，不是bundle缺失、算法或硬件证据。

## 冻结身份

必须保持字节不变：

- Trainer：`0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b`
- Config：`9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Scientific preflight：`b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc`
- Regression：`f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c`
- Monitor：`536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`
- Bundle archive：`3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f`
- Bundle manifest：`99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa`
- Deployment science launcher：`ec60864aaa9940fd61eb1391008b50f2f48402eeae8f78c91c0b0c1fc313a398`
- Deployment preflight launcher：`374d24881d108bbdd08dee0880b7392a7cc3adf6af177f6eb4deaac15535b228`

不得重建或改变bundle内容。

## 唯一允许的代码修正

仅修改clean-room audit harness的`sys.path`判定：

1. 允许一个由harness本次创建并记录的designated empty working directory出现在`sys.path`。
2. 在启动解释器前记录该目录的：

   - 规范化绝对路径；
   - device/inode；
   - ownership及permissions；
   - 递归目录清单；
   - 创建时间。

3. 目录在解释器启动前必须为空，且不得包含`.py`、`.pyc`、共享库、package目录或symlink。
4. import完成后再次扫描；目录仍必须为空且device/inode不变。
5. 对`sys.modules`中所有具有origin的模块生成import-origin manifest，并逐项分类为：

   - 解压后的已验证bundle；
   - 当前Python标准库；
   - 当前固定环境的third-party site-packages；
   - builtin/frozen模块。

6. 任一实际模块origin位于designated empty directory、源仓库checkout、用户临时源码路径或其他未批准位置，都必须失败。
7. 所有repository-local模块必须解析到bundle extraction root，并与bundle manifest SHA匹配。
8. 禁止简单删除全部`sys.path`安全检查、允许任意空目录模式、使用宽泛路径白名单或吞掉异常。

允许使用`-P`/`PYTHONSAFEPATH`作为额外防护，但不得以此替代import-origin审计。

## 有界执行

1. 为上述规则添加本地正反测试：

   - designated目录为空且无模块来源：PASS；
   - 目录含可导入模块：FAIL；
   - symlink或扫描后新增文件：FAIL；
   - repo-local模块从bundle外解析：FAIL。

2. 本地测试通过后，仅执行一次完整remote clean-room audit。
3. 若该audit失败，立即结束为`PRECHECK_BLOCKED`；不得继续修补或重试。
4. Audit通过后，对四个环境各执行一次真实网络preflight。
5. 任一环境preflight失败，结束为`PRECHECK_BLOCKED`；不得启动科学cell。
6. 四环境全部通过后，启动一个新、非覆盖campaign中的：

   - `bigfish-easy-0-10`，seed 0
   - `bossfight-easy-0-10`，seed 0
   - `caveflyer-easy-0-10`，seed 0
   - `coinrun-easy-0-10`，seed 0

7. 每格intended horizon为6M，终点`5,980,160`，每格最多一次科学提交。
8. Executor负责实时资源刷新、nonduplicate检查以及全部host/GPU/partition/concurrency/queue placement。

## 强制preflight

必须继续证明：

- bundle、manifest及每个文件SHA正确；
- 所有本地import origin均位于bundle；
- 四环境production model成功构造；
- structural manifest一致；
- trainable/optimizer及PopArt隔离正确；
- `u_det`和`u_paper`来自相同更新边界；
- `||u_target||₂=||u_paper||₂`；
- actor/shared proposal、global clip、one-step policy/logits/shared delta与Paper bit-identical；
- 仅value-head方向不同；
- 无额外RNG、sample或data-order变化；
- FP64/Jacobi/Cholesky `info=0`且残差有限；
- 内存、OOM、CUDA、NaN/Inf及hard-error检查通过。

## 严格比较与早停

仅与不可变原始Paper RAT中同环境、seed 0、同evaluation semantics和同transition记录比较：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当：

```text
Target reward / Paper reward < 0.60
```

时取消对应cell并记录`EARLY_STOPPED_ALGORITHM`。无精确共同点时不得操作；不得用Paper终点比较中间Target。

## 必需遥测

记录reward、ratio、KL、LR、entropy、value loss/MSE、explained variance、PopArt、advantage统计，以及：

- `||u_det||₂`
- `||u_paper||₂`
- norm-match scale
- `||u_target||₂`
- head-direction cosine
- global pre-clip norm及clip scale
- head/solve residual和Cholesky info

## 验收标准

唯一结论必须为：

- `CANDIDATE_PROMOTE_TO_3SEED`：至少3/4环境达到终点，且达到终点的环境ratio均不低于0.60。
- `CANDIDATE_REJECT`：至少2个环境严格触发早停，或完整终点证据明确否定候选。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：preflight通过且科学运行开始，但基础设施故障阻止充分判定。
- `PRECHECK_BLOCKED`：clean-room audit或任一四环境preflight未通过。

## 禁止事项

- 不得修改算法、科学文件、bundle、deployment launcher或monitor。
- 不得引入第二候选、scale变体、cap、floor、EMA或guard。
- 不得覆盖既有root、重跑Paper或复用Task 14/15失败root。
- 不得retry、requeue或resubmit科学cell。
- 不得使用Jupyter。
- 不得访问`.54`、`ws4090-31`或`10.49.7.54`。
- 不得指定计算资源或触碰无关任务。
- 所有既有算法、preflight、deployment和取消记录必须原样保留。

## 报告、提交与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-SYSPATH-AUDIT-RECOVERY-AND-6M-S0-20260825-16.md`

提交audit-harness修正、测试、model-free证据及报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、全部commit、audit harness SHA、import-origin manifest、clean-room与四环境preflight结果、科学终态、严格阶段比率及failure-ledger增量。
