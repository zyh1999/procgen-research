Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-INTERPRETER-PATH-AUDIT-AND-6M-S0-20260825-17

## 唯一目标

仅修正clean-room auditor对当前Python解释器标准库zip搜索路径的分类；保留NormMatch V2算法、bundle、deployment launchers和全部科学文件。执行一次clean-room audit；通过后执行四环境真实网络preflight；仅当全部通过才启动四环境seed 0预定6M实验。

## 证据判断

Task 16唯一失败路径为：

```text
/usr/lib64/python39.zip
```

该路径是解释器自动生成的标准库zip搜索候选，并非repository源码路径。失败发生在trainer import之前，bundle和designated-empty检查均已通过。因此分类为`infrastructure-failure/clean-room-audit-origin-policy`，没有算法、数值、求解器、内存或reward证据。

正确审计应区分：

- `sys.path`中的合法解释器候选；
- 实际成功提供模块的origin。

标准库zip候选可以存在于搜索路径中，但任何实际repository-local模块仍必须来自冻结bundle。

## 冻结身份

必须保持字节不变：

- Trainer：`0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b`
- Config：`9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Scientific preflight：`b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc`
- Regression：`f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c`
- Monitor：`536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`
- Bundle archive：`3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f`
- Bundle manifest：`99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa`
- Science launcher：`ec60864aaa9940fd61eb1391008b50f2f48402eeae8f78c91c0b0c1fc313a398`
- Preflight launcher：`374d24881d108bbdd08dee0880b7392a7cc3adf6af177f6eb4deaac15535b228`

不得修改、重建或重新打包bundle。

## 唯一允许的代码修正

仅修改Task 16的clean-room origin-policy：

1. 根据当前解释器动态推导标准路径，至少使用：

   - `sys.base_prefix`
   - `sys.base_exec_prefix`
   - `sys.version_info`
   - `sysconfig.get_paths()`

2. 允许当前解释器自动生成的版本化标准库zip候选，但必须满足：

   - 路径由当前prefix和major/minor版本确定性推导；
   - basename严格匹配当前解释器的`pythonXY.zip`形式；
   - 位于解析后的标准库/prefix层级；
   - 不通过硬编码host路径获得。

3. 若候选不存在：

   - 记录为`NONEXISTENT_INTERPRETER_ZIP_CANDIDATE`；
   - 不得有任何module origin指向该路径。

4. 若候选存在：

   - 必须是普通文件而非symlink；
   - 非当前用户可写；
   - 记录owner、mode、size及SHA256；
   - 实际模块origin必须在后续manifest中逐项审计。

5. import后生成完整`import_origin_manifest.json`。所有origin必须分类为：

   - 冻结bundle；
   - 当前解释器标准库；
   - 固定third-party environment；
   - builtin/frozen。

6. 所有repository-local模块必须来自bundle extraction root并匹配manifest SHA。
7. Task 16的designated-empty目录规则及全部负向保护保持不变。

禁止：

- 直接将`/usr/lib64/python39.zip`写入白名单；
- 允许任意`.zip`或整个`/usr`目录；
- 删除pre-import路径检查；
- 跳过post-import origin审计；
- catch、fallback import或动态安装模块。

## 必需回归

远端audit前必须证明：

- 当前解释器推导的合法、但不存在的标准zip候选：PASS；
- 当前解释器推导的安全真实标准zip：PASS；
- 同名zip位于任意临时目录：FAIL；
- Python版本不匹配：FAIL；
- 用户可写或symlink zip：FAIL；
- repository-local模块从bundle外或解释器zip解析：FAIL；
- Task 16的empty-cwd正向及四项负向测试继续通过。

## 有界执行

1. 仅提交上述origin-policy修正及测试。
2. 所有本地测试通过后，只执行一次remote clean-room audit。
3. Audit失败即结束为`PRECHECK_BLOCKED`；不得继续修补或重试。
4. Audit通过后，对四环境各执行一次真实网络preflight。
5. 任一环境preflight失败即结束为`PRECHECK_BLOCKED`，不得启动科学cell。
6. 全部通过后，在全新非覆盖campaign运行：

   - `bigfish-easy-0-10`，seed 0
   - `bossfight-easy-0-10`，seed 0
   - `caveflyer-easy-0-10`，seed 0
   - `coinrun-easy-0-10`，seed 0

7. 每格intended horizon为6M，终点为`5,980,160`，每格最多一次科学提交。
8. Executor负责实时资源检查以及全部host、GPU、partition、并发和queue placement。

## 强制科学preflight

必须继续证明：

- bundle和repository-local import origins正确；
- 四环境production model成功构造且structural manifest一致；
- trainable/optimizer及PopArt隔离正确；
- `u_det`与`u_paper`位于相同更新边界；
- `||u_target||₂=||u_paper||₂`；
- actor/shared proposal、global clip及one-step policy/logits/shared delta与Paper bit-identical；
- 仅value-head方向不同；
- 无额外RNG、sample或data-order变化；
- FP64/Jacobi/Cholesky `info=0`、残差有限；
- memory、OOM、CUDA、NaN/Inf和hard-error检查通过。

## 严格比较与早停

仅与不可变原始Paper RAT同环境、seed 0、同evaluation semantics和同transition记录比较：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当`Target reward / Paper reward < 0.60`时取消对应cell并记录`EARLY_STOPPED_ALGORITHM`。无精确共同记录时不得取消；不得使用Paper终点比较中间Target。

## 验收标准

唯一结论必须为：

- `CANDIDATE_PROMOTE_TO_3SEED`：至少3/4环境达到终点，且达到终点的环境ratio均不低于0.60。
- `CANDIDATE_REJECT`：至少2个环境严格触发早停，或完整终点证据明确否定候选。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：preflight通过并开始科学运行，但基础设施故障阻止充分判定。
- `PRECHECK_BLOCKED`：clean-room audit或任一四环境preflight未通过。

## 禁止事项

- 不得修改算法、科学文件、bundle、deployment launcher或monitor。
- 不得引入第二候选或其他scale规则。
- 不得硬编码已观察到的host路径作为白名单。
- 不得覆盖旧root、重跑Paper或复用失败root。
- 不得retry、requeue或resubmit科学cell。
- 不得使用Jupyter。
- 不得访问`.54`、`ws4090-31`或`10.49.7.54`。
- 不得指定任何具体计算资源。
- 必须保留Task 14–16及全部历史失败和取消记录。

## 报告、提交与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-INTERPRETER-PATH-AUDIT-AND-6M-S0-20260825-17.md`

提交auditor修正、回归、model-free证据和报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、全部commit、auditor SHA、解释器路径推导、import-origin manifest、clean-room及四环境preflight结果、科学终态、严格阶段比率和failure-ledger增量。
