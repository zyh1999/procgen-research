Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-TORCH-GENERATED-ORIGIN-AUDIT-AND-6M-S0-20260825-18

## 唯一目标

仅扩展clean-room auditor，在严格证明来源、内容和生命周期后识别PyTorch生成的`_remote_module_non_scriptable.py`；保持NormMatch V2算法、bundle、deployment launchers及全部科学文件不变。Audit通过后完成四环境preflight，并仅在全部通过时运行四环境seed 0预定6M实验。

## 证据判断

Task 17已通过bundle、empty-cwd和canonical interpreter path检查，并成功开始trainer/third-party imports。唯一阻塞是Torch运行时生成的临时模块不属于现有静态origin类别。

该失败属于`infrastructure-failure/clean-room-audit-origin-policy`，没有算法、数值、求解器、H200、memory或reward证据。不得仅按临时目录或文件名放行。

## 冻结身份

必须保持字节不变：

- Trainer：`0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b`
- Config：`9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Preflight：`b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc`
- Regression：`f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c`
- Monitor：`536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`
- Bundle archive：`3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f`
- Bundle manifest：`99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa`
- Science launcher：`ec60864aaa9940fd61eb1391008b50f2f48402eeae8f78c91c0b0c1fc313a398`
- Preflight launcher：`374d24881d108bbdd08dee0880b7392a7cc3adf6af177f6eb4deaac15535b228`

## 唯一允许的代码修正

仅增加origin类别：

```text
APPROVED_RUNTIME_GENERATED_THIRDPARTY_MODULE
```

`_remote_module_non_scriptable.py`仅在同时满足以下条件时可归类：

1. 在相同冻结Python/PyTorch环境的独立clean process中可复现。
2. 定位负责生成和加载它的installed PyTorch代码，记录：

   - distribution及版本；
   - generator/loader模块与函数；
   - generator源码origin及SHA256；
   - installed-distribution provenance。

3. 临时父目录在当前audit进程启动后创建，属于当前UID、权限受限且不是symlink。
4. 文件在import期间新建，是普通文件且非symlink；记录ctime/mtime、mode、size和SHA256。
5. module name、`__spec__`、loader、package和origin与独立复现一致。
6. 内容与独立复现的规范化内容或已证明的确定性模板匹配，并通过AST/compile检查。
7. 内容不得引用repository checkout、用户源码路径、网络下载或未批准位置。
8. import后重新验证目录、文件、origin和hash未被替换。
9. 所有repository-local模块仍必须来自冻结bundle。

若不能严格建立上述provenance，必须继续拒绝并结束`PRECHECK_BLOCKED`。

禁止硬编码具体临时目录、仅按文件名放行、允许任意`/tmp`、允许任意Torch临时文件或关闭origin审计。

## 必需回归

远端audit前必须证明：

- 同一固定PyTorch环境真实生成的合法模块：PASS；
- 同名预先存在文件：FAIL；
- 内容、AST、hash或loader不匹配：FAIL；
- 父目录或文件为symlink：FAIL；
- 非PyTorch generator/loader：FAIL；
- import后文件被替换：FAIL；
- 文件或模块引用bundle外repository源码：FAIL；
- Task 16/17全部empty-cwd、zip和out-of-bundle负向测试继续通过。

## 有界执行

1. 完成generator provenance调查、单一auditor修正及测试。
2. Provenance无法严格证明时，直接`PRECHECK_BLOCKED`，不得提交远端audit。
3. 全部本地检查通过后，仅执行一次remote clean-room audit。
4. Audit失败即`PRECHECK_BLOCKED`，不得继续修补或重试。
5. Audit通过后，对四环境各执行一次真实网络preflight。
6. 任一preflight失败即`PRECHECK_BLOCKED`，不得启动科学cell。
7. 全部通过后，在全新非覆盖campaign运行四环境seed 0，每格intended horizon 6M、终点`5,980,160`、最多一次科学提交。
8. Executor负责全部实时资源和placement决定。

## 强制科学preflight

必须继续证明：

- repository-local imports全部来自bundle；
- 四环境模型结构一致；
- trainable/optimizer及PopArt隔离正确；
- `u_det`和`u_paper`位于相同边界；
- `||u_target||₂=||u_paper||₂`；
- actor/shared、global clip、one-step policy/logits/shared delta与Paper bit-identical；
- 仅value-head方向不同；
- 无额外RNG、sample或data-order变化；
- FP64/Jacobi/Cholesky `info=0`、残差有限；
- memory和hard-error检查通过。

## 严格比较与早停

只比较原始Paper RAT同环境、seed 0、同evaluation semantics和同transition：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当`Target/Paper < 0.60`时取消对应cell。不得用Paper终点比较中间Target；无精确共同点不得操作。

## 验收标准

唯一结论：

- `CANDIDATE_PROMOTE_TO_3SEED`：至少3/4环境到达终点，且到达终点的环境ratio均不低于0.60。
- `CANDIDATE_REJECT`：至少2个环境严格早停，或完整终点证据明确否定候选。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：科学运行开始后基础设施阻止判定。
- `PRECHECK_BLOCKED`：generator provenance、clean-room audit或任一preflight失败。

## 禁止事项

不得修改算法、bundle、科学文件、launchers或monitor；不得建立通用临时目录白名单；不得引入第二候选或sweep；不得覆盖旧root、重跑Paper、retry科学cell、使用Jupyter、访问`.54`/`ws4090-31`/`10.49.7.54`、指定计算资源或触碰无关任务。全部历史失败必须保留。

## 报告、提交与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-TORCH-GENERATED-ORIGIN-AUDIT-AND-6M-S0-20260825-18.md`

提交auditor修正、generator provenance、回归、model-free证据和报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、commit身份、PyTorch generator/loader provenance、临时模块manifest、clean-room与preflight结果、科学终态、严格阶段比率及failure-ledger增量。
