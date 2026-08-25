Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-BARE-EXEC-NAMESPACE-RECOVERY-AND-6M-S0-20260825-19

## 唯一目标

仅修正Task 18 clean-room prestart在bare `exec` namespace中错误依赖`__file__`的问题；显式传入并验证冻结origin-policy路径。不得改变NormMatch V2、bundle、已证明的Torch临时模块策略或任何科学文件。Audit通过后继续四环境preflight及seed-0预定6M实验。

## 证据判断

Task 18已完整证明Torch生成模块的generator、loader、template、内容SHA、生命周期和负向安全边界。唯一远端失败来自：

```python
os.environ.get(KEY, Path(__file__) ...)
```

函数调用前会求值default表达式，因此即使环境变量存在，bare namespace仍因没有`__file__`而失败。这是`infrastructure-failure/clean-room-prestart-namespace`，不影响已通过的provenance结论。

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
- Task 18 origin policy：`889b914a792132358f90572d5c4f16b561c60bf0615eb8492c9ecf781f601fc1`
- Provenance probe：`917faf124f35ca7a1c4ceef4a8dc43183500cbc2131c71f95f8d2186c51f6c23`
- Generated module：`8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`

## 唯一允许的修正

1. 删除bare-exec路径解析中的所有`__file__`依赖，包括eager default、fallback和日志分支。
2. Audit launcher必须通过必填环境变量显式传入origin-policy绝对路径。
3. 使用lazy分支：

```python
raw_policy_path = os.environ.get(POLICY_PATH_ENV)
if not raw_policy_path:
    raise RuntimeError("missing explicit origin-policy path")
policy_path = Path(raw_policy_path).resolve(strict=True)
```

4. 验证policy文件：

   - 普通文件、非symlink；
   - owner/mode符合预期；
   - SHA256严格等于`889b914a...`；
   - 不位于designated empty cwd；
   - 路径、inode和SHA写入prestart ledger。

5. 如Task 18 audit launcher尚未设置变量，仅允许创建audit-launcher-only变体；science/preflight launchers不得修改。
6. 禁止cwd搜索、glob、文件名猜测、`__file__` fallback或吞掉缺失变量错误。

## 必需回归

必须通过：

- bare namespace无`__file__`、显式变量正确：PASS；
- 变量缺失、路径不存在、symlink、SHA错误：FAIL；
- 设置变量但仍含eager `__file__`求值：FAIL；
- 普通module与bare-exec执行结果一致；
- Task 16–18全部empty-cwd、zip、Torch generator正负测试继续通过；
- 全部冻结身份检查通过。

## 有界执行

1. 仅提交namespace修正、必要的audit-launcher-only变体及测试。
2. 本地全部通过后，只执行一次remote clean-room audit。
3. Audit失败即`PRECHECK_BLOCKED`，不得修补或重试。
4. Audit通过后，对四环境各执行一次真实网络preflight。
5. 任一preflight失败即`PRECHECK_BLOCKED`，不得启动科学cell。
6. 全部通过后，在全新非覆盖campaign运行BigFish、BossFight、CaveFlyer、CoinRun seed 0；每格intended horizon 6M、终点`5,980,160`、最多一次提交。
7. Executor独立负责所有实时资源和placement决定。

## 严格比较与早停

仅比较原始Paper RAT同环境、seed 0、同evaluation semantics和同transition：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当`Target/Paper < 0.60`时取消对应cell。不得使用Paper终点比较中间Target。

## 验收标准

唯一结论：

- `CANDIDATE_PROMOTE_TO_3SEED`：至少3/4环境到达终点，且到达终点的环境ratio均不低于0.60。
- `CANDIDATE_REJECT`：至少2个环境严格早停，或终点证据明确否定候选。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：科学运行开始后基础设施阻止判定。
- `PRECHECK_BLOCKED`：namespace、clean-room或任一preflight失败。

## 禁止事项

不得修改算法、bundle、科学文件、science/preflight launchers、Torch provenance策略或monitor；不得引入第二候选、覆盖旧root、重跑Paper、retry科学cell、使用Jupyter、访问`.54`/`ws4090-31`/`10.49.7.54`、指定计算资源或触碰无关任务。必须保留全部历史失败。

## 报告与推送

更新并提交：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-BARE-EXEC-NAMESPACE-RECOVERY-AND-6M-S0-20260825-19.md`

推送并验证`origin/agent-work`，保持worktree干净。回调必须包含唯一结论、commit身份、audit SHA、policy-path ledger、clean-room/preflight/科学终态、严格阶段比率及failure-ledger增量。
