Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-POLICY-PATH-IDENTITY-RECOVERY-AND-6M-S0-20260825-20

## 唯一目标

仅修正clean-room prestart对合法storage alias的判定：不再要求raw path字符串等于resolved path字符串，而以canonical target、same-file身份及冻结SHA证明policy文件不变。保持算法、bundle、Torch provenance和全部科学文件不变。

## 证据判断

Task 19的raw路径与resolved路径虽然拼写不同，但指向相同文件：

- device：`3592384858`
- inode：`144122242006038476`
- size：`13605`
- owner：UID `778916`
- mode：`0644`
- SHA256：`889b914a792132358f90572d5c4f16b561c60bf0615eb8492c9ecf781f601fc1`

仅字符串相等断言失败。这属于`infrastructure-failure/clean-room-prestart-path-canonicalization`，不是科学或安全身份失败。

## 冻结身份

必须保持字节不变：

- Trainer：`0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b`
- Config：`9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Preflight：`b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc`
- Regression：`f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c`
- Monitor：`536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`
- Bundle：`3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f`
- Manifest：`99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa`
- Science/preflight launchers：`ec60864...` / `374d2488...`
- Origin policy：`889b914a792132358f90572d5c4f16b561c60bf0615eb8492c9ecf781f601fc1`
- Torch module：`8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`

## 唯一允许的修改

仅修改policy-path身份验证：

1. 删除raw字符串必须等于resolved字符串的断言。
2. 对raw路径执行`resolve(strict=True)`。
3. 必须同时证明：

   - `os.path.samefile(raw, resolved)`为真；
   - raw与resolved的`stat` device/inode相同；
   - 最终目标为普通文件且最终文件本身不是symlink；
   - owner、mode、size符合冻结记录；
   - SHA256严格等于`889b914a...`。

4. 使用resolved target或经`fstat`验证的文件描述符加载policy。
5. 执行后再次验证device/inode/size/SHA没有变化。
6. Ledger必须记录raw及resolved拼写、samefile、lstat/stat、device/inode、owner/mode/size和pre/post SHA。
7. 可接受父路径的mount/automount alias，但不得允许目标文件symlink或文件替换。

禁止硬编码`/scratch`、`/net/scratch`，按SHA忽略身份，搜索文件名或放宽其他origin检查。

## 必需回归

必须通过：

- 不同路径拼写、相同device/inode及正确SHA：PASS；
- 内容相同但不同文件：FAIL；
- 最终文件symlink：FAIL；
- resolve或执行后目标替换：FAIL；
- device/inode、owner、mode、size或SHA不符：FAIL；
- 缺失路径：FAIL；
- Task 16–19全部empty-cwd、zip、Torch provenance及bare-exec测试继续通过。

## 有界执行

1. 仅提交上述身份检查修正及测试。
2. 本地全部通过后，只执行一次remote clean-room audit。
3. Audit失败即`PRECHECK_BLOCKED`，不得修补或重试。
4. Audit通过后，对四环境各执行一次真实网络preflight。
5. 任一preflight失败即`PRECHECK_BLOCKED`，不得启动科学cell。
6. 全部通过后，以全新非覆盖root运行四环境seed 0；每格intended horizon 6M、终点`5,980,160`、最多一次科学提交。
7. Executor独立负责全部实时资源及placement。

## 严格早停

仅比较原始Paper RAT同环境、seed 0、同evaluation semantics及同transition：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当`Target/Paper < 0.60`时取消对应cell。不得以Paper终点比较中间Target。

## 验收标准

唯一结论：

- `CANDIDATE_PROMOTE_TO_3SEED`：至少3/4环境到达终点且终点ratio均不低于0.60。
- `CANDIDATE_REJECT`：至少2个环境严格早停，或终点证据明确否定候选。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：科学运行开始后基础设施阻止判定。
- `PRECHECK_BLOCKED`：path identity、clean-room audit或任一preflight失败。

## 禁止事项

不得修改算法、bundle、科学文件、launchers、Torch provenance或monitor；不得引入第二候选、覆盖旧root、重跑Paper、retry科学cell、使用Jupyter、访问`.54`/`ws4090-31`/`10.49.7.54`、指定计算资源或触碰无关任务。全部历史失败必须保留。

## 报告与推送

更新并提交：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-POLICY-PATH-IDENTITY-RECOVERY-AND-6M-S0-20260825-20.md`

推送并验证`origin/agent-work`，保持worktree干净。回调必须包含唯一结论、commit身份、path-identity ledger、clean-room/preflight/科学终态、严格阶段比率及failure-ledger增量。
