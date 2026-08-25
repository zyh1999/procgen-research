Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-PY39-LSTAT-COMPATIBILITY-AND-6M-S0-20260825-21

## 唯一目标

仅修正Task 20 policy-path validator对Python 3.9不支持的`Path.stat(follow_symlinks=False)`调用；使用Python 3.9兼容的`os.lstat`、`os.stat`和`os.fstat`实现同一文件身份与防替换语义。保持算法、bundle、origin policy及全部科学文件不变。

## 证据判断

Task 20唯一失败为：

```text
TypeError: stat() got an unexpected keyword argument 'follow_symlinks'
```

远端Python为3.9.25，其`pathlib.Path.stat()`不接受该参数；本地Python 3.13回归接受，造成版本兼容遗漏。失败发生在第一次raw stat，尚未执行policy或科学preflight。

这是`infrastructure-failure/clean-room-prestart-python-api-compatibility`，不否定Task 20的file-identity设计，也没有算法、数值、求解器或reward证据。

## 冻结身份

以下内容必须保持字节不变：

- Trainer：`0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b`
- Config：`9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Preflight：`b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc`
- Regression：`f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c`
- Monitor：`536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`
- Bundle/manifest：`3da17520...` / `99191542...`
- Science/preflight launchers：`ec60864...` / `374d2488...`
- Task 18 origin policy：`889b914a...`
- Torch module：`8205b169...`

## 唯一允许的代码修正

仅替换不兼容stat调用：

1. 使用`os.lstat(raw_path)`检查raw最终路径组件，不跟随最终symlink。
2. 使用`resolve(strict=True)`获得canonical target。
3. 使用`os.lstat(resolved_path)`证明resolved最终组件不是symlink。
4. 使用`os.stat(raw_path)`和`os.stat(resolved_path)`取得实际target身份并验证：

   - `os.path.samefile(raw, resolved)`；
   - device/inode相同；
   - regular file；
   - UID/GID、mode、size符合记录；
   - SHA256为`889b914a...`。

5. 使用：

```python
fd = os.open(resolved_path, os.O_RDONLY | os.O_CLOEXEC | optional_O_NOFOLLOW)
fd_stat = os.fstat(fd)
```

并证明fd identity与validated target一致。
6. 从已验证fd或resolved target执行policy；执行后重新`fstat/stat/hash`，检测替换。
7. 删除所有`Path.stat(follow_symlinks=...)`调用。
8. 不得改变Task 20的same-file、安全或ledger语义。

## 必需兼容性回归

远端audit前必须：

- 使用Python 3.9实际解释器运行validator正负测试，而非只用较新Python。
- 静态扫描确认不存在Python 3.9不支持的`Path.stat(follow_symlinks=...)`。
- 验证storage alias同文件：PASS。
- 最终symlink、不同inode同内容、替换、错误owner/mode/size/SHA：FAIL。
- `O_NOFOLLOW`不可用时必须显式记录，并仍通过lstat/fstat前后身份检查；不得静默降低保护。
- Task 16–20全部安全回归继续通过。

## 有界执行

1. 仅提交Python 3.9兼容stat修正及测试。
2. 全部兼容性回归通过后，只执行一次remote clean-room audit。
3. Audit失败即`PRECHECK_BLOCKED`，不得修补或重试。
4. Audit通过后，对四环境各执行一次真实网络preflight。
5. 任一preflight失败即`PRECHECK_BLOCKED`，不得启动科学cell。
6. 全部通过后，在全新非覆盖campaign运行四环境seed 0；每格intended horizon 6M、终点`5,980,160`、最多一次科学提交。
7. Executor独立负责实时资源及全部placement决定。

## 严格早停

仅比较原始Paper RAT同环境、seed 0、同evaluation semantics和同transition：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当`Target/Paper < 0.60`时取消对应cell。不得用Paper终点比较中间Target。

## 验收标准

唯一结论：

- `CANDIDATE_PROMOTE_TO_3SEED`：至少3/4环境达到终点且终点ratio均不低于0.60。
- `CANDIDATE_REJECT`：至少2个环境严格早停，或终点证据明确否定候选。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：科学运行开始后基础设施阻止判定。
- `PRECHECK_BLOCKED`：兼容性、clean-room audit或任一preflight失败。

## 禁止事项

不得修改算法、bundle、科学文件、launchers、Torch provenance或monitor；不得放宽file-identity检查；不得引入第二候选、覆盖旧root、重跑Paper、retry科学cell、使用Jupyter、访问`.54`/`ws4090-31`/`10.49.7.54`、指定计算资源或触碰无关任务。全部历史失败必须保留。

## 报告与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-PY39-LSTAT-COMPATIBILITY-AND-6M-S0-20260825-21.md`

提交兼容性修正、Python 3.9回归、model-free证据和报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、commit身份、Python版本、stat/fd identity ledger、audit/preflight/科学终态、严格阶段比率及failure-ledger增量。
