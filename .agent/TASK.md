Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-RUNTIME-GENERATED-CLOSURE-AUDIT-AND-6M-S0-20260825-22

## 唯一目标

严格证明`_classes.py`及同一冻结import/model-construction路径产生的全部runtime-generated模块的generator、loader、内容和生命周期，并据此建立逐文件批准的generated-module closure。不得按文件名或目录宽泛放行，不得改变NormMatch V2或任何冻结科学身份。

## 证据判断

Task 21已经完成Python 3.9、storage alias、same-file、fd及pre/post SHA修复。唯一失败是prestart为空的designated目录在trainer imports期间生成了：

```text
_classes.py
```

其来源尚未证明，因此auditor拒绝正确。该结果属于`infrastructure-failure/clean-room-loaded-module-origin-policy`，没有算法、数值、求解器、H200或reward证据。

## 冻结身份

Trainer、config、preflight、regression、monitor、bundle archive/manifest、science/preflight launchers、Task 18 origin policy、`_remote_module_non_scriptable.py` provenance及Task 21 path-identity逻辑必须全部字节不变。

## 唯一允许的修改

仅扩展clean-room auditor，新增逐文件类别：

```text
APPROVED_RUNTIME_GENERATED_THIRDPARTY_MODULE
```

### Provenance采集

在至少两个独立的冻结Python 3.9/PyTorch clean process中：

1. prestart证明designated目录为空。
2. 记录文件create/write/rename/delete事件和Python调用栈。
3. 记录module name、`__spec__`、loader、package及origin。
4. 定位generator/loader所属installed distribution、版本、RECORD、源码路径及SHA256。
5. 完成trainer import和production model construction后，列举全部新生成文件。
6. 两次运行的规范化generated-artifact closure必须一致。

### `_classes.py`批准条件

必须同时证明：

- 当前进程prestart后创建；
- 父目录为本次UID-owned、mode-restricted、非symlink designated目录；
- 文件为普通文件、非symlink；
- generator和loader来自固定installed distribution；
- generator源码及RECORD hash固定；
- module/spec/loader/package与独立复现一致；
- 内容与确定性template或规范化复现一致；
- 记录精确size及SHA256；
- AST和compile通过；
- 不含repo checkout、用户源码、网络下载或未批准路径引用；
- import后inode、mode、size和SHA未被替换。

Closure中其他generated modules必须分别通过同等级检查；不得因`_classes.py`通过而自动放行任何同目录文件。

Designated目录规则改为：prestart必须为空；post-import只能包含closure中逐文件批准的artifact。

## 必需负向测试

必须拒绝：

- preexisting同名文件；
- 不同generator、distribution、loader或module identity；
- content/template/AST/hash不匹配；
- symlink父目录或文件；
- import后替换；
- 额外未登记文件；
- repository-local模块从bundle外解析；
- 文件引用repo路径、用户源码或网络。

Task 16–21全部安全回归必须继续通过。无法稳定复现完整closure时直接`PRECHECK_BLOCKED`。

## 有界执行

1. 仅提交closure provenance、auditor修正和测试。
2. 本地全部通过后，仅执行一次remote clean-room audit。
3. Audit失败即`PRECHECK_BLOCKED`，不得修补或重试。
4. Audit通过后，对四环境各执行一次真实网络preflight。
5. 任一preflight失败即`PRECHECK_BLOCKED`，不得启动科学cell。
6. 全部通过后，以全新非覆盖root运行四环境seed 0；每格intended horizon 6M、终点`5,980,160`、最多一次提交。
7. Executor独立负责全部实时资源及placement。

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
- `PRECHECK_BLOCKED`：closure provenance、audit或任一preflight失败。

## 禁止事项

不得修改算法、bundle、科学文件、launchers、monitor或既有provenance；不得建立文件名/目录白名单；不得引入第二候选、覆盖旧root、重跑Paper、retry科学cell、使用Jupyter、访问`.54`/`ws4090-31`/`10.49.7.54`、指定资源或触碰无关任务。历史失败必须全部保留。

## 报告与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-RUNTIME-GENERATED-CLOSURE-AUDIT-AND-6M-S0-20260825-22.md`

提交closure provenance、auditor修正、Python 3.9回归、model-free证据和报告，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、commit身份、完整generated-artifact closure、generator/loader/RECORD证据、audit/preflight/科学终态、严格阶段比率及failure-ledger增量。
