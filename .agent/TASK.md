Status: READY

# Task-ID: PROCGEN-NORMMATCH-V2-EXACT-PROBE-ALIAS-RECOVERY-AND-6M-S0-20260825-28R

## 唯一目标

仅修正closure origin scan对其自身冻结文件`runtime_closure_probe_task23.py`的storage-alias误判：证明raw与resolved路径指向同一不可变文件后，只批准该精确probe identity。保持其他origin rejection、NormMatch V2算法及全部科学身份不变。

本任务取代尚未执行的任何更宽泛audit-control manifest方案；不得建立通用审计框架。

## 证据判断

Task 27科学preflight核心已经通过：

- 实际det/Paper对象identity正确；
- 唯一调用及输入不变；
- \(||u_{\rm det}||=.6050832272\)
- \(||u_{\rm Paper}||=.9192549586\)
- scale=`1.519220710`
- \(||u_{\rm target}||=.9192548990\)
- cosine=`.8612535000`
- residual=`8.627e-16`
- Cholesky `info=0`

这说明NormMatch同时面对中等norm差异和非零方向差异，科学实验仍能区分“全局scale校准是否足够”。Deterministic Gaussian GGN可以理论正确；本实验检验其与Paper finite-sample damped update的scale alignment。

唯一closure失败是冻结probe通过`/net/scratch`别名出现，未被现有origin policy识别。它不是trainer依赖或科学异常。

## 冻结身份

以下内容必须字节不变：

- Trainer：`0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b`
- Config：`9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Task 27 preflight：`e43fe7e730e840de07cc467bbc56900591c581fd24d4520abe129b0bad3d2cfb`
- Task 27 identity ledger：`3d020a88c298f4f56f2a99cd71f7c68620dbf549fc924ec559b0069f00651871`
- Bundle/manifest、science/preflight launchers及monitor
- Task 23 hook、Task 25 classifier、Task 26 AST helper和Task 27 semantic binding

不得修改被拒绝的probe内容。

## 唯一允许的代码修改

新增唯一窄类别：

```text
APPROVED_EXACT_FROZEN_CLOSURE_PROBE_ALIAS
```

仅`runtime_closure_probe_task23.py`可以进入该类别，并须同时满足：

1. 从冻结Git提交解析该probe的exact repository-relative path、Git blob和SHA256。
2. Origin basename、module key、spec、loader及package与冻结probe加载方式完全一致。
3. 对reported raw origin执行`resolve(strict=True)`。
4. Raw与resolved路径必须：

   - `os.path.samefile(raw, resolved)`为真；
   - `stat` device/inode相同；
   - 指向普通文件；
   - 最终文件本身均非symlink；
   - UID/GID、mode、size符合冻结记录；
   - SHA256严格等于冻结probe SHA。

5. 使用`O_RDONLY|O_CLOEXEC|O_NOFOLLOW`打开resolved文件并以`fstat`绑定其identity。
6. Origin scan前后重新验证fd/path的device、inode、size和SHA不变。
7. 必须证明该module由closure-audit entrypoint加载，而不是由trainer、bundle或scientific import graph加载。
8. 必须证明formal scientific trainer进程不加载该probe。
9. Ledger记录raw/resolved spelling、samefile、lstat/stat/fstat、UID/GID、mode、size、Git blob及pre/post SHA。

不得通过目录、basename、Task编号、`tools/`前缀、Git checkout或任意storage alias进行批准。

## 必需负向测试

必须拒绝：

- 同名但不同Git blob或SHA；
- 相同字节但不同inode；
- symlink文件或最终组件；
- module key、loader、spec、package或importer不匹配；
- pre/post替换；
- trainer或bundle模块伪装成probe；
- 同目录中的任何其他文件；
- probe泄漏到formal scientific process；
- 任意未批准`/scratch`或`/net/scratch`origin。

Task 16–27全部既有回归必须继续通过。

## 有界执行

1. 仅提交exact-probe alias validator及负向测试。
2. 在Python 3.9真实环境通过后，只执行一次closure job。
3. 两个独立clean process必须完成production model construction并产生一致closure。
4. Closure失败即`PRECHECK_BLOCKED`，不得修补或重试。
5. Closure通过后，只执行既定必要formal clean-room audit；不得新增audit层。
6. Formal audit失败即`PRECHECK_BLOCKED`。
7. Audit通过后，对四环境各执行一次真实网络preflight；任一失败即`PRECHECK_BLOCKED`。
8. 全部通过后，以全新非覆盖root运行：

   - BigFish seed 0
   - BossFight seed 0
   - CaveFlyer seed 0
   - CoinRun seed 0

9. 每格intended horizon为6M，终点`5,980,160`，每格最多一次科学提交。
10. Executor独立负责全部实时资源、scheduler和placement。

## 科学遥测

必须记录：

- det、Paper及NormMatched proposal norm；
- scale和det/Paper cosine；
- global pre/post-clip norm；
- value prediction change；
- value loss/MSE、explained variance；
- advantage及PopArt统计；
- solver residual和Cholesky info。

解释不得称deterministic Gaussian GGN公式错误；应判断其scale、damping、RHS和finite-sample update是否与Paper匹配。

## 严格比较与早停

只与原始Paper RAT同环境、seed 0、同evaluation semantics及同transition比较：

- 首个共同点`>=2,000,000`
- 首个共同点`>=4,000,000`
- 终点`5,980,160`

仅当`Target/Paper < 0.60`时取消该cell。无精确共同点不得操作；不得以Paper终点比较中间Target。

## 验收标准

唯一结论必须为：

- `CANDIDATE_PROMOTE_TO_3SEED`：至少3/4环境到达终点，且到达终点的环境ratio均不低于0.60。
- `CANDIDATE_REJECT`：至少2个环境严格早停，或完整终点证据明确否定候选。
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`：科学运行开始后基础设施阻止充分判定。
- `PRECHECK_BLOCKED`：probe identity、closure、formal audit或任一preflight失败。

## 禁止事项

不得修改算法、trainer、config、preflight、bundle、launchers、monitor、hook、classifier、AST或runtime binding；不得建立通用control manifest或目录白名单；不得引入第二候选、覆盖旧root、重跑Paper、retry科学cell、使用Jupyter、访问`.54`/`ws4090-31`/`10.49.7.54`、指定host/GPU/partition/concurrency或触碰无关任务。历史失败必须全部保留。

## 报告与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-NORMMATCH-V2-EXACT-PROBE-ALIAS-RECOVERY-AND-6M-S0-20260825-28R.md`

提交validator、负向测试、closure及model-free科学证据，保持worktree干净，推送并验证`origin/agent-work`。回调必须包含唯一结论、commit身份、probe Git blob/SHA、raw/resolved/fd identity ledger、closure/formal audit/preflight结果、四环境科学终态、严格阶段比率、proposal norm/cosine及failure-ledger增量。
