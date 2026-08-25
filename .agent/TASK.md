Status: READY

# TASK.md

Task-ID: `PROCGEN-ACTOR-WEIGHTED-GAE-GGN-HEAD-6M-S0-20260825-32`

## 唯一目标

实现并检验一个严格确定性的、actor-relevant critic GGN：

`DET_ACTOR_WEIGHTED_GAE_GGN_HEAD_V1`

该方法不模仿Paper sampled critic update。它保持actor和shared-trunk critic更新与严格Paper control完全相同，只在257个critic-exclusive value-head参数上，将普通MSE/sampled-score更新替换为“actor加权GAE误差”的deterministic GGN。通过四环境seed0、intended 6M实验判断该目标是否改善长期value→GAE→actor控制耦合。

## 科学动机与边界

既有证据表明：

- Hybrid-head单步policy/logits不变，但后续reward仍可能失败；
- BossFight可有更低value loss却更差reward；
- BigFish在2M通过、4M失败；
- CaveFlyer可在6M匹配Paper；
- solver始终健康。

因此本任务只检验：普通value MSE是否缺少actor-relevant时序几何。Paper仅作为性能baseline和因果control，不是要复制的更新目标。

## 精确算法

使用与Paper完全相同的rollout、reward、done mask、bootstrap、GAE参数、PopArt状态、actor update、shared-trunk sampled critic update、minibatch顺序、epochs、momentum/history、adaptive-KL、clip及evaluation语义。

仅修改value head。

### 1. 固定value误差

在冻结的当前PopArt标准化坐标中：

\[
e_t=V_\theta(s_t)-\operatorname{stopgrad}(\mathrm{return}_t).
\]

不得改变return或GAE的生成方式。

### 2. 精确GAE误差算子

构造确定性线性算子 \(D_{\gamma,\lambda,m}\)，使用与现有trainer完全相同的时间顺序、terminal mask和bootstrap边界，使其满足：

\[
q=D_{\gamma,\lambda,m}e.
\]

必须通过对冻结GAE实现进行value有限差分验证，证明 \(D\Delta V\) 等于重新计算GAE后的变化；不得使用简化的跨episode矩阵。

### 3. Actor相关权重

对冻结old policy：

\[
w_t=\left\|\mathbf 1_{a_t}-\pi_{\rm old}(\cdot|s_t)\right\|_2^2.
\]

以minibatch均值归一化至1，权重detached、无clip、无floor、无额外随机数。这是logit空间中policy-score对advantage误差的确定性敏感度。

### 4. GAE-aware目标与GGN

\[
L_{\rm AG}(\theta)
=\frac{1}{2B}\sum_t w_tq_t^2.
\]

令 \(J_h=\partial V/\partial\theta_h\)，其中\(\theta_h\)仅包含
`last_v_layer.weight/bias`，构造：

\[
K=\operatorname{diag}(\sqrt w)\,D J_h,\qquad
r=\operatorname{diag}(\sqrt w)\,q.
\]

求解：

\[
\left(\frac{K^\top K}{B}+0.5I\right)u
=-\frac{K^\top r}{B}.
\]

使用symmetric FP64、Jacobi scaling和Cholesky；保留Paper global clip `.5`及value-head既有momentum/history语义。不得做Paper proposal norm matching、sampled-value score、joint/cross block或low-Fisher guard。

## 允许代码动作

- 新建独立trainer、config、functional preflight、launcher和monitor。
- 从严格Paper control及Hybrid-head V1复制未改变部分，并记录逐字段diff。
- 使用Git哈希、import smoke、真实网络构造及一步回归验证来源。
- 可修复在科学启动前发现的纯代码错误，但每项必须版本化并进入基础设施账本；不得改变上述数学定义。

不得重新使用Task14–31的multiprocessing origin-observer/closure框架作为科学门槛。

## 必需preflight证据

1. GAE算子对多episode、terminal、truncation和bootstrap的finite-difference误差达到FP64容差。
2. actor权重公式、均值归一化及无梯度/RNG证明。
3. exact production network中：

   - actor方向与Paper control bit-identical；
   - shared critic方向与Paper control bit-identical；
   - policy/shared参数的一步delta bit-identical；
   - policy logits bit-identical；
   - 只有257个value-head参数delta不同；
   - value-head对policy Jacobian为0/disconnected。

4. PopArt仿射reward变换回归：对应标准化输入下，方向、prediction change和接受结果一致。
5. GGN矩阵/RHS与显式autograd Hessian-vector及直接小矩阵参考一致。
6. Cholesky info0、finite residual、无NaN/Inf。
7. 与Hybrid V1、NormMatch V2、separate-B、joint-2B及历史expected/no-cross公式的明确diff表。
8. 不得把Paper sampled proposal相似度作为PASS条件。

Preflight失败则停止为`PRECHECK_BLOCKED`，不得启动科学cell。

## 科学执行

Preflight PASS后运行且仅运行：

- BigFish、BossFight、CaveFlyer、CoinRun；
- seed0；
- 每格intended horizon 6M；
- 独立、非覆盖root；
- 原始Paper RAT seed0仅作为严格同阶段reward baseline。

Executor在启动前自行刷新scheduler、GPU、进程、ownership、capacity、artifact及duplicate状态，并自行决定所有放置与并发。

## 同阶段早停协议

只在相同环境、seed0、evaluation语义的精确共同进度比较：

- first common `>=2M`
- first common `>=4M`
- `5,980,160`

只有：

\[
\text{Target reward}/\text{Paper reward}<0.60
\]

才可取消该cell并记录`EARLY_STOPPED_ALGORITHM`。无精确共同row则不得操作；中间Target不得比较Paper terminal。

## 必需科学证据

每个stage保存：

- reward、严格Paper reward及ratio；
- KL、actor LR、entropy；
- value loss；
- GAE mean/variance/RMS；
- \(L_{\rm AG}\)；
- actor权重分布；
- TD residual及return error；
- GGN spectrum摘要、effective rank、condition number；
- parameter/prediction/GAE change norm；
- predicted与realized \(L_{\rm AG}\) change；
- residual、Cholesky info及hard-error扫描。

必须特别分析：

- 更低value loss是否对应更好的GAE和reward；
- BigFish是否再次出现2M通过、4M失效；
- CaveFlyer优势是否保持；
- 环境差异来自GAE目标、谱结构还是reward/PopArt尺度。

## 唯一终局结论

只允许一个：

- `PRECHECK_BLOCKED`
- `CANDIDATE_INCONCLUSIVE_INFRASTRUCTURE`
- `CANDIDATE_REJECT`：至少两个环境发生有效算法早停，或证据已使三环境成功不可能。
- `GAE_GGN_SEED0_PROMISING`：至少三个环境到达5,980,160；最多一个算法早停；至少两个环境终点超过Paper；计入早停stage ratio后的四环境平均ratio大于1；数值与GAE健康证据完整。
- `CANDIDATE_NOT_READY`：科学证据完整但既不满足拒绝也不满足promising标准。

本任务不得启动seeds1–2或正式x3扩展。

## 禁止事项

- 不得继续Task31R或任何`__mp_main__`/origin observer工作。
- 不得修改Paper baseline或重跑Paper。
- 不得做Paper norm/RHS/inverse matching或multi-ε quadrature。
- 不得添加joint/cross、projection、low-Fisher、actor tuning或第二候选。
- 不得进行超参数/算法sweep。
- 不得覆盖历史root、删除失败记录或将基础设施失败重标为算法失败。
- 不得使用Jupyter。
- 不得访问`.54`、`ws4090-31`、`10.49.7.54`。
- 不得规划MuJoCo或Isaac。
- Planner不指定host、GPU、partition、卡数、并发或queue placement。

## 报告、提交与推送

更新：

- `.agent/STATE.md`
- `.agent/AGENT_REPORT.md`
- `.agent/reports/PROCGEN-ACTOR-WEIGHTED-GAE-GGN-HEAD-6M-S0-20260825-32.md`

报告必须包含冻结哈希、完整Paper→Target diff、preflight、四环境stage表、failure ledger、artifact/error扫描、唯一结论、assignment/evidence/Delivery commits及`origin/agent-work`验证。

提交代码和模型无关证据；不得提交model/checkpoint。推送后回调Planner。
