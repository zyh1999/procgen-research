# Procgen strict joint-2B causal audit

Last refreshed: 2026-08-13 (CSF3 live jobs).

## Question

Why does the shared Procgen strict joint actor--critic `2B x 2B` update behave
very differently across environments despite finite, accurately solved systems?

## Matched controls

All listed runs use rollout 4096, minibatch 512, 4 epochs, 1M transitions,
damping 0.5, no momentum, no Kaczmarz, adaptive-LR cap 0.05, and KL upper
threshold 0.02.

| Variant | Rows | lambda_C | c_C | Meaning |
| --- | ---: | ---: | ---: | --- |
| actor-only | B = 512 | 0 | 0 | actor RHS and actor curvature only |
| curvature-only | 2B = 1024 | 1 | 0 | strict stacked metric including all actor--critic cross blocks, no critic RHS |
| full joint clean | 2B = 1024 | 1 | 1 | clean critic score and paired residual RHS |
| full joint Gaussian | 2B = 1024 | 1 | 1 | Gaussian critic score and paired Gaussian residual RHS |

The trainer overwrites optimizer gradients with `flat_dir`. Therefore logged
`loss_v` is diagnostic only: actor-only does **not** include a separate Adam
value-loss update.

## Verified findings

1. **Not a solver/numerical failure.** All observed solves are finite, with
   residuals around 1e-13 to 1e-10 and no OOM, NaN, CUDA, traceback, or
   Cholesky errors.
2. **The ablations have the intended semantics.** Curvature-only retains the
   stacked 1024-row system and its AA/AC/CA/CC blocks; only its critic RHS is
   zero. Actor-only is the 512-row control.
   The full critic construction was also checked algebraically: with
   `critic_residual = return - value`, `H_C = sqrt(lambda_C) D_xi J_C`, and
   `b_C = c_C/sqrt(lambda_C) D_xi residual`, its expectation is
   `lambda_C J_C^T J_C` and `c_C J_C^T residual`.  The later `K/B` and
   reconstructed direction `/B` cancel in the undamped sample-space least
   squares solution.  Thus there is no observed missing batch factor, ratio,
   or sign reversal; at `c_C=1` this is the natural/GGN direction for the
   conventional half-MSE objective.
   The retained MuJoCo shared joint-kernel implementation uses the same
   convention (`joint_H=[H_pi;J_v]`, `joint_rhs=[adv;vf_coef*(return-value)]`,
   kernel divided by its sample count and reconstructed direction divided by
   the same count).  Thus the Procgen construction does not differ from the
   successful MuJoCo path by an accidental factor of two; `c_C` is the same
   kind of direct residual-RHS weight as the MuJoCo `vf_coef`.
3. **Critic metric alone changes the trajectory, but is insufficient for the
   full-joint early learning signal.** At a matched intermediate point,
   BigFish was actor-only 2.32, curvature-only 1.97, full clean 5.21; BossFight
   was actor-only 0, curvature-only 0, full clean 1.50.
4. **Direct actor--critic coupling is time-dependent, not uniformly zero.**
   Late sparse diagnostic samples in full clean have metric cosine in
   [-0.0009, 0.0078] and actor gain from the critic RHS in
   [0.9996, 1.0035]. In contrast, CaveFlyer at 49k transitions has cosine
   -0.124 and actor gain 0.936: a modest but real early negative critic-RHS
   response. It must not be extrapolated from the late diagnostic alone.
5. **Full joint is not simply taking larger policy steps.** Its effective
   `LR * direction-L2` is lower than the controls at matched BigFish and
   BossFight stages, because adaptive KL lowers its LR.

## Current working mechanism (not yet a conclusion)

The strongest remaining explanation is a **long-horizon critic feedback**:
the critic RHS predominantly updates critic/value and shared representation;
that changes the following rollout's baseline and GAE advantage, then changes
subsequent policy/KL/LR dynamics. This can help BigFish/BossFight while
contributing to low entropy in CaveFlyer. It is not yet proven across all four
environments.

Early direct coupling is a contributing diagnostic, but not a sufficient
explanation on its own.  Across the first 100,352 transitions, the minimum
actor gain from the critic RHS is 0.894 (BigFish), 0.565 (BossFight), 0.734
(CaveFlyer), and 0.988 (CoinRun).  BossFight has the strongest observed
instantaneous antagonism yet benefits from the full RHS, so a negative
cross-block response cannot by itself explain CaveFlyer's degradation.  The
remaining distinction must include the induced value/advantage/controller
trajectory rather than only the one-minibatch geometric cosine.

The archived trainer summaries give a concrete candidate for that trajectory:
late CaveFlyer full-clean summaries show `ret_var` falling from values around
2.56 to 0.322 and `adv_var` to 0.0275, alongside `loss_v` 0.019 and reward
1.6.  The still-running curvature-only run near its current prefix instead
reports `ret_var` 1.95--2.66, `adv_var` 1.95--2.66, `loss_v` 2.49--2.96, and
reward around 4.7--5.1.  This is consistent with a critic-RHS/value-feedback
collapse of the later advantage signal, but is not yet causal proof: lower
return variance can also be a consequence of the already collapsed policy.
The queued matched RHS-scale intervention is what distinguishes these two
directions of causality.

The adaptive controller is a measurable amplifier of this feedback. It makes
one decision per rollout from the **last** minibatch's fixed-behavior KL, not
the rollout average. In the first 50 CaveFlyer rollouts, full clean has six
down-triggering terminal KLs above 0.02 (first at rollout 16), thirteen
up-triggers below 0.005, and mean LR 0.0219; curvature-only has one down
trigger (first at 44), 38 up-triggers, and mean LR 0.0439. Full clean also
triggers earlier/more often than curvature-only in BigFish and BossFight, so
this is not a Cave-only root cause, but its timing and hysteresis plausibly
amplify the RHS-induced trajectory difference. The same full-clean Cave run
finishes with mean LR 0.00869 versus 0.02997 for the currently available
curvature-only prefix. A later fixed-LR or rollout-mean-KL control is needed
only if the RHS-scale and head-only interventions do not already resolve the
mechanism.

There is also a narrowly scoped controller-semantics defect to keep separate
from the joint algorithm: the selected `pi_info['kl']` is computed during the
last minibatch forward pass, before that minibatch's `optimizer.step()`. The
subsequent once-per-rollout adaptation therefore measures neither the
post-final-step policy nor an average over the fixed rollout; its comment
claims the former. This is common to every current causal control, so it does
not invalidate their *relative* c_C/head-only comparisons, but it means a
later controller-specific test must recompute KL on the fixed rollout after
all minibatch updates (or deliberately average them), in an isolated run.

The CaveFlyer curvature-only trajectory refines this: around 229k transitions,
actor-only / curvature-only / full-clean rewards were 3.2 / 3.6 / 1.03 and
entropies 2.39 / 2.01 / 1.96. Thus the metric/cross blocks can create a useful
long-horizon entropy change; the much worse CaveFlyer outcome requires adding
the full-amplitude critic RHS. The working question is now its transferable
scale, not whether all joint curvature is intrinsically harmful.

At the latest prefix that is shared by all four variants (seed 0), this split
persists rather than being a single early fluctuation.  At 348,160 CaveFlyer
transitions, actor-only / curvature-only / clean / Gaussian rewards are
3.0 / 3.4 / 1.4 / 0.9, with entropies 1.56 / 1.47 / 1.51 / 1.52.  Conversely,
at 741,376 BigFish transitions the corresponding rewards are
2.71 / 1.82 / 4.77 / 3.10, and at 684,032 BossFight transitions they are
0 / 0 / 1.39 / 0.63.  Thus the full critic RHS is genuinely beneficial in
some environments but harmful in CaveFlyer under an otherwise matched setup;
this is a sign/scale transfer issue, not an unconditionally bad 2B metric.

Nor is CaveFlyer an obvious initial-Gram outlier. At 40,960 transitions in
the matched full-clean runs, normalized cross-block magnitudes are 0.552
(BigFish), 0.152 (BossFight), 0.231 (CaveFlyer), and 0.348 (CoinRun). Cave's
critic RHS L2 is also 14.4, below BigFish/BossFight at 19.8/22.9. Therefore a
large initial actor--critic cross block or raw critic-RHS norm alone cannot
explain its later degradation; the discriminating mechanism is temporal.

## Outstanding evidence

### Pre-registered decision rule

Use the same-transition, seed-0 comparisons first; do not promote a method
claim until matched seeds follow.

| Observation after the pending controls | Supported next diagnosis | Next isolated control only if needed |
| --- | --- | --- |
| `c_C=.1` rescues CaveFlyer while full-head does not | critic RHS scale/controller feedback is dominant | sweep RHS scale or use a normalized residual target; preserve the metric |
| full-head rescues CaveFlyer while `c_C=.1` does not | critic update through the shared trunk is dominant | compare a controlled shared-trunk projection/gate, not another global RHS scale |
| both rescue CaveFlyer | both paths matter; compare their BigFish cost at equal transition | select a Pareto trade-off before any multi-seed run |
| neither rescues, while full joint still hits early KL down-triggers | controller measurement/hysteresis remains a live cause | isolated post-epoch full-rollout-KL controller run |
| neither rescues and controller-corrected run also fails | the remaining explanation is value/GAE target dynamics beyond one-step geometry | instrument post-update value/GAE distributions before changing curvature |

The pending controller control is deliberately narrow.  The original trainer
already makes exactly **one** adaptive-LR decision after all four epochs of a
rollout, using the final-minibatch after-step KL (`pi_info['kl']`) against the
frozen rollout behavior logits.  The post-epoch trainer preserves that
frequency, reference policy, thresholds, and multiplicative LR rule; it only
replaces the observation by the exact categorical behavior-KL averaged over
all 4,096 states from that same fixed rollout.  It must therefore not be
described as a more-frequent controller.

### Same-transition trajectory synthesis (seed 0)

The 0.61M / 0.99M checkpoints sharpen the environment-dependent pattern.
In CaveFlyer, clean full-joint has reward 0.7 / 1.6 and entropy 0.50 / 0.13;
Gaussian-score full-joint has 1.3 / 1.2 and entropy 0.46 / 0.29.  In contrast,
actor-only has 2.4 / 2.3 and entropy 1.63 / 1.64, while curvature-only has
4.8 / 4.8 and entropy 1.54 / 1.89.  Thus critic-score randomization does not
rescue the collapse, whereas retaining the strict 2B critic metric but
removing the critic residual RHS does.  This is stronger evidence against
``Gaussian versus deterministic score sampling`` as the primary cause.

The same full RHS is not uniformly harmful: BigFish at those checkpoints is
6.25 / 7.13 versus actor-only 2.44 / 2.43, and BossFight is 1.82 / 2.30
versus actor-only 0 / 0.  CoinRun also reaches 4.7 / 7.0, albeit with low
entropy.  Hence the hypothesis is specifically an environment-dependent
critic-RHS interaction with the learned shared representation, not an
algorithm-wide numerical instability or a blanket argument for removing the
critic.  The head-only control is the next discriminating intervention.

- Curvature-only CaveFlyer and CoinRun are needed to separate metric/cross-block
  effects from critic-RHS feedback in the two environments where full joint
  has low entropy.
- Matched clean full-joint `c_C=0.1, lambda_C=1` is queued as CSF3 array
  `18552281_[0-3]`. It keeps the strict 2B system while reducing only the
  critic RHS, testing whether the early full-joint direction/entropy split is
  amplitude-driven.
  With a fixed sampled metric, the solve is linear in the RHS: the
  critic-induced direction must scale approximately tenfold down. In CaveFlyer
  at 49k, `c_C=1` had actor gain 0.936; a simple RHS-amplitude explanation
  predicts a gain close to 0.994 at `c_C=.1`, with an early direction and
  entropy closer to curvature-only. Failure of that prediction would point to
  a nonlocal training/controller feedback rather than the one-step RHS scale.
  BigFish has now started as child `18552944` on node864 and its preflight
  verifies `scope=all`, `lambda_C=1`, `c_C=.1`, batch 4096/minibatch 512,
  four epochs, damping .5, no momentum/Kaczmarz. At 40,960 transitions its
  critic RHS L2 is 2.13 versus 19.84 for matched `c_C=1`, while the full
  direction is 0.103 versus 0.267 because the actor component remains. Both
  solves are finite (1.96e-14 versus 2.55e-13), so the isolated scale
  intervention is now confirmed live.
  At 65,536 transitions it has no terminal-minibatch KL down-trigger above
  .02 and all 16 observed rollouts are below the .005 up-trigger threshold;
  it has reached LR .05. The matched c_C=1 BigFish prefix has four down
  triggers in its first 50 rollouts. This directly confirms RHS scale changes
  the KL/LR feedback path, but the c_C=.1 reward (about 1.46 at that early
  prefix) is far too early to support an outcome claim.
  At the later matched 98,304-transition prefix, full c_C=1 / full c_C=.1 /
  curvature-only BigFish rewards are 2.39 / 1.71 / 1.53 and entropies are
  2.44 / 2.67 / 2.69. Thus c_C=.1 removes early controller pressure but also
  largely removes the early full-RHS learning benefit in BigFish; it is an
  environment-scale trade-off, not yet a universal repair.
  The completed curvature-only CaveFlyer control now makes the first half of
  this diagnosis substantially stronger: at 1,007,616 transitions it is
  `PASS`, with reward 3.90, entropy 2.28, behavior KL .0031, and solve
  residual 1.89e-12.  Thus a strict 2B metric with the critic rows and every
  actor--critic cross block retained is compatible with sustained CaveFlyer
  exploration once the critic residual RHS is removed.  This rules out a
  blanket "2B curvature is unstable" explanation.
  The c_C=.1 BigFish intervention has also crossed a useful same-transition
  checkpoint: at 360,448 transitions clean c_C=1 / c_C=.1 /
  curvature-only rewards are 5.88 / 2.35 / 2.17, with LR .00658 / .0222 /
  .05 and entropies .92 / 1.55 / 2.31.  Lowering c_C therefore changes the
  controller and preserves more entropy, but does not retain the beneficial
  BigFish learning effect of the full RHS.  It is not a free stability fix.
  A later matched check excludes a simpler global-update-norm explanation for
  that trade-off.  Near 802,816 transitions, full c_C=1 / c_C=.1 have raw
  joint-direction norms 0.265 / 0.275 and effective norms equal to those raw
  values: neither is clipped by the 0.5 global direction cap.  The c_C=.1
  run has reward 4.35 versus 4.48 for c_C=1 at that checkpoint, while its
  entropy remains higher (.95 versus .48).  The occasional later per-minibatch
  cap activation therefore cannot explain its systematic slower learning;
  the relevant difference remains the changed KL/LR feedback and/or the
  learned shared representation trajectory.
  BigFish `c_C=.1` has now completed cleanly (`PASS`, `rc=0`, Slurm
  `COMPLETED`) at 1,003,520 transitions.  Its terminal trace records reward
  5.35, entropy .718, behavior KL .0113, and solve residual 3.60e-10.
  Thus this particular scale-control result is lifecycle-valid.  In contrast,
  the CoinRun curvature-only exception below remains deliberately excluded.
- A second, orthogonal four-environment control is queued as CSF3 array
  `18552840_[12-15]`: `full_head`, with clean score, `lambda_C=c_C=1`, and
  the same 4096/512/4-epoch/controller setup, but
  `joint_critic_param_scope=head_only`. Its critic Jacobian/RHS contains only
  `last_v_layer.*`, so the shared trunk has no critic pathway and the
  actor--critic cross block is identically zero. It tests whether the
  environment-dependent failure specifically requires shared-representation
  critic updates, independently of the RHS-amplitude intervention.
- This is seed 0 only. Any final method-level claim requires matched multi-seed
  evidence after the causal identity is resolved.

## Non-comparable historical evidence

Earlier CSF3 coefficient sweeps under
`/scratch/h99859yz/procgen_det2b_rhscoef_gpuh_20260811_v1` must not be merged
with this table: they use `train_shared_rat_exact_deterministic_ggn_symfp64.py`,
a different config SHA, BigFish only, and 6M transitions. They motivate the
new sweep but do not answer this campaign's causal question.

## Lifecycle exception: CoinRun curvature-only seed 0

The CoinRun curvature-only worker `18551302_7` reached all 246 planned
rollouts (1,007,616 transitions) and its last recorded metric is reward 4.10,
entropy 1.15, behavior KL .0101, solve residual 6.15e-12.  Its training
stderr has no OOM, NaN, CUDA, Cholesky, or Python traceback.  However Slurm
accounting records `FAILED (exit 2)` and the run directory remains `RUNNING`
without an `rc` file.  The outer launcher was modified while the child was
running: `run_task.sh` was overwritten at 23:40:48 while the child was still
active, and its post-child shell parsing failed at 23:44:32 before status
finalization (`tic_curvature_coef: command not found`, then the remaining
array arguments).  This is the expected signature of bash reading a shared
source while it is being overwritten, not a trainer exception.  Future
post-epoch controls snapshot their task launcher into the allocated worker
and record its SHA before starting Python.
This is a lifecycle/launcher failure, not evidence of an algorithmic failure,
but the run is deliberately excluded from any formal completed-seed table.
The written trajectory may be used only as a diagnostic prefix until a clean
rerun is completed under an immutable launcher.

## Actor-from-critic damping guard: 2M gate and formal validation

The block-median damping intervention alone is not a complete repair.  In the
unprotected `block01_lr02` run, CaveFlyer first reaches a counterfactual
actor-from-critic floor at 978,944 transitions, has sustained 8/10 floor
binding by 1,015,808, crosses entropy .2 at 1,187,840, and finishes at 6M
with reward .048, entropy .041, and behavior KL .081.  CoinRun has sustained
8/10 binding by 1,363,968, crosses entropy .2 at 1,589,248, and later reaches
reward zero with entropy about .019.  Thus scaling each diagonal block by its
own median delays failure but still permits the actor kernel/damping scale to
collapse relative to the critic block.

The repaired gate retains the strict shared full `1024 x 1024` joint system,
clean deterministic critic score, all cross blocks, `c_C=lambda_C=1`, FP64
solve, base damping .5, and block-median floor .1.  Its only optimization-path
addition is
`joint_actor_damping_from_critic_floor=.01`, which lower-bounds actor damping
by one percent of the critic kernel median.  The matched four-environment
seed-0 gate is archived under
`gate_3m_seed0_jupyter_actorcriticguard01_v1/acguard01`; its frozen
`GATE_PASS_2M_AUDIT.txt` reports `AUDIT=PASS` after every environment exceeded
2M transitions, with positive cross blocks, solve residuals near machine
precision, and zero guard violations.

The sharpest same-transition result is CoinRun at 1,589,248 transitions.  The
guarded run has reward 7.60, entropy .741, and behavior KL .0119; the
unprotected block-only run has essentially the same reward (7.70) but entropy
.195 and KL .0408.  By 2,019,328 transitions the guarded run has reward 8.29,
entropy .685, KL .00829, and 69/100 recent guard bindings with zero
violations.  CaveFlyer also crosses 2M with reward about 3.50, entropy .98,
and KL .020 while the guard begins binding near 1.978M.  These results show
that the guard does not merely freeze the policy: reward continues to improve
while KL remains controlled and entropy stays well above the failed regime.

This remains gate evidence rather than a final method claim.  The formal
`4 environments x 3 seeds x 6M` array is CSF3 Job `18644494_[0-11%4]` on
`gpuA`, with root `formal_6m_3seed_jupyter_actorcriticguard01_v1/acguard01`.
The earlier `gpuL` submission `18643851` was cancelled while still pending
with zero runtime and before any run directory existed; the migration avoids
duplicate children while retaining the same immutable algorithm identity.
The `gpuA` submission overrides only scheduler resources
(`gpu:a100_80g:1`, 12 CPUs, 96G) and leaves the launcher body and result root
unchanged.  While it was still pending, its walltime was reduced from the
overly conservative 48 hours to 16 hours without resubmission, preserving its
queue age.  This remains conservative relative to the observed guard-gate
throughput (2--2.8M transitions in about 1.1--1.7 hours) and an observed 6M
child completion in about 4.8 hours.  Its launcher,
trainer, and config SHA256 values are respectively
`93ba823918cca76a7a15104984a8a26aee8409f633acd3e3b39315525344cb89`,
`2709256861583122e61bd0211bb85ab7f3108273455b3f957f98515b463c0475`,
and `3a24a057bdb6898e0ca3e6153eddfc7d6272700f5df083f345d84c9f940ffdb0`.
The trainer differs from the gate only by six retry attempts around metric
trace flush I/O; it does not alter training math.  Final acceptance requires
both `audit_joint2b_completion.py` and the default strict
`audit_joint2b_performance.py` to return `AUDIT=PASS`; the current empty-root
test returns code 2 and `INCOMPLETE`, confirming that both audits fail closed.

The performance audit was strengthened before the formal array started.  In
addition to aligned-horizon reward and final entropy/KL, it now checks the 6M
tail reward mean, the worst seed, and retention relative to the aligned
candidate reward.  Its SHA256 is
`e65fee57202a6937a0c90f3d0ab5a3564eb0d8ad7c206662d44e201a2cec92c5`.
Against the known failed block-only run it flags CaveFlyer and CoinRun with
`STABILITY`, `FINAL_REWARD`, `SEED_COLLAPSE`, and `REWARD_RETENTION`, so a
late reward collapse can no longer pass merely because the early reward was
high.  The one-seed guard gate passes these new late-stage checks, but its
CaveFlyer aligned reward at 1,007,616 transitions is 2.318 versus 2.510 for
sampled-B, so the strict short-horizon `REWARD` check remains unresolved until
the formal three-seed mean is available.
