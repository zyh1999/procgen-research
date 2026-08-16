# Procgen strict joint-2B diagnosis (2026-08-13)

## Question

Why does the shared Procgen strict `2B x 2B` actor-RAT + critic-GGN update
underperform, even though a related shared MuJoCo joint system can work?

## Controlled Procgen identity

- Environments: BigFish, BossFight, CaveFlyer, CoinRun; seed 0.
- Rollout 4096, minibatch 512, 4 epochs, 1M transitions.
- Strict stacked actor/critic system: 1024 rows.
- Damping 0.5, no momentum, no Kaczmarz, clean deterministic critic score.
- Fixed rollout behavior reference and Procgen rollout-level adaptive LR.

## Evidence already established

1. The failure is not explained by inaccurate linear solves. Full-fp64 and
   symmetrized/Jacobi solves achieve tiny residuals but do not recover reward.
2. The effect of the critic objective coefficient is environment-dependent.
   Strong critic RHS helps BigFish/BossFight early, while CaveFlyer benefits
   more from actor-only or curvature-only variants.
3. Removing explicit actor-critic kernel cross blocks is not sufficient:
   block-diagonal sample-space runs can still interfere through shared
   parameters and a shared optimizer/clip.
4. Restricting critic reconstruction to the 257-column value head after the
   full solve is diagnostic, not a valid repair. Roughly 99% of the original
   critic reconstruction norm lies in the shared trunk. The post-solve mask
   can also destroy critic descent.
5. At matched 634,880 transitions, reconstruction-head gives BigFish 4.97
   versus full-joint 6.54, and BossFight 0.18 versus full-joint 1.16. Later the
   two environments diverge sharply: over the ten rollouts ending near 807k,
   reconstruction-head BigFish reaches mean reward 9.93, while BossFight stays
   at 0.157.
6. At the latest valid reconstruction-head diagnostic, critic RHS changes the
   actor metric response by 1.084x on BigFish but 0.676x on BossFight. The
   effect is not uniformly harmful. The diagnostic points had no additional
   global-clip attenuation.
7. The late divergence tracks value fit. BigFish has value MSE about 0.041
   against return variance 0.677, while BossFight has MSE about 2.81 against
   variance 1.60 and entropy about 0.53. The head-only critic is adequate after
   BigFish discovers a useful policy, but it cannot learn the representation
   needed by BossFight. Thus critic trunk writes are simultaneously a useful
   representation-learning channel and a possible source of policy
   interference; deleting them is environment-dependent, not a general fix.
8. The complete reconstruction-head trajectories do not retain their peak
   performance. BigFish peaks at reward 10.56 near 807k but finishes with a
   last-10 mean of 2.98; BossFight finishes at 0.002 with entropy 0.25. During
   the BigFish decline, actor/critic block Frobenius norms grow from roughly
   2.5e3 near the peak to 1.1e4/9.2e3 at the end. BossFight reaches about
   9.6e4/1.6e5. Solve residuals remain tiny and global clipping is usually
   inactive, so accurate solution of a dynamically rescaled metric does not
   guarantee a stable policy update.
9. The rollout-level adaptive-LR implementation reads the final minibatch's
   behavior KL. Auditing all 246 rollouts finds only one missed above-threshold
   excursion for BigFish and four for BossFight, and no case where the final
   KL would request an LR increase while another minibatch exceeded 0.02.
   Therefore last-minibatch observation is imperfect but is not the leading
   explanation for these collapses. This is consistent with the prior
   whole-rollout post-epoch KL control failing to rescue CaveFlyer.
10. Frobenius growth alone is not causal. The original full-joint control has
    actor/critic block norms two to four orders of magnitude larger than the
    reconstruction-head control and nevertheless performs better. At the end
    of BossFight, full joint is roughly 2.4e6/1.2e6 with reward 2.19, while
    reconstruction-head is only 7.6e4/1.6e5 with reward 0.01. The remaining
    scale question is therefore diagonal distribution and damped condition,
    not raw block norm.
11. The reconstruction-head restriction is already disadvantaged near 0.1M
    transitions on the two later environments: CaveFlyer is 1.2 versus 2.4
    actor-only and 2.2 full-joint; CoinRun is 2.5 versus 2.6 actor-only and 3.6
    full-joint. Entropy remains about 2.70 and KL is small, so this early gap
    precedes entropy collapse and supports loss of useful critic-driven visual
    representation rather than an immediate controller failure.
12. The post-solve reconstruction mask also breaks the descent guarantee of
    the isolated critic component.  On every sparse causal diagnostic in the
    completed BigFish and BossFight full-joint controls, the measured critic
    self-response `g_C^T d_C` is positive (246/246 for each environment).  With
    the reconstruction-head mask applied after the same coupled solve, it is
    negative on 135/246 BigFish diagnostics (54.9%) and 137/246 BossFight
    diagnostics (55.7%).  This is expected mathematically: an SPD solve makes
    the unmasked preconditioned component a descent direction, but coordinate
    projection after reconstruction does not preserve that property.  This
    explains why reconstruction-head itself is not a valid repair, but it does
    **not** explain the original unmasked full-joint underperformance.
13. Removing only the explicit sample-space cross blocks is not a uniform
    repair.  At matched completion, block diagonalization improves BigFish
    last-10 reward from about 6.38 to 12.67, but lowers BossFight from 1.98 to
    0.57; CaveFlyer and CoinRun were stopped early at about 0.02 and 3.49.
    Mean normalized cross-block magnitude is likewise not monotone with the
    outcome, and mean actor gain from the critic RHS stays near one in both
    full and block-diagonal runs.  This rules out the scalar hypothesis
    "larger cross block is always worse."  It does not yet isolate shared
    parameter interference: the block-diagonal dual solve still reconstructs
    both components into the same trunk and clips their sum globally.
14. The matched critic-RHS-scale sweep points to cumulative exploration loss,
    not an acutely uncontrolled local KL.  With `c_C=1`, BigFish and
    BossFight have the best final reward among the tested RHS scales, while
    CaveFlyer/CoinRun finish with entropy about 0.16/0.09 and last-10 rewards
    1.54/6.94.  With the critic RHS removed but critic curvature retained,
    CaveFlyer/CoinRun keep entropy about 1.94/1.41 and reach 4.57/4.64.
    Nevertheless, the last-10 behavior KL remains near 0.01 in all of these
    runs.  The leading mechanism is therefore that the critic RHS can
    cumulatively reshape the shared policy representation and exhaust
    exploration while each individual rollout-relative update still appears
    KL-controlled.  The separate-capacity control will determine whether this
    is direct shared-trunk leakage or merely joint global-clip attenuation.
15. The full-joint inverse-metric diagnostics do not show large instantaneous
    actor/critic opposition.  Median metric cosines are near zero in every
    environment, no diagnostic has absolute cosine above 0.5, and mean actor
    response gain from the critic RHS is only about 0.999--1.017.  Hence the
    likely effect is many small critic-driven representation changes altering
    future policy gradients, rather than a large anti-actor step in each
    minibatch.
16. Entropy collapse is environment-dependent rather than intrinsically a
    failure.  Under `c_C=1`, CaveFlyer entropy falls from about 2.52 at 0.1M to
    0.09 at 1M while reward deteriorates to 1.3.  CoinRun entropy also falls,
    from about 2.11 to 0.14, while reward improves from 3.6 to 7.3.  The causal
    claim is therefore that the critic RHS changes the exploration trajectory;
    whether that helps depends on the environment.  A shared-trunk isolation
    test is still needed before calling the representation change harmful in
    general.
17. Global L2 clipping is unlikely to be the primary transmission mechanism.
    Reconstructing its exact scale from effective/raw direction norms shows
    that `c_C=1` clips only about 7.3% of CaveFlyer and 6.8% of CoinRun
    minibatches, with mean scale about 0.990 and median one.  Conversely,
    BigFish curvature-only clips about 20% of minibatches without showing the
    same entropy-collapse trajectory.  The separate control will still log
    actor-only versus joint attenuation, but existing evidence rules out
    persistent global clipping as the leading explanation.
18. The critic component is structurally a trunk update, not mainly a value-
    head update.  Before applying the reconstruction-head mask, the median
    fraction of critic reconstruction L2 norm in non-value-head columns is
    about 98--99% in all four environments; completed BigFish/BossFight means
    are about 95.5%/95.7%.  Thus changing `c_C` primarily changes how strongly
    the critic rewrites the visual representation shared with the policy.
    This provides a direct structural bridge between critic RHS scale and the
    cumulative exploration trajectories observed above.
19. The reference shared Exact-RAT critic is not a deterministic critic-GGN
    update.  Source audit of `train_shared.py` shows that each B-row feature is
    the *sum* of the policy score and a randomized value-likelihood score.  For
    `sample_v = v + xi`, `vf_logp=-(v-stopgrad(sample_v))^2`, this critic score
    is exactly `2 xi J_C`, so the reference Gram is built from
    `H_ref = G_A + 2 diag(xi) J_C`, `xi~N(0,1)`.  The B x B system solves an
    advantage RHS and an all-ones RHS together.  Actor reconstruction uses the
    policy score/loss only; the second coefficients weight an ordinary MSE
    backward pass without a second ratio multiplication.  The strict joint
    method instead stacks deterministic actor and critic rows into 2B, solves a
    residual critic RHS, and reconstructs a GGN-preconditioned critic direction.
    Therefore successful shared Exact-RAT isolates a materially different
    randomized combined metric and critic update.
20. A matched reference-Exact-RAT control has now been corrected and queued.
    It keeps the current DMLP1024 architecture, 4096/512/4-epoch/1M workload,
    damping 0.5, rollout-relative controller, LR 0.004 with 0.05 cap, and
    global clip 0.5.  It constructs the exact combined randomized B-row score
    `G_A + 2 xi J_C`, solves advantage/all-ones RHS columns in the same
    `K D + mu I` system, reconstructs actor movement through `G_A`, and uses
    the all-ones coefficients in an ordinary weighted-MSE backward pass.  A
    random linear-model autograd equivalence test matches the constructed
    ascent direction to relative error `1.81e-16` (B rows, two RHS columns).
    This control will distinguish the reference randomized combined geometry
    from the deterministic stacked critic-GGN update.
21. The strict `head_only` critic-scope control completed all four environments
    at 1,007,616 transitions with RC 0 and no OOM/NaN/Traceback.  Its last-10
    rewards were BigFish 1.25, BossFight 0.04, CaveFlyer 0.00 and CoinRun 0.60
    (peaks 2.08, 1.69, 5.71 and 8.10).  All traces report 1024 rows,
    `joint_critic_param_scope=head_only` and exactly zero cross-block Frobenius
    norm.  Restricting the critic Jacobian to its head therefore does *not*
    rescue CaveFlyer or the other tasks.  This rejects the simple hypothesis
    that damage is caused only by the critic directly writing through the
    shared visual trunk; the joint actor update/controller and the choice of
    GGN-preconditioned critic target remain live explanations.
22. The capacity-matched separate control is structurally matched as intended.
    The shared model has 1,464,544 total parameters: a 1,460,432-parameter
    backbone plus actor/critic heads totalling 4,112 parameters.  The separate
    model has two identical 1,460,432-parameter backbones and the same two
    heads, for 2,924,976 total parameters.  Thus each task receives exactly the
    original backbone capacity; the parameter-count increase is the intended
    duplication that removes shared columns, not a wider per-task network.
    BigFish and BossFight have produced real metrics with 1024 rows, zero cross
    block, FP64 residuals around 1e-13--1e-14, and no extra joint actor clip;
    CaveFlyer and CoinRun are waiting for the two running tasks to release the
    account's current GPU slots.
23. The `head_only` failure is not by itself proof that shared-trunk coupling is
    irrelevant: it also removes the critic's ability to learn its visual
    representation.  The completed trajectories collapse late (CaveFlyer from
    a peak 5.71 to 0; CoinRun from 8.10 to 0.60), while their final KL remains
    inside the controller band.  The capacity-matched separate control is the
    clean test because it removes shared columns while retaining a complete
    critic backbone.  At the first comparable early checkpoint it is healthy:
    BigFish 98,304 transitions (last-10 1.78), BossFight 94,208 (0.05), and
    CaveFlyer 20,480 (2.58); entropy remains about 2.67--2.71, KL is below
    0.0025, clip scale is exactly 1, and no numerical/runtime error is present.
    CoinRun is pending solely on the account's three-GPU `AssocGrpGRES` limit.
24. Reducing only the critic RHS coefficient to `c_C=0.1` is not a universal
    repair.  All four runs completed cleanly at 1,007,616 transitions; last-10
    rewards are BigFish 5.58, BossFight 0.46, CaveFlyer 1.60 and CoinRun 7.90.
    Relative to the matched `c_C=1` full joint results (6.95, 1.55, 1.30,
    7.30), this improves CaveFlyer only slightly and trades away BigFish and
    BossFight.  CoinRun reaches high reward despite entropy 0.076, whereas
    CaveFlyer remains poor at entropy 0.364.  Consequently a single claim that
    critic RHS magnitude or low entropy alone is the root cause is contradicted
    by the cross-environment evidence.
25. FP64-to-FP32 coefficient application is measurable but does not explain the
    CaveFlyer failure.  Although the logged FP64 solve residual is around
    `1e-11`, the applied residual after casting coefficients is larger.  When
    normalized by the joint RHS norm at matched logged updates, its median /
    90th-percentile / maximum is `2.0e-5 / 7.4e-5 / 1.3e-4` for CaveFlyer.
    This is actually smaller than BossFight (`2.3e-4 / 6.7e-4 / 7.5e-4`) and
    CoinRun (`2.2e-4 / 6.5e-4 / 2.1e-3`), which achieve better final rewards.
    Thus coefficient down-casting is not the environment-selective root cause,
    though the new spectral run continues to log it as a numerical guardrail.
26. Sparse full-joint inverse-metric diagnostics also reject strong immediate
    actor/critic opposition.  Across 246 diagnostics per environment, median
    actor gain from adding the critic RHS is 1.0003--1.0018 and median metric
    cosine is within about 0.007 of zero; critic self-response is positive in
    the unmodified full solve.  A reconstruction-control decomposition shows
    the critic component L2 is generally smaller than the actor component and
    global clipping adds little actor attenuation.  Any critic effect is
    therefore more plausibly cumulative representation/target drift than a
    single-minibatch direction cancellation or clip bottleneck.
27. The first capacity-matched separate implementation had a seed-identity
    confound: it initialized actor backbone, critic backbone, policy head and
    value head in that order, whereas `SharedActorCritic` initializes shared
    backbone, policy head and value head.  Thus its policy head was at a
    different RNG position even though the actor backbone matched.  Those
    partial runs are retained under their original root but marked
    `INVALID_INIT_ORDER`; array `18562456` was cancelled and is not formal
    causal evidence.  The corrected constructor initializes backbone -> policy
    head -> value head in the exact shared order and then uses a no-RNG deep
    copy for the critic backbone.  A direct test verifies bitwise equality of
    actor backbone, policy head, value head and forward outputs (`maxerr=0`),
    equality of the cloned backbone values, and disjoint parameter identities.
    The corrected four-environment array is `18563567_[0-3%4]`, trainer SHA256
    `a0ff7552f4b238d126d5a29e54a906d637fd1ffc1796bb4378ce64bce3fbaa66`;
    BigFish/BossFight/CaveFlyer are running and CoinRun is pending on the
    three-GPU group limit.  The shared spectral job dependency was atomically
    moved to this corrected array before cancelling the superseded one.
28. The corrected array has passed its runtime identity gate and produced real
    metrics.  At the first 4,096-transition rollout, episodic rewards are
    exactly identical to both shared full-joint and actor-only controls in all
    three running environments (BigFish 1.4166666, BossFight 0, CaveFlyer
    5.7142859).  Separate-policy entropy, ratios and behavior KL agree with the
    actor-only trace to floating-point noise; e.g. BigFish ratio min/max are
    bitwise identical and KL differs by only about `2.4e-9`.  Each trace also
    confirms `actor_critic_initialization_mode=shared_exact_then_copy_backbone`,
    1024 rows, 1,460,432 columns per backbone, exact-zero cross block, FP64
    residual near `2e-14`, and no runtime error.  This makes subsequent
    trajectory divergence attributable to the controlled shared-vs-separate
    geometry/value-learning difference rather than seed initialization.
29. Sparse runtime decomposition in the corrected separate run gives the exact
    structural invariants expected from disjoint parameter columns: actor /
    critic component cosine is 0, inverse-metric actor gain from critic RHS is
    exactly 1, actor-alone and joint clip scales are both 1, and cross-block
    Frobenius is exactly 0.  At 36,864 transitions these invariants hold in
    BigFish, BossFight and CaveFlyer with healthy FP64 solves.  This proves that
    the running control is not merely labelled “separate”; its applied actor
    direction has no direct critic-RHS or shared-global-clip contamination.
30. The first matched trajectory divergence identifies a concrete controller
    mechanism.  Around 49k--53k transitions, corrected separate behavior KL is
    0.00010 (BigFish), 0.00042 (BossFight) and 0.00018 (CaveFlyer), while the
    original shared full-joint values at the same transitions are 0.00211,
    0.00138 and 0.00669: roughly 21x, 3x and 38x larger.  Separate entropy and
    reward remain close to actor-only, both use LR 0.05 and the separate
    diagnostic proves actor gain=1/cross=0/no clip.  The extra shared KL must
    therefore come from applying the critic component to policy-producing
    shared parameters, not from a stronger actor solve.  This also explains
    why deleting sample-space cross blocks is insufficient: the block-diagonal
    control still sums actor and critic parameter directions into the same
    trunk and shows elevated KL.  The rollout KL scheduler cannot distinguish
    useful actor movement from critic-induced policy drift; it adjusts the one
    shared LR using their sum.  The remaining long-run and weighted-critic
    controls must determine whether this drift is specifically harmful for the
    GGN critic direction rather than ordinary shared critic learning.
31. Measuring controller KL once on the complete frozen rollout after all four
    epochs is not a repair.  Code audit confirms that the post-epoch control
    recomputes exact categorical KL over all 4,096 rollout observations and
    makes one LR decision, rather than using a final-minibatch estimate.  All
    four runs completed cleanly, but last-10 rewards are BigFish 4.31,
    BossFight 1.85, CaveFlyer 1.10 and CoinRun 6.50 versus 6.95, 1.55, 1.30
    and 7.30 for the matched original controller.  It trades outcomes rather
    than rescuing CaveFlyer.  Hence last-minibatch noise or adjustment timing
    alone is excluded; the controller's deeper problem is that its observed
    policy KL already contains critic-induced shared-parameter movement.
32. The KL/controller divergence persists beyond initialization.  At matched
    119k--123k transitions, separate versus shared full-joint behavior KL is
    0.00157 vs 0.00725 (BigFish), 0.00262 vs 0.00622 (BossFight), and 0.00056
    vs 0.00816 (CaveFlyer).  The scheduler has consequently reduced the shared
    LR to 0.0333, 0.0222 and 0.0148, while separate and actor-only remain at
    0.05.  Shared entropy is already lower, most strongly in CaveFlyer (2.489
    vs separate 2.701).  The shared block-diagonal control shows the same high
    KL / reduced-LR pattern, confirming that deleting Gram cross blocks does
    not remove critic-induced policy drift through the common parameter
    vector.  This is now replicated across three environments and two shared
    kernel constructions rather than being a single early checkpoint.
33. Historical five-seed shared Exact-RAT curves provide supporting, but not
    matched, evidence that generic shared critic learning is viable.  Around
    983k transitions their means are BigFish 9.62, BossFight 1.34, CaveFlyer
    4.31 and CoinRun 6.10; final means over the full 6M schedule are 16.3,
    1.74, 4.66 and 9.56.  All five seeds per environment completed.  These
    runs used the later `7a0698e`-derived Procgen snapshot and its smaller
    architecture / historical controller details, so they cannot prove the
    current DMLP1024 causal comparison.  They do, however, contradict a broad
    claim that any critic gradient through a shared Procgen trunk must fail.
    The queued current-architecture weighted-MSE control remains necessary to
    attribute the present drift specifically to the GGN-preconditioned critic.
34. The corrected source audit makes clear that “sampled B” and deterministic
    stacked `2B` solve different optimization problems, not merely different
    estimators of an otherwise identical update:

    - reference RAT uses one coefficient per sample and
      `H_ref = G_A + 2 diag(xi) J_C`; its B x B kernel includes AA, CC and
      randomized AC/CA terms in their *sum*;
    - stacked GGN uses independent actor/critic coefficient vectors in the
      2B block system `[[GG^T,GJ^T],[JG^T,JJ^T]]` and separately enforces an
      actor advantage RHS and critic residual RHS;
    - RAT reconstructs actor movement with `G_A^T D alpha_adv` only and obtains
      the critic update from ordinary weighted MSE; stacked GGN reconstructs
      the full deterministic `G_A^T alpha_A + J_C^T alpha_C` direction;
    - the reference value-score factor is `2 xi`, so its expected critic
      curvature contribution is `4 J_C^T J_C`, but its critic *objective*
      remains a first-order weighted residual gradient.  Matching only
      `lambda_C` cannot make the two algorithms equivalent.

    Thus the decisive control must preserve the exact combined-score solve and
    loss reconstruction, as job `18563864` now does; a sampled `2B` system is
    not a faithful proxy for original RAT.

## MuJoCo comparison audit

The core Woodbury construction matches:

```text
K = H H^T / B
(K D + mu I) alpha = b
d = H^T D alpha / B
```

However, the successful staged MuJoCo full-joint experiment is not a
hyperparameter-matched control. Its recorded identity is:

- minibatch 1024 (2048-row joint system), 4 epochs;
- `c_C = vf_coef = 4` on critic RHS;
- damping 0.03;
- LR 0.05;
- classic momentum 0.5 or 0.9;
- global L2 clip 0.5.

The current Procgen causal control instead uses `c_C=1`, damping 0.5, no
momentum and a 512-row minibatch. Therefore “MuJoCo works but Procgen does
not” does not yet isolate the environment; regularization scale, RHS scale,
history and sample-space dimension all differ.

## Decisive next control

Run a capacity-matched separate actor/critic model while preserving the full
critic GGN/RHS and the strict 1024-row solve. Independent parameter columns
make the cross block exactly zero without deleting critic representation
learning. Record:

- kernel diagonal min/median/max and damping-to-median ratio;
- damped spectral condition estimate;
- actor-alone and joint clip scales;
- value MSE versus return variance;
- actor/critic self-responses and a validity flag for metric cosine.

Interpretation gate:

- recovery with good value fit => shared-trunk geometry is causal;
- no recovery plus extra actor attenuation => shared global clip is causal;
- no recovery without clip attenuation, but extreme damping/diagonal ratio =>
  regularization-scale mismatch is causal;
- no recovery with reasonable scale and fit => critic-preconditioned update or
  adaptive-KL feedback remains the leading cause.

## Active jobs

- Reconstruction-head array: `18561291_[0-3%4]`, completed.
- Strict initialization-matched separate array: `18563567_[0-3%4]`;
  BigFish, BossFight and CaveFlyer are running, while CoinRun is pending on the
  three-GPU association limit.  Superseded `18562456` is marked invalid and
  cancelled because of its head-initialization RNG-order confound.
- Shared full-joint spectral logging control: `18562717_[0-3%4]`, dependent on
  the capacity-matched separate array.  It preserves the original update and
  adds kernel-diagonal, damping-ratio and damped-spectrum diagnostics, plus
  full-parameter and shared-trunk cosine/norm ratios between the isolated GGN
  critic component and ordinary `J_C^T e_C / B`.  Trainer SHA256 after this
  logging-only addition:
  `1f7eb7e432749822dd3f63dcf05288d52b12caea674852a38d4d4049f42105c1`.
- Matched reference-RAT weighted-MSE critic control: `18563864_[0-3%4]`,
  dependent on the spectral array.  Trainer SHA256
  `b4daa547ad0a25c0d6234a65bc1e9e9a10f554553fe51305b858c31c2b6878c9`;
  config SHA256
  `bbd4a81317d9e46e7c9116c9eac99fd7fb7fac6fde41b28375da49a058dfd773`;
  launcher SHA256
  `9e0c2fa1f6ff488e9c1d9a2eaaaaabc8ea53294dab09b08f98f0e2c9be6ab76b`.
  Superseded pending snapshots `18563018`, `18563141`, `18563824` were cancelled
  at zero elapsed time before allocation.  `18563141` incorrectly used an
  actor-only Gram; `18563824` had the corrected trainer but a config identity
  that still said `clean` while the CLI selected Gaussian reference sampling.
  None produced a run or result.
- Strict separate trainer SHA256:
  `a0ff7552f4b238d126d5a29e54a906d637fd1ffc1796bb4378ce64bce3fbaa66`.
- Strict separate launcher SHA256:
  `fc0c814bf12690baa7207ddee4127c4c962f25021ba8afa6632eb67bcdceb76d`.

## Live strict-separate checkpoint (2026-08-13, about 315k transitions)

The corrected initialization-matched separate control is numerically healthy:
all three allocated A100s are at 100% utilization with about 37.1 GiB used,
there are no OOM/NaN/Traceback/CUDA/Cholesky errors, and the solve residuals
are between `7.8e-13` and `1.5e-10`.  CoinRun remains pending only because the
account currently permits three concurrent A100s.

At exactly 315,392 transitions (last ten rollout-update `eprewmean` values):

| environment | separate reward / H / KL / LR | shared-full reward / H / KL / LR | actor-only reward / H / KL / LR |
|---|---|---|---|
| BigFish | 2.30 / 2.45 / .0048 / .05 | 4.03 / .75 / .0125 / .00988 | 2.33 / 2.51 / .0061 / .05 |
| BossFight | .41 / 2.25 / .0116 / .05 | 1.26 / 1.68 / .0115 / .00988 | 0 / 1.44 / .0086 / .0333 |
| CaveFlyer | 1.79 / 2.11 / .0086 / .0148 | 1.78 / 1.55 / .0103 / .00658 | 3.66 / 1.83 / .0072 / .0222 |

This is not yet a final reward ranking, but it strengthens the mechanism
diagnosis: removing shared parameters preserves much higher entropy and keeps
the adaptive LR much higher in BigFish/BossFight.  Hence the critic update
through the shared trunk creates policy drift that the behavior-KL controller
attributes to the common optimizer.  The block-diagonal sample-space Gram did
not generally repair this because actor and critic directions still write the
same trunk.  The remaining spectral and exact reference-RAT controls are still
required to distinguish an anomalous deterministic-GGN direction from the
broader shared-optimizer/controller interaction.

## Spectral diagnostic v2 queue correction

Before the spectral job allocated, its logging was extended with an exact
policy-Fisher decomposition of the two isolated RHS responses:

```text
q_joint = q_actor + 2 q_cross + q_critic
```

It now records `q_actor`, `q_critic`, the actor/critic cross term, their
reconstruction error against the complete joint direction, and the diagnostic
ratio `q_critic / |q_joint|`.  All operations are deterministic tensor
algebra after the applied direction is formed; they consume no RNG and do not
change parameters, gradients, optimizer state, clipping or KL control.  The
versioned trainer SHA256 is
`11f7cc88d16c1b5600a034c2f76d60079561c341acaf0fb18c16bc7c25d98e9a`.

The old zero-runtime pending spectral/RAT jobs `18562717` and `18563864` were
cancelled and replaced without losing data:

- spectral v2: `18564119_[0-3%4]`, dependent on `18563567_*`;
- exact reference sampled-B RAT: `18564133_[0-3%4]`, dependent on
  `18564119_*`.

Both dependency chains were verified in Slurm.  The spectral v2 launcher has
an immutable trainer-SHA gate; its SHA256 is
`297ed0c4b0c4901472e17c9534375073d5d8dc06483e553eaeb27d277c2d2f6d`.

## Correction: RHS drift versus curvature geometry

The older sparse full-joint decomposition already contains enough evidence to
reject the simplest version of the earlier explanation.  Up to 430,080
transitions, the median isolated critic-RHS actor-Fisher quadratic divided by
the complete joint actor-Fisher quadratic is only:

| mode | BigFish | BossFight | CaveFlyer |
|---|---:|---:|---:|
| full joint | 0.17% | 0.92% | 0.02% |
| block diagonal | 22.3% | 38.0% | 32.9% |

The full-joint cross blocks therefore usually *suppress* the direct policy
effect of the critic RHS; deleting them can expose a much larger critic-driven
policy component.  The shared full run still has much higher mean rollout KL
and faster Fisher/entropy collapse than the separate network, but this cannot
be attributed simply to a large additive critic-RHS direction.  The leading
mechanism is now that critic curvature rows change the actor preconditioner
and its optimizer/controller trajectory, with an environment-dependent role
for critic learning.

The completed curvature-only (`lambda_C=1, c_C=0`) trajectories support that
environment dependence:

| environment | curvature-only final last-10 reward | full joint last-10 reward |
|---|---:|---:|
| BigFish | 2.54 | 6.95 |
| BossFight | 0.015 | 1.55 |
| CaveFlyer | 4.57 | 1.30 |
| CoinRun | 4.64 | 7.30 |

BigFish/BossFight/CoinRun need critic learning, while CaveFlyer is specifically
harmed when the critic RHS is added to otherwise useful critic curvature.
CoinRun has a complete 1,007,616-transition trace but its outer lifecycle file
was left `RUNNING` by the known shared-launcher overwrite, so it is trajectory
evidence rather than a formal lifecycle PASS.

## CaveFlyer separate-network timeline and spectral v3

The strict separate control makes the critic-learning feedback visible without
any parameter-sharing path.  CaveFlyer evolves as follows:

| transitions | reward | entropy | categorical Fisher trace | behavior KL | LR | value MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 98,304 | 1.20 | 2.70 | .933 | .00037 | .05 | .208 |
| 249,856 | 2.33 | 2.49 | .903 | .0283 | .05 | .0336 |
| 499,712 | 1.90 | 1.06 | .446 | .0114 | .00658 | .0292 |

At 499,712 transitions, the matched critic-frozen curvature-only control is at
reward 3.80 and entropy 2.08.  Thus critic learning can trigger the CaveFlyer
collapse through the value/GAE/advantage-data path even when actor and critic
parameters are disjoint.  Shared trunk coupling accelerates this effect but is
not required for it.

The spectral diagnostic was therefore upgraded once more before allocation.
On each sparse diagnostic minibatch it now solves the exact actor-only
512-row counterfactual using the same `H_pi`, advantage, ratio, damping and
solve dtype, and compares that direction to the actor-RHS component from the
full 1024-row system.  This directly isolates how critic curvature/cross blocks
change the actor preconditioner independently of critic RHS.  No RNG or
training state is changed.

- spectral v3 job: `18564386_[0-3%4]`, trainer SHA256
  `6deda39584d8570e045ec784983695acb45a3d6fe68120a645ffca81693fe8be`;
- launcher SHA256:
  `b551ebddbf12e6f677dd672ecaec04733ed3c5910eb581ba227e3f8b79dea5cc`;
- exact reference sampled-B RAT successor: `18564393_[0-3%4]`.

The superseded v2 spectral/RAT jobs `18564119`/`18564133` were cancelled at
zero elapsed time.  Slurm dependencies for the v3 chain were reverified.

The initial v3 chain still used a whole-array wildcard dependency, which would
have forced BigFish/BossFight/CaveFlyer to wait for the later-starting CoinRun.
It was replaced at zero elapsed time with corresponding-element dependencies:

- spectral v3: `18564677_[0-3%4]`, `aftercorr:18563567`;
- reference sampled-B RAT: `18564681_[0-3%4]`, `aftercorr:18564677`.

Each environment can now advance to the next diagnostic as soon as the same
environment in the preceding array succeeds.  Slurm expanded and verified all
four element-wise dependencies.  Superseded `18564386`/`18564393` produced no
run or result.

## Advantage-scale feedback hypothesis

The separate CaveFlyer trace reveals a second mechanism after critic fitting.
Raw minibatch GAE variance falls from `.859` at 98k to `.166` at 250k and
`.0361` at 500k, while value MSE falls from `.208` to about `.03`.  However,
`Advantage_Update` recenters and RMS-normalizes every shuffled minibatch in
every optimizer epoch.  Consequently the actor RHS norm remains exactly
`sqrt(512) = 22.627` at all three points even though the raw residual signal
has shrunk roughly 24-fold.

This creates a plausible feedback loop:

```text
critic fit improves -> raw GAE residual shrinks -> minibatch RMS normalization
restores full actor-RHS magnitude -> residual ranking/noise receives a large
natural-gradient step -> KL spike -> one-rollout-late LR reduction -> lower
entropy/Fisher trace and degraded exploration
```

It is not yet sufficient as a global explanation: BigFish also reaches small
raw advantage variance and continues improving, while BossFight benefits from
critic learning.  The exact sampled-B RAT control is necessary because it uses
the same advantage normalization; recovery there would show that normalization
becomes harmful specifically with the strict joint preconditioner/critic
update, rather than being independently wrong.

## Reference sampled-B RAT implementation audit

The queued reference control was checked line-by-line against Desktop
`trust-region/train_shared.py` and with a toy shared-network autograd identity.
It matches the public-code semantics:

- sampled per-example score `H_i = G_{A,i} + 2 xi_i J_{C,i}` with
  `xi_i ~ N(0,1)`;
- `K = H H^T / B` and `(K diag(ratio) + mu I)`;
- the advantage and all-ones RHS are solved against the same matrix;
- actor reconstruction uses only `G_A^T diag(ratio) alpha_A / B`;
- critic minimizes `mean(alpha_C * (v-ret)^2)` with no second ratio factor;
- actor and critic descent gradients are combined and globally L2-clipped once.

For a float64 toy shared network (`B=11`), the queued manual construction and
the literal public-code combined-loss backward agree to absolute error
`1.85e-16`, relative error `9.46e-17`:

```text
REFERENCE_RAT_FULL_BACKWARD_EQUIV ... B 11 rhs 2 score_scale 2
```

Thus a difference between job `18564393` and strict 2B cannot be dismissed as
an actor/critic sign, factor-of-two, ratio-placement or reconstruction bug in
the reference control.

## Current corresponding-element pipeline and historical reference caveat

The active immutable pipeline is now:

- initialization-matched separate actor/critic: `18563567_[0-3%4]`;
- spectral-v3 strict shared 2B: `18564677_[0-3%4]`, element-wise
  `aftercorr:18563567`;
- public-code-equivalent sampled-B RAT: `18564681_[0-3%4]`, element-wise
  `aftercorr:18564677`.

At 708,608--716,800 transitions, the three allocated separate controls remain
finite and error-free.  Their latest point rewards are BigFish `3.45`,
BossFight `0.50`, and CaveFlyer `2.10`; entropies are `.863`, `1.238`, and
`.603`, and solve residuals remain between `6.8e-12` and `1.7e-10`.
CoinRun is still waiting only because the account currently permits three
concurrent A100 jobs.  These are not final results, but they already show that
removing all shared actor--critic columns merely delays, rather than removes,
the CaveFlyer entropy/Fisher collapse.

At the matched 815,104-transition checkpoint, using the last ten *distinct
rollout transitions* (not the last ten minibatch records), the comparison is:

| environment | separate reward / entropy / Fisher | actor-only | curvature-only | shared full |
|---|---:|---:|---:|---:|
| BigFish | 3.47 / .887 / .394 | 2.43 / 1.468 / .542 | 2.60 / 1.204 / .504 | 4.34 / .345 / .169 |
| BossFight | 1.72 / 1.624 / .685 | 0.00 / 2.173 / .776 | .04 / 1.187 / .476 | 2.01 / 1.180 / .515 |
| CaveFlyer | 1.43 / .187 / .097 | 2.76 / 1.392 / .539 | 4.08 / 1.683 / .620 | 1.70 / .267 / .139 |

This is a sharper causal split than a global method ranking.  Independent
critic learning is beneficial on BossFight, somewhat useful on BigFish, and
catastrophic on CaveFlyer.  CaveFlyer collapses to almost the same low-entropy
state as shared full even though its cross block is exactly zero.  Shared
joint geometry therefore accelerates some collapses but is not a necessary
cause; the learned critic's effect on the rollout/GAE/normalized-advantage
data stream is independently sufficient on CaveFlyer.

The older five-seed public Exact-RAT curves are useful as a capability check,
not yet a strict causal control.  Near 1.02M transitions their means are about
BigFish `8.95`, BossFight `1.31`, CaveFlyer `4.66`, and CoinRun `6.04`.
However, that published-style batch used initial/max LR `.5`, momentum `.1`,
and the original 256-wide decision representation, whereas the current causal
chain fixes initial LR `.004`, max LR `.05`, momentum `0`, and DMLP1024.
Therefore the queued sampled-B control intentionally keeps the *current*
hyperparameters and architecture.  It first answers whether the strict-2B
geometry/update is the cause.  A later public-hyperparameter control is needed
only if sampled-B also fails under the matched settings.

That fallback is staged but deliberately not submitted.  It keeps DMLP1024
and the identical sampled-B/weighted-critic algorithm while restoring only
the historical Procgen optimizer/controller values: initial/max LR `.5`, KL
upper `.04`, and PyTorch SGD momentum `.1`.  A versioned trainer was required
because the current controlled branch intentionally accepted only momentum
`0` or `.9`; the fallback changes only this validation whitelist to include
`.1` and otherwise differs byte-for-byte from the audited reference trainer.

- fallback trainer:
  `train_shared_rat_weightedcritic_reference_publicmomentum.py`, SHA256
  `14eccfbe7de1a971fe7f4dbeec6d3d74c9d732b3dae3035c84ed50321127bb1e`;
- fallback config SHA256:
  `954a27f46564861f1d22267f09ea953fbe98148ca1a7ab5710cead83aebea826`;
- fallback launcher SHA256:
  `0941bb3b865e3197677d9b9e37a76d3f96755f6362729b2ee4f50b5f01b10d23`.

All three artifacts are staged and hash-verified on CSF3; no fallback job has
been submitted, so it cannot consume capacity or contaminate the current
causal chain.

## Initialization-matched separate control: first terminal results

BigFish and CaveFlyer completed formally (`PASS`, `rc=0`, Slurm
`COMPLETED`) at 1,007,616 transitions with no OOM, NaN, traceback, CUDA,
Cholesky or nonfinite error.  Distinct-rollout last-10 and terminal diagnostics
are:

| environment | last-10 reward | final reward | entropy | Fisher trace | behavior KL | LR | solve residual | value MSE | raw advantage variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BigFish | 3.58 | 3.84 | .472 | .237 | .00844 | .00658 | 3.96e-10 | .0156 | .0197 |
| CaveFlyer | 1.27 | 1.40 | .116 | .066 | .01168 | .00658 | 8.18e-11 | .0410 | .0493 |
| BossFight | .954 | 1.10 | 1.208 | .545 | .00969 | .00658 | 2.34e-10 | .0104 | .0137 |

The CaveFlyer result is decisive against the hypothesis that shared
actor--critic parameter columns or explicit cross blocks are necessary for
collapse: both are absent exactly, yet entropy and categorical Fisher still
collapse and reward is far below the matched actor-only/curvature-only
controls.  The remaining causal path is the independently learned critic
changing returns/GAE and hence future normalized actor data.  This does not
yet prove that path is the *only* defect in shared strict 2B; spectral v3 will
measure the additional same-minibatch preconditioner distortion.

BossFight subsequently also completed formally (`PASS`, `rc=0`, Slurm
`COMPLETED`) without scanned errors.  Its critic fits extremely closely while
raw advantage variance becomes `.0137`; nevertheless it retains much more
entropy/Fisher than CaveFlyer.  The feedback is therefore environment
dependent, not a universal consequence of low critic error alone.

The separate control still has one possible residual coupling: actor and
critic directions pass through one global Euclidean clip.  The sparse
component diagnostics rule this out as the primary CaveFlyer mechanism.
Compared with clipping the actor component alone, the combined direction
attenuates actor on only 8.5% of CaveFlyer diagnostic minibatches; the mean
extra scale ratio is `.9981`, the minimum `.7993`, and only one of 246
diagnostics is below `.8`.  BigFish is affected on .4% of diagnostics;
BossFight on 10.2% with mean ratio `.9883`.  Thus joint clipping exists but is
too rare and too small on average to explain CaveFlyer's sustained collapse.

## Spectral-v3 first real diagnostics

Corresponding elements 0/1 started on A100 (`18564677_0` BigFish and
`18564677_1` BossFight) after their separate controls completed.  Preflight,
trainer SHA `6deda395...8be8be`, 4096/512/4, damping `.5`, clean full
1024-row system and real metric records all match.  The component quadratic
reconstruction error is already below `6e-7`, validating the logged
decomposition numerically.

During the first 12,288 transitions, the actor-RHS direction under the full
strict-2B metric is almost identical to the exact same-minibatch actor-only
counterfactual:

- BigFish: cosine `>= .999998`, norm ratio within `7e-6` of one after the
  first rollout, delta fraction `.00013--.00181`;
- BossFight: after a first-minibatch delta `.0693`, cosine is `>= .999998`,
  norm ratio within `4e-6` of one, delta fraction `.00066--.00206`.

The direct critic-RHS policy gain is likewise near one and the immediate
metric actor/critic cosine is near zero.  At initialization, therefore,
critic rows and cross blocks do **not** materially distort the actor natural
direction.  The isolated critic GGN direction is generally aligned with the
ordinary critic gradient but damped: BigFish cosine about `.999`, norm ratio
`.60--.61`; BossFight quickly reaches cosine `.993--.994`, norm ratio `.618`
(the very first minibatch is less aligned).  Later checkpoints remain
necessary because the hypothesized defect may emerge only after entropy,
Fisher and advantage scale change.

## Minimal intervention queued after the identity controls

The next causal intervention changes exactly one training operation in strict
shared 2B: it keeps minibatch advantage centering but disables the extra RMS
division.  PopArt/rollout normalization, critic GGN/RHS, 1024-row geometry,
cross blocks, damping, controller, clipping, architecture and all batch sizes
remain unchanged.  It logs both the pre-normalization RMS and whether the
optional normalization ran, so identity can be verified from every metric.

- trainer `train_shared_joint2b_no_extra_adv_rms.py`, SHA256
  `0a145a7d904ef283016d25641500074a72c4fed541161e574b8f609f19a6d26a`;
- config SHA256
  `8782f7788e08d7be07a85fd4eb235941303b8a5528065fd5416479c928a3a4c3`;
- launcher SHA256
  `b56bf041d35962f68d084470361690bbb9975def584b63ab3ba214d4aa382c87`;
- Slurm array `18566246_[0-3%4]`, corresponding-element dependency
  `aftercorr:18564681`.

The source diff against spectral v3 contains only the optional advantage RMS
branch and two logging fields; both local and remote `py_compile`, launcher
syntax, and hashes passed.  The job cannot start before the same environment's
matched sampled-B RAT completes, preserving the one-factor causal order.
# 2026-08-13 head-only and scale diagnosis update

- `18552840_[12-15]` head-only strict joint completed all four environments
  with `PASS`, `rc=0`, `joint_system_rows=1024`,
  `critic_param_scope=head_only`, and `cross_block_fro=0`.
- Final last-10 reward means were BigFish 1.376, BossFight 0.024,
  CaveFlyer 0.000, and CoinRun 0.220.  Therefore eliminating shared-trunk
  critic columns and all actor-critic cross blocks does not rescue CaveFlyer.
- The earlier whole-rollout post-epoch KL control (`18559260_[24-27]`) was
  already complete.  Its CaveFlyer last-10 reward was 1.210, versus 1.540 for
  the clean control, so final-minibatch KL noise is not the primary failure.
- Reducing only the critic objective/RHS coefficient gave a monotonic rescue
  on CaveFlyer: `c_C=0.1` last-10 1.715 and `c_C=0.03` last-10 3.874, while
  the curvature-only control was 4.570.  This is the strongest current causal
  evidence for critic drive/curvature/damping scale mismatch rather than a
  numerical solve failure, controller timing, or shared cross block.
- Spectral v3 diagnostics independently show fixed damping 0.5 falling from
  hundreds of median kernel diagonals to roughly 0.01--0.06 while the joint
  actor direction rotates away from the same-minibatch actor-only direction.
- A minimal scale repair is now queued as job `18566456_[0-3]`, dependent on
  matched sampled-B RAT job `18564681`: it preserves the full clean 2B system,
  RHS, advantage normalization, and all cross blocks, but uses
  `effective_damping=max(0.5, median(diag(KD)))`.  Trainer SHA is
  `99c8c6e4d7489eb4c507d00d3a0af081c4a2b49e8e112134658f1d38e69cfbd1`.
- The live spectral-v3 relationship is strong rather than anecdotal.  Across
  the first ten diagnostic checkpoints (through 372,736 transitions), the
  correlation between `log10(damping / median_diag)` and the full-vs-actor
  direction delta was -0.925 on BigFish and -0.987 on BossFight; correlation
  with actor-direction cosine was +0.848 and +0.953.  Once the ratio fell
  below one, mean direction deltas were 0.634 and 0.673.  Solver residuals
  stayed around 1e-11--1e-10, so this is geometric scale drift, not an
  inaccurate linear solve.
- A concrete source for this scale drift is now under test: Procgen calls
  `PopArt.update(ret)` every rollout, which rescales the normalized value-head
  weight by `old_std/new_std`.  Since the strict critic rows are Jacobians of
  that normalized value output, this reparameterization directly changes
  `J_v J_v^T` and hence the meaning of a fixed absolute damping.  The pending
  relative-damping trainer now logs PopArt mean/std and value-head weight norm
  without changing the algorithm; its updated SHA is
  `8d9d8dff8635a6020b72ab3f08a567590c4d5e52dc24aa19bcea72bcb49d8730`.
- To avoid waiting for the full A100 dependency chain, an isolated CaveFlyer
  H200 causal probe of the identical relative-damping code was submitted as
  `18566659_2`.  It has a separate result root and is explicitly labelled as
  hardware-mismatched preliminary evidence; the formal A100 four-environment
  job `18566456_[0-3]` remains queued.

## Relative-damping probe and late spectral evidence

The CaveFlyer H200 relative-damping probe has now passed 372,736 transitions
without OOM, nonfinite values, Cholesky failure, or an inaccurate solve.  At
that matched checkpoint it has reward `3.43`, entropy `1.903`, behavior KL
about `.0082` at the rollout-summary point, and solve residual `6.7e-14`.
The clean fixed-damping A100 control at the same transition has reward `1.20`
and entropy `1.419`; `c_C=.03`, actor-only, and curvature-only have rewards
`2.70`, `2.50`, and `5.10`, respectively.  Because this probe is on H200, the
reward comparison is preliminary, but it is already a positive causal signal
for scale-aware damping rather than a formal hardware-matched result.

The intervention is materially active rather than nominal.  On CaveFlyer the
kernel diagonal median grows from `.00228` initially to `64.27` at 372,736;
the effective damping therefore grows from `.5` to `64.27`.  At 372,736 the
full-vs-actor-only direction cosine is `.9767` and relative direction delta is
`.2148`, while the fixed-damping spectral controls show progressively larger
distortion as `0.5 / median(diag K)` shrinks.  By about 598k transitions the
fixed-damping BigFish/BossFight ratios are `.00207` and `.000131`, direction
deltas `.750` and `1.204`, and cosines `.726` and `.525`, despite residuals in
the `1e-10` range.  This strongly separates geometric scale drift from linear
solver error.

Head-only also reveals the longer-horizon feedback path explicitly.  Its
CaveFlyer cross block is identically zero, yet final raw normalized advantage
means are roughly `-1.45` to `-1.50`, variances `.011--.023`, reward zero, and
entropy about `.39--.60` in the final summaries.  In contrast `c_C=.03` ends
near reward `4.17` with advantage variance still intermittently nontrivial.
The actor update centers each minibatch, so the negative mean itself is not a
direct policy-gradient bias; the important remaining issue is that a
near-fitted critic can leave a tiny residual advantage and the second RMS
normalization restores that residual/noise to unit actor-RHS scale.

Two fast CaveFlyer H200 probes are now serialized after the relative-damping
probe to distinguish the remaining alternatives before the formal A100 chain
finishes:

- `18566766_2`: strict clean 1024-row joint 2B with only the second
  per-minibatch advantage RMS division disabled;
- `18566767_2`: matched public-code-equivalent sampled-B RAT with Gaussian
  value score, 512-row combined kernel, and two RHS columns.

Both use separate result roots, immutable trainer/config hashes, and are
tagged as H200 causal probes rather than formal A100 comparisons.

The relative-damping H200 probe has since completed `PASS`, `rc=0`, at
1,007,616 transitions.  CaveFlyer last-10 reward is `4.10`, final reward
`4.70`, last-10 entropy `1.738`, and last-10 behavior KL `.00956`.  The
matched A100 controls have last-10 reward/entropy `1.54/.155` for fixed clean,
`3.874/1.085` for `c_C=.03`, `2.093/1.459` for actor-only, and
`4.57/1.944` for curvature-only.  At the final relative-damping minibatch the
kernel median is `185.79`, effective damping is `185.79`, and the original
fixed `.5` is only `.00269` times that median; solve residual remains
`1.0e-13`.  Although the hardware mismatch still requires the formal A100
replication, this is now a complete intervention result rather than an early
curve: maintaining a fixed damping-to-kernel ratio prevents the CaveFlyer
entropy collapse and restores reward to the low-critic-drive range.

After this completion, `18566766_2` started automatically on the same H200.
Its first real metrics verify `joint_system_rows=1024`, clean full joint,
`normalize_advantage_rms=0`, and an unnormalized actor RHS norm of `15.73`
when the pre-normalization advantage RMS is `.695`; no launch or numerical
error is present.  `18566767_2` remains correctly dependent on its successful
completion.

The logging-only spectral-v3 controls have now completed BigFish and
BossFight.  Their final kernel medians are `603.0` and `8265.8`, making fixed
damping `.5` only `8.29e-4` and `6.05e-5` of the median diagonal; final
entropies are `.231` and `.299`.  Thus the loss of fixed-damping scale is not
specific to CaveFlyer.  The remaining three H200 relative-damping probes were
queued as `18566946_[0,1,3%1]` after the sampled-B CaveFlyer probe.  They reuse
the exact immutable relative-damping trainer/config and result root while
excluding the already complete CaveFlyer directory; the serial `%1` limit
respects the current H200 account concurrency.  These remain preliminary
hardware-mismatched cross-environment evidence pending formal A100 job
`18566456`.

At roughly 549k transitions the no-extra-RMS CaveFlyer probe still has reward
`2.9` (recent-10 `3.126`) but entropy has fallen to `.685` (recent-10 `.854`).
Its actor RHS has contracted to `4.78`, yet kernel median is `525.8` and fixed
damping is only `9.51e-4` of that median.  Removing the RMS therefore provides
useful RHS self-scaling but does not stop the geometric scale loss; it is not
a sufficient repair by itself.

To test whether it remains a secondary amplifier after fixing geometry, a
strict combined CaveFlyer probe was added without modifying either existing
trainer: relative median damping plus no extra minibatch advantage RMS.
Trainer/config/launcher SHAs are respectively
`1cc005f0ad21ab4870eab1ed73a7d95ea833d2eceeffbd6a531a0af6b7c6ed0d`,
`16bb5eb190fa48cf9cb1244d2e5d67458269554348cf7c322744b631d90b3c5f`,
and `51d6a02b0d6edcb2a5b185f9ce5eb8974e7d5bcae98649777dfbbabef365e1a6`.
Local and remote compile, syntax, and hash checks passed.  H200 job
`18567006_2` depends on successful matched sampled-B job `18566767_2`; the
remaining relative-damping environments `18566946_[0,1,3%1]` were updated to
depend on the combined probe so the one-GPU causal order is unambiguous.

The mechanism is more precise than "the critic simply makes steps too big."
Using the per-minibatch proxy `lr * sqrt(d^T F_actor d)` at matched CaveFlyer
transitions, fixed clean decays from `.0106` at 102k to `.00207` at 754k;
no-extra-RMS is only `.002--.005` in the later phase, while relative damping
stays about `.010--.013` from 204k through 754k.  Relative damping therefore
does not merely shrink all updates.  It prevents the growing critic/cross
geometry from rotating and suppressing the actor direction; the KL controller
can then retain a high LR and a nonvanishing policy-space step.  Fixed clean
and no-extra-RMS both lose effective policy movement as entropy collapses.

The no-extra-RMS CaveFlyer probe has now completed `PASS`, `rc=0`.  Its
last-10 reward is `2.44` versus `1.54` for fixed clean, but last-10 entropy is
only `.272` (final `.148`) versus `1.738` for relative damping.  Final
advantage RMS is `.104`, actor RHS norm `2.35`, kernel median `362.9`, and
fixed damping/median `.00138`.  Thus retaining the natural advantage scale
modestly improves reward but does not prevent entropy collapse; it is a
secondary mitigation, not the primary repair.

Matched sampled-B H200 job `18566767_2` then started successfully.  Its first
real minibatch verifies the intended public-code-equivalent identity:
`joint_system_rows=512`, Gaussian value-score noise std `.987`, two RHS
columns, `rat_reference_combined_sampled_b`, policy-score-only actor
reconstruction, weighted-MSE critic backward, momentum zero and no
Kaczmarz.  Actor and critic solve residuals are `1.7e-14` and `2.5e-14`.

At 196,608 transitions the sampled-B probe remains numerically healthy but
is not yet a quality result: reward is `1.23`, entropy `2.665`, behavior KL
`.000777`, LR `.05`, and solve residual `3.99e-13`.  Its raw joint direction
is `6.46` and is globally clipped to `.5`; the matched effective actor-space
step proxy `lr*sqrt(d_eff^T F_A d_eff)` is `.0410`.  At the same transition
that proxy is `.00615` for fixed strict-2B, `.00317` for no-extra-RMS,
`.02134` for relative damping, `.01451` for actor-only, and `.01679` for
curvature-only.  Thus sampled-B has so far retained entropy through a much
more aggressive clip-dominated policy update, not by simply reproducing a
better-conditioned version of the strict-2B update.  A full 1M trajectory is
required before deciding whether its 512-row combined formulation avoids the
late collapse.

The formal A100 chain also advanced: spectral-v3 CaveFlyer task
`18564677_2` is running on node857, while sampled-B BigFish/BossFight tasks
`18564681_[0-1]` are runnable but still pending resources.  No OOM, NaN,
Traceback, CUDA, or Cholesky error is present in the H200 sampled-B probe.

The separate-actor/critic capacity-matched A100 control is now complete in
all four environments, with `joint_system_rows=1024` and exactly zero
actor--critic cross block.  Last-10 reward/entropy are BigFish
`3.58/.466`, BossFight `.954/1.039`, CaveFlyer `1.27/.125`, and CoinRun
`5.16/.194`.  Final fixed-damping-to-kernel-median ratios are respectively
`.0139`, `.000246`, `.00476`, and `.000324`.  CaveFlyer and CoinRun therefore
still undergo strong entropy collapse even when parameter sharing and the
explicit cross block are mathematically absent.  This independently rules
out shared-trunk coupling as the primary cause and strengthens the geometric
scale/controller explanation: a fixed `.5` damping becomes negligible for
the growing per-sample kernel even in a block-diagonal separate network.

The separate CaveFlyer trajectory also supplies the temporal ordering.  In
100k-transition bins, fixed damping divided by the kernel median falls from
about `.651` in the 200k bin to `.00512` in the 300k bin.  Over the same
transition the mean LR drops from `.0364` to `.0107`, entropy drops from
`2.434` to `1.869`, and it subsequently decays to `.100` by the endpoint;
the effective actor-space step falls from `.0124` toward `.0026`.  The
relative-damping probe holds the damping/median ratio at exactly `1` after
the initial floor-controlled phase, keeps mean LR near `.035--.05`, and
maintains an effective actor-space step around `.011--.016` with endpoint
entropy near `1.70`.  This ordering is consistent with geometry-scale loss
preceding controller slowdown and entropy collapse, rather than entropy loss
being the original cause of the kernel change.

Formal A100 spectral-v3 CaveFlyer now reproduces the same mechanism within
the first 100k transitions.  At 86,016 transitions the kernel median is
`17.52`, fixed damping/median is `.02855`, and the damped sample-space
condition estimate is `1.95e4`.  The full-joint actor direction relative to
the actor-only metric direction has already fallen to cosine `.8785` with
relative difference `.489`; solve residual remains numerical-noise small.
By 90,112 transitions LR is `.0148` and entropy `2.164`.  At the same
transition the H200 relative-damping intervention has effective
damping/median `1`, LR `.05`, and entropy `2.504`.  The exact values are not
a hardware-matched performance comparison, but the A100 trajectory verifies
that rapid geometry growth and direction distortion are not H200-specific.
Across the first 42 valid CaveFlyer diagnostic updates through 172,032
transitions, Pearson correlation between `log10(damping/kernel-median)` and
the full-vs-actor-only direction cosine is `+0.831`; correlation with the
relative direction difference is `-0.942`.  At 167,936 transitions the ratio
is `.00587`, damped condition estimate `4.11e4`, cosine `.858`, and relative
direction difference `.529`.  This is the third environment showing the
same monotone scale-to-direction relationship after BigFish and BossFight.

The sampled-B trajectory exposes a distinct reason that its high entropy
must not yet be interpreted as successful learning.  At 524,288 transitions
the raw joint direction norm is `7.02`, decomposed as actor `.472` and critic
`6.98`, so the critic-dominated vector is globally clipped to `.5`.  LR is
already capped at `.05`, yet behavior KL is only `.000390` and reward `2.09`.
Its current-step KL is `.000543`, larger than relative damping's `.000131` at
the same transition, while relative damping has accumulated behavior KL
`.0142`.  A plausible mechanism is therefore temporal cancellation: fresh
Gaussian critic scores perturb the actor direction through the combined
sampled-B kernel at every minibatch, producing nontrivial individual steps
that do not build a coherent displacement from the fixed rollout policy.
This explains the current combination of preserved entropy, maxed LR, tiny
behavior KL, and slow reward growth, but remains an inference until the full
trajectory (or a temporal-direction-cosine diagnostic) is available.

The matched H200 sampled-B trajectory has now completed `PASS`, `rc=0`, at
1,007,616 transitions.  Last-10 reward is only `1.142`, entropy `2.683`, and
behavior KL `.000268`; final LR is capped at `.05`.  Final raw joint/actor/
critic direction norms are `5.60/.381/5.57`, again globally clipped to `.5`,
with actor and critic solve residuals `1.34e-13/3.31e-13`.  It therefore
avoids collapse but is worse in reward than fixed clean (`1.54`) and far
below relative damping (`4.10`) or curvature-only (`4.57`) at the matched
endpoint.  This complete result supports the qualitative diagnosis that the
public sampled-B update preserves entropy by making incoherent,
critic-clip-dominated policy movements rather than repairing the deterministic
joint geometry.

The combined relative-damping plus no-extra-adv-RMS H200 job `18567006_2`
started automatically after sampled-B and has emitted real metrics.  Its
preflight and first 28,672 transitions verify both interventions together:
1024 rows, clean full joint, median-diagonal damping floor `1`, centered but
not RMS-normalized advantages (`normalize_advantage_rms=0`), rollout
4096/minibatch512/four epochs, no momentum/Kaczmarz.  At 28,672 transitions
the kernel median is still below the `.5` floor, entropy is `2.707`, LR is
`.0456`, reward `3.26`, and solve residual `7.36e-16`; no runtime error is
present.  This is early identity/health evidence, not yet a performance
conclusion.

By 81,920 transitions the combined probe has entered its adaptive-damping
regime: kernel median and effective damping are both `1.734`, while solve
residual remains `2.36e-13`.  However, removing RMS normalization is not
obviously additive.  Its actor RHS is `18.9`, raw joint direction `.629` is
clipped to `.5`, behavior KL is `.0182`, and raw ratio range is
`.379--7.19`; reward is `1.10`, entropy `2.678`.  At the same transition the
single relative-damping run has kernel median/effective damping `11.14`, raw
direction `.146` without clipping, behavior KL `.0131`, ratio range
`.504--2.38`, reward `2.30`, entropy `2.524`.  The no-extra-RMS trajectory has
therefore altered the evolving kernel spectrum enough that a median-matched
scalar damping still permits a much larger and more extreme early update.
This is a risk signal, not an early-stop decision; the run is below 100k and
has no hard failure.

The early spike did not persist.  By 184,320 transitions the combined run's
raw direction is `.0718` (no global clipping), effective policy-step proxy is
`.01505`, and behavior KL `.00966`; the single relative-damping run at the
same transition has `.0545`, `.01605`, and `.00655`.  Combined entropy is
higher (`2.542` versus `2.349`) while reward is lower (`2.13` versus `2.90`).
Thus the 82k ratio excursion was a transient during rapid RHS/kernel
rescaling, not evidence of sustained instability.  The two methods now have
similar policy-space step magnitude, so the remaining trajectory can test
whether retaining natural advantage RMS changes learning quality after the
primary geometry repair.

At 245,760 transitions the combined-versus-relative comparison is becoming
directional.  Both have matched effective damping to kernel median
(`48.7` and `51.8`) and LR `.05`, but the natural-scale combined RHS is
`16.59` at advantage RMS `.733`, versus normalized RHS `22.63`.  Its
effective policy-step proxy is consequently `.0111` versus `.0135`, entropy
is higher (`2.441` versus `2.133`), and reward lower (`2.10` versus `3.33`).
This suggests that once geometry is repaired, RMS normalization is useful
for maintaining a consistent learning rate in policy space; removing it
mostly slows exploitation while preserving more entropy.  This remains a
mid-trajectory observation pending the endpoint.

At 307,200 transitions the scale mechanism is explicit rather than merely a
reward difference.  The no-RMS run's current advantage RMS has contracted to
`.151`, actor RHS to `3.41`, effective policy-step proxy to `.00290`, and
behavior KL to `.000796`, despite LR `.05` and correctly matched damping
`49.61/49.61`; reward is `2.10`, entropy `2.483`.  The normalized relative
run at the same transition has fixed actor RHS `22.63`, effective step
`.01446`, behavior KL `.01006`, reward `3.70`, and entropy `2.216`, with
matched damping `55.24/55.24`.  Thus extra RMS normalization is not the cause
of strict-2B collapse.  Once geometric scale is repaired, it prevents the
policy-space learning rate from inheriting large, state-dependent swings in
the raw advantage RMS.  Removing it intermittently stalls the actor even
though the KL controller has already saturated LR at its upper bound.

At 364,544 transitions the no-RMS run's instantaneous reward has caught up
(`3.30` versus `3.33`) and entropy remains higher (`2.479` versus `2.077`),
but its update remains much weaker: advantage RMS `.180`, actor RHS `4.08`,
behavior KL `.000892`, and effective policy-step proxy `.00429`, versus
normalized relative damping's RHS `22.63`, KL `.01194`, and step `.01111`.
Both have LR saturated at `.05` and exactly median-matched damping.  The
controller therefore cannot compensate for the missing RMS normalization
because it has no LR headroom.  Any endpoint equality would represent slower,
higher-entropy learning, not evidence that normalization caused the original
failure.

At 417,792 transitions this interpretation persists under trailing windows:
last-10 reward is `2.97` versus normalized relative damping's `3.253`, entropy
`2.349` versus `2.065`, and behavior KL `.00237` versus `.01185`.  The current
effective policy-step proxy is `.00754` versus `.01950`; both maintain exact
median-matched damping and LR `.05`.  Removing RMS has not reintroduced
geometric instability, but it produces a slower, higher-entropy trajectory
because raw advantage scale directly gates actor movement.

At 483,328 transitions the last-10 gap widens again: no-RMS reward `1.79`,
entropy `2.307`, behavior KL `.00201`, and effective policy-step `.00516`,
versus normalized relative damping's `2.914`, `1.989`, `.01151`, and
`.01389`.  Damping remains exactly median-matched and residual is `1.8e-14`,
so this is not recurrence of the joint-geometry failure.  It is continued
under-updating caused by natural advantage RMS (`~.30` over the recent
window) directly scaling the actor RHS while LR is capped.

Formal A100 sampled-B BigFish has reached 204,800 transitions and reproduces
the completed H200 mechanism: LR `.05`, last-10 entropy `2.701`, last-10
behavior KL only `4.33e-5`, raw direction `9.10` with actor `.179` and critic
`9.07`, clipped to `.5`.  Thus the sampled-B critic-clip dominance and tiny
cumulative movement are hardware-independent rather than an H200 artifact.

Formal A100 sampled-B BigFish task `18564681_0` has obtained node853 and
written a valid immutable preflight and its first 32 real minibatch metrics,
so it is now a verified active result.  The preflight matches the
H200 probe: rollout `4096`, minibatch `512`, four epochs, 1M transitions,
damping `.5`, no momentum/Kaczmarz, 512-row combined randomized-score
system, two RHS columns, unit-Gaussian value score, policy-only actor
reconstruction, and reference-RAT weighted-MSE critic backward.  Trainer and
config hashes match the H200 probe (`b4daa...8730`, `bbd4...d773`).
At the first A100 BigFish minibatch the raw joint direction is `2.45`, with
actor `.103` and critic `2.44`, and is clipped to `.5`; by the end of the
first update behavior KL is only `6.24e-8`.  This independently verifies that
critic-dominated global clipping is part of the sampled-B update identity,
not an H200-specific artifact.  Only deprecation warnings are present.

At 761,856 transitions the combined relative-damping/no-extra-RMS run remains
numerically healthy but is now clearly an under-update control rather than a
better repair.  Its last-10 reward/entropy/behavior-KL/effective-direction are
`2.70/2.212/.00224/.0179`, with natural advantage RMS `.367`, LR already at
the `.05` ceiling, kernel median/effective damping `92.31/92.31`, and solve
residual `4.61e-14`.  The normalized relative-damping run at the identical
transition has `3.00/1.721/.00440/.0257` (instantaneous KL `.00614`, LR
`.0333`).  Therefore removing advantage RMS does not repair an otherwise
bad geometry: after relative damping has repaired the geometry, it reduces
and destabilizes actor movement according to the raw advantage scale.  The
normalization is beneficial rate control and should remain in the minimal
repair.

The formal A100 controls continue to reproduce both mechanisms.  CaveFlyer
fixed-damping spectral task `18564677_2` is at 585,728 transitions with
kernel median `209.19`, fixed damping/kernel ratio `.00239`, entropy `.755`,
behavior KL `.00696`, and solve residual `2.55e-11`; the geometry becomes
poor despite a numerically accurate solve.  BigFish sampled-B task
`18564681_0` is at 430,080 transitions with LR `.05`, entropy `2.704`,
behavior KL only `5.41e-5`, rows `512`, and residual `1.58e-13`.  This is the
same high-entropy/tiny-policy-movement pattern seen on H200.  No OOM, NaN,
Traceback, CUDA, Cholesky, or nonfinite error is present in either active
control.

The combined relative-damping/no-extra-RMS run completed `PASS` at 1,007,616
transitions with no numerical error.  Using the last minibatch of each of the
last ten rollout updates, its reward/entropy/behavior-KL/effective-direction
are `3.62/2.121/.00117/.00761`, versus `4.10/1.738/.00956/.02570` for the
otherwise matched normalized relative-damping run.  At the final minibatch
the natural advantage RMS has fallen to `.156`; LR is already capped at
`.05`, yet behavior KL is only `.000507` and the effective direction `.00552`.
This closes the combination test: removing RMS normalization neither reveals
nor fixes the strict-2B defect.  It turns the repaired method into an
intermittently under-updating variant.  The minimal CaveFlyer repair should
keep the existing advantage RMS normalization and change only damping scale.

The cross-environment H200 relative-damping chain has now started real
BigFish metrics (`18566946_0`), with immutable hashes matching the successful
CaveFlyer run and the same 4096/512/four-epoch/1M configuration.  At 184,320
transitions its trailing-10-update reward/entropy/KL are
`2.30/2.472/.00360`, compared with the fixed-damping trajectory's
`2.863/1.705/.01416` at the same point.  The relative run has kernel
median/effective damping `6.15/6.15`, LR `.05`, and residual `1.93e-12`.
This is healthy but learns more slowly early in BigFish; it prevents rapid
entropy loss rather than giving a uniform early reward improvement.  The
endpoint is required before deciding whether a floor ratio of exactly one is
too conservative outside CaveFlyer.

At 290,816 transitions the BigFish relative-damping probe remains healthy
but deliberately conservative: trailing-10-update reward/entropy/KL are
`2.48/2.379/.00416`, kernel median/effective damping `7.59/7.59`, LR `.05`,
and residual `1.60e-12`.  This is still below the fixed trajectory at the
same early stage while retaining substantially more entropy.  It reinforces
that the diagnosis (fixed damping becomes negligible) and the best universal
coefficient are separate questions: a ratio floor of one repairs CaveFlyer
but may over-regularize BigFish.  No coefficient change should be chosen
until its endpoint and BossFight/CoinRun evidence arrive.

Formal A100 CaveFlyer fixed-damping spectral evidence has reached 749,568
transitions.  Kernel median is now `380.96`, so `.5/median=.00131`; trailing
entropy is `.355` and behavior KL `.0112`, while solve residual remains only
`6.67e-11`.  The same run therefore exhibits severe entropy collapse as the
fixed damping becomes three orders of magnitude smaller than the observed
kernel scale, despite an accurate solve.  Formal A100 sampled-B BigFish has
reached 610,304 transitions with trailing reward/entropy/KL
`2.063/2.695/.000134`, LR `.05`, rows512 and residual `2.92e-13`.  BossFight
sampled-B has also started with a valid matched preflight and real metrics.
These concurrent A100 controls continue to separate two failure modes:
fixed strict-2B rotates/suppresses the actor as damping loses scale, whereas
sampled-B preserves entropy but produces extremely small coherent policy
movement under critic-dominated global clipping.

At 475,136 transitions the BigFish ratio-1 relative-damping run has
trailing-10-update reward/entropy/KL `2.907/2.303/.00526`, while the matched
fixed-damping trajectory at the same point has `4.460/.787/.01220`.
Relative damping is therefore preserving exploration but is now clearly too
strong for BigFish at this horizon.  This does not contradict the scale
diagnosis: it separates the need for scale-aware damping from the choice of
its dimensionless coefficient.

To identify a cross-environment coefficient rather than extrapolate from one
task, a strictly matched H200 sweep has been staged as Slurm array `18567905`
with dependency `afterany:18566946` and serial throttle `%1`.  It runs
BigFish and CaveFlyer at damping/median floors `.1` and `.3`, keeping
rollout4096, minibatch512, four epochs, 1M transitions, clean full-joint
rows1024, RMS-normalized advantages, base damping `.5`, no momentum, and no
Kaczmarz.  Config hashes are
`96d766d06db59517ceeabe20fdb3cfae9a1ed2de2306b696c873929687db1c44`
and `5eaca375a085a925d5f639da186b121de0d62be3bef9b142d93d9e8f0020258b`;
launcher hash is
`50cf264495c500690a06ab5ae57d9c85444a4815eb991b53394e606723133193`.
The dependency preserves the current BigFish/BossFight/CoinRun ratio-1
sequence before testing coefficient refinements.

At 770,048 transitions the BigFish ratio-1 run still has only
`3.017/2.104/.00490` trailing reward/entropy/KL, with kernel
median/effective damping `30.17/30.17` and residual `1.04e-12`.  The slower
learning is therefore not merely an early transient; a unit median ratio is
too conservative for BigFish at the fixed 1M sample budget.  The queued
`.1/.3` sweep is now necessary to distinguish scale-aware repair from
over-regularization.

Formal A100 CaveFlyer spectral has reached 917,504 transitions with trailing
entropy `.252`, kernel median `383.19`, fixed damping/kernel ratio `.00130`,
and residual `2.02e-11`.  This is a near-endpoint reproduction of collapse
under an accurately solved but badly scaled metric.  The next spectral task,
CoinRun `18564677_3`, has concurrently acquired node850 and emitted a valid
matched preflight plus real rows1024 metrics; at 12,288 transitions its
kernel median is still only `.00221`, so the base `.5` damping dominates
early (`damping/median=226.7`).  This directly exposes the temporal inversion
being tested: base damping starts overwhelmingly strong and later becomes
negligible as the learned critic/kernel grows.

The full CaveFlyer A100 time course makes that inversion quantitative.  At
4,096 transitions the kernel median is `.00227`, so fixed damping/median is
`219.8`; at 40,960 it is still `56.3`.  By 81,920 the kernel median has
jumped to `16.11`, the ratio has crossed to `.0310`, behavior KL is `.0334`,
and LR begins retreating from its `.05` ceiling.  At 163,840 the ratio is
`.00869` with KL still `.0333`; by 327,680 it is `.00211`, entropy has fallen
to `1.78`, and LR to `.00988`.  At 655,360 entropy is `.453`; at 929,792 the
ratio is `.00151`, entropy `.194`, and LR only `.00293`.  Thus the same fixed
`.5` ridge changes by roughly 146,000-fold relative to the learned kernel
over one run.  The KL controller reacts only after the metric transition and
reduces scalar LR; it cannot restore the lost actor-direction geometry or
exploration.  This temporal ordering is the clearest current root-cause
evidence for the fixed-damping strict-2B failure.

The H200 BigFish ratio-1 run has now completed `PASS`, `rc=0`, at 1,007,616
transitions.  Its trailing ten rollout updates yield reward/entropy/KL
`3.372/1.868/.00438`, versus the matched fixed-damping run's
`6.382/.272/.01312`.  Final kernel median/effective damping are
`39.30/39.30`, LR is capped at `.05`, and residual is `4.00e-13`.
Consequently ratio one is conclusively over-regularized for BigFish at the
1M sample budget: it prevents collapse but also prevents the task's useful
low-entropy exploitation.  This endpoint strengthens the case for the
already queued `.1/.3` dimensionless sweep.

BossFight ratio one (`18566946_1`) has automatically acquired node820 after
BigFish and emitted a valid matched preflight plus real metrics.  At 45,056
transitions the kernel is still tiny (median `.00619`), so the base `.5`
damping remains active and corresponds to `80.8` times the median; entropy is
`2.699`, LR `.05`, and residual `1.85e-14`.  This is verified execution, not
merely an array state.

Formal A100 CaveFlyer spectral has completed `PASS` at 1,007,616
transitions.  Trailing reward/entropy/KL are `2.050/.295/.01187`; final
kernel median is `168.90`, fixed damping/median `.00296`, LR `.00439`, and
residual `1.69e-11`.  This supplies the formal endpoint: the fixed strict-2B
trajectory collapses exploration under a well-solved but scale-drifting
system.  The concurrent CoinRun spectral task has already moved from an
initial kernel median near `.0022` to `115.43` by 114,688 transitions;
damping/median has inverted to `.00433`, entropy has fallen to `1.87`, and
LR to `.0148`.  The same abrupt kernel-growth transition therefore appears
in another environment, rather than being unique to CaveFlyer.

Formal A100 sampled-B BigFish has completed `PASS`, `rc=0`, at 1,007,616
transitions.  Trailing reward/entropy/KL are `2.138/2.693/.000216`, with LR
capped at `.05`.  The trailing raw joint/actor/critic norms are
`1.918/.198/1.908`; the joint direction is clipped to `.5` essentially every
update, while solve residual remains `1.80e-13`.  Compared at the same formal
endpoint, fixed clean reaches reward `6.382`, ratio-1 relative damping
`3.372`, and sampled-B only `2.138`.  The A100 endpoint rules out an H200
artifact and confirms the sampled-B failure mode: critic-dominated norm
clipping plus randomized reconstruction preserves entropy but yields almost
no coherent actor movement (tiny KL even at maximum LR).

At 118,784 transitions BossFight ratio one has trailing reward/entropy/KL
`.071/2.571/.00633`, versus matched fixed damping's
`.172/2.234/.01535`.  It shows the same early over-regularization seen in
BigFish.  Scale drift is common across environments, but the coefficient
must not force all tasks to retain the CaveFlyer-optimal exploration level.

At 397,312 transitions BossFight ratio one has trailing reward/entropy/KL
`.718/1.717/.00578`, versus fixed damping's
`1.571/1.610/.01026` at the identical transition.  Ratio one now has nearly
the same entropy yet less than half the reward, so its deficit is genuine
over-damping rather than simply delayed entropy collapse.

CoinRun spectral clarifies that scale drift and task harm are distinct.  By
225,280 transitions its kernel median is already `1409.1`, fixed
damping/median only `.000355`, trailing entropy `.757`, and current entropy
`.469`; nevertheless trailing reward is `5.92`.  Fixed damping therefore
under-regularizes the learned metric in multiple environments, but rapid
low-entropy exploitation can be beneficial in CoinRun/BigFish and damaging
in exploration-sensitive CaveFlyer.  A usable repair needs a dimensionless
coefficient that controls the degree of metric regularization without
forcing a universal entropy trajectory; this is exactly what the `.1/.3`
BigFish/CaveFlyer bracket is designed to identify.

Cross-environment correlations over every available spectral diagnostic make
the geometry claim independent of selected checkpoints.  For BigFish,
CaveFlyer, and CoinRun respectively, Pearson correlation between
`log10(damping/kernel median)` and actor-only/full-joint direction cosine is
`+.877/+.763/+.928`; correlation with direction delta is
`-.928/-.874/-.975`, and with entropy `+.830/+.593/+.905`.  Pooled over 57
diagnostics, the corresponding values are `+.656/-.796/+.718`.  Thus a
falling dimensionless damping ratio consistently rotates the full-joint
actor direction away from its actor-only counterpart and lowers entropy.
Reward is deliberately not universal: its within-environment correlation
with log ratio is `-.845` for BigFish, `+.606` for CaveFlyer, and `-.960` for
CoinRun.  The same geometric transition benefits exploitation-heavy tasks
but harms exploration-sensitive CaveFlyer, which explains why a scale-aware
coefficient must be tuned as a controlled dimensionless regularizer rather
than assumed to be one.

Matched-update curve averages show that the endpoint contrast is not a
last-window artifact.  Over all 246 rollout updates, BigFish mean reward is
`4.313` for fixed clean, `2.611` for ratio-one relative damping, and `1.917`
for sampled-B; after 500k the corresponding means are
`5.187/2.936/2.092`.  CaveFlyer gives the opposite fixed-versus-relative
ordering: all-update means `2.545/3.404/1.662`, and post-500k means
`2.585/3.756/1.554`.  (The formal fixed Cave trajectory is A100 while the
relative causal probe is H200, so this is strong directional/sample-grid
evidence rather than a hardware-identical formal ranking.)  Ratio one raises
CaveFlyer's entire sample-efficiency curve and lowers BigFish's, while
sampled-B is poor throughout both tasks.

Blockwise spectral logs refine the phrase "critic/kernel growth."  The
critic block is usually larger late, but the scale drift is not critic-only.
For BigFish, actor/critic block Frobenius norms move from `.55/1.14` at 4k
to `329706/831388` at 987k; for CaveFlyer, `.475/1.12` to
`22258/180120`; for CoinRun, `.467/5.29` to `342148/1207519` by 332k.
Thus actor and critic Jacobian Grams both grow by four to six orders of
magnitude, with the critic often dominating the joint scale.  Cross-block
coupling also grows but its normalized magnitude generally falls later, and
the matched separate-network ablation already failed with cross block zero.
The most accurate root cause is therefore learned Jacobian/Gram scale drift
under a fixed absolute ridge (critic-dominated in many phases), not a purely
critic-GGN explosion or an off-diagonal-coupling bug.

The actor-block growth is also not caused by a growing categorical output
Fisher.  BigFish categorical Fisher trace falls from `.933` to `.153`,
CaveFlyer from `.933` to `.164`, and CoinRun from `.933` to `.117`, while
their actor-block Frobenius norms grow from order one to tens or hundreds of
thousands.  Therefore the exploding Gram scale is generated by the learned
parameter-to-logit/value Jacobians (network feature/weight scale), despite a
shrinking output-distribution Fisher trace.  PopArt/output normalization does
not by itself normalize these parameter-space Jacobians.  This explains why
a fixed absolute damping constant cannot remain meaningful during training
and why a dimensionless kernel-relative ridge is the targeted intervention.

Recorded eigenvalues allow a counterfactual conditioning check before the
coefficient sweep runs.  From 82k onward, median fixed-ridge condition numbers
are about `350936/195566/1542788` for BigFish/CaveFlyer/CoinRun.  Replacing
the ridge by `max(.5, .1*median_diag)` would reduce them to roughly
`9655/4373/6238`; coefficient `.3` to `3219/1458/2080`; coefficient one to
`966/438/625`.  The 90th-percentile pattern is similar.  Thus `.1/.3`
already improve conditioning by two to three orders of magnitude while
leaving substantially more task-specific geometry than the demonstrably
over-regularized coefficient one.  This provides a spectral—not merely
reward-based—justification for the queued bracket.

Source inspection confirms that the scale statistic is taken from the exact
ratio-weighted system being solved, not from an unweighted proxy.  The code
forms `K=HH^T/B`, `weighted_K=K D_ratio` (actor rollout ratios, critic ratio
one), computes the median of `diag(weighted_K)`, solves
`(K D_ratio + mu I) alpha = b` in FP64, and reconstructs with
`H^T D_ratio alpha`.  Relative damping changes only
`mu=max(.5,c*median_diag)`; RHS, ratio semantics, 2B rows, cross blocks, and
reconstruction remain unchanged.  The spectral logger uses the similar
symmetric representative `sqrt(D) K sqrt(D)`, so its eigenvalues correspond
to the actual nonsymmetric weighted solve.  Missing-ratio or mismatched-
matrix explanations are therefore ruled out for this intervention.

BossFight ratio one has completed `PASS`, `rc=0`, at 1,007,616 transitions.
Its all-update reward mean is `.683` versus fixed clean's `1.410`; post-500k
mean `1.010` versus `1.823`; trailing-10 reward `1.347` versus `1.976`.
Final entropy is `1.564`, kernel median/effective damping `145.90/145.90`, LR
`.05`, and residual `3.43e-13`.  Ratio one is therefore conclusively
over-regularized in a second environment, not just BigFish.  CoinRun ratio
one (`18566946_3`) has automatically acquired node820 and emitted a valid
matched preflight plus real rows1024 metrics; at 16,384 transitions the base
`.5` ridge is still active and equals `226` times the tiny initial median.

The matched `head_only` strict full-joint array `18552840_[12-15%4]` has now
completed all four environments with `PASS`.  Identity checks confirm
`joint_system_rows=1024`, `joint_critic_param_scope=head_only`, and exactly
zero `cross_block_fro`.  Nevertheless trailing-10 rewards are only
`1.25/.04/0/.60` for BigFish/BossFight/CaveFlyer/CoinRun, versus
`6.95/1.55/1.30/7.30` for the matched full-clean run.  CaveFlyer is also far
below actor-only's `3.00`.  Removing shared-trunk critic curvature and all
actor--critic cross blocks therefore does not rescue training; cross-block
coupling is not the dominant failure mechanism.

Source inspection exposes a remaining coupling even when the Gram matrix is
block diagonal: the trainer solves for actor-RHS and critic-RHS responses,
sums both in the shared parameter vector, and applies one global Euclidean
clip of `.5`.  Thus a large critic response can determine the normalized
joint direction and consume the same clipping budget that bounds the actor
response even when `cross_block_fro=0`.  To test this without increasing the
maximum total step, a matched component-clipping trainer was created from
the exact formal trainer SHA.  It obtains both RHS responses with the same
1024-row joint inverse (cross blocks unchanged), clips each response to `.5`,
sums them, and retains the original final global `.5` safety cap.  All other
configuration remains fixed.  Trainer SHA is
`731fac7f8cc9467ca1d7c6caaa8037dfddabd50aabfb91f2454932671cd4c9ee`;
the four-environment A100 array is `18568591_[0-3%4]`, currently pending
priority.  New trace fields record raw actor/critic component norms and both
component clip scales, so improvement can be attributed to direction balance
rather than an unobserved total-step increase.

Because Slurm estimates the four-A100 formal array will not start until
14-Aug-2026 16:24, an isolated CaveFlyer L40S probe was also submitted as
`18568671_2`, with its own
`probe_1m_seed0_jupyter_componentclip_l40s_v1` root.  It is presently pending
`QOSGrpGRES` (not a code/configuration failure) and will use only the next
available account slot; the A100 formal array remains intact for matched
cross-environment evidence.

The coefficient-one CoinRun relative-damping probe has also completed `PASS`.
Its trailing-10 reward is `8.00`, entropy `1.125`, behavior KL `.00254`, and
solve residual `1.94e-13`.  Unlike BigFish and BossFight, CoinRun tolerates or
benefits from this strong scale-aware ridge, reinforcing that the geometry
failure is real but the reward-optimal regularization strength is
environment-dependent.  The queued coefficient sweep has advanced to its
first `.1` BigFish run (`18567905_0`): at 311,296 transitions trailing reward
is `3.40`, entropy `2.035`, KL `.00846`, kernel median `60.07`, and effective
damping `6.007`, exactly maintaining damping/median `.1` with a stable
`1.96e-12` residual.  This is healthy early evidence, not yet a final ranking.

Further temporal evidence falsifies component clipping as the primary cause.
For CaveFlyer head-only, the global `.5` clip fires on exactly zero
minibatches through 200k transitions, yet its reward is already below the
actor-only control by 100--200k.  Over the whole trajectory, clip activation
is only `9.4%`; the sparse exact decomposition gives mean
`actor_gain_from_critic_rhs` approximately `1.000` and mean
`critic_induced_actor_quadratic` only `2.7e-4`.  The separate-network matched
control also fails with cross block zero and disjoint parameters.  Therefore
critic consumption of a shared clipping budget can be a late amplifier but
cannot explain the initial CaveFlyer divergence.  The queued component jobs
were cancelled before formal completion.  The brief L40S probe reached 143k;
neither component was clipped at its endpoint (actor/critic norms
`.356/.060`, both scales one), which is consistent with the timeline result.

The completed critic-objective sweep identifies the stronger causal axis.
With critic curvature fixed at one, CaveFlyer trailing-10 reward/entropy for
`c_C=0/.03/.1/1` are `3.90/2.25`, `3.74/1.05`, `1.60/.36`, and `1.30/.10`.
Reducing the critic RHS monotonically rescues exploration-sensitive
CaveFlyer, while BigFish prefers the larger RHS (`2.75/4.25/5.58/6.95`
reward for the same sequence).  Direct same-minibatch actor projection is
near zero, so the likely path is critic learning changing the value baseline
and hence the next rollout's GAE, not a large instantaneous cross-block
rotation.

To cut precisely that cross-rollout path, trainer
`train_shared_joint2b_frozen_advcritic.py` keeps the full clean 1024-row
joint solve, critic RHS, curvature, cross blocks, optimizer, and controller,
but builds rollout GAE from an immutable initial value snapshot while actions
and logits remain live.  SHA is
`2743fe3732560acf23e9195625cfd3b57d43218a2527442b93b5a7a289563cd0`.
Serial four-environment L40S array `18568966_[0-3%1]` has begun BigFish on
node878.  At 40,960 transitions it reports rows1024, source
`frozen_initial`, live/frozen value RMS gap `.148`, reward `1.46`, entropy
`2.703`, KL `.00048`, and residual `3.2e-14`; fixed clean at the same point
has reward `1.50`, entropy `2.688`, KL `.00419`.  This verifies intervention
identity but remains too early for a causal outcome.

At 507,904 transitions, the coefficient `.1` relative-damping BigFish run
has trailing reward `4.75`, entropy `1.869`, KL `.00513`; fixed clean at the
same point is `5.06/.725/.00569`, and coefficient one is
`2.41/2.381/.00285`.  Thus `.1` currently preserves most BigFish reward while
avoiding the severe entropy collapse, whereas coefficient one is clearly
over-regularized.  This is the first evidence of a plausible cross-task
scale-aware setting, pending CaveFlyer and endpoints.

The `.1` BigFish run has now completed `PASS` at 1,007,616 transitions:
trailing reward `4.71`, entropy `1.429`, KL `.00431`, LR `.0222`, kernel
median/effective damping `134.67/13.47`, ratio exactly `.1`, residual
`4.35e-12`.  Endpoint fixed clean is `6.95/.267`, whereas coefficient one is
`2.81/1.948`; `.1` is a genuine intermediate tradeoff but does not recover
all of BigFish's exploitation-heavy reward.  The array has moved to `.1`
CaveFlyer.  At 593,920 transitions its reward/entropy/KL are
`3.00/1.983/.01048`, versus fixed `.90/.277/.00861` and coefficient one
`3.70/2.019/.01562` at the identical transition.  Scale-aware damping is
therefore a strong CaveFlyer repair, while `.1` controls KL better than one;
the endpoint is pending.

The whole-rollout/post-epoch KL-controller control has also completed all
four environments with `PASS`.  Endpoint rewards are `4.31/1.85/1.10/6.50`
for BigFish/BossFight/CaveFlyer/CoinRun.  CaveFlyer is not rescued relative
to fixed clean (`1.30`), so minibatch-frequency LR adjustment is not the
primary failure.  It changes some environment outcomes, but cannot explain
the strict-2B CaveFlyer collapse.

Removing the extra minibatch advantage RMS normalization alone gives a
CaveFlyer endpoint of `2.10` (entropy `.148`), while coefficient-one relative
damping gives `4.70` (entropy `1.69`) and the combination gives `3.20`
(entropy `1.97`).  Thus the extra normalization is a secondary scaling
factor, not the main cause; scale-aware damping has the larger causal effect.

The frozen-GAE intervention is live and correctly identified.  At 249,856
BigFish transitions it reports reward `2.35`, entropy `1.338`, behavior KL
`.01086`, full 1024-row joint solve residual `1.19e-10`, and a live-versus-
frozen value RMS gap `.341`.  This remains preliminary until the four serial
environments complete, but the intervention is now materially separated
from the matched live-value trajectory.

Based on the independent coefficient and damping evidence, a matched
minimal-repair combination was prepared and queued on H200 as array
`18569202_[0-3%1]`, dependent on completion of the current floor sweep.  It
uses the unchanged clean strict 1024-row joint system, rollout 4096,
minibatch 512, four epochs, seed zero, base damping `.5`, no momentum or
Kaczmarz, with only `c_C=.03` and a relative-damping floor of `.1`.  The
config SHA is
`63bb3052c4c2ea6ef963658f837cd8b152264995e771a0fd8af1c8241ad9e2c2`;
the launcher SHA is
`1a964541e367baf90b97666217ddfcafc28d4a88fa8267f65e262a474892c116`.

The existing CaveFlyer spectral trajectory makes the ridge-scale failure
direct.  At initialization the kernel median is only `.00228`, so base
damping `.5` is `219x` the median.  By 249,856 transitions the median is
`124.8` and `.5` is only `.0040x`; later medians are commonly `100--370`,
with damping/median around `.001--.005`.  The corresponding damped condition
estimate rises from `3.24` initially to roughly `6.6e4` by 250k and
`1.4e5--4.0e5` later.  Linear residuals can remain tiny while this happens:
the solver accurately solves an increasingly under-regularized problem.
This distinguishes a geometry-definition failure from a numerical-solver
failure and explains why a relative ridge has a large causal effect.

The coefficient `.3` relative-damping BigFish run has now completed `PASS`
at 1,007,616 transitions.  Its endpoint reward is `8.03`, entropy `.804`,
behavior KL `.01161`, LR `.05`, kernel median `113.62`, effective damping
`34.09`, and solve residual `7.28e-13`.  It exceeds both fixed damping
(`6.95`) and the `.1` floor (`4.71`) on BigFish, while remaining numerically
stable.  This strengthens the scale-drift diagnosis: changing only the ridge
from an absolute constant to a median-relative floor reverses the apparent
strict-joint degradation.  The paired `.3` CaveFlyer run has started with
the same trainer SHA, rows1024, full clean joint geometry, and matched
4096/512/4-epoch/1M configuration.  At 86,016 transitions its reward is
`2.90`, entropy `2.53`, KL `.01130`; the kernel median has already grown to
`23.54`, activating effective damping `7.06` and preserving the requested
damping/median ratio `.3` with residual `2.39e-13`.

Frozen-GAE BigFish has reached 888,832 transitions with reward `3.22`,
entropy `1.07`, KL `.00779`, live/frozen value RMS gap `.495`, and residual
`1.99e-10`.  It is clearly worse than matched fixed/live-value BigFish and
does not constitute a universal repair.  The decisive remaining observation
is the serial CaveFlyer member: improvement there would isolate an
environment-dependent critic-to-next-rollout-GAE path; failure there would
leave scale drift as the dominant explanation.

The `.3` CaveFlyer run has also completed `PASS`.  Its endpoint is reward
`3.16`, entropy `1.742`, KL `.00475`, LR `.0222`, kernel median `252.17`,
effective damping `75.65`, and residual `2.45e-13`.  Thus scale-relative
damping repairs the severe fixed-damping exploration collapse, but the
reward-optimal floor remains environment-dependent: `.3` is best on BigFish
(`8.03`) while `.1` is better on CaveFlyer (`3.91` versus `3.16`).

Frozen-GAE BigFish has completed `PASS` at 1,007,616 transitions with reward
`2.98`, entropy `1.311`, KL `.00977`, live/frozen value RMS gap `.422`, and
residual `2.20e-10`.  This is substantially worse than the matched live-value
fixed-clean endpoint (`6.95`), so critic-to-next-rollout GAE feedback is not a
global primary cause.  The serial array has moved to BossFight with the same
trainer SHA and verified 1024-row full joint identity; CaveFlyer remains the
environment-specific causal test.

An implementation-scale audit found no missing minibatch normalization.  The
trainer forms `K = H H^T / B`, solves `(K D + mu I) alpha = b`, and reconstructs
`H^T D alpha / B`; the two `1/B` factors are therefore consistent with the
parameter-space regularized least-squares system.  PopArt is also internally
consistent here: returns and advantages are normalized after the PopArt
statistics update, and `compute_value` differentiates the normalized value
head used by the normalized regression target.  The observed scale drift is
therefore not explained by an omitted `B` factor or by differentiating an
unnormalized value against a normalized target.  It is genuine learned
Jacobian/parameterization drift in the actor-critic network.  At matched
endpoints both actor and critic blocks are large; relative damping changes
the entire trajectory and keeps `mu / median(diag K)` controlled even when
the raw median remains large.

The frozen-GAE trainer was also diff-audited against its causal baseline.
The only functional change is a deep-copied immutable value model exposed to
the Runner while policy logits/actions remain live; the strict joint solve,
critic residual/RHS, optimizer, PopArt update of the trained model, and KL
controller are unchanged.  Remaining differences are diagnostic logging.
Thus the CaveFlyer member is a valid intervention on the cross-rollout
`critic update -> value estimate -> GAE` path rather than a policy-only or
optimizer substitute.

Frozen-GAE BossFight completed `PASS` with reward `0`, entropy `1.547`, KL
`.01427`, and residual `1.19e-9`.  Its live/frozen value RMS gap is only
`2.19e-4`, so this is effectively a null intervention on BossFight rather
than evidence that freezing a materially different baseline causes failure.
The array has moved to CaveFlyer.  There the intervention is already real:
the gap briefly reaches `.60` around 110k and is `.148` at 212,992.  At that
matched transition frozen-GAE reward/entropy are `1.20/1.34`, versus
`3.00/2.48` for `c_C=.03` and `2.50/2.21` for floor `.1`; historical fixed
replicates span `0.83--2.13` reward.  Frozen initial values therefore do not
show an early rescue and materially reduce exploration.  Endpoint evidence
is still required, but the cross-rollout GAE path is becoming less likely as
the primary strict-joint failure mechanism.

Event-order analysis strengthens the direction of the scale mechanism.  In
the fixed-clean spectral CaveFlyer trajectory, median `diag(K)` first crosses
`1/10/100` at `61k/78k/188k` transitions, while entropy first falls below
`2/1/.5` only at `225k/520k/606k`.  BigFish likewise crosses kernel medians
`1/10` at `65k/106k`, before entropy first falls below two at `164k`.
Therefore the scale loss precedes the exploration collapse; it is not merely
a statistic that grows after entropy has already failed.  Together with the
relative-damping intervention, this supports the causal sequence
`Jacobian/kernel growth -> fixed ridge becomes negligible -> oversized or
poorly regularized joint direction -> entropy/policy collapse`.

The combination BossFight member has completed `PASS` with reward `.63`,
entropy `.141`, KL `.01307`, kernel median `42.60`, effective damping `4.26`,
and residual `1.31e-12`; it does not improve on isolated `c_C=.03` (`.83`).
The array has moved to CaveFlyer with verified identical SHA and configuration
and has produced real metrics.  This is a second environment in which the
two conservative controls are not additively beneficial.

Frozen-GAE CaveFlyer now provides a strong intermediate causal result.  At
286,720 transitions the live/frozen raw-value RMS gap is `.763`, reward is
`.83`, entropy has already fallen to `.310`, KL is `.0114`, and the solve
residual is `7.44e-11`.  The intervention is therefore substantial but does
not rescue exploration; it makes collapse earlier than either `c_C=.03` or
relative floor `.1`.  Subject to the endpoint, this rejects the hypothesis
that learned critic values entering the next rollout's GAE are the primary
source of strict-joint failure.  Learned critic values are instead providing
a useful baseline even though the critic RHS strength inside the joint update
remains an environment-sensitive control.

The minimal combination array `18569202` is now running after the floor sweep
completed.  Its BigFish preflight and command verify exactly
`c_C=.03`, `lambda_C=1`, relative-damping floor `.1`, full clean 1024-row
joint geometry, rollout 4096, minibatch 512, four epochs, 1M transitions,
base damping `.5`, and no momentum/Kaczmarz.  It has produced real metrics
with the expected config SHA; at 45,056 transitions reward is `1.71`, entropy
`2.706`, KL `.00018`, and residual `1.82e-14`.  This is identity evidence,
not yet outcome evidence.

A baseline-identity audit found that historical fixed-clean seed-zero endpoints
come from several trainer snapshots and hardware allocations and are materially
dispersed (for example BigFish endpoints `4.94` and `6.95`).  They remain useful
trajectory diagnostics but are not a sufficiently strict causal denominator
for the H200 floor sweep.  An exact control was therefore queued as
`18571925_[0-1%1]`, dependent on the four-environment combination array.  It
uses the identical relative-damping trainer SHA
`8d9d8dff8635a6020b72ab3f08a567590c4d5e52dc24aa19bcea72bcb49d8730`,
the same H200 account, BigFish/CaveFlyer, 4096/512/four epochs/1M, full clean
rows1024, `c_C=lambda_C=1`, and changes only the relative floor to zero.  The
launcher SHA is
`cee289a7d04e0eb9c1061cbef3502ba4be25814e74b6ffdba0c3c33c19fb5961`.
This control is required before interpreting the floor `.1/.3` endpoint gap
as a hardware-independent causal effect.

The combination run has completed its first environment.  BigFish is `PASS`
at 1,007,616 transitions with reward `2.48`, entropy `.886`, KL `.01870`,
kernel median `154.41`, effective damping `15.44` (ratio `.1`), and residual
`4.28e-12`.  It is worse than either isolated intervention (`c_C=.03` gives
`4.25`; floor `.1` gives `4.71` in their existing formal trajectories).
At 94k the floor had not activated and the combination matched the weak-RHS
trajectory; after roughly 500k the median exceeded five and the relative
floor activated.  Performance then remained below both isolated controls.
Therefore weak critic RHS and scale-aware ridge are not independent additive
repairs: both reduce the effective joint update along the same feedback loop,
and combining them at these strengths over-regularizes BigFish.  The array
has moved to BossFight; CaveFlyer remains necessary to determine whether the
combination trades BigFish reward for a more robust exploration-sensitive
setting.

Frozen-GAE CaveFlyer has now reached 528,384 transitions with reward `.10`,
entropy `.300`, KL `.01414`, live/frozen value RMS gap `.175`, and solve
residual `2.06e-10`.  There are no OOM, NaN, traceback, CUDA, or Cholesky
errors.  Since the intervention remains numerically healthy but exploration
has already collapsed, the endpoint is now unlikely to reverse the causal
conclusion: the `critic update -> next-rollout value estimate -> GAE` path is
not the primary failure source and freezing it is actively harmful on
CaveFlyer.

Frozen-GAE CaveFlyer has completed `PASS` at 1,007,616 transitions with
reward `.30`, entropy `.338`, KL `.00473`, live/frozen value RMS gap `.224`,
and solve residual `7.08e-10`.  It had already fallen below entropy `.3` by
roughly 528k and never recovered.  This is endpoint-level evidence that
cutting the learned-critic-to-next-rollout-GAE path does not rescue the strict
joint update; it materially worsens the exploration-sensitive environment.
The serial array has advanced to CoinRun.

The `c_C=.03` plus relative-floor `.1` CaveFlyer member is simultaneously at
557,056 transitions with reward `4.10`, entropy `1.724`, KL `.00425`, and
solve residual `3.88e-13`, also without runtime errors.  This is a healthier
intermediate point than frozen-GAE and fixed damping, but it is not yet an
endpoint: BigFish (`2.48`) and BossFight (`.63`) already show that the same
combination is non-additive and can over-regularize.  The strict same-trainer,
same-H200 floor-zero control `18571925` remains dependency-pending and will be
the decisive denominator once this serial four-environment array completes.

The combination CaveFlyer member has now completed `PASS` at 1,007,616
transitions with reward `1.06`, entropy `.519`, KL `.01183`, kernel median
`221.37`, effective damping `22.14`, and residual `8.57e-13`.  Its trajectory
was temporarily healthy at 557k (reward `4.10`, entropy `1.72`) but collapsed
by 893k (reward `.43`, entropy `.31`) before a small endpoint rebound.  Thus
the combined intervention delays but does not prevent the late exploration
collapse; exact solves and a damping/median ratio of `.1` are insufficient at
this coefficient pair.  The array has advanced to CoinRun.  This completes
three environments showing that `c_C=.03 + floor=.1` is not a general repair:
BigFish and BossFight are worse than their isolated interventions, while
CaveFlyer still fails late.

The exact floor-zero matched control has now genuinely started as Slurm
`18571925_0` on the same H200 node class.  BigFish preflight verifies trainer
SHA `8d9d8dff8635a6020b72ab3f08a567590c4d5e52dc24aa19bcea72bcb49d8730`,
config SHA `9adf8e292648e80e756d0748ba21a1068ae308ed9e63d477dd47e7f597309d39`,
4096/512/four epochs/1M, full-clean 1024-row joint geometry,
`c_C=lambda_C=1`, base damping `.5`, no momentum/Kaczmarz, and relative floor
exactly zero.  At 40,960 transitions it has reward `1.48`, entropy `2.690`,
KL `.00414`, kernel median `.155`, effective damping `.5`, and residual
`3.13e-13`, with no runtime errors.  This establishes launch identity and
real metric production; endpoints for BigFish and then CaveFlyer remain the
last causal denominator needed for the relative-damping claim.

The exact matched floor-zero BigFish control has completed `PASS` at
1,007,616 transitions with reward `6.74`, entropy `.479`, KL `.00759`, kernel
median `588.62`, fixed damping `.5` (damping/median `8.49e-4`), and residual
`1.13e-10`, with no runtime errors.  The same-array CaveFlyer member has
started on the same H200.  This refines rather than removes the scale result:
floor zero allows severe exploration contraction on BigFish, but its endpoint
reward can exceed floor `.1` (`4.71`); floor `.3` remains the higher-reward
BigFish trajectory (`8.03`).  Relative damping is therefore a stability and
exploration control whose reward-optimal coefficient is environment-specific,
not a monotone reward improvement.  CaveFlyer is the decisive matched test
because its fixed-damping failure is an actual reward-and-entropy collapse.

The four-environment `c_C=.03 + floor=.1` array has fully completed.  CoinRun
ends at reward `7.30`, entropy `.899`, KL `.00988`, kernel median `391.96`,
effective damping `39.20`, and residual `5.60e-12`.  The four endpoints are
therefore BigFish `2.48`, BossFight `.63`, CaveFlyer `1.06`, and CoinRun
`7.30`: the setting is usable on CoinRun but is not a general strict-joint
repair.  Frozen-GAE CoinRun is still running and is not required for the
already endpoint-supported rejection of the critic-to-GAE primary-cause
hypothesis on CaveFlyer.

Frozen-GAE CoinRun subsequently completed `PASS` at 1,007,616 transitions
with reward `1.10`, entropy `.250`, KL `.01983`, live/frozen value RMS gap
`.672`, and residual `9.05e-10`.  Together with CaveFlyer, this confirms that
freezing the rollout value baseline is not a repair and can itself produce
low-entropy behavior.

## Final causal conclusion

The exact same-trainer, same-H200 floor controls are now complete.  At
1,007,616 transitions:

| environment | damping floor | reward | entropy | kernel median | effective damping | damping / median |
|---|---:|---:|---:|---:|---:|---:|
| BigFish | 0 | 6.74 | .479 | 588.62 | .5 | .000849 |
| BigFish | .1 | 4.71 | 1.439 | 134.67 | 13.47 | .1 |
| BigFish | .3 | 8.03 | .804 | 113.62 | 34.09 | .3 |
| CaveFlyer | 0 | 1.30 | .150 | 392.73 | .5 | .001273 |
| CaveFlyer | .1 | 3.91 | 1.289 | 475.71 | 47.57 | .1 |
| CaveFlyer | .3 | 3.16 | 1.742 | 252.17 | 75.65 | .3 |

All six solves are accurate (`1e-10` or better residual), and the matched
floor-zero runs have no OOM, NaN, traceback, CUDA, or Cholesky error.  In the
floor-zero CaveFlyer trajectory the kernel median reaches `34.3` by 90k,
`103` by 172k, and roughly `393` at the endpoint while damping remains `.5`.
Entropy subsequently falls to `.150`, with a transient behavior-KL spike of
`.0537` near 508k.  Setting a scale-relative ridge causally prevents most of
this exploration collapse and raises CaveFlyer reward from `1.30` to
`3.91`/`3.16`.

The primary strict-joint Procgen failure is therefore not an inaccurate or
singular solve.  It is loss of *relative* regularization as the learned joint
actor-critic Jacobian/kernel scale grows by orders of magnitude: fixed
`damping=.5` becomes negligible, the accurately solved joint direction is
underregularized, KL excursions occur, and the policy contracts to low
entropy.  The critic RHS strength modulates this feedback in an
environment-dependent way but is not a standalone cause or universal repair.
Cross blocks, head/trunk coupling, KL-adjustment timing, extra advantage RMS,
and critic-to-next-rollout GAE feedback have each been falsified as the
primary mechanism by matched interventions.

The technically correct remedy is to define damping relative to a kernel
scale statistic (or equivalently normalize/precondition the joint system)
rather than use an absolute `.5`.  A single relative coefficient is not
reward-optimal across all environments: `.1` is better for CaveFlyer, while
`.3` is better for BigFish.  Thus scale control fixes the pathological
collapse, after which its coefficient remains an ordinary environment-level
hyperparameter rather than a universal constant.
