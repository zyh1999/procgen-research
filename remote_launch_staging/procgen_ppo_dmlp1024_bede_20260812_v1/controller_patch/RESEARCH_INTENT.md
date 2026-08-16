# Authoritative Research Intent for the Three-Domain Controller

Updated: 2026-08-11. This file records the user's current experimental intent. Live scheduler, process, metric, source/config, and run-root evidence still determines whether a particular action is safe.

## Supersession notice

This file overrides older decisions copied into `BOARD.md`, `CYCLE_SUMMARY.md`, action logs, or failure tables. In particular, the MuJoCo 128,450,560 versus nominal 130M counter difference is resolved and is not a pending mismatch, not a reason for `NEEDS_USER`, and not a launch blocker.

## Controller role

The controller is not only a passive monitor. It should maintain a bounded research loop:

1. Audit active and terminal runs and preserve exact provenance.
2. Compare only matched runs.
3. Stop hard failures and persistently valueless runs using the configured evidence rules.
4. Identify the next experiment that adds the most missing evidence.
5. Submit it only through an exact vetted launcher after proving non-duplication and available capacity.
6. Reassess the result and either extend the seed matrix, change to the next planned comparison, or retire the failed configuration.

Do not invent a new algorithm or hyperparameter family. Prefer finishing an already defined comparison and its missing seeds. Do not optimize merely for GPU occupancy.

## Global priorities and decision policy

- Keep Procgen, MuJoCo, and Isaac identities separate. Do not transfer reward scales or baselines between tasks, algorithms, evaluation conventions, hardware families, or progress points.
- Evidence priority: complete a missing matched seed or interrupted continuation; then complete a missing environment/task in an established matrix; then run a planned matched control; only then consider a new hyperparameter point already defined by an existing study.
- A completed or early-stopped run must remain in `RESULTS_TABLE.tsv`. A failed configuration must remain in `FAILED_CONFIGS.tsv`, even if a replacement succeeds.
- Hard failures include verified OOM, CUDA failure, NaN/nonfinite, traceback, or dead process with incomplete output. Stop only the exact job element or exact owned process tree and preserve logs.
- Value early stop: robust score below 3/5 of the highest positive matched baseline, after the domain minimum progress, at least three points, and two consecutive controller cycles. A single noisy reward point is not sufficient.
- When a run is promising but noisy, continue it to the planned comparison point. When matching is uncertain, improve provenance/baseline registration instead of stopping.
- Avoid Jupyter for training. Never start a new Jupyter allocation. The independent watchdog may cancel a Jupyter-named job after continuous strong idle evidence for roughly 50 minutes, which is safely within the user's one-hour limit.
- `ws4090-31` / `10.49.7.54` is quarantined as an entire host until the user explicitly reverses that decision.

## Procgen: two experiment programs

### P1. Shared Exact GGN / shared-RAT program

- Current formal identity includes `adv_resnet_shared_exact_deterministic_ggn_symfp64.yaml` and the four environments CoinRun, BigFish, BossFight, and CaveFlyer.
- Preserve the deterministic critic-GGN, symmetric FP64/Jacobi, solver geometry, RHS, clipping, scale, seed, source/config hash, and hardware identity. Do not merge this with PPG actor-ablation evidence.
- Seed-0 symmetric-FP64/Jacobi completed in all four environments. The seed-1 CoinRun, BigFish, and BossFight runs were interrupted by the procgen-3090 host shutdown near 5,529,600/5,980,160 transitions; CaveFlyer was interrupted near 2,048,000/5,980,160. Their stale `RUNNING` status files are not proof of a live process or completion, and there are no resumable checkpoints.
- Do not automatically relaunch those four seed-1 roots. They are retired from the executable catalog while the user prioritizes the PPG decision-network study. Preserve them as infrastructure-interrupted evidence rather than algorithm failures.
- A new seed is useful only if the current run has valid finite solver/task diagnostics and is not an exact duplicate. Do not use another environment's reward as its baseline.

### P2. Phasic EF/PPG actor-ablation and adaptive-KL program

- The established BigFish comparison contains official-schedule `E_v2` as the matched PPG baseline; the controlled actor ablation is ACTOR_G, ACTOR_H, ACTOR_I, and ACTOR_J. ACTOR_K is a separate Exact-RAT adaptive-KL experiment and must not replace or be conflated with ACTOR_I.
- ACTOR_G is the entropy-NG line, ACTOR_H is the policy-KL/Fisher-clip line, ACTOR_I is the one-actor-epoch line, ACTOR_J combines entropy-NG, one actor epoch, and the policy Fisher/KL clip, and ACTOR_K is the separate Exact-RAT adaptive-KL line.
- Preserve the official schedule, actor/critic boundary, auxiliary phase, clone KL, entropy semantics, curvature choice, Fisher clipping, seed, and progress point.
- The original E_v2, G/H/I/J, and K BigFish seed matrices are complete. Preserve their terminal results; do not relaunch them as missing work.
- Track auxiliary EV/MSE and clone-KL/clip-scale diagnostics in addition to reward. A numerically finite solve with a collapsed auxiliary head is a failed method configuration, not a success.

### P2a. Active 1.465M decision-network study (highest current Procgen priority)

- The user explicitly requested a wider decision network. The active architecture is the unchanged IMPALA encoder followed by a shared `256 -> 1024 -> 256` ReLU decision MLP, then the policy, true-value, and auxiliary-value heads. Exact total parameter count is 1,464,804.
- The isolated root is `/root/procgen_ppg_dmlp1024_20260810_v1` on `procgen-3090`. E-v2 uses the independent official Adam implementation under `e_impl`; ACTOR_J uses the independent EF/NPG implementation at the root. Do not mix those two trainer entry points.
- Eight seed-0 formal runs are supervised now: E-v2 on GPUs0-3 and ACTOR_J on GPUs4-7, each across BigFish, BossFight, CaveFlyer, and CoinRun. Exact labels are E_BIGFISH, E_BOSSFIGHT, E_CAVEFLYER, E_COINRUN, J_BIGFISH, J_BOSSFIGHT, J_CAVEFLYER, and J_COINRUN. `status/PIDS.tsv`, per-run status files, `status/HEARTBEAT.tsv`, and `status/PROVENANCE.sha256` define their ownership and provenance.
- Both trainer implementations passed 14 tests, exact parameter/forward/parameter-group validation, and short full policy-plus-auxiliary smoke cycles. Formal E-v2 retains the official 32-policy-update/6-aux-epoch schedule; formal J retains its established 16-policy-update/6-aux-epoch schedule. Smoke-only shortened schedules are not formal results.
- First priority is to monitor all eight seed-0 runs through at least 1M transitions with reward plus auxiliary EV/MSE and policy/clone-KL diagnostics. Never compare reward values across different environments.
- BigFish old E-v2/J curves are architecture controls, but any value stop must compare at matched progress and use robust windows in two cycles. BossFight, CaveFlyer, and CoinRun have no completed same-architecture PPG baseline yet; do not ratio-stop the E-v2 baseline there. Do not stop the sole matched baseline needed to interpret J.
- Seeds1-2 are ranked future candidates only after seed-0 establishes value. No continuation is executable until an exact guarded launcher, fixed root, source/config hashes, non-duplication proof, and idle GPU are entered by the external operator into the approved catalog.

#### 2026-08-11 recovery handoff to the dual-RTX-5060-Ti node

- The E-v2 DMLP1024 seed-0 baselines are now terminal complete on BigFish, BossFight, CaveFlyer, and CoinRun. Preserve them as the same-architecture matched baselines.
- The former procgen-3090 ACTOR_J DMLP1024 seed-0 BigFish, CaveFlyer, and CoinRun processes died incomplete at 4,505,600, 4,423,680, and 4,464,640 transitions. They have no resumable checkpoint and remain failed infrastructure records. The user explicitly authorized clean replacement runs on the new dual-RTX-5060-Ti host.
- BossFight ACTOR_J seed0 is not a recovery target. It was ledger-confirmed `EARLY_STOPPED_FAILED` at 4,096,000 transitions with a robust sampled score 5.7933 versus the matched E-v2 score 10.60, ratio 0.5465. Keep it in reports and do not relaunch it.
- The fixed nonoverwriting recovery root is `/home/zzz/rlstack5060/campaigns/procgen_actorj_dmlp1024_recovery_20260811_v1`. GPU0 owns BigFish seed0 followed serially by CoinRun seed0; GPU1 owns CaveFlyer seed0. The complete source archive SHA256 is `cb9d908997aadfa5d98b5bc7a14808b27f925407704416c784ee06e539ad578b`; it supersedes the initial archive that omitted the lineage-identical `vec_env` package. Exact trainer and config hashes are recorded under that root.
- CSF3 is still the research control plane. `procgen-5060` telemetry is injected into `state/live_snapshot.txt` through a forced-command SSH key. That key cannot start work or provide a shell; in addition to status it accepts only the standard structured stop command for an exposed `owned_child` PID, exact workspace `/home/zzz/rlstack5060/workspaces/procgen`, and one of the three fixed recovery containers. Treat live fixed worker queues as active duplicates. Do not infer authorization for additional seeds or variants from this three-task recovery handoff.

### P2b. Pure-PPO DMLP1024 architecture control on Bede

- The user explicitly authorized a pure-PPO control for the same widened decision architecture and asked that all bundles be submitted to Bede at once. Preserve the unchanged IMPALA/ResNet encoder, shared `256 -> 1024 -> 256` ReLU decision MLP, linear policy head, and PopArt critic head. Pure PPO has no PPG auxiliary head or auxiliary phase; its exact active parameter count is 1,464,547, versus 1,464,804 for the PPG implementation with its additional auxiliary-value head.
- The fixed matrix is BigFish, BossFight, CaveFlyer, and CoinRun, each with seeds0-2 and 6,000,000 transitions per child. PPO hyperparameters remain fixed: Adam learning rate 0.001, clip 0.2, four epochs, eight minibatches, entropy coefficient 0, max gradient norm 0.5, 16 environments, and 256 rollout steps. Do not reinterpret this control as E-v2 or ACTOR_J and do not introduce an auxiliary update.
- Bede root: `/nobackup/projects/bdman37/yihe/procgen_ppo_dmlp1024_bede_20260812_v1`; formal root: `formal_4env_x3seed_6m_20260812_v1`. Each environment is one one-GPU sbatch containing exactly three concurrent, independently logged seed children. The preflight smoke job 1070572 completed successfully with all three children PASS, finite progress, three checkpoints, and observed GPU use.
- Formal jobs were submitted without dependencies on 2026-08-11T23:02:47Z: BigFish 1070573 (`pg-pd-bf`), BossFight 1070574 (`pg-pd-bo`), CaveFlyer 1070575 (`pg-pd-cf`), and CoinRun 1070576 (`pg-pd-cr`). Pending Bede jobs are active duplicates; Slurm may start them in any order. Do not submit replacements merely because a bundle remains pending.
- Exact SHA256 identities are trainer `989ea7f7607261872f753a8b4630eeb24b436ca01b668ee57f7e69e18ced90e5`, formal config `35a7ac93189f7174b317040746556f3e3689e1c666527bfd062650fa1240a26b`, bundle launcher `65947f5fd90e8f91fb7d5897309f375b9214b4ff8b8b94aefdf76da76e0ae0be`, and submitter `d7bc93b636536a1043c923cd2172faa8adbaffdb5f0defc2452f26979f9a0ccd`.
- A Bede job ID owns three seeds. Never cancel the whole bundle because one seed has a weak or noisy point. A value-based whole-bundle stop requires all three children to independently satisfy the matched evidence rule for two consecutive controller cycles; a verified bundle-wide hard failure may be stopped sooner. Preserve every early-stopped or failed child in the reporting tables.

## MuJoCo: two experiment programs

### M1. Large-batch curvature and K-opt program

- Keep `curv256` critic-GGN ktrue/k-opt work separate from `full-EF/full-GGN` work, even when both use clip/momentum studies.
- The established matrix spans Ant, HalfCheetah, Hopper, Walker2d, Humanoid, HumanoidStandup, and other already enumerated study tasks. Complete missing matched seeds/trials inside existing studies before adding a new study.
- Bede is ppc64le and Slurm-only; its V100 evidence must retain hardware provenance and must not be silently merged with CSF3 or RTX 4090 results.
- Treat Bede's concurrent GPU count and its Slurm queue depth as different quantities. The controller may submit more valuable approved jobs than can run simultaneously so that additional jobs wait in `PENDING`; four running jobs is not a submission ceiling.
- Keep Bede pre-submission bounded to roughly the next one or two GPU waves of exact, non-duplicate, already planned matrix work. Let Slurm decide when resources become available, but do not create speculative variants or a large stale backlog merely to keep the queue nonempty.
- The trainer endpoint at 128,450,560 transitions for a nominal 130M budget is an expected final full-update boundary, only about 1.2 percent short. Treat it as COMPLETE when terminal artifacts and the final planned update are present. It is not an algorithm failure and must not block further K-opt workers or trigger a rerun solely for the counter difference.
- Genuine blockers remain NaN/nonfinite, solver failure, OOM/CUDA error, missing terminal artifacts, or task-quality collapse under the matched early-stop rule.

### M2. Small-batch / energy-free and momentum program

- The current formal family compares momentum 0.5 versus 0.9 and dual-energy-free 255p1 versus curv256/full controls across the already prepared seven environments, two seeds, and 10M budget.
- The current CSF3 array already represents the planned matrix. Let its pending elements drain; do not submit a duplicate array. Replace only an exact failed/missing element after preserving its failure record.
- Keep the ws4090-76 dual-energy-free 255p1 independent-anchor line separate and finish its missing Humanoid seeds before inventing another anchor or momentum variant.
- Rank configurations using matched task return plus solver/clip diagnostics, not raw throughput alone.

## Isaac: matched policy-optimization programs

- Preserve two identities: Emp256/PPO100 rolling-old line-search u500, and EnergyFree255p1/manual controls.
- Tasks of interest are Unitree A1, Isaac Ant, and Anymal-C. Seed 42 alone is single-seed evidence, never a multi-seed conclusion.
- Avoid duplicate Optuna trials and shared-study worker oversubscription. Prefer completing the existing three-task matched matrix and then add missing matched seeds if the current configuration is valid.
- Preserve PPO warm-up state, environment/network/log-std/normalizer/critic/stat buffers when evaluating a switch to Emp256. Compare at matched updates and include AUC/learning efficiency, not only a final noisy point.
- Do not start new Isaac work inside Jupyter. Use vetted CSF3 Slurm or locked direct-host launchers only.

## Choosing the next submission

Every cycle must write a ranked research-candidate list in `CYCLE_SUMMARY.md`, even when every GPU is busy or execution is gated. Research ranking answers what evidence should be added next; executable ranking separately answers what can safely start now. Select at most one heavy executable run per free GPU and use this order:

1. Exact continuation/replacement of an interrupted run whose configuration is still valuable.
2. Missing seed in an active matched matrix.
3. Missing environment/task in an active matched matrix.
4. An already defined control required to interpret a promising result.
5. Another Optuna worker only when the shared study is incomplete, has no duplicate worker risk, and the study's trial budget supports it.

Before submission prove: exact configuration and seed, no active/complete duplicate, expected run root, source/config identity, available GPU, and exact approved launcher. If any proof is absent, record the concrete candidate and missing proof rather than substituting a different experiment.
