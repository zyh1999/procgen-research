# Executor report

Task: `PROCGEN-FULL-SHARED-JOINT2B-PPO500K-RAT-SCHEDULER-6M-S0-20260826-50`

Current conclusion: `CANDIDATE_NOT_READY`

Task50 is frozen as the single rollout-level LR scheduler variant of Task49.
The Joint optimizer is created cleanly at the fixed switch with LR `.004`;
every Joint rollout records one behavior hash, one constant minibatch LR, exact
full-class behavior-to-final KL and exactly one bounded next-rollout LR update.
The sole Bede gate `1075026` passed on gpu015 with finite strict Joint2B/cross
evidence.

At exact 2,007,040, BF `1075028` passed `10.48/9.28=1.1293103448` and Coin
`1075031` passed `8.80/3.70=2.3783783784`; both remain RUNNING. Boss `1075029`
recorded `.39/2.92=.1335616438` and Cave `1075030` recorded
`2.10/4.45=.4719101124`. Each frozen monitor wrote one
`EARLY_STOPPED_ALGORITHM` ledger row and returned rc3. Scheduler-authoritative
terminal state for both is CANCELLED by639800874, exit `0:0`, elapsed
`02:14:05`, node gpu016; root RUNNING/absent rc are stale. Exact-stage solves
were finite with Cholesky info0, relative residuals `5.55e-14/8.54e-15`, hard
error scan0 and no checkpoint. Complete bounded model-free evidence is
archived. Task49 and Task50 BF/Coin were untouched; no repeat apply, retry,
requeue or resubmit exists, and the sole 20-minute automation remains active.


Task: `PROCGEN-FULL-SHARED-JOINT2B-PPO500K-WARMUP-6M-S0-20260826-49`

Current conclusion: `CANDIDATE_NOT_READY`

The user-authorized atomic CSF3-to-Bede migration is complete. CSF3 gate
`19441667` was verified zero-step with no root/process/artifact and cancelled
once; it is preserved as
`CANCELLED_FOR_USER_AUTHORIZED_ZERO_STEP_BEDE_MIGRATION`.

Bede gate `1074924` completed `0:0` on gpu006 with `PRECHECK_PASS`: a real PPO
update, the single fixed boundary switch, and a finite full Joint-2B solve all
ran; Cholesky info is zero and relative residual is `3.17e-15`.

At exact 2,007,040, BigFish `1074926` passed `.9310344828`, BossFight
`1074927` passed `.6061643836` (strictly above `.60`), and CoinRun `1074929`
passed `2.4324324324`; all remain RUNNING. CaveFlyer `1074928` recorded
`0/4.45=0`, so the frozen monitor wrote one `EARLY_STOPPED_ALGORITHM` ledger
row and returned rc3. Scheduler-authoritative terminal state is CANCELLED by
639800874, exit `0:0`, elapsed `01:56:15`, node gpu006; the root RUNNING marker
and absent rc are stale. Its switch count is one, exact-stage Cholesky info is
zero, relative residual is `9.09e-15`, finite scan passed, and hard-error scan
is zero. Complete bounded model-free evidence is archived; no model/checkpoint,
repeat apply, retry, requeue or resubmit exists. Task50 was not changed and the
sole 20-minute automation continues for all remaining live cells.


Task: `PROCGEN-FULL-SHARED-JOINT2B-SCIENCE-LAUNCH-20260826-45`

Current conclusion: `CANDIDATE_NOT_READY`

The user explicitly authorized direct Task45 science with no further preflight
or audit. Frozen trainer/config/science-launcher/oracle identities and the
normalized command remain exact. Deployment freeze
`9f0fcc2b076693964ac331477e4d1b8977660313` routes only fresh Task45 roots.

Exactly four gpuH seed0 intended-6M cells were submitted once: BigFish
`19409681`, BossFight `19409682`, CaveFlyer `19409683`, CoinRun `19409684`.
All are initially `RUNNING` on node820 with scientific-start markers, trainer
PIDs and active minibatches; no immediate hard error is present. Task43's
unresolved preflight discrepancies remain recorded and were not called PASS.

Full launch evidence and monitoring identities are in
`.agent/reports/PROCGEN-FULL-SHARED-JOINT2B-SCIENCE-LAUNCH-20260826-45.md`
and `remote_launch_staging/procgen_full_shared_joint2b_science_launch_20260826_45/`.
# Executor report

Task: `PROCGEN-FULL-SHARED-JOINT2B-ACTOR-SCALE-FLOOR-6M-S0-20260826-46`

Current conclusion: `CANDIDATE_NOT_READY`

The critic-anchored `.01` actor scale floor is implemented and frozen at
`829c58773c2b6a9bc01db2546f0145c24fb118d0`. Minimal syntax/import/hash,
command, duplicate, root and live gpuH placement checks passed. Exactly four
seed0 intended-6M jobs were submitted once: BigFish `19424173`, BossFight
`19424174`, CaveFlyer `19424175`, and CoinRun `19424176`. BigFish initially
runs on node820; the other three wait on `AssocMaxJobsLimit`.

Task45 remains disjoint and untouched. Full frozen identities, roots and the
replacement monitoring contract are in
`.agent/reports/PROCGEN-FULL-SHARED-JOINT2B-ACTOR-SCALE-FLOOR-6M-S0-20260826-46.md`.

# Task46 BossFight archive update

BossFight `19424174` is terminal `EARLY_STOPPED_ALGORITHM` at exact 2,007,040:
`0/2.92=0`; scheduler is CANCELLED by 778916 after 00:50:44 on node821. Full
model-free ledger, before/after scheduler evidence, numerical snapshot and
artifact hashes are committed. The actor scale floor was active and numerical
telemetry remained finite, so this is an algorithm reward early stop rather
than an infrastructure/numerical failure. Remaining Task46 cells and all
Task45 cells were untouched; Task46 remains `CANDIDATE_NOT_READY`.


# Task45 bounded monitor archive

Task45 exact 2M decisions are now preserved: BigFish PASS
`10.09/9.28=1.0872844828`, BossFight `EARLY_STOPPED_ALGORITHM`
`0/2.92=0` and scheduler CANCELLED, and CoinRun PASS
`6.20/3.70=1.6756756757`. Cave remains the prior numerical failure.

CoinRun is still live but has a verified low-Fisher numerical degeneration at
about 2.91M (actor scale `1.546e-52`, critic scale `2.643e5`, Inf direction and
quadratics, NaN predicted KL, finite solver residual). There is no authorized
cancel action before an eligible exact 4M row, so it was preserved and left
running. Task46 was not modified.

# Task45 4M archive update

BigFish `19409681` is now `EARLY_STOPPED_ALGORITHM` at exact 4,014,080:
`3.34/13.28=.2515060241`; scheduler is CANCELLED by 778916. CoinRun
`19409684` passed its exact 4M reward gate, `6.10/8.00=.7625`, and remains
RUNNING despite continuing low-Fisher Inf/NaN numerical telemetry. No endpoint
or authorized cancellation exists yet, so the live Task45 conclusion remains
`CANDIDATE_NOT_READY`. Task46 was read-only and unchanged.

# Task45 final / Task46 BigFish archive

Task45 is terminal `CANDIDATE_REJECT`. CoinRun `19409684` completed
scientifically with Slurm COMPLETED/0:0 and root PASS/rc0 before its endpoint
monitor invocation; the later exact endpoint ratio `5.50/9.40=.585106` is
below threshold but caused no scheduler cancellation. BigFish and BossFight
remain earlier algorithm stops and CaveFlyer remains the numerical failure.

Task46 BigFish `19424173` is now an exact 4M algorithm early stop,
`1.61/13.28=.12123494`, scheduler CANCELLED by 778916 with clean finite
numerical evidence. Task46 CaveFlyer and CoinRun remain running and untouched;
the sole automation cadence is 20 minutes.

# Task46 Cave archive / Task47 launch

Task46 CaveFlyer `19424175` is now an exact 2M reward early stop,
`0/4.45=0`, scheduler CANCELLED by 778916 with clean finite numerical evidence.
Task46 CoinRun `19424176` remains the only live Task46 cell.

Task47 `FULL_SHARED_DETGGN_BLOCKDIAG_BXB_V1` preserves frozen Task06 raw
full-shared actor/critic rows and removes only the two dual cross blocks. Its
sole production preflight `19425914` passed with two independent 512x512
Cholesky solves and full P938976 coverage. Four science cells were submitted
once: BF `19425987`, Boss `19425988`, Cave `19425989` are RUNNING and Coin
`19425990` is naturally PENDING on the four-H200 user limit. Current Task47
conclusion is `QUEUED_RESOURCE_WAIT`; the existing automation was updated in
place and no duplicate monitor exists.

# Task47 exact 2M archive update

BigFish `19425987` passed exact 2M at `8.28/9.28=.8922413793` and remains
RUNNING. BossFight `19425988` (`.07/2.92=.0239726`) and CaveFlyer `19425989`
(`2.50/4.45=.56179775`) were each cancelled once by the correct frozen Task47
monitor and are terminal `EARLY_STOPPED_ALGORITHM`. Both scheduler records are
`CANCELLED by 778916`, exit `0:0`, elapsed 00:34:55 on node821; stale root
RUNNING markers are not live.

Both cells retained finite actor/critic BxB solves, Cholesky info0, residuals
below `1.67e-13` and zero hard-error matches, so these are reward failures.
Task47 CoinRun `19425990` started naturally and remains RUNNING with BigFish.
Task46 CoinRun `19424176` independently passed exact 4M at `.8` and remains
RUNNING. Task47 is nonterminal `CANDIDATE_NOT_READY`; the sole automation
continues and no live cell was touched by archival.

# Task47 BigFish exact 4M archive

BigFish `19425987` passed exact 2M, then failed exact 4M at
`7.50/13.28=.5647590361`. The correct frozen Task47 monitor wrote
`EARLY_STOPPED_ALGORITHM` and returned rc3; scheduler state is CANCELLED by
778916 / `0:0`, elapsed 00:54:42 on node820. Root RUNNING/absent rc are stale.
The exact-stage actor and critic BxB solves were finite with Cholesky info0,
relative residuals at most `2.32e-14`, finite scan PASS and no hard-error
matches. No checkpoint, retry, requeue or resubmit exists.

BossFight, CaveFlyer and BigFish are now algorithm early stops, fixing the
Task47 conclusion as `CANDIDATE_REJECT`. Task47 CoinRun `19425990` and Task46
CoinRun `19424176` remain RUNNING on node822 and untouched. The sole 20-minute
monitor continues for those two live cells.

# Task46 / Task47 terminal delivery

Task46 CoinRun `19424176` completed the endpoint with scheduler COMPLETED/0:0,
root PASS/rc0 and exact ratios `1.6756756757`, `.8`, `.6595744681` at 2M, 4M
and 5,980,160. Endpoint telemetry remained finite with Cholesky info0 and
residual `3.77e-16`; the actor floor was active after actor-row collapse.
Checkpoint bytes remain only on scratch; Git records metadata and hashes only.
Task46 is terminal `CANDIDATE_REJECT` because its other three cells were legal
algorithm early stops.

Task47 CoinRun `19425990` stopped at exact2M, `2.20/3.70=.5945945946`.
The frozen monitor returned rc3 and Slurm reports CANCELLED by778916 /0:0,
elapsed00:39:48 on node822. Exact-stage actor/critic solves were finite,
Cholesky info0, residuals at most `3.37e-14`, and hard-error scan zero. Task47
is fully terminal `CANDIDATE_REJECT`: BF stopped at4M and all other cells at2M.

No retry, requeue, resubmit, new candidate, model or checkpoint was committed.
All jobs bound to the sole `procgen-3090` automation are terminal, so it may be
retired after this origin-verified delivery.

# Task49 active implementation

Task48 has been proven absent and recorded `SUPERSEDED_BEFORE_EXECUTION`.
Task49 versions the exact Task06 deterministic full-shared Joint-2B trainer
with one scientific change: Paper-matched PPO for 123 complete rollouts through
503,808 transitions, then exactly one boundary switch to untouched Joint-2B.
PPO has an independent Adam state and Joint-2B optimizer state is asserted
clean at switch. Local compile, config and launcher checks passed; the sole
minimal production gate is the next action.

The frozen implementation was pushed at
`e0dc2e5ca4efd85419e974e42561eea11145c96f`. The one production gate is job
`19441667`, currently PENDING `AssocGrpGRES` on gpuH with StartTime Unknown.
No science jobs or roots exist and no retry/requeue/resubmit occurred. Current
conclusion is `QUEUED_RESOURCE_WAIT`. The coordinator's sole automation
`monitor-procgen-task49-ppo-warmup` tracks this gate at20-minute cadence and
will wake the same Executor; no duplicate automation was created.
