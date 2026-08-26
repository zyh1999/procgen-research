# Task45: Full-Shared Joint-2B Direct Science Launch

## Frozen identity and user override

- Task: `PROCGEN-FULL-SHARED-JOINT2B-SCIENCE-LAUNCH-20260826-45`
- Method: `FULL_SHARED_JOINT2B_SCALE_RECOVERY_V1`
- Implementation/deployment freeze: `9f0fcc2b076693964ac331477e4d1b8977660313`
- Trainer SHA256: `b13b3eae445f83f0e1d6565cb942c8390b6cd143fcc7f366410406c370aa0815`
- Config SHA256: `1b0cf73885dcb7c61078e463f826d6b296abfedb5a6a019f6782c6b899896b52`
- Frozen science-launcher SHA256: `59dea11cb0fed21974f4985d2eb2016703dd33f06d2e11daa7ca419d5a888fb4`
- Oracle SHA256: `62b1b10a81fc89ec621f0ceaf60735864cc899fafd6bda074c7414744506d303`
- Task45 deployment-only launcher SHA256: `ba864c8ceddf86b40dea3295f348f35308caef0f9018219daefb9048e342ab30`
- Frozen exact-stage monitor SHA256: `7315fce97c9164dcd07b209ffbd6048aa0fab20dc3e97ad6b16aff2865d4488e`

The user explicitly authorized direct science without another preflight or
audit. The unresolved Task43 actor vmap/reference discrepancy remains recorded
and was not relabeled PASS. The accepted Task43 production-reference finite
error was max absolute `1.9206858326015208e-14` and max relative
`1.1036722475186101e-08`, with no hard infrastructure/nonfinite signature.

## Minimal launch checks

- Remote trainer/config/frozen-launcher hashes matched exactly.
- Frozen and Task45 commands were identical after deployment-path/root routing:
  `$PY -u $TRAINER --config $(basename $CONFIG) --env_name $ENV_NAME --seed 0 --device 0`.
- All four Task45 roots were absent and no Task45 duplicate job/process existed.
- gpuH, account `gpu-h200-fse-pgdr` and QOS `gpu-h200-fse` were live with mixed H200 capacity.
- No shape/oracle/gather rebuild, local gate, negative test, production preflight or provenance audit ran.

## Submitted matrix

Campaign:
`/scratch/h99859yz/procgen_full_shared_joint2b_science_launch_20260826_45`

| environment | job | initial scheduler | root |
|---|---:|---|---|
| BigFish | `19409681` | `RUNNING`, node820 | `runs/FULL_SHARED_JOINT2B_SCALE_RECOVERY_V1/bigfish-easy-0-10/seed0/6m` |
| BossFight | `19409682` | `RUNNING`, node820 | `runs/FULL_SHARED_JOINT2B_SCALE_RECOVERY_V1/bossfight-easy-0-10/seed0/6m` |
| CaveFlyer | `19409683` | `RUNNING`, node820 | `runs/FULL_SHARED_JOINT2B_SCALE_RECOVERY_V1/caveflyer-easy-0-10/seed0/6m` |
| CoinRun | `19409684` | `RUNNING`, node820 | `runs/FULL_SHARED_JOINT2B_SCALE_RECOVERY_V1/coinrun-easy-0-10/seed0/6m` |

All four roots contain `scientific_started.marker`, `RUNNING`, trainer PID,
frozen identities and the exact normalized command. Initial training minibatches
are executing with no immediate OOM/CUDA/NCCL/disk/quota/NaN/Inf failure.

## Monitoring contract

Only these four job/root pairs may be monitored. Use the immutable original
Paper seed0 baselines at exact first common >=2M, first common >=4M and
5,980,160. Cancel only the individual cell when exact Target/Paper <0.60.
Preserve scheduler/root/process/progress/trace/checkpoint/numerical evidence and
keep infrastructure, numerical and algorithm classifications separate.

Final conclusion: `CANDIDATE_REJECT`.

## Exact 2M and numerical callback (2026-08-26 09:54Z)

Scheduler and frozen-monitor reconciliation produced:

| environment | exact transition | Target | Paper | ratio | scheduler/action |
|---|---:|---:|---:|---:|---|
| BigFish | 2,007,040 | 10.09 | 9.28 | 1.0872844828 | PASS; RUNNING |
| BossFight | 2,007,040 | 0.00 | 2.92 | 0 | `EARLY_STOPPED_ALGORITHM`; CANCELLED by 778916 |
| CaveFlyer | n/a | n/a | n/a | n/a | FAILED/1:0 numerical at about 536k |
| CoinRun | 2,007,040 | 6.20 | 3.70 | 1.6756756757 | PASS; RUNNING |

BossFight `19409682` is scheduler-authoritatively terminal after 00:50:17 on
node820. Its root `RUNNING` marker and absent launcher rc are stale consequences
of the monitor cancellation, not evidence that it remains live. The exact
ledger, Paper/Target hashes and scheduler terminal row are preserved; it must
never be retried.

CoinRun `19409684` remains scheduler/root RUNNING, but its latest trace at
2,908,160 transitions is algorithmically/numerically degenerate: entropy
`2.550e-28`, actor raw scale `1.546e-52` versus critic scale `2.64341e5`,
direction/gradient/actor and critic quadratics `Inf`, predicted KL `NaN`, clip
scale `0`, LR `.5`, while the solve residual remains healthy at `7.44e-16`.
This mirrors the low-Fisher singular amplification that terminated CaveFlyer.
It is classified `RUNNING_NUMERICAL_DEGENERATION_NO_AUTHORIZED_CANCEL`, not an
infrastructure failure and not a completed scientific result. Because its
eligible exact 2M comparison passed and no exact >=4M comparison exists, the
Executor did not cancel it.

BigFish remains a healthy 2M PASS and continues. Task45 is therefore still
nonterminal with unique current conclusion `CANDIDATE_NOT_READY`.

## Exact 4M callback (2026-08-26 10:23Z)

BigFish `19409681` reached the exact 4,014,080 common row with Target `3.34`
and Paper `13.28`, ratio `0.2515060241`. The frozen Task45 monitor recorded
`EARLY_STOPPED_ALGORITHM`, returned rc3 under `--apply`, and Slurm now reports
`CANCELLED by 778916`, elapsed 01:16:26 on node820. The root `RUNNING` marker
and absent launcher rc are stale scheduler-kill artifacts. Exact ledgers,
progress/trace/log hashes, command/provenance and scheduler terminal evidence
are preserved; no checkpoint existed and the hard-error scan was clean.

CoinRun `19409684` reached the exact 4,014,080 row with Target `6.10` and Paper
`8.00`, ratio `.7625`, so the frozen rule recorded PASS and authorized no
cancellation. It remains scheduler RUNNING. The independent numerical-health
classification remains degraded: at 4,403,200 transitions entropy was
`1.280e-28`, actor raw scale `7.753e-53` versus critic `2.81891e5`, direction
and gradient were `Inf`, predicted KL `NaN`, clip scale `0`, LR `.5`, while
the solve residual remained `6.41e-16`. This is algorithm/numerical evidence,
not infrastructure evidence, but the exact reward rule requires continued
monitoring toward endpoint.

Task45 can no longer satisfy its promising criterion: BigFish and BossFight
are algorithm early stops and CaveFlyer is a numerical failure. CoinRun is
still nonterminal, so the live campaign remains `CANDIDATE_NOT_READY` until its
endpoint or terminal event; it was not cancelled merely from the aggregate
outcome.

## Final terminal reconciliation (2026-08-26 11:00Z)

CoinRun `19409684` completed normally before the endpoint comparator ran:
Slurm `COMPLETED/0:0` at CSF3 11:51, elapsed 01:45:15 on node820; root
`PASS/rc0`, exact 5,980,160 progress row, metric trace and a 3,766,013-byte
checkpoint were present with hard-error scan zero. The checkpoint is retained
only in scratch and is not committed to Git.

The frozen monitor was invoked after completion. Artifact mtimes are 11:51:15,
whereas the endpoint ledger mtime is 11:58:13. It then compared exact 5,980,160
Target `5.50` with Paper `9.40`, ratio `.5851063829787234`, and recorded
`EARLY_STOPPED_ALGORITHM`/rc3. Since Slurm had already completed, no scheduler
cancellation occurred. The authoritative classification is
`COMPLETED_SCIENTIFIC_ENDPOINT_BELOW_THRESHOLD`, not scheduler-cancelled.

Task45 final matrix:

| environment | terminal evidence | classification |
|---|---|---|
| BigFish | 4M `3.34/13.28=.251506` | `EARLY_STOPPED_ALGORITHM`, scheduler cancelled |
| BossFight | 2M `0/2.92=0` | `EARLY_STOPPED_ALGORITHM`, scheduler cancelled |
| CaveFlyer | numerical failure near 536k | `ALGORITHM_NUMERICAL_FAILURE` |
| CoinRun | endpoint `5.50/9.40=.585106` | completed endpoint below threshold |

The unique final conclusion is `CANDIDATE_REJECT`: no environment exceeded
Paper at endpoint, two cells were algorithm early stops, one failed
numerically, and the only completed endpoint was below the `.60` threshold.
No retry, requeue or resubmission is authorized.
