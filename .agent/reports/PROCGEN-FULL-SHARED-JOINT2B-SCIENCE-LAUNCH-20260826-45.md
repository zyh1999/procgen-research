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

Current nonterminal conclusion: `CANDIDATE_NOT_READY`.
