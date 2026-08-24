# PROCGEN-HYBRID-HEAD-STRUCTURAL-MANIFEST-RECOVERY-6M-MISSING3-20260824-12

## Scope and immutable provenance

Assignment `05fe72ba8d13217217a3039990cdba2ec5432279` authorized only the split
of environment-independent structure from environment/input-dependent
connectivity evidence, starting from harness freeze
`26b2252527076df4bfe537a8612446317cbdcf3a`. The corrected preflight was frozen
and pushed before remote work at
`570cca72136a8a8dc1972d0eadee7167d236f93a`.

Scientific identity stayed byte-identical:

| Artifact | SHA256 |
|---|---|
| trainer | `7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54` |
| config | `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda` |
| scientific launcher | `ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e` |
| stage monitor | `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e` |
| corrected preflight harness | `704278e8b5802498b8e065b9f12945e2cb72a665cdd28845b2401091b2e993ea` |
| corrected validation launcher | `6dcd0757b88b065eef2befbe862bc28a6da269eb86ea733a24c92919f39ad76b` |

The old full-file SHA assertion was removed; no observed environment SHA was
hard-coded or allow-listed. The harness now writes `structural_manifest.json`
and `connectivity_probe.json`. A caller-requested legacy
`parameter_partition.json` is only a byte-identical structural copy, preserving
the frozen scientific launcher's call contract without restoring probe values.

## Four-environment no-training validation

Live inspection showed no owned target Procgen GPU job. Task 11 scheduler and
root evidence remained immutable. Exactly one validation per environment was
submitted to gpuH; none performed a scientific training step.

| Environment | Job | Node | Slurm terminal | Structural SHA | Connectivity SHA | Semantic result |
|---|---:|---|---|---|---|---|
| BigFish | `19232320` | node820 | `COMPLETED/0:0`, 26s | `3f91f5c3...d623` | `7e47569374695c6cf13e468e2929890ace42a8ad03309f5255e85606e422ff05` | PASS |
| BossFight | `19232321` | node822 | `COMPLETED/0:0`, 25s | `3f91f5c3...d623` | `f54549e607ec966226f24f78dfb8dc284c36d928935297552206c3ddc2984dbd` | PASS |
| CaveFlyer | `19232322` | node820 | `COMPLETED/0:0`, 19s | `3f91f5c3...d623` | `76759122f3eb9d47d87cbd5334248f33af510214170e1e9467bb9b9adc19ee78` | PASS |
| CoinRun | `19232323` | node823 | `COMPLETED/0:0`, 2m50s | `3f91f5c3...d623` | `2558c07c01a904053683e22913a56f9d62f5ea831a3b99430304628222620dca` | PASS |

All four structural files compare byte-for-byte equal. Full common SHA256:
`3f91f5c313480c089d300a7dd5aff4a664f28ff9d9f4718bbab430200298d623`.
Their exact counts are total29 tensors/938,979 elements; trainable26/938,976;
policy-exclusive2/3,855; shared22/934,864; critic-exclusive2/257; and
non-trainable PopArt state3/3. Critic-exclusive names are exactly
`last_v_layer.weight` and `last_v_layer.bias`.

Every connectivity probe has `semantic_pass=true`,
`partition_names_match_structural=true`, and `nan_inf_or_fallback=false`.
Both critic-head parameters are policy-disconnected with Jacobian L2 exactly
zero and are value-connected with finite positive Jacobian L2. The full error
scan found no Traceback, RuntimeError, AssertionError, OOM, CUDA/NCCL, disk,
NaN, Inf, or fallback signature; stderr contains only known deprecation and
CUDA-primary-context warnings.

Within each environment the preflight, scientific-launcher dry-run, and trainer
entry resolved JSON files are byte-identical. Pretty-JSON SHA256 values are
BigFish `61f8ebe38443acbdbf141981f4e9921435dccd5d4abb6a63959e3d4bdb9232ab`,
BossFight `da5a3e8de2eeb6c9f5aaf005af2a857f83a2fada5f4c0bf0cf824fad8f15ce0d`,
CaveFlyer `f10ac37e3dbc3364aee75002ad00cb9017f13bd81fcfcb67a3788f5c98cafaef`,
and CoinRun `cc97e8566f0e4c15d4147b8991d00574ef4ffe443a33b27ce328bfdd9eeceb53`;
the necessary environment field accounts for cross-environment differences.
BigFish reproduces the task's named `61f8...` identity exactly.

All four runs also prove trainable/optimizer itemwise object identity, PopArt
state exclusion/invariance, Paper actor direction and sampled shared-critic
direction bit identity, one-step policy parameters/logits/shared deltas bit
identity, value-head-only delta difference, H200 150,111,977,472-byte memory
with 2,045,893,120-byte peak, Cholesky info0, and FP64 relative residual
`8.627e-16`.

## Mandatory launchability gate and exact blocker

The structural/connectivity recovery succeeded, but the mandatory fresh-root
gate cannot be satisfied under the remaining frozen constraints:

1. Frozen launcher SHA `ae7104e7...` contains the literal campaign
   `/scratch/h99859yz/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08`.
2. It constructs the only root as
   `$CAMPAIGN/runs/$METHOD/$ENV_NAME/seed0/6m` and exposes no campaign/root
   override.
3. It aborts with exit90 when that path exists.
4. The BigFish, BossFight, CaveFlyer and CoinRun Task 11 roots all exist.
5. Task 12 forbids changing the launcher and forbids overwriting or moving any
   Task 11 root.

Therefore exactly three new non-overwriting roots cannot be reached using the
required byte-identical launcher. Modifying/copy-editing the launcher or moving
the historical roots would exceed explicit authority. No scientific job was
submitted, no monitor was requested, and all Task 11 evidence remains intact.

## Evidence inventory and ledger

Model-free validation artifacts are tracked under
`remote_launch_staging/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/evidence_task12_preflight/<job>/`.
They include status/rc, timestamps, host/GPU, input hashes, static/regression
outputs, three resolved configs, structural manifest, connectivity probe,
optimizer/PopArt audit, compatibility stdout/stderr, and scheduler accounting.
No model or checkpoint is included.

Task 11 provenance is unchanged: BigFish `19228676` remains a valid 4M
`EARLY_STOPPED_ALGORITHM`; Boss/Cave/Coin `19228677-19228679` remain immutable
pre-training infrastructure/preflight-design failures. Task 12 adds four
successful no-training validations and no algorithm, numerical, or scientific
result.

## Conclusion

The cross-environment evidence split passes, but the frozen launcher and
immutable-root requirements make the requested scientific submissions
unrepresentable without new Planner authority for a scheduler/root-only launch
material change. The bounded task stops before science.

`PRECHECK_BLOCKED`
