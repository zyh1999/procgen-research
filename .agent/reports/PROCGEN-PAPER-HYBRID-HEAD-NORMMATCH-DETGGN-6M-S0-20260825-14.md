# PROCGEN-PAPER-HYBRID-HEAD-NORMMATCH-DETGGN-6M-S0-20260825-14

## Conclusion

`PRECHECK_BLOCKED`

The V2 static and synthetic numerical audits passed, but the mandatory real
production-network preflight failed in all four environments before model
construction. Per the task's one-shot gate, no preflight repair and no
scientific submission were performed.

## Identity and frozen inputs

- Assignment: `cc58bea2b9a817cb0b5c44484e97f947f67be34b`
- Method: `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`
- Hybrid-Head V1 freeze: `fe4b8a58812e80689705abec11364457cae31e26`
- V1 trainer SHA256: `7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54`
- V2 trainer SHA256: `0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b`
- V1/V2 config SHA256 (byte-identical): `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- scientific launcher SHA256: `85e12886ce5cf81fd98647aa5163319a50174a39210cbeea1ccfde015aaf9d19`
- preflight harness SHA256: `b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc`
- regression SHA256: `f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c`
- preflight launcher SHA256: `ee2634e386b8422d5adbae6d782d80e8620195d05519625be745706a9f901caa`
- stage monitor SHA256: `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`

## Sole scientific diff

At the same post-history, pre-global-clip minibatch boundary, V2 forms the
already-available counterfactual Paper head proposal and the V1 deterministic
head proposal. It applies exactly:

```text
s = ||u_paper||_2 / ||u_det||_2
u_target = s * u_det
```

Both zero maps to zero; zero deterministic with nonzero Paper hard-fails;
zero Paper maps to zero. No cap, floor, coefficient, EMA, guard, fallback,
sample, RNG draw or data-order change was added. The target global pre-clip
norm is checked against the counterfactual Paper norm before reusing the
literal Paper clip coefficient. Actor/shared gradients remain Paper's; only
the 257-parameter value-head proposal direction differs. Added telemetry is
limited to proposal norms/scale/cosine, global norms, value/advantage and
PopArt summaries required by the task.

The static AST/diff audit reports `NORMMATCH_V2_STATIC_AUDIT_PASS`. The remote
non-training regression reports:

- exhaustive disjoint policy/shared/head partition;
- exact-zero disconnected head policy Jacobian;
- bit-identical Paper actor and sampled shared-critic algebra;
- bit-identical one-step policy parameters and logits;
- only the value-head delta differs;
- exact head proposal norm match and literal Paper global clip reuse;
- unchanged RNG state/data order and all zero-boundary rules;
- FP64/Jacobi/Cholesky residual `2.616e-16` locally and `1.501e-16` in each
  allocated job's regression phase;
- rejection of joint/shared-GGN/cross/low-Fisher/projection/Kaczmarz and free
  scale fields.

These passed checks are structural/regression evidence only; they do not
replace the mandatory production-network gate.

## Mandatory four-environment preflight

Exactly one no-training job was submitted for each environment on gpuH. All
were owned by `h99859yz`, received one H200, and reached immutable terminal
`FAILED/1:0`:

| Environment | Job | Node | Elapsed | Root status | Scientific start |
|---|---:|---|---:|---|---|
| BigFish | 19238126 | node820 | 00:00:22 | PRECHECK_BLOCKED | no |
| BossFight | 19238127 | node820 | 00:00:22 | PRECHECK_BLOCKED | no |
| CaveFlyer | 19238128 | node822 | 00:00:19 | PRECHECK_BLOCKED | no |
| CoinRun | 19238129 | node823 | 00:00:20 | PRECHECK_BLOCKED | no |

Every job completed the regression phase, then failed identically while the
canonical production harness imported the frozen V2 trainer:

```text
File ".../train_shared_paper_hybrid_head_detggn_papernorm_v2.py", line 15
  import utils.logger as logger
ModuleNotFoundError: No module named 'utils'
```

The production code tree's `utils` package was not present in the fresh
campaign deployment. Thus canonical config resolution, actual production
model construction, actual-network norm-match/one-step/PopArt/structural
proof, production memory footprint and final solver gate were not reached.
This is a preflight/deployment infrastructure failure, not algorithm,
numerical, solver, environment or reward evidence. Task 14 explicitly forbids
field repair after any mandatory preflight failure, so the missing package was
not added and the preflight was not retried.

## Scientific matrix and early-stop table

No V2 scientific job, root, trainer process, transition, progress, metric
trace, checkpoint or model exists. Therefore no Target/Paper 2M, 4M or
5,980,160 ratio exists and no reward early-stop action was eligible. No
scientific monitor was requested or created.

## Evidence and safety

Model-free evidence is tracked under
`remote_launch_staging/procgen_paper_hybrid_head_normmatch_detggn_6m_s0_20260825_14/evidence_preflight/`.
It contains Slurm accounting, per-job hashes, H200 identity, timestamps,
status/rc, regression output and exact compatibility traceback. The staged
payload contains no checkpoint/model. Historical joint-2B, separate-B,
Hybrid V1, low-Fisher, P1, ACTOR_J and infrastructure evidence remains
unchanged.

No retry, requeue, resubmit, scientific launch, Jupyter, quarantined host,
Paper rerun, sweep, second candidate, overwrite or unrelated job mutation
occurred.

PRECHECK_BLOCKED
