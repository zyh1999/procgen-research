# Task43: Full-Shared Joint-2B Structural-Zero Recovery

## Frozen identity

- Task: `PROCGEN-FULL-SHARED-JOINT2B-STRUCTURAL-ZERO-RECOVERY-AND-6M-S0-20260826-43`
- Method: `FULL_SHARED_JOINT2B_SCALE_RECOVERY_V1`
- Parent delivery: Task42 `113799764cd835ac97ee6d06295fb1433ce4aeca`
- Trainer SHA256: `b13b3eae445f83f0e1d6565cb942c8390b6cd143fcc7f366410406c370aa0815`
- Config SHA256: `1b0cf73885dcb7c61078e463f826d6b296abfedb5a6a019f6782c6b899896b52`
- Science-launcher SHA256: `59dea11cb0fed21974f4985d2eb2016703dd33f06d2e11daa7ca419d5a888fb4`
- Canonical Task41 oracle SHA256: `62b1b10a81fc89ec621f0ceaf60735864cc899fafd6bda074c7414744506d303`
- Task40 production-shape, Task41 manifest-oracle and Task42 gather PASS artifacts were reused without rebuilding.
- Task38 remains `SUPERSEDED_BEFORE_EXECUTION`; jobs `19407505`, `19407880`, `19408491` and `19408837` remain immutable.

The frozen trainer, config, science launcher, strict full-shared Joint-2B formula,
natural cross blocks, block-relative normalization and dimensionless damping
`0.5` were not modified.

## Bounded preflight-only change

The Task43 helper uses `torch.autograd.grad(..., allow_unused=True)` only in the
explicit preflight reference and retains the complete Task41 ordered 26-tensor,
938,976-column trainable collection. A `None` actor gradient is permitted only
for `CRITIC_EXCLUSIVE` tensors and a `None` critic gradient only for
`POLICY_EXCLUSIVE` tensors; permitted values are materialized with
`zeros_like`. Shape, dtype and device must match the original parameter, and
the disconnected-role tensor must be strictly zero even if autograd returns a
tensor rather than `None`. No column is removed or reordered.

Preflight-only SHA256 identities:

- structural-zero helper: `ee5ca8f929efdec86b9d9fcd4f61913e3bc93497c79650f5c3bd548d2954c94d`;
- production 512-row equivalence gate: `db200b9f4efd9e7a0c97bfe8b259b32884a658d134deed4fb08858139f997a98`;
- negative regression: `7125d2014bef1de9e51046623464347ce7717d93c755c566c2658be4dfba5508`;
- local-gate launcher: `52844d78e1d64e96b538623bc27ffe14592aece7e6afc5d45be589392592e4aa`;
- original unused production-preflight launcher: `8be0d03ca3dabfd44bf5ee4af4e8629011526ffd5ce9408b8507eb90c9667569`;
- user-override shortest-path deployment launcher: `369aceab773b6c1185df44b3c2bafab72d23442dfb9cf461d6e5e7156d169714`.

## Required one-shot equivalence gate

Before submission, live gpuH state showed the authorized account/QOS, mixed
H200 capacity, no Task43 duplicate and an absent Task43 campaign root. The
Task41 root reported `LOCAL_GATE_PASS`, its oracle hash was exact, and Task42's
frozen tensor gather evidence contained
`TASK42_GATHER_VALUE_LOGITS_GRAD_NEGATIVE_PASS`.

Exactly one Task43 equivalence gate was submitted:

| job | partition/account/QOS | scheduler | elapsed | node | root status/rc |
|---|---|---|---:|---|---|
| `19409128` | `gpuH` / `gpu-h200-fse-pgdr` / `gpu-h200-fse` | `FAILED/1:0` | 15 s | node820 | `LOCAL_EQUIVALENCE_FAIL/1` |

The structural-zero positive and negative rules passed. The negative suite
rejected actor `None` on shared/policy tensors, critic `None` on shared/value
tensors, nonzero disconnected-role tensors, zero-column deletion, column
reordering, wrong shape/dtype/device and a connected-gradient-zero control.

The production gate reconstructed the real model and matched the Task41
oracle before entering the 512-row comparison. It then failed at the first
actor vmap/reference parameter comparison, the shared tensor
`backbone_net.conv_layers.0.weight` with shape `[8,3,3,3]` and 216 elements:

```text
Mismatched elements: 216 / 216 (100.0%)
Greatest absolute difference: 0.8025436401367188 at index (5, 2, 0, 0)
Greatest relative difference: 1.9264323711395264 at index (5, 0, 2, 1)
```

Thus actor gather Jacobian versus explicit per-sample reference equivalence
was not established, and the gate stopped before a complete 512-row actor and
critic comparison. Consequently there is no valid complete structural-unused
count, strict `1024 x 938976` reference identity, block/cross/direct-reference,
solver or actual-delta evidence for Task43. The traceback is the sole hard
error match; there is no OOM, CUDA, NCCL, disk/quota or NaN/Inf signature.

This is classified as `local-gate/preflight-reference-equivalence-failure`,
not algorithm, numerical-solver, GPU or scientific evidence. It was not
repaired or rerun.

## User override and sole production preflight

After the gate failed and its evidence was preserved, the user explicitly
superseded its stop rule: no more micro/negative/audit gates, use only the
already-passed structural-zero compatibility checks to prevent an immediate
crash, and submit the sole production preflight. The frozen scientific files
were unchanged. A minimal deployment launcher checked that compatibility
marker and the unchanged oracle hash; it did not rerun or rebuild shape,
oracle or gather evidence.

Exactly one production preflight was submitted:

| job | partition/account/QOS | scheduler | elapsed | node | root status/rc |
|---|---|---|---:|---|---|
| `19409435` | `gpuH` / `gpu-h200-fse-pgdr` / `gpu-h200-fse` | `FAILED/1:0` | 14 s | node820 | `PRECHECK_FAIL/1` |

It reconstructed the production model and entered the Joint-2B numerical
reference, then the frozen strict equality assertion failed:

```text
Mismatched elements: 1035714 / 1048576 (98.8%)
Greatest absolute difference: 1.9206858326015208e-14 at index (612, 902)
Greatest relative difference: 1.1036722475186101e-08 at index (47, 627)
```

The failure is a production-preflight direct-reference equality failure. It
does not establish complete Jacobian/block/cross/solver/delta PASS evidence and
is not a scientific result. The traceback is the only hard-error match; there
is no OOM, CUDA, NCCL, disk/quota or NaN/Inf signature. The preflight was not
modified or retried.

## Science absence

Because the sole production preflight failed:

- no Task43 science job/root/process/transition/progress/trace/checkpoint/model exists;
- no Paper comparison, cancellation or monitor exists;
- no historical Task39--42 job was retried, requeued, resubmitted, overwritten or relabeled.

Downloaded model-free evidence is under
`remote_launch_staging/procgen_full_shared_joint2b_structural_zero_recovery_6m_s0_20260826_43/evidence_remote/`.
The terminal stderr SHA256 is
`cf0e61215e610022f7070f50123a5a4d26f8c1af17fb037fac010382f7f9ad99`
for the local gate and
`9b67e27c4646bbcf7435b4b6ee20d3dafb79444b2cc6b53f00c90412beed4381`
for the production preflight.
No model or checkpoint is included.

## Conclusion

`PRECHECK_BLOCKED`
