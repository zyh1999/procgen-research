# Task42: Full-Shared Joint-2B Actor Gather Recovery

## Frozen identity

- Task: `PROCGEN-FULL-SHARED-JOINT2B-ACTOR-GATHER-RECOVERY-AND-6M-S0-20260826-42`
- Method: `FULL_SHARED_JOINT2B_SCALE_RECOVERY_V1`
- Trainer SHA256: `b13b3eae445f83f0e1d6565cb942c8390b6cd143fcc7f366410406c370aa0815`
- Config SHA256: `1b0cf73885dcb7c61078e463f826d6b296abfedb5a6a019f6782c6b899896b52`
- Science-launcher SHA256: `59dea11cb0fed21974f4985d2eb2016703dd33f06d2e11daa7ca419d5a888fb4`
- Reused canonical Task41 oracle SHA256:
  `62b1b10a81fc89ec621f0ceaf60735864cc899fafd6bda074c7414744506d303`.
- Task40 production HWC/CHW/image-size evidence and Task41 oracle evidence
  were reused without rebuilding or extension.
- Task38 remains `SUPERSEDED_BEFORE_EXECUTION`; Task39–41 terminal jobs and
  roots remain immutable.

## Sole preflight-only change

The Task41 preflight source SHA256
`cd87d3bbc9030df2eed536cea1b0acf8c88bf3f6b858757e69ced040ca0f9660`
contains exactly one actor selection expression:

```python
torch.log_softmax(logits, -1)[0, action]
```

The bounded Task42 wrapper verifies that frozen source hash and replaces only
that unique expression with the specified last-dimension `torch.gather`, then
returns the selected single-sample log-prob as a scalar. No trainer, actor
sign/probability/RHS, parameter, data, RNG, reduction or scientific command was
changed.

Preflight-only SHA256 identities:

- gather helper: `cf81caa69cf8b3424d0498f62793a4bf2f02f27c0927e88d9f08cd3a1b20f705`;
- production equivalence gate: `4421cd40513e06994975b7f196205205b9e10d12582d423a73dedf4a4ac79b77`;
- exact source-replacement wrapper: `29b2b3c2b3fd5166443ee3322859ea3f85e25365cdfaf436372482ac4f457250`;
- tensor/negative regression: `2c50cd3b397973fdf29345c2d05ed9adec50cb7ad3275bb4e39b2aed8e1d992f`;
- local-gate launcher: `110f1d8cb6ded7a7c07834ce9a334af8e530f32c678914577ca7fb3d2d9a3ea1`;
- unused formal-preflight launcher: `0c145a574c202c13878fc6e32ea4719fbb53507c06c02661a62bf939dd5a5277`.

## Required equivalence gate

After fresh gpuH, duplicate and root checks, exactly one required equivalence
gate was submitted:

| job | scheduler | elapsed | node | root status/rc |
|---|---|---:|---|---|
| `19408837` | `FAILED/1:0` | 15 seconds | node820 | `LOCAL_EQUIVALENCE_FAIL/1` |

The fixed-logits/action portion passed completely:

- gather versus explicit value maximum error: `0`;
- gather versus explicit logits-gradient maximum error: `0`;
- first action `0`, last action `14` and internal boundary actions were used;
- wrong dtype, negative/out-of-range action, wrong dimension/reshape,
  sign/reduction changes and forward-only equality with a changed Jacobian
  were all rejected.

The production-network portion rebuilt neither shape nor oracle; it reused the
Task41 construction and verified the canonical ordered 26-tensor/938,976-column
collection. It then failed at the first explicit actor parameter-gradient
reference because `torch.autograd.grad` was invoked over the complete ordered
trainable collection with `allow_unused=False`, while the critic-exclusive
value-head tensors are structurally absent from the actor log-prob graph:

```text
RuntimeError: One of the differentiated Tensors appears to not have been used
in the graph. Set allow_unused=True if this is the desired behavior.
```

Therefore no complete production-parameter gradient or 512-row Jacobian
equivalence evidence was produced. This is classified
`local-gate/preflight-test-structural-unused-value-head`, not algorithm,
solver, GPU or scientific evidence. The traceback is the only hard-error
signature; there is no OOM, CUDA, NCCL, disk/quota or NaN/Inf evidence.

Per the one-shot gate rule, this failure was not repaired or retried. No formal
Task42 production preflight was submitted.

## Science and artifacts

Because the required equivalence gate failed, no Task42 science job, root,
process, transition, progress, trace, checkpoint/model, Paper comparison,
cancellation or monitor exists. Post-terminal checks confirmed the formal
preflight root and science root tree are absent and no Task42 trainer remains.

## Conclusion

`PRECHECK_BLOCKED`
