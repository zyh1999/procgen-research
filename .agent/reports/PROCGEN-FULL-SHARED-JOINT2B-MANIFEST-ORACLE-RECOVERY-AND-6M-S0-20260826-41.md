# Task41: Full-Shared Joint-2B Manifest Oracle Recovery

## Identity

- Task: `PROCGEN-FULL-SHARED-JOINT2B-MANIFEST-ORACLE-RECOVERY-AND-6M-S0-20260826-41`
- Method: `FULL_SHARED_JOINT2B_SCALE_RECOVERY_V1`
- Frozen trainer SHA256: `b13b3eae445f83f0e1d6565cb942c8390b6cd143fcc7f366410406c370aa0815`
- Frozen config SHA256: `1b0cf73885dcb7c61078e463f826d6b296abfedb5a6a019f6782c6b899896b52`
- Frozen science-launcher SHA256: `59dea11cb0fed21974f4985d2eb2016703dd33f06d2e11daa7ca419d5a888fb4`
- Task38 remains `SUPERSEDED_BEFORE_EXECUTION`; no Task38 implementation, job,
  root or artifact exists.
- Task39 preflight `19407505` and Task40 preflight `19407880` remain immutable
  terminal failures and were not retried or relabeled.

## Production oracle local gate

The oracle is generated only through the frozen real production path:

`ProcgenEnv -> VecExtractDictObs(rgb) -> VecMonitor -> build_resnet -> SharedActorCritic`

The observed space is HWC `(64,64,3)`, the model input is CHW `(3,64,64)`,
and ResNet receives image size `64`. The Task40 negative shape gate remains
PASS. Two independent clean Python 3.9/PyTorch production constructions in
gpuH job `19408345` emitted byte-identical canonical JSON:

- scheduler: `COMPLETED/0:0`, node820, 22 seconds;
- root: `LOCAL_GATE_PASS/rc0`;
- oracle SHA256:
  `62b1b10a81fc89ec621f0ceaf60735864cc899fafd6bda074c7414744506d303`.

The ordered model has 29 parameter tensors and 938,979 parameter elements.
The ordered trainable/autograd/Joint-2B column collection has 26 tensors and
938,976 elements. The exact three-element difference is production model state
with `requires_grad=False`:

| name | numel | production meaning |
|---|---:|---|
| `last_v_layer.mean` | 1 | PopArt running mean |
| `last_v_layer.mean_sq` | 1 | PopArt running second moment |
| `last_v_layer.debiasing_term` | 1 | PopArt debiasing state |

All three remain in the production model and optimizer parameter container,
but are excluded from its ordered trainable subset, autograd Jacobians,
Joint-2B columns, solver and delta. There are no separately registered model
buffers in this construction. The oracle records every name, order, shape,
numel, dtype, requires-grad flag, role, optimizer membership and Joint-2B
membership. The optimizer objects are item-by-item identical to the model
objects, and its filtered trainable order is item-by-item identical to the
autograd and Joint-2B column order.

The required negative suite passed, rejecting missing, extra, duplicate,
reordered, shape/numel/dtype/requires-grad/role changes, optimizer/Joint-2B
membership mismatch, nontraining state in the solver, same-total different
membership, trainer/config/construction drift, and edited/hash-mismatched
oracle data.

## Production preflight

The preflight/oracle implementation was frozen and pushed before the formal
job at commit `a5743fb`. Preflight-only SHA256 identities are:

- oracle implementation: `f52ad732b2b213aef35e377f0bc84026e4fa9f43f55313059dbbf22aa829c1b5`;
- production preflight: `cd87d3bbc9030df2eed536cea1b0acf8c88bf3f6b858757e69ced040ca0f9660`;
- negative regression: `f797324ddd3e69a8dcd94b8a9e24235db5794325c9572019fbc85348583ef81d`;
- formal launcher: `ce6e14a01764ec1d0f9f4af969eb38be11d4ff8408f9023cebb71cf3073ecfdb`.

Exactly one formal production preflight was submitted after fresh gpuH,
ownership, duplicate and root checks:

| job | scheduler | elapsed | node | root status/rc |
|---|---|---:|---|---|
| `19408491` | `FAILED/1:0` | 14 seconds | node820 | `PRECHECK_FAIL/1` |

The production model construction, exact source hashes and frozen oracle
comparison completed successfully; the earlier Task39 image-size failure and
Task40 count mismatch did not recur. The job then failed while constructing
the actor per-sample Jacobian, before actor/critic Jacobians, strict 1024-row
system or solver evidence was complete. The exact exception is:

```text
RuntimeError: vmap: It looks like you're either (1) calling .item() on a Tensor
or (2) attempting to use a Tensor in some data-dependent control flow ...
```

It arose from `torch.log_softmax(logits, -1)[0, action]`, where the vmapped
scalar action was used as an indexing operand. This is classified
`precheck-failure/preflight-harness-vmap-data-dependent-indexing`, not an
algorithm, solver, numerical, GPU or scientific result. The required
hard-error scan contains the expected Python traceback but no OOM, CUDA,
NCCL, disk/quota or NaN/Inf failure signature. No `preflight.json` or PASS
marker was produced.

The single-preflight rule was honored: the harness was not repaired and job
`19408491` was not retried, requeued or resubmitted.

## Science matrix

Because the sole production preflight did not PASS, no Task41 science job,
root, process, transition, progress, trace, checkpoint, model, Paper comparison,
cancellation or monitor exists. A post-terminal scheduler/process/artifact
scan confirmed that only the unrelated multicore tunnel remained live.

## Conclusion

`PRECHECK_BLOCKED`
