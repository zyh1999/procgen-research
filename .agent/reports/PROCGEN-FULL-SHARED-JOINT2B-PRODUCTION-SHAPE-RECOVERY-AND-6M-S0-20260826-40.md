# PROCGEN-FULL-SHARED-JOINT2B-PRODUCTION-SHAPE-RECOVERY-AND-6M-S0-20260826-40

## Current conclusion

`PRECHECK_BLOCKED`

Task40 changes only the production-network preflight shape resolver. Task39
trainer/config/science launcher and method remain byte-identical. Task38 is
still `SUPERSEDED_BEFORE_EXECUTION`; Task39 job `19407505` remains immutable
`FAILED/1:0` and is not retried.

The corrected harness creates the real Procgen environment using the frozen
config, reads its HWC RGB observation space, passes the derived height to the
same `build_resnet` call used by training, and records the CHW model-input
shape. Negative gates reject channel-as-image-size, missing/swapped/nonproduction
dimensions, wrong channels, and a production parameter-count drift.

## Frozen identities

| Item | SHA256 |
|---|---|
| Task39 trainer | `b13b3eae445f83f0e1d6565cb942c8390b6cd143fcc7f366410406c370aa0815` |
| Task39 config | `1b0cf73885dcb7c61078e463f826d6b296abfedb5a6a019f6782c6b899896b52` |
| Task39 science launcher | `59dea11cb0fed21974f4985d2eb2016703dd33f06d2e11daa7ca419d5a888fb4` |
| corrected Task40 preflight | `418bb8bb720c1adc69b8da76b9fe8bb60cf844ee521b8483b6dbe3aa1ee23871` |
| Task40 preflight launcher | `f670cfa774ced7462cfc9bbc29acf70c2bbdb9ce00474e460b9a12cb4f5133fd` |
| shape regression | `7a15d932a044f7647d336f3bf6bd0a5a8fd2450cfc8b198855766c68e38cd06f` |

The three scientific files remained byte-identical. Assignment/preflight-only
freeze commit is `7208d6c2e5aa45ec5971625548ee3ee467ab33b1`.

## Shape regression and construction chain

The minimal Python 3.9/PyTorch regression emitted
`TASK40_SHAPE_SEMANTICS_PASS`. It proves production HWC `(64,64,3)` resolves
to ResNet image-size argument `64` and model input CHW `(3,64,64)`. It rejects
channel-as-image-size, missing dimensions, channels-first/swapped layout,
nonproduction spatial dimensions and wrong channel count.

The corrected formal harness creates the real `ProcgenEnv` from the frozen
config, applies `VecExtractDictObs(...,"rgb")` and `VecMonitor`, reads its
actual observation/action spaces, and then calls the same `build_resnet` and
`SharedActorCritic` construction used by the trainer. No mock, reduced network
or alternate encoder is used. This corrected path successfully passed the
former zero-spatial-size failure and constructed the production model.

## Sole production preflight

The only Task40 formal preflight was gpuH job `19407880`: scheduler
`FAILED/1:0`, elapsed `00:00:20`, node820; root `PRECHECK_FAIL/1`. The root
records the exact frozen trainer/config/preflight hashes and an NVIDIA H200.

The mandatory parameter-manifest gate then measured `938,976` trainable
parameters, while the preflight-only frozen expectation was `938,979`, and
raised before per-sample Jacobians or the Joint-2B solve:

```text
RuntimeError: production parameter manifest drift: 938976 != 938979
```

This is `precheck-failure/parameter-manifest-expected-count-mismatch`. It is a
preflight-only identity assertion failure, not algorithm, solver, numerical,
GPU or scientific evidence. No OOM, CUDA, NCCL, disk/quota or NaN/Inf failure
is present. Per the explicit one-shot contract, the expectation was not
corrected and the preflight was not resubmitted.

## Scheduler, artifacts and science

Task39 job `19407505` remains immutable `FAILED/1:0`; it was not retried or
relabelled. Task38 remains `SUPERSEDED_BEFORE_EXECUTION` with no implementation,
job, root or artifact. Task40 produced only its preflight status/rc, GPU,
hashes, stdout/stderr and scheduler evidence. No Task40 science job/root,
trainer process, transition, trace, checkpoint/model, Paper comparison,
cancellation or monitor exists.

## Failure ledger and conclusion

| Item | State | Classification |
|---|---|---|
| Task38 | absent | `SUPERSEDED_BEFORE_EXECUTION` |
| Task39 `19407505` | unchanged FAILED/1:0 | historical harness image-size failure |
| Task40 shape regression | PASS | preflight-only regression |
| Task40 `19407880` | FAILED/1:0; PRECHECK_FAIL/1 | expected parameter-count mismatch before Jacobians |
| four Task40 science cells | not submitted | correctly blocked |

The unique terminal conclusion is `PRECHECK_BLOCKED`.
