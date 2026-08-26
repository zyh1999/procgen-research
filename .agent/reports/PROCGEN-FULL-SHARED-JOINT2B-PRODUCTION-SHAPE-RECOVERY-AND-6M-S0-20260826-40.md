# PROCGEN-FULL-SHARED-JOINT2B-PRODUCTION-SHAPE-RECOVERY-AND-6M-S0-20260826-40

## Current conclusion

`CANDIDATE_NOT_READY`

Task40 changes only the production-network preflight shape resolver. Task39
trainer/config/science launcher and method remain byte-identical. Task38 is
still `SUPERSEDED_BEFORE_EXECUTION`; Task39 job `19407505` remains immutable
`FAILED/1:0` and is not retried.

The corrected harness creates the real Procgen environment using the frozen
config, reads its HWC RGB observation space, passes the derived height to the
same `build_resnet` call used by training, and records the CHW model-input
shape. Negative gates reject channel-as-image-size, missing/swapped/nonproduction
dimensions, wrong channels, and a production parameter-count drift.

Preflight and science identities/results will be appended after the single
authorized production preflight.
