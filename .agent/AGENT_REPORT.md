# Executor report

Task: `PROCGEN-FULL-SHARED-JOINT2B-MANIFEST-ORACLE-RECOVERY-AND-6M-S0-20260826-41`

Conclusion: `PRECHECK_BLOCKED`

Task39 trainer/config/science launcher remained byte-identical. The Task40
production shape gate remained PASS. gpuH local-gate job `19408345` completed
`0:0` and two clean production constructions generated a byte-identical
ordered oracle, SHA256 `62b1b10a81fc89ec621f0ceaf60735864cc899fafd6bda074c7414744506d303`.
It proves 29 model parameter tensors/938,979 elements versus 26 ordered
trainable autograd/Joint-2B tensors/938,976 elements. The exact three-element
difference is `last_v_layer.mean`, `mean_sq` and `debiasing_term`: PopArt
nontraining state retained in the model/optimizer container and excluded from
Jacobians, solver columns and delta. All mandated negative tests passed.

The preflight-only implementation was frozen and pushed at `a5743fb`. The sole
production preflight, gpuH job `19408491`, is scheduler `FAILED/1:0`, elapsed
14 seconds on node820, root `PRECHECK_FAIL/1`. Model/oracle checks passed, then
the actor per-sample Jacobian stopped on PyTorch vmap data-dependent indexing
at `[0, action]`, before complete Jacobian/1024-row/solver evidence. This is a
preflight-harness failure, not algorithm, solver, GPU or scientific evidence.
There is no OOM/CUDA/NCCL/disk/quota/NaN/Inf signature.

No repair or retry was made. No science job/root/process/transition/trace/
checkpoint/model/comparison/cancellation/monitor exists. Task39 `19407505`
and Task40 `19407880` remain unchanged; Task38 remains
`SUPERSEDED_BEFORE_EXECUTION` and absent. Full model-free evidence is in
`.agent/reports/PROCGEN-FULL-SHARED-JOINT2B-MANIFEST-ORACLE-RECOVERY-AND-6M-S0-20260826-41.md`
and `remote_launch_staging/procgen_full_shared_joint2b_manifest_oracle_recovery_6m_s0_20260826_41/`.
