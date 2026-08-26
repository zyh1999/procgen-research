# Executor report

Task: `PROCGEN-FULL-SHARED-JOINT2B-STRUCTURAL-ZERO-RECOVERY-AND-6M-S0-20260826-43`

Conclusion: `PRECHECK_BLOCKED`

Frozen trainer/config/science-launcher identities, Task40 production shape,
Task41 canonical oracle and Task42 gather PASS evidence remained exact and
were reused without rebuilding. The Task43 preflight-only helper retained the
complete ordered 26-tensor/938,976-column space, materialized only permitted
structural `None` gradients with `zeros_like`, and passed all role, nonzero,
deletion/reordering and shape/dtype/device negative tests.

Exactly one required gpuH equivalence gate, job `19409128`, failed
`FAILED/1:0` after 15 seconds on node820 with root
`LOCAL_EQUIVALENCE_FAIL/1`. Production construction and oracle identity passed,
but the first actor vmap/reference comparison mismatched 216/216 elements of
shared tensor `backbone_net.conv_layers.0.weight`; maximum absolute error was
`0.8025436401367188`. Complete 512-row actor/critic equivalence was not proven.

This is a local preflight-reference equivalence failure, not algorithm,
solver, GPU or scientific evidence. It was not repaired or rerun. The user
then overrode the stop rule and prohibited further micro/audit gates. The sole
production preflight `19409435` failed `FAILED/1:0` after 14 seconds on node820:
its strict Joint-2B reference equality mismatched 1,035,714/1,048,576 elements,
with maximum absolute difference `1.9206858326015208e-14`. It was not repaired
or retried.

No science job/root/process/transition/trace/checkpoint/model/comparison/
cancellation/monitor exists. Full model-free evidence is in
`.agent/reports/PROCGEN-FULL-SHARED-JOINT2B-STRUCTURAL-ZERO-RECOVERY-AND-6M-S0-20260826-43.md`
and `remote_launch_staging/procgen_full_shared_joint2b_structural_zero_recovery_6m_s0_20260826_43/`.
