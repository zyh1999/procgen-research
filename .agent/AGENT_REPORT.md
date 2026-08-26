# Executor report

Task: `PROCGEN-FULL-SHARED-JOINT2B-ACTOR-GATHER-RECOVERY-AND-6M-S0-20260826-42`

Conclusion: `PRECHECK_BLOCKED`

Trainer/config/science launcher, Task40 shape semantics and Task41 canonical
oracle SHA `62b1b10a81fc89ec621f0ceaf60735864cc899fafd6bda074c7414744506d303`
remained exact and were reused without rebuilding. Task42 versions only the
preflight actor selection as last-dimension tensor gather and its bounded
equivalence tests.

Exactly one required gpuH equivalence gate, job `19408837`, failed
`FAILED/1:0` after 15 seconds on node820 with root
`LOCAL_EQUIVALENCE_FAIL/1`. Fixed gather versus explicit values and logits
gradients are bit-identical with maximum errors zero; boundary actions and all
required tensor-level negative cases pass. The production ordered collection
then stopped at the first explicit full-parameter actor-gradient reference:
`torch.autograd.grad(..., allow_unused=False)` rejected the structurally unused
critic-exclusive value-head parameters. No complete ordered parameter-gradient
or 512-row Jacobian equivalence was produced.

This is a local preflight-test structural-unused-value-head failure, not
algorithm, solver, GPU or scientific evidence. Per the one-shot rule it was
not repaired or retried, and no formal production preflight or science job/root
/process/transition/trace/checkpoint/model/comparison/cancellation/monitor
exists. Task39–41 jobs remain unchanged; Task38 remains
`SUPERSEDED_BEFORE_EXECUTION`. Full model-free evidence is in
`.agent/reports/PROCGEN-FULL-SHARED-JOINT2B-ACTOR-GATHER-RECOVERY-AND-6M-S0-20260826-42.md`
and `remote_launch_staging/procgen_full_shared_joint2b_actor_gather_recovery_6m_s0_20260826_42/`.
