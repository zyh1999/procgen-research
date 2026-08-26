# Task-ID: PROCGEN-FULL-SHARED-JOINT2B-STRUCTURAL-ZERO-RECOVERY-AND-6M-S0-20260826-43

Status: READY

Recover only Task42's explicit actor/critic reference handling for structurally
disconnected parameters. Use `allow_unused=True` only in the preflight
reference, permit actor `None` only for critic-exclusive value-head tensors and
critic `None` only for policy-exclusive tensors, and materialize each allowed
entry as `zeros_like` without deleting or reordering any of the 938,976 columns.
Keep trainer, config, science launcher, Task40 shape, Task41 oracle, Task42
gather, strict full-shared Joint-2B natural cross blocks, block-relative
normalization, relative damping .5 and all scientific semantics byte-identical.

Before the production preflight, the production model must prove complete
512-row actor and critic vmap Jacobians equal explicit per-sample references,
with actor value-head 257 columns and critic policy-head 3,855 columns strictly
zero, shared columns connected in both, full oracle order/metadata preserved,
and input/model/RNG/optimizer/PopArt state unchanged. Reject disallowed-role
None, connected gradients replaced by zero, deleted/reordered columns and
wrong shape/dtype/device. Reuse Task40/41/42 PASS artifacts without rebuilding.

Only after the equivalence gate PASS, submit exactly one production preflight.
It must reuse and match the Task41 oracle, verify the complete production
512+512 strict Joint-2B system and 938,976 columns, full shared actor/critic
coverage, nonzero direct-reference cross blocks, reconstruction, normalized
scales, positive block-rescaling and PopArt invariance, FP64 Cholesky/residual
and hard-error/nonfinite gates. Any failure is terminal PRECHECK_BLOCKED with
no repair or retry.

Only after PRECHECK_PASS, submit exactly one fresh seed0 intended-6M cell for
BigFish, BossFight, CaveFlyer and CoinRun. Compare only exact same-transition
immutable Paper seed0 rows at first common >=2M, first common >=4M and
5,980,160; cancel a cell only for Target/Paper < .60. No retry, requeue,
resubmit, sweep, second candidate, extra seed, Paper rerun, provenance-observer
framework, Jupyter or quarantined access.

Task38 remains SUPERSEDED_BEFORE_EXECUTION. Task39–42 jobs remain immutable and
must not be retried or relabeled. Update the Task43 report, STATE and
AGENT_REPORT; push model-free evidence only to
origin/agent-work, verify the Delivery SHA and callback Planner/coordinator
with exactly one allowed conclusion.

## User execution override after local gate 19409128

The user superseded the local-gate stop rule after its terminal evidence was
preserved. Do not add or rerun micro/negative/audit gates. Use only the already
passed structural-zero role/shape/dtype/device compatibility checks as the
immediate-crash guard, then run the sole production preflight. If production
model/Jacobian/solver construction completes without an immediate hard error,
submit the four authorized science cells once.
