# Task-ID: PROCGEN-FULL-SHARED-JOINT2B-ACTOR-GATHER-RECOVERY-AND-6M-S0-20260826-42

Status: READY

Recover only Task41 preflight's actor per-sample log-prob tensor indexing by
replacing `[0, action]` with the specified equivalent `torch.gather`. Keep the
trainer, config, science launcher, Task41 canonical oracle, ordered parameter
collection, Task40 production shape, strict full-shared Joint-2B natural cross
blocks, block-relative normalization, relative damping .5 and all scientific
semantics byte-identical.

Before the production preflight, fixed logits/actions and the actual frozen
production model must prove gather equality for values, logits gradients,
ordered trainable-parameter gradients and the complete 512-row actor Jacobian
against explicit non-vmap indexing. Preserve action boundaries, row/column
order, shape, dtype, inputs and RNG. Reject wrong dtype/range/dimension/reshape,
sign/reduction changes, parameter reorder and forward-only equality with a
different Jacobian. Reuse Task40/41 PASS artifacts without rebuilding them.

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

Task38 remains SUPERSEDED_BEFORE_EXECUTION. Task39 job 19407505, Task40 job
19407880 and Task41 job 19408491 remain immutable and must not be retried or
relabeled. Update the Task42 report, STATE and AGENT_REPORT; push model-free evidence only to
origin/agent-work, verify the Delivery SHA and callback Planner/coordinator
with exactly one allowed conclusion.
