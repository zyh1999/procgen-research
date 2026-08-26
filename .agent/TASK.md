# Task-ID: PROCGEN-FULL-SHARED-JOINT2B-MANIFEST-ORACLE-RECOVERY-AND-6M-S0-20260826-41

Status: READY

Recover only Task39/40's production-network parameter-manifest preflight by
replacing the unexplained trainable-count scalar with a versioned ordered
oracle generated from the frozen trainer/config and real Procgen production
model-construction path. Explain the 938,979 versus 938,976 difference by exact
name and production semantics. Keep trainer, config, science launcher, method,
Task40 production shape semantics, strict full-shared Joint-2B block-relative
normalization, relative damping .5 and all science semantics byte-identical.

The oracle must bind the frozen source hashes, production construction entry,
HWC/CHW shape, ordered names/shapes/numel/dtypes/requires-grad/roles, actual
optimizer membership, ordered optimizer-trainable/autograd/Joint-2B column
membership, and nontraining state. Two independent clean production
constructions must emit byte-identical JSON. Mandatory negative gates reject
missing/extra/duplicate/reordered or changed members, role/membership drift,
nontraining state entering the solver, source/construction drift and edited or
hash-mismatched oracle data.

Only after all local gates PASS, submit exactly one production preflight. It
must regenerate and compare the oracle item-by-item, then verify the complete
production 512+512 strict Joint-2B system, full shared actor/critic coverage,
nonzero direct-reference cross blocks, reconstruction, normalized scales,
positive block-rescaling and PopArt invariance, FP64 Cholesky/residual and
nonfinite/hard-error gates. Any failure is terminal PRECHECK_BLOCKED with no
repair or retry.

Only after PRECHECK_PASS, submit exactly one fresh seed0 intended-6M cell for
BigFish, BossFight, CaveFlyer and CoinRun. Compare only exact same-transition
immutable Paper seed0 rows at first common >=2M, first common >=4M and
5,980,160; cancel a cell only for Target/Paper < .60. No retry, requeue,
resubmit, sweep, second candidate, extra seed, Paper rerun, provenance-observer
framework, Jupyter or quarantined access.

Task38 remains SUPERSEDED_BEFORE_EXECUTION. Task39 job 19407505 and Task40 job
19407880 remain immutable and must not be retried or relabeled. Update the
Task41 report, STATE and AGENT_REPORT; push model-free evidence only to
origin/agent-work, verify the Delivery SHA and callback Planner/coordinator
with exactly one allowed conclusion.
