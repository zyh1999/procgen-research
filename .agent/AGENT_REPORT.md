# Executor Report

## Metadata

- Task-ID: `PROCGEN-PAPER-HYBRID-HEAD-DETGGN-6M-S0-20260824-08`
- Method: `PAPER_HYBRID_SHARED_PAPER_HEAD_DETGGN_V1`
- Planner: ChatGPT thread `6a8309ee-bb34-83eb-9512-72acc5913334`
- Assigned task commit: `72c59e4125e824c63c3555a6235acc17fdf52125`
- Frozen implementation commit: `fe4b8a58812e80689705abec11364457cae31e26`
- Preserved correction/evidence commit: `896f54459b53f9f489951fb3c9f9ed5fec32c11e`
- Repository target: `origin/agent-work`

## Result

The frozen static identity audit and CSF3 numerical regression passed. They
proved exact Paper actor and sampled shared-trunk critic preservation, an
exhaustive mutually exclusive policy/shared/critic-head partition, an exactly
zero-disconnected policy-logit Jacobian for the critic-exclusive head,
bit-identical one-step policy parameters/logits, and a changed value-head delta
only. Historical expected/no-cross/block-trace, joint-2B, and separate-B
formula distinctness also passed.

The mandatory actual-network H200 preflight did not pass. Job `19220448`
failed before compatibility testing because the launcher omitted the staged
code directory from Python's import path. After an infrastructure-only,
versioned correction, job `19220752` imported the target but failed before the
actual-network partition and memory proof because the preflight constructor
namespace omitted required `norm_obs`. Both jobs are immutable
`infrastructure-failure/preflight-design`, FAILED/1:0 after 15 seconds on
node820. Neither is an algorithm, numerical, solver, or hardware compatibility
result.

Because mandatory conditions 2--3 and the actual-network/H200 realization of
condition 8 remain unproven, no scientific job was submitted. Final scheduler,
root, and process reconciliation found no live target, root, trainer,
transition, progress, trace, checkpoint, or model. The four exact-stage table
is therefore not evaluable, and all cells remain unlaunched rather than failed
scientifically.

Full hashes, diff identity, regression output, resource/duplicate audit,
failure ledger, and raw preflight text evidence are in
`.agent/reports/PROCGEN-PAPER-HYBRID-HEAD-DETGGN-6M-S0-20260824-08.md` and
`remote_launch_staging/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/`.
No model/checkpoint is committed. Joint-2B `GATE_FAIL`, separate-B
`CANDIDATE_NOT_READY`, low-Fisher `GUARD_NOT_HELPFUL`, P1/ACTOR_J failures,
and prior infrastructure/cancellation provenance remain immutable in STATE.

PRECHECK_BLOCKED
