#!/usr/bin/env python3
"""Exact Task32-to-Task33 single-causal-delta audit."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
task32 = root.parent / "procgen_actor_weighted_gae_ggn_head_6m_s0_20260825_32"
source32 = task32 / "train_shared_det_actor_weighted_gae_ggn_head_v1.py"
source33 = root / "train_shared_det_gae_ggn_head_widentity_v1.py"
config32 = task32 / "adv_resnet_shared_det_actor_weighted_gae_ggn_head_v1_6m.yaml"
config33 = root / "adv_resnet_shared_det_gae_ggn_head_widentity_v1_6m.yaml"
expected_diff = root / "TASK32_TO_TASK33_TRAINER.diff"

for path in (source32, source33, config32, config33, expected_diff):
    assert path.is_file() and path.stat().st_size > 0, path

text32, text33 = source32.read_text(), source33.read_text()
tree32, tree33 = ast.parse(text32), ast.parse(text33)
actual_diff = "".join(difflib.unified_diff(
    text32.splitlines(keepends=True), text33.splitlines(keepends=True),
    fromfile="Task32/train_shared_det_actor_weighted_gae_ggn_head_v1.py",
    tofile="Task33/train_shared_det_gae_ggn_head_widentity_v1.py",
))
changed_lines = lambda value: [
    line for line in value.splitlines()
    if line[:1] in {"+", "-"} and line not in {"+", "-"}
    and not line.startswith(("+++", "---"))
]
assert changed_lines(actual_diff) == changed_lines(expected_diff.read_text())
assert config32.read_bytes() == config33.read_bytes()

functions32 = {node.name: node for node in tree32.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
functions33 = {node.name: node for node in tree33.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
assert "actor_score_weights" in functions32 and "actor_score_weights" not in functions33
assert "validate_actor_weighted_gae_config" in functions32
assert "validate_gae_widentity_config" in functions33
for unchanged in ("gae_error_operator_apply", "solve_gae_head_primal_fp64", "flat_group", "categorical_kl_from_logits"):
    assert ast.dump(functions32[unchanged], include_attributes=False) == ast.dump(functions33[unchanged], include_attributes=False)

for forbidden in (
    "actor_score_weights", "actor_weight_", "sqrt_weights", "match_head_proposal_norm",
    "paper_head_proposal", "weight clipping", "weight floor", "norm_match",
):
    assert forbidden not in text33, forbidden
for required in (
    "head_rows = selected_J", "head_rhs = selected_q",
    "head_critic_implicit_weight=1.0",
    "head_critic_kernel_mode='identity_weighted_exact_gae_value_head_primal'",
    "gae_unweighted_objective=lag_objective.item()",
):
    assert required in text33, required

ledger = {
    "task_id": "PROCGEN-GAE-GGN-HEAD-WIDENTITY-6M-S0-20260825-33",
    "method": "DET_GAE_GGN_HEAD_WIDENTITY_V1",
    "task32_trainer_sha256": hashlib.sha256(source32.read_bytes()).hexdigest(),
    "task33_trainer_sha256": hashlib.sha256(source33.read_bytes()).hexdigest(),
    "task32_task33_config_byte_identical": True,
    "task32_task33_config_sha256": hashlib.sha256(config33.read_bytes()).hexdigest(),
    "exact_unified_diff_sha256": hashlib.sha256(actual_diff.encode()).hexdigest(),
    "sole_scientific_delta": "diag(sqrt(actor_score_weight)) removed; W=I for K=D J_h and r=q",
    "actor_score_or_probability_weight_path": "ABSENT",
    "implicit_weight_min": 1.0,
    "implicit_weight_max": 1.0,
    "task32_max_weight_512_concentration_path": "IMPOSSIBLE_ABSENT",
}
(root / "task32_task33_diff_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
print("TASK33_EXACT_TASK32_SINGLE_CAUSAL_DELTA_PASS")
print(json.dumps(ledger, sort_keys=True))
