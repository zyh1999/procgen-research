#!/usr/bin/env python3
"""Static single-causal-change and forbidden-mechanism audit for Task14."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "procgen_paper_hybrid_head_detggn_6m_s0_20260824_08" / "train_shared_paper_hybrid_head_detggn_v1.py"
V2 = HERE / "train_shared_paper_hybrid_head_detggn_papernorm_v2.py"
V1_CONFIG = HERE.parent / "procgen_paper_hybrid_head_detggn_6m_s0_20260824_08" / "adv_resnet_shared_paper_hybrid_head_detggn_v1_6m.yaml"
V2_CONFIG = HERE / "adv_resnet_shared_paper_hybrid_head_detggn_papernorm_v2_6m.yaml"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(V1) == "7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54"
assert V1_CONFIG.read_bytes() == V2_CONFIG.read_bytes()
assert sha(V2_CONFIG) == "9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda"

v1_text = V1.read_text()
v2_text = V2.read_text()
v1_tree = ast.parse(v1_text)
v2_tree = ast.parse(v2_text)
v1_functions = {node.name: node for node in v1_tree.body if isinstance(node, ast.FunctionDef)}
v2_functions = {node.name: node for node in v2_tree.body if isinstance(node, ast.FunctionDef)}
assert set(v2_functions) == set(v1_functions) | {"match_head_proposal_norm"}
for name in sorted(set(v1_functions) - {"validate_hybrid_head_config", "learn"}):
    assert ast.dump(v1_functions[name], include_attributes=False) == ast.dump(
        v2_functions[name], include_attributes=False
    ), name

helper = v2_functions["match_head_proposal_norm"]
helper_text = ast.get_source_segment(v2_text, helper)
for forbidden in ("rand", "clamp", "ema", "cap", "floor", "fallback"):
    assert forbidden not in helper_text.lower(), forbidden
for required in (
    "paper_norm / det_norm", "det_proposal * scale",
    "zero deterministic head proposal with nonzero Paper proposal",
):
    assert required in helper_text, required

required_runtime = (
    "paper_head_proposal = -torch.cat",
    "match_head_proposal_norm(\n            head_direction, paper_head_proposal",
    "target_global_preclip_norm",
    "paper_global_preclip_norm",
    "parameter.grad = (-head_target_proposal",
    "paper_clip_scale",
    "head_det_paper_proposal_cosine",
    "value_explained_variance",
    "popart_debiasing",
    "advantage_finite",
)
for token in required_runtime:
    assert token in v2_text, token

removed = [
    line[1:] for line in difflib.unified_diff(
        v1_text.splitlines(), v2_text.splitlines(), lineterm=""
    ) if line.startswith("-") and not line.startswith("---")
]
allowed_removed = {
    "        name in forbidden_exact or name.startswith('joint_') or",
    "        # Preserve Paper's global clipping coefficient for every unchanged",
    "        # policy/shared delta. Only afterward replace the critic-head gradient.",
    "            parameter.grad = (-head_direction[offset:offset + count].view_as(parameter) * paper_clip_scale).to(parameter.dtype)",
}
assert set(removed) == allowed_removed, sorted(set(removed) - allowed_removed)

config_text = V2_CONFIG.read_text().lower()
for forbidden in (
    "normmatch", "head_scale", "scale_cap", "scale_floor", "ema", "guard",
    "joint_", "cross", "shared_ggn", "kaczmarz", "projection",
):
    assert forbidden not in config_text, forbidden

result = {
    "result": "NORMMATCH_V2_STATIC_AUDIT_PASS",
    "v1_trainer_sha256": sha(V1),
    "v2_trainer_sha256": sha(V2),
    "v1_v2_config_byte_identical": True,
    "config_sha256": sha(V2_CONFIG),
    "sole_change": "history-corrected value-head proposal Paper-norm matching plus telemetry",
    "extra_rng_or_data_access": False,
    "free_scale_cap_floor_ema_guard": False,
}
print(json.dumps(result, indent=2, sort_keys=True))
