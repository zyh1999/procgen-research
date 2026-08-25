#!/usr/bin/env python3
"""Static and numerical historical-scaling audit for Task34R."""
import ast
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml


root = Path(__file__).resolve().parent
staging = root.parent
task07 = Path(os.environ.get(
    "TASK07_SOURCE_ROOT", staging / "procgen_paper_separateb_detggn_6m_s0_20260824_07"
))
task13 = Path(os.environ.get(
    "TASK13_SOURCE_ROOT", staging / "procgen_paper_hybrid_head_detggn_6m_s0_20260824_08"
))
task32 = Path(os.environ.get(
    "TASK32_SOURCE_ROOT", staging / "procgen_actor_weighted_gae_ggn_head_6m_s0_20260825_32"
))

paths = {
    "task07_trainer": task07 / "train_shared_paper_separateb_detggn_v1.py",
    "task13_trainer": task13 / "train_shared_paper_hybrid_head_detggn_v1.py",
    "task32_trainer": task32 / "train_shared_det_actor_weighted_gae_ggn_head_v1.py",
    "target_trainer": root / "train_shared_det_standard_mse_ggn_head_cvlm_v1.py",
    "target_config": root / "adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml",
}
for path in paths.values():
    assert path.is_file() and path.stat().st_size > 0, path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


target_text = paths["target_trainer"].read_text()
target_tree = ast.parse(target_text)
config = yaml.safe_load(paths["target_config"].read_text())["algo_config"]
assert config["cvlm_alpha_init"] == 1.0
assert config["cvlm_alpha_min"] == 2.0 ** -20
assert config["cvlm_alpha_max"] == 2.0 ** 20
assert config["cvlm_max_trials"] == 4
assert config["cvlm_accept_lower"] == 0.25
assert config["cvlm_accept_upper"] == 0.75
for forbidden in (
    "actor_score_weights", "gae_error_operator_apply", "solve_gae_head_primal_fp64",
    "match_head_proposal_norm", "paper_head_proposal", "critic_curvature_coef",
    "critic_objective_coef",
):
    assert forbidden not in target_text, forbidden
for required in (
    "train_G64 = train_J64.t() @ train_J64 / float(num_sa)",
    "train_g64 = train_J64.t() @ train_error64 / float(num_sa)",
    "validation_G64 = validation_J64.t() @ validation_J64 / float(num_sa)",
    "cvinds = minibatch_blocks[(block_index + 1) % len(minibatch_blocks)]",
    "rho64 = ared_validation64 / pred_train64",
    "parameter.grad = (-1e-6 * old_momentum)",
):
    assert required in target_text, required

# Derive Task13 rather than trusting config labels.
task13_text = paths["task13_trainer"].read_text()
assert "critic_h_weight = math.sqrt(critic_curvature_coef)" in task13_text
assert "head_rows = critic_h_weight * J_head" in task13_text
assert "head_rhs = (critic_objective_coef / critic_h_weight) * (_ret - _vals).detach()" in task13_text
assert "critic_curvature_coef: 0.1" in (task13 / "adv_resnet_shared_paper_hybrid_head_detggn_v1_6m.yaml").read_text()

# Deterministic numerical alignment in standard MSE coordinates.
generator = torch.Generator().manual_seed(340825)
J = torch.randn(512, 257, generator=generator, dtype=torch.float64)
error = torch.randn(512, generator=generator, dtype=torch.float64)
G = J.t() @ J / 512.0
g = J.t() @ error / 512.0
identity = torch.eye(257, dtype=torch.float64)

# Target standard coordinates: (G + mu I)u = -g.
spectrum_trace_mean = torch.trace(G) / 257.0
mu_target = float(spectrum_trace_mean)
u_target = torch.linalg.solve(G + mu_target * identity, -g)

# Task13 source-derived coordinates: (0.1G + .5I)u = -g, or
# (G + 5I)u = -10g after division by curvature coefficient.
u_task13_direct = torch.linalg.solve(0.1 * G + 0.5 * identity, -g)
u_task13_standard = torch.linalg.solve(G + 5.0 * identity, -10.0 * g)
assert torch.allclose(u_task13_direct, u_task13_standard, rtol=1e-12, atol=1e-12)

# Task07 and Task13 share the same .1/1.0 scaling but differ in parameter scope.
task07_text = paths["task07_trainer"].read_text()
assert "critic_h_weight = math.sqrt(critic_curvature_coef)" in task07_text
assert "critic_objective_coef / critic_h_weight" in task07_text

# Task32 is explicitly a different D/W construction, not a standard-MSE scale.
task32_text = paths["task32_trainer"].read_text()
assert "gae_error_operator_apply" in task32_text
assert "actor_score_weights" in task32_text


def cosine(a, b):
    return float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))


ledger = {
    "task_id": "PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-6M-S0-20260825-34R",
    "method": "DET_STANDARD_MSE_GGN_HEAD_CVLM_V1",
    "source_sha256": {name: sha(path) for name, path in paths.items()},
    "standard_objective": "||V-stopgrad(R_lambda)||^2/(2B)",
    "standard_gradient": "J^T e/B",
    "standard_ggn": "J^T J/B",
    "gaussian_precision": 1.0,
    "task13_source_curvature_coefficient": 0.1,
    "task13_source_objective_coefficient": 1.0,
    "task13_source_damping": 0.5,
    "task13_standard_coordinate_effective_damping": 5.0,
    "task13_standard_coordinate_rhs_multiplier": 10.0,
    "task07_scaling_same_as_task13_scope_different": True,
    "task32_not_scale_equivalent": "D=GAE temporal operator and W=actor score weights",
    "target_damping_rule": "mu=alpha*max(trace(G)/257,epsilon_fp64)",
    "target_initial_alpha": 1.0,
    "synthetic": {
        "trace_G": float(torch.trace(G)),
        "trace_G_per_parameter": mu_target,
        "gradient_norm": float(torch.linalg.vector_norm(g)),
        "target_raw_solve_norm": float(torch.linalg.vector_norm(u_target)),
        "task13_raw_solve_norm": float(torch.linalg.vector_norm(u_task13_direct)),
        "target_task13_cosine": cosine(u_target, u_task13_direct),
        "task13_direct_vs_standard_max_abs": float((u_task13_direct - u_task13_standard).abs().max()),
    },
    "same_minibatch_ared_pred_is_degenerate_and_never_controls_lm": True,
    "cross_minibatch_validation_controls_lm": True,
}
output_path = Path(os.environ.get(
    "TASK34R_AUDIT_OUTPUT", root / "historical_scaling_ledger.json"
))
output_path.write_text(
    json.dumps(ledger, indent=2, sort_keys=True) + "\n"
)
print("TASK34R_HISTORICAL_SCALING_AUDIT_PASS")
print(json.dumps(ledger, sort_keys=True))
