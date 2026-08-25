#!/usr/bin/env python3
"""Dependency-free frozen-source checks for the Task34R scientific boundary."""
import ast
from pathlib import Path


root = Path(__file__).resolve().parent
trainer = root / "train_shared_det_standard_mse_ggn_head_cvlm_v1.py"
config = root / "adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml"
text = trainer.read_text()
ast.parse(text)

required = (
    "def solve_standard_mse_head_fp64",
    "train_G64 = train_J64.t() @ train_J64 / float(num_sa)",
    "train_g64 = train_J64.t() @ train_error64 / float(num_sa)",
    "validation_G64 = validation_J64.t() @ validation_J64 / float(num_sa)",
    "cvinds = minibatch_blocks[(block_index + 1) % len(minibatch_blocks)]",
    "rho64 = ared_validation64 / pred_train64",
    "if rho64 > float(algo_config.cvlm_accept_upper)",
    "parameter.grad = (-1e-6 * old_momentum)",
)
for token in required:
    assert token in text, token
for token in (
    "def actor_score_weights", "def gae_error_operator_apply",
    "def solve_gae_head_primal_fp64", "def match_head_proposal_norm",
):
    assert token not in text, token

cfg = config.read_text()
for token in (
    "cvlm_alpha_init: 1.0", "cvlm_alpha_min: 9.5367431640625e-7",
    "cvlm_alpha_max: 1048576.0", "cvlm_max_trials: 4",
    "cvlm_accept_lower: 0.25", "cvlm_accept_upper: 0.75",
):
    assert token in cfg, token
print("TASK34R_STATIC_PASS")
