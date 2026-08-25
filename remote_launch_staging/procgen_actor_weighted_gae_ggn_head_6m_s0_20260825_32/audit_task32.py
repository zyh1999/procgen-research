#!/usr/bin/env python3
"""Static single-causal-delta and forbidden-mechanism audit for Task32."""
import ast
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
trainer = root / "train_shared_det_actor_weighted_gae_ggn_head_v1.py"
config = root / "adv_resnet_shared_det_actor_weighted_gae_ggn_head_v1_6m.yaml"
monitor = root / "stage_monitor.py"
launcher = root / "actor_weighted_gae_ggn_head_6m_gpuh.sbatch"
preflight = root / "gpuh_preflight.py"

for path in (trainer, config, monitor, launcher, preflight):
    assert path.is_file() and path.stat().st_size > 0, path
    if path.suffix == ".py":
        ast.parse(path.read_text(), filename=str(path))

text = trainer.read_text()
required = (
    "class GAEMetadataRunner(Runner)",
    "def gae_error_operator_apply(",
    "def actor_score_weights(",
    "def solve_gae_head_primal_fp64(",
    "rollout_latents = actor_critic.backbone_net(current_rollout_obs)",
    "head_critic_kernel_mode='actor_weighted_exact_gae_value_head_primal'",
    "parameter_partition='policy_exclusive_shared_critic_exclusive'",
    "paper_clip_scale=paper_clip_scale.item()",
)
for needle in required:
    assert needle in text, needle

for forbidden in (
    "match_head_proposal_norm", "paper_head_proposal", "multi_epsilon",
    "__mp_main__",
    "origin_safety", "runtime_closure_probe",
):
    assert forbidden not in text, forbidden

config_text = config.read_text()
for forbidden in ("critic_curvature_coef", "critic_objective_coef", "norm_match"):
    assert forbidden not in config_text, forbidden
for exact in (
    "lr: 0.5", "epochs: 4", "minibatches: 8", "cg_damping: 0.5",
    "max_grad_norm: 0.5", "with_popart: True", "adv_type: gae",
):
    assert exact in config_text, exact

print("TASK32_STATIC_SINGLE_CAUSAL_DELTA_PASS")
for path in (trainer, config, preflight, launcher, monitor):
    print(f"{path.name} {hashlib.sha256(path.read_bytes()).hexdigest()}")
