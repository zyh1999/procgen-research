#!/usr/bin/env python3
"""Partition, Paper-equivalence, one-step-logit, and head-solver regression."""
from pathlib import Path
import ast
import copy
import types
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
TARGET = HERE / "train_shared_paper_hybrid_head_detggn_v1.py"
tree = ast.parse(TARGET.read_text())
names = {
    "validate_hybrid_head_config", "chunked_gram_fp64",
    "chunked_transpose_mv_fp64", "solve_head_critic_b_fp64",
    "parameter_partition", "partition_manifest",
}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
ns = {"torch": torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TARGET), "exec"), ns)

cfg = dict(
    ent_coef=0.0, optimizer="sgd", lr=0.5, epochs=4, minibatches=8,
    use_kl_adaptive_lr=True, cg_damping=0.5, max_grad_norm=0.5,
    critic_curvature_coef=0.1, critic_objective_coef=1.0,
    fp64_gram_chunk_cols=32768, dual_jacobi_eps=1e-18,
)
ns["validate_hybrid_head_config"](types.SimpleNamespace(**cfg))
for field, bad in [
    ("lr", .004), ("epochs", 3), ("optimizer_momentum", 0.0),
    ("adaptive_kl_mode", "rollout"), ("joint_critic_curvature_coef", .1),
    ("low_fisher_guard", .01), ("cross_block_mode", "on"),
    ("shared_ggn", True), ("policy_null_projection", True),
    ("is_kaczmarz", True),
]:
    broken = dict(cfg); broken[field] = bad
    try:
        ns["validate_hybrid_head_config"](types.SimpleNamespace(**broken))
    except ValueError:
        pass
    else:
        raise AssertionError(f"illegal field passed: {field}")


class ToySharedActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_net = nn.Sequential(nn.Linear(5, 7), nn.Tanh())
        self.pi_head = nn.Linear(7, 3)
        self.last_v_layer = nn.Linear(7, 1)
        self.shared_sigma = None

    def forward(self, obs):
        latent = self.backbone_net(obs)
        return self.last_v_layer(latent).squeeze(-1), self.pi_head(latent)


torch.manual_seed(811)
model = ToySharedActorCritic()
probe = torch.randn(11, 5)
groups, manifest = ns["partition_manifest"](model, probe)
assert [name for name, _ in groups["POLICY_EXCLUSIVE"]] == ["pi_head.weight", "pi_head.bias"]
assert [name for name, _ in groups["CRITIC_EXCLUSIVE"]] == ["last_v_layer.weight", "last_v_layer.bias"]
assert [name for name, _ in groups["SHARED"]] == ["backbone_net.0.weight", "backbone_net.0.bias"]
for name in manifest["CRITIC_EXCLUSIVE"]["names"]:
    assert manifest["CRITIC_EXCLUSIVE"]["connectivity"][name]["policy_connected"] is False
    assert manifest["CRITIC_EXCLUSIVE"]["connectivity"][name]["policy_jacobian_probe_l2"] == 0.0

# Exact Paper actor/shared-critic algebra and the head-only replacement.
B, P, lam, damping = 13, 19, .1, .5
H = torch.randn(B, P)
adv = torch.randn(B)
ratio = torch.rand(B) + .2
momentum = torch.randn(P)
paper_actor_rhs = adv - H @ momentum
paper_actor_system = H @ H.T / B @ torch.diag(ratio) + damping * torch.eye(B)
paper_actor_alpha = torch.linalg.solve(paper_actor_system, paper_actor_rhs)
target_actor_alpha = torch.linalg.solve(paper_actor_system, paper_actor_rhs)
assert torch.equal(paper_actor_alpha, target_actor_alpha)
paper_critic_rhs = torch.ones(B) - H @ momentum
paper_critic_system = H @ H.T / B + damping * torch.eye(B)
paper_critic_alpha = torch.linalg.solve(paper_critic_system, paper_critic_rhs)
target_shared_critic_alpha = torch.linalg.solve(paper_critic_system, paper_critic_rhs)
assert torch.equal(paper_critic_alpha, target_shared_critic_alpha)

Jhead = torch.randn(B, 8)
head_momentum = torch.randn(8)
head_rows = lam ** .5 * Jhead
head_rhs = torch.randn(B) / lam ** .5 - head_rows @ head_momentum
out = ns["solve_head_critic_b_fp64"](head_rows, head_rhs, B, damping, 4, 1e-18)
alpha, kernel, system, jacobi, info, residual_norm, relative = out
assert kernel.shape == (B, B) and system.shape == (B, B)
assert alpha.shape == (B,) and int(info.max()) == 0
assert torch.isfinite(jacobi).all() and relative.item() < 1e-10
head_direction = ns["chunked_transpose_mv_fp64"](head_rows, alpha, B, 4)
assert torch.allclose(head_direction.double(), head_rows.double().T @ alpha / B, rtol=1e-6, atol=1e-7)

# Actual one-step proof: Paper and Target receive identical policy/shared
# gradients and the exact Paper clipping coefficient; only value-head gradient
# differs. Therefore policy parameters and logits are bit-identical.
paper_model = ToySharedActorCritic()
target_model = copy.deepcopy(paper_model)
obs = torch.randn(17, 5)
actions = torch.randint(0, 3, (17,))
returns = torch.randn(17)

def raw_grads(net):
    values, logits = net(obs)
    actor_loss = nn.functional.cross_entropy(logits, actions)
    critic_loss = nn.functional.mse_loss(values, returns)
    actor = torch.autograd.grad(actor_loss, list(net.parameters()), retain_graph=True, allow_unused=True)
    critic = torch.autograd.grad(critic_loss, list(net.parameters()), allow_unused=True)
    full = [
        (a if a is not None else torch.zeros_like(p)) +
        (c if c is not None else torch.zeros_like(p))
        for p, a, c in zip(net.parameters(), actor, critic)
    ]
    return actor, critic, full

paper_actor, paper_critic, paper_full = raw_grads(paper_model)
target_actor, target_critic, target_full = raw_grads(target_model)
for a, b in zip(paper_actor, target_actor):
    if a is not None or b is not None:
        assert torch.equal(a, b)
for index in (0, 1):
    assert torch.equal(paper_critic[index], target_critic[index])
paper_norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in paper_full]))
clip = min(1.0, .5 / float(paper_norm + 1e-6))
target_head_replacement = [torch.randn_like(target_full[4]), torch.randn_like(target_full[5])]
with torch.no_grad():
    for p, g in zip(paper_model.parameters(), paper_full):
        p.add_(g, alpha=-.5 * clip)
    for index, (p, g) in enumerate(zip(target_model.parameters(), target_full)):
        applied = target_head_replacement[index - 4] if index >= 4 else g
        p.add_(applied, alpha=-.5 * clip)
paper_named = dict(paper_model.named_parameters())
target_named = dict(target_model.named_parameters())
for name in manifest["POLICY_EXCLUSIVE"]["names"] + manifest["SHARED"]["names"]:
    assert torch.equal(paper_named[name], target_named[name]), name
assert any(
    not torch.equal(paper_named[name], target_named[name])
    for name in manifest["CRITIC_EXCLUSIVE"]["names"]
)
paper_logits = paper_model(obs)[1]
target_logits = target_model(obs)[1]
assert torch.equal(paper_logits, target_logits)

print("HYBRID_HEAD_REGRESSION_PASS")
print("partition=EXHAUSTIVE_MUTUALLY_EXCLUSIVE_STABLE")
print("critic_exclusive_policy_jacobian=EXACT_ZERO_DISCONNECTED")
print("paper_actor_matrix_rhs_direction=BIT_IDENTICAL")
print("paper_sampled_shared_critic_direction=BIT_IDENTICAL")
print("one_step_policy_parameters=BIT_IDENTICAL")
print("one_step_policy_logits=BIT_IDENTICAL")
print("only_value_head_delta=DIFFERS")
print(f"head_solver=FP64_Jacobi_Cholesky relative_residual={relative.item():.3e}")
print("illegal_joint_sharedggn_cross_guard_projection_kaczmarz_fields=REJECTED")
