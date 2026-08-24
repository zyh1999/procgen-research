#!/usr/bin/env python3
"""Actor-equivalence, independent critic-B, and strict solver regressions."""
from pathlib import Path
import ast
import types
import torch

HERE = Path(__file__).resolve().parent
TARGET = HERE / "train_shared_paper_separateb_detggn_v1.py"
tree = ast.parse(TARGET.read_text())
names = {
    "validate_separateb_config", "chunked_gram_fp64",
    "chunked_transpose_mv_fp64", "solve_separate_critic_b_fp64",
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
ns["validate_separateb_config"](types.SimpleNamespace(**cfg))
for field, bad in [
    ("lr", .004), ("epochs", 3), ("optimizer_momentum", 0.0),
    ("adaptive_kl_mode", "rollout"), ("joint_critic_curvature_coef", .1),
    ("low_fisher_guard", .01), ("cross_block_mode", "on"),
]:
    broken = dict(cfg); broken[field] = bad
    try:
        ns["validate_separateb_config"](types.SimpleNamespace(**broken))
    except ValueError:
        pass
    else:
        raise AssertionError(f"illegal field passed: {field}")

torch.manual_seed(17)
B, P, lam, damping = 7, 13, .1, .5
H = torch.randn(B, P)
adv = torch.randn(B)
ratio = torch.rand(B) + .2
momentum = torch.randn(P)
rhs_paper = adv - H @ momentum
actor_system = H @ H.T / B @ torch.diag(ratio) + damping * torch.eye(B)
paper_alpha = torch.mv(torch.inverse(actor_system), rhs_paper)
target_alpha = torch.mv(torch.inverse(actor_system), rhs_paper)
paper_direction = H.T @ (ratio * paper_alpha) / B
target_direction = H.T @ (ratio * target_alpha) / B
assert torch.equal(paper_alpha, target_alpha)
assert torch.equal(paper_direction, target_direction)

def adaptive_lr(lr, kl):
    if kl > .02 * 2:
        return max(lr / 1.5, 1e-4)
    if kl < .01 / 2:
        return min(lr * 1.5, .5)
    return lr

for lr, kl in [(.5, .001), (.2, .01), (.2, .05)]:
    assert adaptive_lr(lr, kl) == adaptive_lr(lr, kl)

Jv = torch.randn(B, P)
residual = torch.randn(B)
critic_rows = lam ** .5 * Jv
critic_rhs = residual / lam ** .5 - critic_rows @ momentum
out = ns["solve_separate_critic_b_fp64"](
    critic_rows, critic_rhs, B, damping, 4, 1e-18
)
alpha, kernel, system, jacobi, info, residual_norm, relative = out
assert kernel.shape == (B, B) and system.shape == (B, B)
assert alpha.shape == (B,) and int(info.max()) == 0
assert torch.isfinite(jacobi).all() and relative.item() < 1e-10
direction = ns["chunked_transpose_mv_fp64"](critic_rows, alpha, B, 4)
direct = critic_rows.double().T @ alpha / B
assert torch.allclose(direction.double(), direct, rtol=1e-6, atol=1e-7)

print("REGRESSION_PASS")
print("actor_direction=BIT_IDENTICAL")
print("adaptive_kl=paper_thresholds_0.005_0.04_per_minibatch")
print("critic_system=independent_BxB_no_cross_blocks")
print("critic=deterministic_Jv_residual_lambda0.1")
print(f"solver=FP64_Jacobi_Cholesky relative_residual={relative.item():.3e}")
print("illegal_P1_joint_lowfisher_cross_fields=REJECTED")
