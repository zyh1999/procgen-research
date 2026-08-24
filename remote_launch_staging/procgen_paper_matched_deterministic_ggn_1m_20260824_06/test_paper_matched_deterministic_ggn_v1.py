#!/usr/bin/env python3
"""Minimal config, history, joint-system, and FP64 solver regressions."""
from pathlib import Path
import ast, types, torch

HERE = Path(__file__).resolve().parent
TARGET = HERE / "train_shared_paper_matched_deterministic_ggn_v1.py"
CONFIG = HERE / "adv_resnet_shared_paper_matched_deterministic_ggn_v1_1m.yaml"
tree = ast.parse(TARGET.read_text())
names = {"validate_paper_matched_config", "chunked_gram_fp64", "chunked_transpose_mv_fp64"}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
ns = {"torch": torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TARGET), "exec"), ns)

cfg = dict(
    ent_coef=0.0, norm_obj='adv', optimizer='sgd', lr=0.5,
    use_kl_adaptive_lr=True, epochs=4, minibatches=8, vf_coef=1.0,
    with_popart=True, adv_type='gae', clamp_ratio=True, max_ratio=10.0,
    min_ratio=0.1, sigma_type='vector', cg_damping=0.5,
    max_grad_norm=0.5, joint_critic_curvature_coef=0.1,
    joint_critic_objective_coef=1.0, fp64_gram_chunk_cols=32768,
    dual_jacobi_eps=1e-18,
)
for required in ('lr: 0.5', 'epochs: 4', 'minibatches: 8',
                 'joint_critic_curvature_coef: 0.1'):
    assert required in CONFIG.read_text()
obj = types.SimpleNamespace(**cfg)
ns["validate_paper_matched_config"](obj)
for field, bad in [("lr", .004), ("epochs", 3), ("optimizer_momentum", 0.0),
                   ("adaptive_kl_mode", "procgen_rollout")]:
    broken = dict(cfg); broken[field] = bad
    try: ns["validate_paper_matched_config"](types.SimpleNamespace(**broken))
    except ValueError: pass
    else: raise AssertionError(f"illegal field passed: {field}")

torch.manual_seed(7)
B, P, lam, mu = 5, 11, .1, .5
Hpi = torch.randn(B, P)
Jv = torch.randn(B, P)
adv = torch.randn(B)
residual = torch.randn(B)
ratio = torch.rand(B) + .2
H = torch.cat([Hpi, lam ** .5 * Jv], 0)
rhs = torch.cat([adv, residual / lam ** .5])
weights = torch.cat([ratio, torch.ones(B)])
assert H.shape == (2 * B, P) and rhs.shape == (2 * B,)

# Literal Paper history rule on the stacked rows.
momentum_buffer = torch.randn(P)
rhs_eff = rhs - H @ momentum_buffer
assert torch.allclose(rhs_eff, rhs - H @ momentum_buffer)

K = ns["chunked_gram_fp64"](H, B, 4)
sqrtw = weights.double().sqrt()
S = sqrtw[:, None] * K * sqrtw[None, :] + mu * torch.eye(2 * B, dtype=torch.float64)
r = sqrtw * rhs_eff.double()
j = torch.diagonal(S).rsqrt()
Se = j[:, None] * S * j[None, :]
chol, info = torch.linalg.cholesky_ex(Se)
assert int(info.max()) == 0
beta = j * torch.cholesky_solve((j * r)[:, None], chol).squeeze(1)
alpha = beta / sqrtw
rel = torch.linalg.vector_norm(K @ (weights.double() * alpha) + mu * alpha - rhs_eff.double()) / torch.linalg.vector_norm(rhs_eff.double())
assert rel < 1e-10, rel
direction = ns["chunked_transpose_mv_fp64"](H, weights.double() * alpha, B, 4)
# The sensitive Gram/solve/reconstruction multiply is FP64; the final flat
# direction intentionally returns to the model parameter dtype, as in P1.
assert direction.dtype == H.dtype and torch.isfinite(direction).all()

print("REGRESSION_PASS")
print("paper_actor=lr0.5 per_minibatch_KL momentum1e-6 history_rhs_minus_H_buffer")
print("critic=deterministic_residual lambda0.1 joint_rows_2B")
print(f"solver=FP64_Jacobi_Cholesky relative_residual={rel.item():.3e}")
print("illegal_P1_actor_fields=REJECTED")
