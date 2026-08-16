#!/usr/bin/env python3
"""Numerical invariants for deterministic RHS-aligned 2B -> B reduction."""

import torch


torch.manual_seed(20260816)
dtype = torch.float64
batches = 17
parameters = 29

actor_h = torch.randn(batches, parameters, dtype=dtype)
critic_h = torch.randn(batches, parameters, dtype=dtype)
actor_rhs = torch.randn(batches, dtype=dtype)
critic_rhs = torch.randn(batches, dtype=dtype)
ratio_actor = torch.exp(0.2 * torch.randn(batches, dtype=dtype))
ratio_critic = torch.ones(batches, dtype=dtype)
actor_damping = 0.03 + 0.07 * torch.rand(batches, dtype=dtype)
critic_damping = 0.5 + 0.10 * torch.rand(batches, dtype=dtype)

sqrt_actor_ratio = torch.sqrt(ratio_actor)
sqrt_critic_ratio = torch.sqrt(ratio_critic)
transformed_actor_h = sqrt_actor_ratio[:, None] * actor_h
transformed_critic_h = sqrt_critic_ratio[:, None] * critic_h
transformed_actor_rhs = sqrt_actor_ratio * actor_rhs
transformed_critic_rhs = sqrt_critic_ratio * critic_rhs

pair_norm = torch.sqrt(
    transformed_actor_rhs.square() + transformed_critic_rhs.square()
)
qa = transformed_actor_rhs / pair_norm
qc = transformed_critic_rhs / pair_norm
compressed_h = qa[:, None] * transformed_actor_h + qc[:, None] * transformed_critic_h
compressed_rhs = pair_norm
compressed_damping = qa.square() * actor_damping + qc.square() * critic_damping
compressed_system = compressed_h @ compressed_h.T / batches + torch.diag(
    compressed_damping
)
beta = torch.linalg.solve(compressed_system, compressed_rhs)

joint_transformed_h = torch.cat(
    [transformed_actor_h, transformed_critic_h], dim=0
)
joint_transformed_rhs = torch.cat(
    [transformed_actor_rhs, transformed_critic_rhs], dim=0
)
joint_symmetric_alpha = torch.cat([qa * beta, qc * beta])
full_damping = torch.cat([actor_damping, critic_damping])
full_system = (
    joint_transformed_h @ joint_transformed_h.T / batches
    + torch.diag(full_damping)
)
full_residual = full_system @ joint_symmetric_alpha - joint_transformed_rhs
projected_full_residual = qa * full_residual[:batches] + qc * full_residual[batches:]

rhs_reconstruction = torch.cat([qa * compressed_rhs, qc * compressed_rhs])
full_primal_rhs = joint_transformed_h.T @ joint_transformed_rhs
compressed_primal_rhs = compressed_h.T @ compressed_rhs

checks = {
    "pair_unit_error": torch.max((qa.square() + qc.square() - 1.0).abs()),
    "rhs_projection_relative_error": torch.linalg.vector_norm(
        rhs_reconstruction - joint_transformed_rhs
    ) / torch.linalg.vector_norm(joint_transformed_rhs),
    "primal_rhs_relative_error": torch.linalg.vector_norm(
        compressed_primal_rhs - full_primal_rhs
    ) / torch.linalg.vector_norm(full_primal_rhs),
    "reduced_residual": torch.linalg.vector_norm(
        compressed_system @ beta - compressed_rhs
    ),
    "galerkin_residual": torch.linalg.vector_norm(projected_full_residual),
}

for name, value in checks.items():
    print(f"{name}={value.item():.17g}")
    if not torch.isfinite(value) or value.item() > 1e-11:
        raise SystemExit(f"TEST=FAIL {name}={value.item()}")
print("TEST=PASS")
