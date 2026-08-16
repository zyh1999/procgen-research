"""Actor-only EF/NPG ablation helpers.

The critic and official PPG auxiliary phase are intentionally absent here.
This module only adds controlled actor-side operations that can be unit tested
without launching Procgen.
"""

from typing import Dict, Tuple

import torch
from torch import Tensor


def damped_empirical_fisher_inverse(
    actor_jacobian: Tensor,
    ratios: Tensor,
    gradient: Tensor,
    damping: float,
    *,
    solve_dtype: torch.dtype = torch.float64,
) -> Tuple[Tensor, Dict[str, float]]:
    """Apply ``(J^T D J / B + damping I)^-1`` to a flat gradient.

    The implementation uses the same B x B sample-space system as the
    existing EF/NPG advantage direction.  Unlike the advantage gradient, an
    exact categorical-entropy gradient need not lie in the sampled Jacobian
    row span, so the damped orthogonal component is retained via Woodbury.
    """
    if actor_jacobian.ndim != 2:
        raise ValueError("actor_jacobian must be rank two")
    batch_size, parameter_size = actor_jacobian.shape
    if ratios.shape != (batch_size,):
        raise ValueError("ratios must have one entry per Jacobian row")
    if gradient.shape != (parameter_size,):
        raise ValueError("gradient size must match the Jacobian width")
    if damping <= 0.0:
        raise ValueError("damping must be strictly positive")
    if torch.any(ratios <= 0.0):
        raise ValueError("importance ratios must be strictly positive")

    jacobian = actor_jacobian.to(solve_dtype)
    solve_ratios = ratios.to(solve_dtype)
    solve_gradient = gradient.to(solve_dtype)
    kernel = jacobian @ jacobian.T / float(batch_size)
    matrix = (
        kernel * solve_ratios.unsqueeze(0)
        + float(damping)
        * torch.eye(batch_size, device=jacobian.device, dtype=solve_dtype)
    )
    rhs = jacobian @ solve_gradient
    alpha = torch.linalg.solve(matrix, rhs)
    direction = (
        solve_gradient
        - jacobian.T @ (solve_ratios * alpha) / float(batch_size)
    ) / float(damping)
    residual = torch.linalg.vector_norm(matrix @ alpha - rhs)
    diagnostics = {
        "entropy_inverse_solve_residual": float(residual.item()),
        "entropy_gradient_l2": float(
            torch.linalg.vector_norm(solve_gradient).item()
        ),
        "entropy_direction_l2": float(
            torch.linalg.vector_norm(direction).item()
        ),
    }
    return direction.to(actor_jacobian.dtype), diagnostics
