#!/usr/bin/env python3
"""Verify that the deterministic B-row Schur solve equals direct joint 2B.

The production solver works with ``K @ D_rho`` rather than an explicitly
symmetric kernel.  This test therefore covers the actual nonsymmetric linear
representation, separate actor/critic damping, the recovered critic dual, and
the final primal parameter direction.
"""

import torch


def solve_schur(system: torch.Tensor, rhs: torch.Tensor, batch: int) -> torch.Tensor:
    aa = system[:batch, :batch]
    ac = system[:batch, batch:]
    ca = system[batch:, :batch]
    cc = system[batch:, batch:]
    rhs_a = rhs[:batch]
    rhs_c = rhs[batch:]

    # One factorization of the critic block supplies both C-side elimination
    # and RHS elimination.  No inverse is formed in the implementation.
    eliminated = torch.linalg.solve(
        cc,
        torch.cat((ca, rhs_c[:, None]), dim=1),
    )
    cc_inv_ca = eliminated[:, :batch]
    cc_inv_rhs = eliminated[:, batch]
    schur = aa - ac @ cc_inv_ca
    reduced_rhs = rhs_a - ac @ cc_inv_rhs
    alpha_a = torch.linalg.solve(schur, reduced_rhs)
    alpha_c = cc_inv_rhs - cc_inv_ca @ alpha_a
    return torch.cat((alpha_a, alpha_c))


def main() -> None:
    torch.manual_seed(20260816)
    dtype = torch.float64
    batch = 13
    parameters = 31

    actor_rows = torch.randn(batch, parameters, dtype=dtype)
    critic_rows = 2.0 * torch.randn(batch, parameters, dtype=dtype)
    rows = torch.cat((actor_rows, critic_rows), dim=0)
    kernel = rows @ rows.T / batch

    # Actor columns use an off-policy ratio while critic columns stay at one,
    # matching the Procgen joint solver's actor_ratio_critic_unit convention.
    actor_ratio = torch.exp(0.35 * torch.randn(batch, dtype=dtype))
    density = torch.cat((actor_ratio, torch.ones(batch, dtype=dtype)))
    actor_damping = 0.013 + 0.01 * torch.rand(batch, dtype=dtype)
    critic_damping = 0.41 + 0.17 * torch.rand(batch, dtype=dtype)
    damping = torch.cat((actor_damping, critic_damping))
    system = kernel * density[None, :] + torch.diag(damping)
    rhs = torch.randn(2 * batch, dtype=dtype)

    direct = torch.linalg.solve(system, rhs)
    reduced = solve_schur(system, rhs, batch)
    relative_dual_error = torch.linalg.vector_norm(reduced - direct) / torch.linalg.vector_norm(direct)
    direct_residual = torch.linalg.vector_norm(system @ reduced - rhs) / torch.linalg.vector_norm(rhs)

    # The actual optimizer consumes H^T (D_rho alpha), so equality must also
    # hold after reconstruction, not merely in output-space coefficients.
    direct_direction = rows.T @ (density * direct)
    reduced_direction = rows.T @ (density * reduced)
    relative_direction_error = (
        torch.linalg.vector_norm(reduced_direction - direct_direction)
        / torch.linalg.vector_norm(direct_direction)
    )

    assert relative_dual_error.item() < 1e-12, relative_dual_error.item()
    assert direct_residual.item() < 1e-12, direct_residual.item()
    assert relative_direction_error.item() < 1e-12, relative_direction_error.item()
    print(
        "PASS exact deterministic Schur B ",
        f"dual_error={relative_dual_error.item():.3e} ",
        f"direction_error={relative_direction_error.item():.3e} ",
        f"residual={direct_residual.item():.3e}",
        sep="",
    )


if __name__ == "__main__":
    main()
