#!/usr/bin/env python3
"""Algebraic checks for the deterministic RAT B-row block-trace kernel."""

import torch


def main():
    torch.manual_seed(7)
    batch = 19
    params = 31
    actor_h = torch.randn(batch, params, dtype=torch.float64)
    critic_j = torch.randn(batch, params, dtype=torch.float64)

    actor_kernel = actor_h @ actor_h.T / batch
    critic_kernel = 4.0 * (critic_j @ critic_j.T) / batch
    blocktrace_kernel = actor_kernel + critic_kernel

    # The strict 2B operator has diagonal task blocks actor_kernel and
    # critic_kernel.  Taking the trace over that two-dimensional task axis is
    # exactly the proposed B-row kernel.
    strict_kernel = torch.cat([
        torch.cat([
            actor_kernel,
            2.0 * (actor_h @ critic_j.T) / batch,
        ], dim=1),
        torch.cat([
            2.0 * (critic_j @ actor_h.T) / batch,
            critic_kernel,
        ], dim=1),
    ], dim=0)
    task_trace = strict_kernel[:batch, :batch] + strict_kernel[batch:, batch:]
    torch.testing.assert_close(task_trace, blocktrace_kernel)

    # This identity is a proof only; the implementation never samples or
    # alternates signs at runtime.
    plus = actor_h + 2.0 * critic_j
    minus = actor_h - 2.0 * critic_j
    symmetric_pair_average = 0.5 * (
        plus @ plus.T + minus @ minus.T
    ) / batch
    torch.testing.assert_close(symmetric_pair_average, blocktrace_kernel)

    eigenvalues = torch.linalg.eigvalsh(blocktrace_kernel)
    assert eigenvalues.min().item() >= -1e-12

    for damping in (0.03, 0.5):
        system = blocktrace_kernel + damping * torch.eye(
            batch, dtype=torch.float64
        )
        rhs = torch.randn(batch, dtype=torch.float64)
        alpha = torch.linalg.solve(system, rhs)
        relative_residual = torch.linalg.vector_norm(system @ alpha - rhs) / (
            torch.linalg.vector_norm(rhs) + 1e-30
        )
        assert relative_residual.item() < 1e-12

    print("blocktrace_rows={}".format(blocktrace_kernel.shape[0]))
    print("min_eigenvalue={:.6e}".format(eigenvalues.min().item()))
    print("TEST=PASS")


if __name__ == "__main__":
    main()
