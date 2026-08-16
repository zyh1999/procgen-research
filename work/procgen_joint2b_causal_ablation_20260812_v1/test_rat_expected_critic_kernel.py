"""Monte-Carlo identity check for deterministic expected-score RAT kernel."""

import torch


def main():
    torch.manual_seed(20260816)
    dtype = torch.float64
    batch, params = 9, 13
    policy_h = torch.randn(batch, params, dtype=dtype)
    value_j = torch.randn(batch, params, dtype=dtype)

    policy_k = policy_h @ policy_h.T / batch
    expected_critic_diag = 4.0 * value_j.square().sum(dim=1) / batch
    analytic_k = policy_k + torch.diag(expected_critic_diag)

    draws = 200_000
    chunk = 2_000
    sampled_sum = torch.zeros_like(analytic_k)
    for _ in range(draws // chunk):
        xi = torch.randn(chunk, batch, dtype=dtype)
        combined = policy_h.unsqueeze(0) + 2.0 * xi.unsqueeze(2) * value_j
        sampled_sum += torch.einsum('nbp,ncp->bc', combined, combined) / batch
    sampled_mean = sampled_sum / draws

    relative_error = torch.linalg.vector_norm(sampled_mean - analytic_k) / (
        torch.linalg.vector_norm(analytic_k) + 1e-30
    )
    offdiag_critic_error = torch.linalg.vector_norm(
        (sampled_mean - policy_k)
        - torch.diag(torch.diagonal(sampled_mean - policy_k))
    ) / (torch.linalg.vector_norm(analytic_k) + 1e-30)

    ratio = torch.exp(0.1 * torch.randn(batch, dtype=dtype))
    advantage = torch.randn(batch, dtype=dtype)
    actor_damping = 0.03
    critic_damping = 0.5
    weighted = analytic_k * ratio.unsqueeze(0)
    actor_system = weighted + actor_damping * torch.eye(batch, dtype=dtype)
    critic_system = weighted + critic_damping * torch.eye(batch, dtype=dtype)
    actor_alpha = torch.linalg.solve(actor_system, advantage)
    critic_alpha = torch.linalg.solve(
        critic_system, torch.ones(batch, dtype=dtype)
    )
    actor_residual = torch.linalg.vector_norm(
        actor_system @ actor_alpha - advantage
    )
    critic_residual = torch.linalg.vector_norm(
        critic_system @ critic_alpha - torch.ones(batch, dtype=dtype)
    )

    print(f'monte_carlo_relative_error={relative_error.item():.6e}')
    print(f'offdiag_critic_error={offdiag_critic_error.item():.6e}')
    print(f'actor_solve_residual={actor_residual.item():.6e}')
    print(f'critic_solve_residual={critic_residual.item():.6e}')
    assert relative_error < 8e-3
    assert offdiag_critic_error < 8e-3
    assert actor_residual < 1e-10
    assert critic_residual < 1e-10
    print('TEST=PASS')


if __name__ == '__main__':
    main()
