import torch

from train_shared_joint2b_rownorm_correlation import (
    correlation_relative_damping_diagonal,
)


def main():
    torch.manual_seed(7)
    dtype = torch.float64
    rows, cols = 18, 31
    h = torch.randn(rows, cols, dtype=dtype)
    rhs = torch.randn(rows, dtype=dtype)
    ratio = torch.exp(torch.randn(rows, dtype=dtype) * 0.7)
    normalized_damping = 0.3
    eps = 1e-12

    kernel = h @ h.t() / float(rows)
    raw_diag = torch.diagonal(kernel)
    row_scale = raw_diag.clamp_min(eps).rsqrt()

    normalized_kernel = (
        row_scale[:, None] * kernel * row_scale[None, :]
    )
    normalized_system = (
        normalized_kernel * ratio.unsqueeze(0)
        + normalized_damping * torch.eye(rows, dtype=dtype)
    )
    normalized_alpha = torch.linalg.solve(
        normalized_system,
        row_scale * rhs,
    )
    normalized_backscaled_alpha = row_scale * normalized_alpha

    damping_diagonal = correlation_relative_damping_diagonal(
        raw_diag,
        normalized_damping,
        eps,
    )
    raw_system = (
        kernel * ratio.unsqueeze(0)
        + torch.diag(damping_diagonal)
    )
    raw_alpha = torch.linalg.solve(raw_system, rhs)

    torch.testing.assert_close(
        raw_alpha,
        normalized_backscaled_alpha,
        rtol=2e-11,
        atol=2e-11,
    )
    normalized_direction = (
        h.t() @ (ratio * normalized_backscaled_alpha) / float(rows)
    )
    raw_direction = h.t() @ (ratio * raw_alpha) / float(rows)
    torch.testing.assert_close(
        raw_direction,
        normalized_direction,
        rtol=2e-11,
        atol=2e-11,
    )

    # The damping is lambda * diag(K), not lambda * diag(K D_rho).
    torch.testing.assert_close(
        damping_diagonal,
        normalized_damping * raw_diag,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.allclose(
        damping_diagonal,
        normalized_damping * raw_diag * ratio,
    )

    for bad_damping, bad_eps in ((0.0, eps), (-0.1, eps), (0.3, 0.0)):
        try:
            correlation_relative_damping_diagonal(
                raw_diag,
                bad_damping,
                bad_eps,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                'invalid correlation damping arguments were accepted: '
                f'{bad_damping=}, {bad_eps=}'
            )

    print('PASS: raw correlation-relative solve matches normalized solve')


if __name__ == '__main__':
    main()
