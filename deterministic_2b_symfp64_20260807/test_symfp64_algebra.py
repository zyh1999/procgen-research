import torch


def chunked_gram_fp64(rows, denominator, chunk_cols):
    result = torch.zeros(
        (rows.shape[0], rows.shape[0]), dtype=torch.float64
    )
    for start in range(0, rows.shape[1], chunk_cols):
        block = rows[:, start:start + chunk_cols].double()
        result.addmm_(
            block, block.t(), beta=1.0, alpha=1.0 / denominator
        )
    return result


def transformed_solve(K, ratio, rhs, damping):
    sqrt_ratio = ratio.sqrt()
    system = sqrt_ratio[:, None] * K * sqrt_ratio[None, :]
    system = system + damping * torch.eye(K.shape[0], dtype=K.dtype)
    transformed_rhs = sqrt_ratio * rhs
    scale = torch.diagonal(system).rsqrt()
    equilibrated = scale[:, None] * system * scale[None, :]
    y = torch.linalg.solve(equilibrated, scale * transformed_rhs)
    beta = scale * y
    return beta / sqrt_ratio


def main():
    torch.manual_seed(7)
    rows = torch.randn(14, 37, dtype=torch.float32)
    ratio = torch.cat([
        torch.rand(7, dtype=torch.float64) * 9.9 + 0.1,
        torch.ones(7, dtype=torch.float64),
    ])
    rhs = torch.randn(14, dtype=torch.float64)
    damping = 0.5
    K = chunked_gram_fp64(rows, 7.0, 5)
    direct = torch.linalg.solve(
        K * ratio.unsqueeze(0)
        + damping * torch.eye(14, dtype=torch.float64),
        rhs,
    )
    transformed = transformed_solve(K, ratio, rhs, damping)
    alpha_error = torch.linalg.vector_norm(direct - transformed)
    direct_direction = rows.double().t().mv(ratio * direct) / 7.0
    transformed_direction = rows.double().t().mv(ratio * transformed) / 7.0
    direction_error = torch.linalg.vector_norm(
        direct_direction - transformed_direction
    )
    residual = torch.linalg.vector_norm(
        K.mv(ratio * transformed) + damping * transformed - rhs
    )
    assert alpha_error < 1e-11, alpha_error
    assert direction_error < 1e-11, direction_error
    assert residual < 1e-11, residual
    print({
        'alpha_error': alpha_error.item(),
        'direction_error': direction_error.item(),
        'residual': residual.item(),
    })


if __name__ == '__main__':
    main()
