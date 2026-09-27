"""RAT-aligned sample coefficients: free anchors and fixed b_N/mu tail.

Same symmetric sample system as Dual127+1: M=sqrt(D) H H.T sqrt(D)/B+mu I,
h=sqrt(D)b, x=sqrt(D)alpha. Unlike Dual, the tail has no free rho.
Optional ray minimization scales ALL coefficients, not a parameter direction.
"""
import torch
from .algebra import (
    exact_selected_kernel_tail_statistics_vectorized_fp32,
    exact_weighted_directions_vectorized_fp32, exact_projection_mixed,
    exact_kernel_fp32)
from .subcurvature import _solve_spd, select_factor_rows


def restricted_coefficients(gram, cross, targets, ratios, anchors, damping, B):
    """FP64 affine restriction of exactly the Dual full sample system."""
    b, d = targets.double(), ratios.double()
    root = d[anchors].sqrt()
    columns, info = [], []
    for j in range(b.shape[1]):
        M = gram.double() / B * root[:, j, None] * root[None, :, j]
        M = (M + M.T) * 0.5
        M.diagonal().add_(damping)
        # cross = H_S H_N.T (D_N b_N); tail alpha_N=b_N/mu.
        rhs = root[:, j] * (b[anchors, j] - cross[:, j].double() / (B*damping))
        xs, chol = _solve_spd(M, rhs)
        alpha = b[:, j] / damping
        alpha = alpha.clone()
        alpha[anchors] = xs / root[:, j]
        res = (M @ xs-rhs).norm()/rhs.norm().clamp_min(1e-30)
        if not torch.isfinite(alpha).all() or not torch.isfinite(res):
            raise FloatingPointError('Task280 nonfinite coefficient solve')
        columns.append(alpha)
        info.append(dict(reduced_residual=res.item(), cholesky_info=chol))
    return torch.stack(columns, 1), info


@torch.no_grad()
def solve_coeff128(factors, targets, ratios, anchors, damping, B, ray_scale=True):
    assert targets.shape == ratios.shape and targets.shape[0] == B
    assert torch.isfinite(targets).all() and torch.isfinite(ratios).all()
    assert (ratios > 0).all() and damping > 0
    mask = torch.ones(B, device=targets.device, dtype=torch.bool)
    mask[anchors] = False
    nonanchors = torch.arange(B, device=targets.device)[mask]
    if len(nonanchors):
        gram, cross, _ = exact_selected_kernel_tail_statistics_vectorized_fp32(
            factors, anchors, nonanchors,
            (ratios[nonanchors].double()*targets[nonanchors].double()).float())
    else:
        gram = exact_kernel_fp32(select_factor_rows(factors, anchors))
        cross = torch.zeros(len(anchors), targets.shape[1], device=targets.device,
                            dtype=torch.float64)
    alpha, info = restricted_coefficients(gram, cross, targets, ratios, anchors, damping, B)
    b, d = targets.double(), ratios.double()
    # H.T D alpha is used ONLY to measure the SAMPLE residual, never installed
    # as the training gradient. The trainer reconstructs through real losses.
    directions = exact_weighted_directions_vectorized_fp32(factors, (d*alpha).float(), B)
    for j in range(targets.shape[1]):
        root = d[:, j].sqrt()
        h = root*b[:, j]
        response = root*(exact_projection_mixed(factors, directions[j])+damping*alpha[:, j])
        norm2 = response.square().sum()
        if not torch.isfinite(response).all() or not torch.isfinite(norm2) or norm2 <= 0:
            # Exact zero RHS has an exact zero solution; no arbitrary scale.
            if h.square().sum() == 0 and torch.isfinite(response).all() and norm2 == 0:
                gamma = torch.ones((), device=b.device, dtype=torch.float64)
            else:
                raise FloatingPointError('Task280 nonfinite/zero sample response')
        else:
            gamma = torch.dot(h, response)/norm2 if ray_scale else torch.ones_like(norm2)
        before = (h-response).square().sum()
        after = (h-gamma*response).square().sum()
        if not torch.isfinite(gamma) or after > before + 1e-8*(1+before):
            raise FloatingPointError('Task280 sample ray residual gate')
        alpha[:, j] *= gamma
        info[j].update(ray_gamma=gamma.item(), sample_residual2_before=before.item(),
                       sample_residual2_after=after.item(), sample_rhs2=h.square().sum().item(),
                       sample_residual_relative=(after.sqrt()/h.norm().clamp_min(1e-30)).item(),
                       full_rhs_rows=B, anchors=len(anchors), system_divisor=B,
                       reconstruction_divisor=B, ray_scale=bool(ray_scale))
    if not torch.isfinite(alpha).all():
        raise FloatingPointError('Task280 final coefficients nonfinite')
    return alpha.float().detach(), info
