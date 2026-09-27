"""Task280 sample-coefficient update, streamed without storing all B factors.

The mathematical system and whole-coefficient residual-ray rule are unchanged.
Only 128 anchor factors and O(P) gradient sums survive between chunks.
"""
from collections import OrderedDict
import torch
from .algebra import (LayerwiseIO, exact_selected_kernel_projections_fp32,
                      exact_projection_mixed)
from .coefficient128 import restricted_coefficients


def solve_streamed_coeff128(model, obs, actions, targets, ratios, noise, anchors,
                           chunk=128, damping=0.5, ray_scale=True):
    device = actions.device
    batch, branches = targets.shape
    assert len(anchors) == 128 and anchors.unique().numel() == 128
    assert ratios.shape == targets.shape and torch.all(ratios > 0)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def score(ids, collect=False):
        x = obs[ids.cpu()].to(device)
        if collect:
            with LayerwiseIO(model) as io:
                values, logits = model(x)
        else:
            values, logits = model(x)
        z = logits.log_softmax(-1).gather(1, actions[ids, None]).squeeze(1)
        z = z + 2 * noise[ids] * values
        return (z, io) if collect else z

    def accumulate(weights):
        sums = [OrderedDict((n, torch.zeros_like(p)) for n, p in params)
                for _ in range(branches)]
        for start in range(0, batch, chunk):
            ids = torch.arange(start, min(start + chunk, batch), device=device)
            z = score(ids)
            for j in range(branches):
                grads = torch.autograd.grad((z * weights[ids, j]).sum(),
                    [p for _, p in params], retain_graph=j < branches-1, allow_unused=True)
                for (n, _), g in zip(params, grads):
                    if g is not None:
                        sums[j][n].add_(g.detach())
        return sums

    tail_weights = ratios * targets
    tail_weights = tail_weights.clone()
    tail_weights[anchors] = 0
    bases = accumulate(tail_weights)
    z, io = score(anchors, collect=True)
    factors = io.factors(z, retain_graph=False)
    io.assert_parameter_coverage()
    # Process one layer at a time, including the anchor factor contraction.
    saved = OrderedDict((n, (m, x.detach().cpu(), None if d is None else d.detach().cpu()))
                        for n, (m, x, d) in factors.items())
    del factors, io, z
    gram = torch.zeros(128, 128, device=device)
    cross = torch.zeros(128, branches, device=device, dtype=torch.float64)
    for name, (module, x, delta) in saved.items():
        if delta is None:
            continue
        g, c = exact_selected_kernel_projections_fp32(
            OrderedDict([(name, (module, x.to(device), delta.to(device)))]), bases)
        gram.add_(g)
        cross.add_(c)
    alpha, diagnostics = restricted_coefficients(gram, cross, targets, ratios,
                                                 anchors, damping, batch)
    # H.T D alpha / B is ONLY an auxiliary vector for SAMPLE residuals.
    # It is never installed as the optimizer's parameter direction.
    directions = accumulate((ratios.double() * alpha).float())
    for direction in directions:
        for name in direction:
            direction[name].div_(batch)
    response = torch.empty_like(alpha)
    for start in range(0, batch, chunk):
        ids = torch.arange(start, min(start + chunk, batch), device=device)
        z, io = score(ids, collect=True)
        factors = io.factors(z, retain_graph=False)
        for j in range(branches):
            response[ids, j] = ratios[ids, j].double().sqrt() * (
                exact_projection_mixed(factors, directions[j]) + damping * alpha[ids, j])
        del factors, io, z
    for j in range(branches):
        h = ratios[:, j].double().sqrt() * targets[:, j].double()
        r = response[:, j]
        norm2 = r.square().sum()
        if not torch.isfinite(r).all() or not torch.isfinite(norm2):
            raise FloatingPointError('Nonfinite sample response')
        if norm2 == 0:
            if h.square().sum() != 0:
                raise FloatingPointError('Zero response with nonzero RHS')
            gamma = torch.ones_like(norm2)
        else:
            gamma = torch.dot(h, r) / norm2 if ray_scale else torch.ones_like(norm2)
        before, after = (h-r).square().sum(), (h-gamma*r).square().sum()
        if not torch.isfinite(gamma) or after > before + 1e-8*(1+before):
            raise FloatingPointError('Sample residual ray gate')
        alpha[:, j] *= gamma
        diagnostics[j].update(ray_gamma=gamma.item(), sample_residual2_before=before.item(),
            sample_residual2_after=after.item(), sample_rhs2=h.square().sum().item(),
            sample_residual_relative=(after.sqrt()/h.norm().clamp_min(1e-30)).item(),
            full_rhs_rows=batch, anchors=128, system_divisor=batch,
            reconstruction_divisor=batch, ray_scale=bool(ray_scale))
    if not torch.isfinite(alpha).all():
        raise FloatingPointError('Nonfinite final coefficients')
    return alpha.float().detach(), diagnostics
