"""B-normalized Full-RHS and ratio-consistent 127+1 coefficient solvers.

Actor ratios enter actor curvature. Critic receives ones, not actor ratios.
Selection uses no inverse-inclusion (HT) weights.
"""
from collections import OrderedDict
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .algebra import (exact_selected_kernel_projections_fp32,
    exact_selected_kernel_tail_statistics_vectorized_fp32,
    exact_weighted_directions_vectorized_fp32, named_dot_fp64)

def select_factor_rows(factors, indices):
    """Return an exact factor view containing only ``indices`` rows."""
    selected = OrderedDict()
    for name, (module, layer_input, delta) in factors.items():
        selected[name] = (
            module,
            layer_input.index_select(0, indices),
            None if delta is None else delta.index_select(0, indices),
        )
    return selected


def _parameter_layout(model):
    layout = []
    offset = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        layout.append((name, tuple(parameter.shape), count, offset))
        offset += count
    return layout, offset


def parameter_columns_fp32(model, factors, flat_indices, conv_chunk=64):
    """Materialize selected columns of the exact per-sample score matrix."""
    if flat_indices.ndim != 1:
        raise ValueError("flat_indices must be one-dimensional")
    layout, parameter_count = _parameter_layout(model)
    if flat_indices.numel() == 0:
        first = next(iter(factors.values()))[1]
        return torch.empty((first.shape[0], 0), device=first.device, dtype=torch.float32)
    if bool((flat_indices < 0).any()) or bool((flat_indices >= parameter_count).any()):
        raise ValueError("parameter column index is out of range")
    if torch.unique(flat_indices).numel() != flat_indices.numel():
        raise ValueError("parameter column indices must be unique")

    batch_size = next(iter(factors.values()))[1].shape[0]
    device = flat_indices.device
    columns = torch.empty(
        (batch_size, flat_indices.numel()), device=device, dtype=torch.float32
    )
    factor_by_name = dict(factors)
    for full_name, shape, count, offset in layout:
        selected = (flat_indices >= offset) & (flat_indices < offset + count)
        if not bool(selected.any()):
            continue
        output_columns = selected.nonzero(as_tuple=False).squeeze(1)
        local = flat_indices.index_select(0, output_columns) - offset
        module_name, parameter_name = full_name.rsplit(".", 1)
        if module_name not in factor_by_name:
            raise RuntimeError(f"missing layerwise factors for {module_name}")
        module, layer_input, delta = factor_by_name[module_name]
        if delta is None:
            columns.index_fill_(1, output_columns, 0.0)
            continue
        if layer_input.dtype != torch.float32 or delta.dtype != torch.float32:
            raise TypeError("Task253 leverage factors must remain FP32")

        if parameter_name == "bias":
            if isinstance(module, nn.Conv2d):
                values = delta.flatten(2).sum(dim=2).index_select(1, local)
            else:
                values = delta.reshape(batch_size, -1).index_select(1, local)
            columns.index_copy_(1, output_columns, values)
            continue
        if parameter_name != "weight":
            raise ValueError(f"unsupported parameter in score layout: {full_name}")

        if isinstance(module, nn.Conv2d):
            if module.groups != 1:
                raise NotImplementedError("grouped convolution is not supported")
            patches = F.unfold(
                layer_input,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            ).transpose(1, 2)
            delta_flat = delta.flatten(2)
            input_width = int(module.weight[0].numel())
            for start in range(0, int(local.numel()), int(conv_chunk)):
                end = min(start + int(conv_chunk), int(local.numel()))
                local_chunk = local[start:end]
                out_index = torch.div(local_chunk, input_width, rounding_mode="floor")
                in_index = local_chunk.remainder(input_width)
                delta_chunk = delta_flat.index_select(1, out_index)
                patch_chunk = patches.index_select(2, in_index).transpose(1, 2)
                values = (delta_chunk * patch_chunk).sum(dim=2)
                columns.index_copy_(1, output_columns[start:end], values)
        else:
            input_2d = layer_input.reshape(batch_size, -1)
            delta_2d = delta.reshape(batch_size, -1)
            input_width = int(shape[1])
            out_index = torch.div(local, input_width, rounding_mode="floor")
            in_index = local.remainder(input_width)
            values = delta_2d.index_select(1, out_index) * input_2d.index_select(
                1, in_index
            )
            columns.index_copy_(1, output_columns, values)
    return columns


@torch.no_grad()
def approximate_rank_leverage_distribution(
    model,
    factors,
    ratio,
    parameter_samples=2000,
    rank=128,
    oversample=32,
    power_iterations=1,
    uniform_mixture=0.05,
    parameter_indices=None,
):
    """Estimate actor-ratio-weighted rank-``rank`` row leverage."""
    batch_size = int(ratio.numel())
    if ratio.shape != (batch_size,) or not bool(torch.isfinite(ratio).all()):
        raise ValueError("ratio must contain one finite entry per sample")
    if bool((ratio <= 0).any()):
        raise ValueError("ratio must be positive")
    if not 0 < rank < batch_size:
        raise ValueError("rank must be in (0, batch_size)")
    if not 0.0 <= uniform_mixture < 1.0:
        raise ValueError("uniform_mixture must be in [0,1)")
    _layout, parameter_count = _parameter_layout(model)
    selected_count = min(int(parameter_samples), parameter_count)
    if parameter_indices is None:
        parameter_indices = torch.randperm(
            parameter_count, device=ratio.device
        )[:selected_count]
    if parameter_indices.shape != (selected_count,):
        raise ValueError("parameter_indices shape mismatch")

    if ratio.is_cuda:
        torch.cuda.synchronize(ratio.device)
    started = time.perf_counter()
    sketch = parameter_columns_fp32(model, factors, parameter_indices)
    sketch.mul_(ratio.detach().float().sqrt()[:, None])
    approximation_rank = min(
        int(rank + oversample), batch_size, selected_count
    )
    left, _singular, _right = torch.pca_lowrank(
        sketch,
        q=approximation_rank,
        center=False,
        niter=int(power_iterations),
    )
    leverage = left[:, :rank].square().sum(dim=1)
    leverage_sum = leverage.sum()
    if not bool(torch.isfinite(leverage).all()) or float(leverage_sum.item()) <= 0:
        raise FloatingPointError("invalid leverage scores")
    probabilities = leverage / leverage_sum
    probabilities.mul_(1.0 - float(uniform_mixture)).add_(
        float(uniform_mixture) / float(batch_size)
    )
    probabilities.div_(probabilities.sum())
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities <= 0).any()):
        raise FloatingPointError("invalid leverage probabilities")
    if ratio.is_cuda:
        torch.cuda.synchronize(ratio.device)
    entropy = -(probabilities * probabilities.log()).sum()
    return probabilities, {
        "leverage_parameter_samples": selected_count,
        "leverage_parameter_count": parameter_count,
        "leverage_rank": int(rank),
        "leverage_oversample": int(oversample),
        "leverage_power_iterations": int(power_iterations),
        "leverage_uniform_mixture": float(uniform_mixture),
        "leverage_probability_min": float(probabilities.min().item()),
        "leverage_probability_max": float(probabilities.max().item()),
        "leverage_probability_ess": float(
            probabilities.square().sum().reciprocal().item()
        ),
        "leverage_probability_entropy_fraction": float(
            (entropy / math.log(batch_size)).item()
        ),
        "leverage_seconds": time.perf_counter() - started,
    }


def fixed_size_inclusion_probabilities(probabilities, sample_size):
    """Convert positive PPS weights to first-order inclusion probabilities."""
    if probabilities.ndim != 1 or not 0 < sample_size < probabilities.numel():
        raise ValueError("invalid probabilities or sample size")
    probabilities = probabilities.double()
    probabilities = probabilities / probabilities.sum()
    inclusion = torch.zeros_like(probabilities)
    active = torch.ones_like(probabilities, dtype=torch.bool)
    remaining = int(sample_size)
    while remaining > 0:
        active_indices = active.nonzero(as_tuple=False).squeeze(1)
        active_weights = probabilities.index_select(0, active_indices)
        candidate = remaining * active_weights / active_weights.sum()
        certain = candidate >= 1.0
        if not bool(certain.any()):
            inclusion.index_copy_(0, active_indices, candidate)
            break
        certain_indices = active_indices.index_select(
            0, certain.nonzero(as_tuple=False).squeeze(1)
        )
        inclusion[certain_indices] = 1.0
        active[certain_indices] = False
        remaining -= int(certain_indices.numel())
    if abs(float(inclusion.sum().item()) - int(sample_size)) > 1e-9:
        raise FloatingPointError("inclusion probabilities do not sum to sample size")
    if not bool(torch.isfinite(inclusion).all()) or bool((inclusion <= 0).any()):
        raise FloatingPointError("invalid inclusion probabilities")
    return inclusion


def pivotal_sample(inclusion):
    """Draw an exact-size pivotal PPS sample without replacement."""
    expected_size = int(round(float(inclusion.sum().item())))
    work = inclusion.detach().double().clone()
    tolerance = 1e-12
    while True:
        fractional = ((work > tolerance) & (work < 1.0 - tolerance)).nonzero(
            as_tuple=False
        ).squeeze(1)
        pair_count = int(fractional.numel()) // 2
        if pair_count == 0:
            break
        left = fractional[: 2 * pair_count : 2]
        right = fractional[1 : 2 * pair_count : 2]
        a = work.index_select(0, left)
        b = work.index_select(0, right)
        total = a + b
        random = torch.rand(pair_count, device=work.device, dtype=work.dtype)
        below_one = total < 1.0
        choose_left_below = random < (a / total).clamp(0.0, 1.0)
        left_below = torch.where(choose_left_below, total, torch.zeros_like(total))
        right_below = total - left_below
        denominator = (2.0 - total).clamp_min(torch.finfo(work.dtype).eps)
        choose_left_one = random < ((1.0 - b) / denominator).clamp(0.0, 1.0)
        left_above = torch.where(choose_left_one, torch.ones_like(total), total - 1.0)
        right_above = total - left_above
        work.index_copy_(0, left, torch.where(below_one, left_below, left_above))
        work.index_copy_(0, right, torch.where(below_one, right_below, right_above))
    work = torch.where(work < tolerance, torch.zeros_like(work), work)
    work = torch.where(work > 1.0 - tolerance, torch.ones_like(work), work)
    if bool(((work != 0.0) & (work != 1.0)).any()):
        raise FloatingPointError("pivotal sampling left a fractional inclusion")
    selected = (work == 1.0).nonzero(as_tuple=False).squeeze(1)
    if selected.numel() != expected_size or torch.unique(selected).numel() != expected_size:
        raise FloatingPointError("pivotal sampling returned the wrong sample")
    return selected


@torch.no_grad()
def select_anchors(
    model,
    factors,
    ratio,
    sample_size,
    sampling_mode,
    leverage_parameter_samples=2000,
    leverage_oversample=32,
    leverage_power_iterations=1,
    leverage_uniform_mixture=0.05,
):
    batch_size = int(ratio.numel())
    if not 0 < int(sample_size) < batch_size:
        raise ValueError("sample_size must be in (0,B)")
    if sampling_mode == "uniform":
        anchors = torch.randperm(batch_size, device=ratio.device)[:sample_size]
        inclusion = torch.full(
            (batch_size,), float(sample_size) / float(batch_size),
            device=ratio.device, dtype=torch.float64,
        )
        diagnostics = {}
    elif sampling_mode == "leverage":
        probabilities, diagnostics = approximate_rank_leverage_distribution(
            model,
            factors,
            ratio,
            parameter_samples=leverage_parameter_samples,
            rank=int(sample_size),
            oversample=leverage_oversample,
            power_iterations=leverage_power_iterations,
            uniform_mixture=leverage_uniform_mixture,
        )
        inclusion = fixed_size_inclusion_probabilities(probabilities, sample_size)
        anchors = pivotal_sample(inclusion)
    else:
        raise ValueError(f"unsupported sampling mode: {sampling_mode}")
    if anchors.shape != (sample_size,) or torch.unique(anchors).numel() != sample_size:
        raise RuntimeError("anchor selection is not exact-size and unique")
    selected_inclusion = inclusion.index_select(0, anchors)
    diagnostics.update({
        "anchor_count": int(sample_size),
        "anchor_unique_count": int(torch.unique(anchors).numel()),
        "sampling_mode": sampling_mode,
        "selected_inclusion_min": float(selected_inclusion.min().item()),
        "selected_inclusion_max": float(selected_inclusion.max().item()),
        "inverse_inclusion_correction": 0,
    })
    return anchors, diagnostics


def _solve_spd(system, rhs):
    chol, info = torch.linalg.cholesky_ex(system)
    if int(info.max().item()) != 0:
        raise RuntimeError(f"Task253 reduced system is not SPD; cholesky_info={int(info.max())}")
    vector_rhs = rhs.ndim == 1
    if rhs.ndim not in (1, 2):
        raise ValueError("SPD right-hand side must be a vector or matrix")
    solution = torch.cholesky_solve(rhs[:, None] if vector_rhs else rhs, chol)
    return (solution.squeeze(1) if vector_rhs else solution), int(info.max().item())


@torch.no_grad()
def solve_fullrhs_named_directions_exactfusion(
    factors,
    full_rhs_named_list,
    ratios,
    anchors,
    damping,
    batch_size,
):
    """Task253 Full-RHS path: selected Gram and all ``H @ g`` in one pass."""
    if len(full_rhs_named_list) != len(ratios) or not full_rhs_named_list:
        raise ValueError("Full-RHS directions and ratios must be nonempty/matched")
    selected = select_factor_rows(factors, anchors)
    base_gram_fp32, projected_columns = (
        exact_selected_kernel_projections_fp32(
            selected, full_rhs_named_list
        )
    )
    base_gram = base_gram_fp32.double() / float(batch_size)
    reconstruction_weights = []
    solve_payloads = []
    for rhs_index, (full_rhs_named, ratio) in enumerate(
        zip(full_rhs_named_list, ratios)
    ):
        if ratio.shape != (batch_size,):
            raise ValueError("Full-RHS ratio shape mismatch")
        root_ratio = ratio.index_select(0, anchors).double().sqrt()
        system = base_gram * root_ratio[:, None] * root_ratio[None, :]
        system = (system + system.T) * 0.5
        system.diagonal().add_(float(damping))
        reduced_rhs = (
            root_ratio * projected_columns[:, rhs_index] / float(batch_size)
        )
        coefficient, cholesky_info = _solve_spd(system, reduced_rhs)
        reconstruction_weights.append((root_ratio * coefficient).float())
        solve_payloads.append(
            (full_rhs_named, system, reduced_rhs, coefficient, cholesky_info)
        )

    corrections = exact_weighted_directions_vectorized_fp32(
        selected,
        torch.stack(reconstruction_weights, dim=1),
        denominator=1.0,
    )
    directions = []
    diagnostics = []
    for correction, payload in zip(corrections, solve_payloads):
        full_rhs_named, system, reduced_rhs, coefficient, cholesky_info = payload
        direction = OrderedDict(
            (
                name,
                (full_rhs_named[name] - correction[name]) / float(damping),
            )
            for name in full_rhs_named
        )
        reduced_residual = (
            (system @ coefficient - reduced_rhs).norm()
            / reduced_rhs.norm().clamp_min(1e-30)
        )
        if not math.isfinite(float(reduced_residual.item())):
            raise FloatingPointError(
                "nonfinite Task253 Full-RHS reduced residual"
            )
        directions.append(direction)
        diagnostics.append({
            "reduced_residual": float(reduced_residual.item()),
            "cholesky_info": cholesky_info,
            "full_rhs_norm": float(
                named_dot_fp64(full_rhs_named, full_rhs_named).sqrt().item()
            ),
            "direction_norm": float(
                named_dot_fp64(direction, direction).sqrt().item()
            ),
        })
    return directions, diagnostics


@torch.no_grad()
def solve_dual_coefficients_exactfusion(
    factors,
    targets,
    ratios,
    anchors,
    damping,
    batch_size,
):
    """Task253 dual path with vectorized actor/critic layer contractions."""
    if targets.ndim != 2 or ratios.ndim != 2 or targets.shape != ratios.shape:
        raise ValueError("dual targets/ratios must have matching shape [B,M]")
    if targets.shape[0] != batch_size or targets.shape[1] < 1:
        raise ValueError("dual target batch/RHS count mismatch")
    device = targets.device
    q = int(anchors.numel())
    mask = torch.ones(batch_size, device=device, dtype=torch.bool)
    mask[anchors] = False
    nonanchors = torch.arange(batch_size, device=device)[mask]

    target64 = targets.detach().double()
    ratio64 = ratios.detach().double()
    if not bool(torch.isfinite(target64).all()) or not bool(
        torch.isfinite(ratio64).all()
    ):
        raise FloatingPointError("dual target/ratio contains nonfinite values")
    if bool((ratio64 <= 0).any()):
        raise ValueError("dual ratios must be positive")
    target_s = target64.index_select(0, anchors)
    target_n = target64.index_select(0, nonanchors)
    ratio_s = ratio64.index_select(0, anchors)
    ratio_n = ratio64.index_select(0, nonanchors)
    tail_weights = (ratio_n * target_n).float()

    base_gram_fp32, raw_cross, base_norm2 = (
        exact_selected_kernel_tail_statistics_vectorized_fp32(
            factors,
            anchors,
            nonanchors,
            tail_weights,
        )
    )
    base_gram = base_gram_fp32.double() / float(batch_size)
    coefficient_columns = []
    diagnostics = []
    zero = torch.zeros((), device=device, dtype=torch.float64)

    for rhs_index in range(targets.shape[1]):
        root_ratio_s = ratio_s[:, rhs_index].sqrt()
        system = base_gram * root_ratio_s[:, None] * root_ratio_s[None, :]
        system = (system + system.T) * 0.5
        system.diagonal().add_(float(damping))
        h = root_ratio_s * target_s[:, rhs_index]
        energy = torch.dot(
            ratio_n[:, rhs_index] * target_n[:, rhs_index],
            target_n[:, rhs_index],
        )

        if float(energy.item()) == 0.0:
            u, cholesky_info = _solve_spd(system, h)
            rho = zero
            numerator = zero
            denominator = zero
            y_s = u
            reduced = system
            reduced_rhs = h
        else:
            c = (
                root_ratio_s * raw_cross[:, rhs_index]
                / float(batch_size)
            )
            cap_c = (
                base_norm2[rhs_index] / float(batch_size)
                + float(damping) * energy
            )
            uv, cholesky_info = _solve_spd(
                system, torch.stack((h, c), dim=1)
            )
            u, v = uv[:, 0], uv[:, 1]
            numerator = energy - torch.dot(c, u)
            denominator = cap_c - torch.dot(c, v)
            if not bool(torch.isfinite(denominator)) or float(
                denominator.item()
            ) <= 0:
                raise RuntimeError(
                    "ratio-consistent Schur denominator must be positive, "
                    f"got {denominator.item()}"
                )
            rho = numerator / denominator
            y_s = u - rho * v
            reduced = torch.cat(
                (
                    torch.cat((system, c[:, None]), dim=1),
                    torch.cat((c, cap_c[None]))[None, :],
                ),
                dim=0,
            )
            reduced_rhs = torch.cat((h, energy[None]))

        coefficients = torch.empty(
            batch_size, device=device, dtype=torch.float32
        )
        coefficients.index_copy_(
            0, anchors, (y_s / root_ratio_s).float()
        )
        coefficients.index_copy_(
            0,
            nonanchors,
            (rho * target_n[:, rhs_index]).float(),
        )
        reduced_coefficients = (
            torch.cat((y_s, rho[None])) if reduced.shape[0] == q + 1 else y_s
        )
        residual = (
            (reduced @ reduced_coefficients - reduced_rhs).norm()
            / reduced_rhs.norm().clamp_min(1e-30)
        )
        if not bool(torch.isfinite(coefficients).all()) or not bool(
            torch.isfinite(residual)
        ):
            raise FloatingPointError("nonfinite Task253 dual coefficient solve")
        coefficient_columns.append(coefficients)
        diagnostics.append({
            "combined_reduced_residual": float(residual.item()),
            "cholesky_info": cholesky_info,
            "rho": float(rho.item()),
            "schur_numerator": float(numerator.item()),
            "schur_denominator": float(denominator.item()),
            "nonanchor_energy": float(energy.item()),
        })
    return torch.stack(coefficient_columns, dim=1), diagnostics


def named_from_gradients(model, gradients):
    """Map ``autograd.grad`` output to a complete FP32 named direction."""
    result = OrderedDict()
    for (name, parameter), gradient in zip(
        ((n, p) for n, p in model.named_parameters() if p.requires_grad), gradients
    ):
        result[name] = (
            torch.zeros_like(parameter, dtype=torch.float32)
            if gradient is None
            else gradient.detach().to(dtype=torch.float32)
        )
    return result


def assign_named_gradients(model, direction):
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name not in direction:
            raise RuntimeError(f"missing gradient for {name}")
        parameter.grad = direction[name].to(dtype=parameter.dtype).clone()
