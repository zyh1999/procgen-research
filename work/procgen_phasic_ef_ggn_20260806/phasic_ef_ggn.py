"""Deterministic phasic critic GGN and actor-Fisher safety utilities.

The MVP in this module intentionally materializes the same-batch critic
Jacobian.  It does not implement sampled critic scores, a joint 2B system,
sketching, Kaczmarz, or optimizer processing of the GGN direction.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap
from torch.nn import functional as F


def partition_named_parameters(model: nn.Module) -> Dict[str, Tuple[str, ...]]:
    """Return exhaustive, disjoint shared/actor-head/critic-head name groups."""
    groups = {"shared": [], "actor_head": [], "critic_head": []}
    unknown = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone_net.") or name.startswith("shared."):
            groups["shared"].append(name)
        elif (
            name.startswith("pi_head.")
            or name.startswith("actor_head.")
            or name == "shared_sigma"
        ):
            groups["actor_head"].append(name)
        elif (
            name.startswith("last_v_layer.")
            or name.startswith("critic_head.")
            or name.startswith("value_head.")
        ):
            groups["critic_head"].append(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(f"Unclassified trainable parameters: {unknown}")
    return {key: tuple(value) for key, value in groups.items()}


def _value_output(output) -> Tensor:
    if hasattr(output, "value"):
        value = output.value
    elif isinstance(output, (tuple, list)):
        value = output[0]
    else:
        value = output
    return value.reshape(value.shape[0], -1).squeeze(-1)


def _policy_logits(output) -> Tensor:
    if hasattr(output, "logits"):
        return output.logits
    if hasattr(output, "policy_logits"):
        return output.policy_logits
    if isinstance(output, (tuple, list)):
        return output[1]
    raise TypeError("Model output does not expose categorical policy logits")


def flatten_named_tensors(
    tensors: Mapping[str, Tensor], names: Sequence[str]
) -> Tensor:
    if not names:
        sample = next(iter(tensors.values()))
        return sample.new_zeros((0,))
    return torch.cat([tensors[name].reshape(-1) for name in names], dim=0)


def unflatten_named_tensor(
    flat: Tensor,
    named_reference: Mapping[str, Tensor],
    names: Sequence[str],
) -> "OrderedDict[str, Tensor]":
    result: "OrderedDict[str, Tensor]" = OrderedDict()
    offset = 0
    for name in names:
        reference = named_reference[name]
        size = reference.numel()
        result[name] = flat[offset : offset + size].view_as(reference)
        offset += size
    if offset != flat.numel():
        raise ValueError(f"Unflatten consumed {offset} of {flat.numel()} values")
    return result


def compute_per_sample_value_jacobian(
    model: nn.Module,
    observations: Tensor,
    aux_param_names: Sequence[str],
    chunk_size: Optional[int] = None,
) -> Tensor:
    """Materialize J=[grad V_i] over only shared + critic-head parameters."""
    named_params = OrderedDict(model.named_parameters())
    missing = set(aux_param_names).difference(named_params)
    if missing:
        raise KeyError(f"Unknown auxiliary parameters: {sorted(missing)}")
    selected = OrderedDict((name, named_params[name]) for name in aux_param_names)
    other = OrderedDict(
        (name, parameter)
        for name, parameter in named_params.items()
        if name not in selected
    )
    buffers = OrderedDict(model.named_buffers())

    def single_value(selected_params, obs_i):
        merged = OrderedDict(other)
        merged.update(selected_params)
        output = functional_call(
            model,
            (merged, buffers),
            (obs_i.unsqueeze(0),),
        )
        return _value_output(output).reshape(())

    single_grad = grad(single_value)
    batch_size = observations.shape[0]
    chunk_size = batch_size if chunk_size is None else int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    rows = []
    for start in range(0, batch_size, chunk_size):
        grad_tree = vmap(single_grad, in_dims=(None, 0))(
            selected, observations[start : start + chunk_size]
        )
        rows.append(
            torch.cat(
                [
                    grad_tree[name].reshape(grad_tree[name].shape[0], -1)
                    for name in aux_param_names
                ],
                dim=1,
            )
        )
    return torch.cat(rows, dim=0)


def cholesky_solve_with_retry(
    gram: Tensor,
    rhs: Tensor,
    damping: float,
    max_retries: int = 5,
    damping_multiplier: float = 10.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """Solve (K + B*mu*I)x=rhs without forming an inverse."""
    if max_retries <= 0:
        raise ValueError("max_retries must be positive")
    if damping < 0.0 or not math.isfinite(damping):
        raise ValueError("damping must be finite and non-negative")
    batch_size = gram.shape[0]
    if gram.shape != (batch_size, batch_size):
        raise ValueError("gram must be square")
    eye = torch.eye(batch_size, device=gram.device, dtype=gram.dtype)
    effective_damping = float(damping)
    last_info = None
    for retry in range(max_retries):
        system = 0.5 * (gram + gram.T) + batch_size * effective_damping * eye
        factor, info = torch.linalg.cholesky_ex(system)
        last_info = info
        if int(info.max().item()) == 0:
            solution = torch.cholesky_solve(rhs.reshape(-1, 1), factor).squeeze(1)
            return solution, {
                "effective_damping": effective_damping,
                "cholesky_success": 1.0,
                "cholesky_retries": float(retry),
                "cholesky_min_diagonal": float(factor.diagonal().min().item()),
            }
        # Multiplying exactly zero would never recover.  This only affects the
        # debug/fallback case; formal configurations use positive damping.
        if effective_damping == 0.0:
            scale = max(float(gram.diagonal().abs().max().item()), 1.0)
            effective_damping = max(torch.finfo(gram.dtype).eps * scale, 1e-12)
        else:
            effective_damping *= float(damping_multiplier)
    raise RuntimeError(
        "Critic GGN Cholesky failed after "
        f"{max_retries} attempts; final info={last_info}"
    )


def compute_critic_ggn_direction(
    model: nn.Module,
    observations: Tensor,
    value_targets: Tensor,
    aux_param_names: Sequence[str],
    damping: float,
    jacobian_chunk_size: Optional[int] = None,
    cholesky_max_retries: int = 5,
    cholesky_damping_multiplier: float = 10.0,
    linear_solve_dtype: torch.dtype = torch.float64,
) -> Tuple["OrderedDict[str, Tensor]", Dict[str, float]]:
    """Compute d=J^T(JJ^T+B*mu*I)^-1(V-y) for mean MSE."""
    values = _value_output(model(observations))
    targets = value_targets.reshape_as(values).detach()
    residual = (values - targets).detach()
    jacobian = compute_per_sample_value_jacobian(
        model, observations, aux_param_names, jacobian_chunk_size
    )
    solve_jacobian = jacobian.to(linear_solve_dtype)
    solve_residual = residual.to(linear_solve_dtype)
    gram = solve_jacobian @ solve_jacobian.T
    raw_symmetry_error = torch.linalg.vector_norm(gram - gram.T) / (
        torch.linalg.vector_norm(gram) + 1e-12
    )
    beta, chol_diag = cholesky_solve_with_retry(
        gram,
        solve_residual,
        damping=damping,
        max_retries=cholesky_max_retries,
        damping_multiplier=cholesky_damping_multiplier,
    )
    flat_direction = (solve_jacobian.T @ beta).to(jacobian.dtype)
    response = jacobian @ flat_direction
    response_cosine = torch.dot(response, residual) / (
        torch.linalg.vector_norm(response)
        * torch.linalg.vector_norm(residual)
        + 1e-12
    )
    named_params = OrderedDict(model.named_parameters())
    direction = unflatten_named_tensor(flat_direction, named_params, aux_param_names)
    diagnostics = {
        "critic_mse_before": float((0.5 * residual.square().mean()).item()),
        "residual_norm": float(torch.linalg.vector_norm(residual).item()),
        "ggn_direction_l2_norm": float(torch.linalg.vector_norm(flat_direction).item()),
        "ggn_response_norm": float(torch.linalg.vector_norm(response).item()),
        "response_residual_cosine": float(response_cosine.item()),
        "gram_trace": float(torch.trace(gram).item()),
        "gram_max_diagonal": float(gram.diagonal().max().item()),
        "gram_symmetry_error": float(raw_symmetry_error.item()),
    }
    diagnostics.update(chol_diag)
    return direction, diagnostics


def embed_full_direction(
    model: nn.Module, direction_by_name: Mapping[str, Tensor]
) -> "OrderedDict[str, Tensor]":
    result: "OrderedDict[str, Tensor]" = OrderedDict()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            result[name] = direction_by_name.get(name, torch.zeros_like(parameter))
    return result


def actor_fisher_vector_product(
    model: nn.Module,
    observations: Tensor,
    direction_by_name: Mapping[str, Tensor],
    actor_param_names: Sequence[str],
) -> Tuple["OrderedDict[str, Tensor]", Tensor]:
    """Pure categorical actor Fisher/Hessian-KL product (no damping)."""
    named_params = OrderedDict(model.named_parameters())
    actor_params = [named_params[name] for name in actor_param_names]
    actor_direction = [direction_by_name[name] for name in actor_param_names]
    with torch.no_grad():
        reference_logits = _policy_logits(model(observations)).detach()
        reference_log_probs = F.log_softmax(reference_logits, dim=-1)
        reference_probs = reference_log_probs.exp()

    current_logits = _policy_logits(model(observations))
    current_log_probs = F.log_softmax(current_logits, dim=-1)
    mean_kl = (
        reference_probs * (reference_log_probs - current_log_probs)
    ).sum(dim=-1).mean()
    first = torch.autograd.grad(
        mean_kl,
        actor_params,
        create_graph=True,
        allow_unused=True,
    )
    dot = sum(
        (gradient * vector).sum()
        for gradient, vector in zip(first, actor_direction)
        if gradient is not None
    )
    second = torch.autograd.grad(
        dot,
        actor_params,
        retain_graph=False,
        allow_unused=True,
    )
    fvp: "OrderedDict[str, Tensor]" = OrderedDict()
    quadratic = observations.new_zeros(())
    for name, parameter, vector, product in zip(
        actor_param_names, actor_params, actor_direction, second
    ):
        product = torch.zeros_like(parameter) if product is None else product
        fvp[name] = product
        quadratic = quadratic + torch.sum(vector * product)
    return fvp, torch.clamp(quadratic, min=0.0)


def fisher_clip_scale(
    fisher_quadratic: Tensor,
    learning_rate: float,
    target_kl: Optional[float] = None,
    fisher_radius: Optional[float] = None,
    eps: float = 1e-12,
    enabled: bool = True,
) -> Tensor:
    if not enabled:
        return fisher_quadratic.new_ones(())
    q = torch.clamp(fisher_quadratic, min=0.0)
    if target_kl is not None:
        scale = torch.sqrt(
            q.new_tensor(2.0 * float(target_kl))
            / (float(learning_rate) ** 2 * q + eps)
        )
    elif fisher_radius is not None:
        scale = q.new_tensor(float(fisher_radius)) / torch.sqrt(q + eps)
    else:
        raise ValueError("target_kl or fisher_radius is required when clipping")
    return torch.clamp(scale, max=1.0)


@torch.no_grad()
def apply_direction(
    model: nn.Module,
    direction_by_name: Mapping[str, Tensor],
    learning_rate: float,
    allowed_names: Optional[Iterable[str]] = None,
) -> None:
    allowed = None if allowed_names is None else set(allowed_names)
    for name, parameter in model.named_parameters():
        if name not in direction_by_name:
            continue
        if allowed is not None and name not in allowed:
            continue
        parameter.add_(direction_by_name[name], alpha=-float(learning_rate))


def categorical_policy_kl(reference_logits: Tensor, current_logits: Tensor) -> Tensor:
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)
    reference_probs = reference_log_probs.exp()
    current_log_probs = F.log_softmax(current_logits, dim=-1)
    return (
        reference_probs * (reference_log_probs - current_log_probs)
    ).sum(dim=-1).mean()


def categorical_entropy(logits: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1).mean()


def explained_variance(targets: Tensor, predictions: Tensor) -> Tensor:
    target_variance = torch.var(targets, unbiased=False)
    if float(target_variance.item()) <= 1e-12:
        return targets.new_tensor(float("nan"))
    return 1.0 - torch.var(targets - predictions, unbiased=False) / target_variance


def policy_phase_critic_mse(
    shared_features_fn,
    value_from_features_fn,
    observations: Tensor,
    value_targets: Tensor,
    *,
    detach_shared_features: bool,
) -> Tuple[Tensor, Tensor]:
    """Return policy-phase critic MSE with an explicit detach boundary.

    The callable interface keeps this helper independent of a particular
    actor-critic class while making the PPG-style stop-gradient unit-testable.
    """
    features = shared_features_fn(observations)
    if detach_shared_features:
        features = features.detach()
    values = value_from_features_fn(features).reshape_as(value_targets)
    return F.mse_loss(values, value_targets), values


def run_auxiliary_critic_ggn_step(
    model: nn.Module,
    observations: Tensor,
    value_targets: Tensor,
    fisher_observations: Tensor,
    *,
    damping: float,
    learning_rate: float,
    target_kl: Optional[float],
    fisher_radius: Optional[float],
    use_actor_fisher_clip: bool,
    jacobian_chunk_size: Optional[int],
    cholesky_max_retries: int,
    cholesky_damping_multiplier: float,
    linear_solve_dtype: torch.dtype = torch.float64,
) -> Dict[str, float]:
    """Run one deterministic auxiliary step and return all MVP diagnostics."""
    groups = partition_named_parameters(model)
    aux_names = groups["shared"] + groups["critic_head"]
    actor_names = groups["shared"] + groups["actor_head"]
    with torch.no_grad():
        critic_before = _value_output(model(observations)).detach()
        fisher_logits_before = _policy_logits(model(fisher_observations)).detach()
        entropy_before = categorical_entropy(fisher_logits_before)
    direction_aux, diagnostics = compute_critic_ggn_direction(
        model,
        observations,
        value_targets,
        aux_names,
        damping,
        jacobian_chunk_size,
        cholesky_max_retries,
        cholesky_damping_multiplier,
        linear_solve_dtype,
    )
    direction_full = embed_full_direction(model, direction_aux)
    _, q_actor = actor_fisher_vector_product(
        model, fisher_observations, direction_full, actor_names
    )
    scale = fisher_clip_scale(
        q_actor,
        learning_rate,
        target_kl=target_kl,
        fisher_radius=fisher_radius,
        enabled=use_actor_fisher_clip,
    )
    clipped_direction = OrderedDict(
        (name, direction * scale) for name, direction in direction_full.items()
    )
    actor_zero_max = max(
        [
            float(clipped_direction[name].abs().max().item())
            for name in groups["actor_head"]
        ]
        or [0.0]
    )
    if actor_zero_max != 0.0:
        raise RuntimeError("Auxiliary actor-head direction is not exactly zero")
    apply_direction(
        model,
        clipped_direction,
        learning_rate,
        allowed_names=aux_names,
    )
    with torch.no_grad():
        critic_after = _value_output(model(observations)).detach()
        fisher_logits_after = _policy_logits(model(fisher_observations)).detach()
        mse_before = 0.5 * (critic_before - value_targets).square().mean()
        mse_after = 0.5 * (critic_after - value_targets).square().mean()
        actual_kl = categorical_policy_kl(
            fisher_logits_before, fisher_logits_after
        )
        entropy_after = categorical_entropy(fisher_logits_after)
        predicted_kl = (
            0.5
            * float(learning_rate) ** 2
            * scale.square()
            * q_actor
        )
        diagnostics.update(
            {
                "critic_mse_before": float(mse_before.item()),
                "critic_mse_after": float(mse_after.item()),
                "critic_mse_change": float((mse_after - mse_before).item()),
                "explained_variance_before": float(
                    explained_variance(value_targets, critic_before).item()
                ),
                "explained_variance_after": float(
                    explained_variance(value_targets, critic_after).item()
                ),
                "actor_fisher_quadratic": float(q_actor.item()),
                "actor_fisher_norm": float(torch.sqrt(q_actor + 1e-12).item()),
                "clip_scale": float(scale.item()),
                "clip_triggered": float(scale.item() < 1.0 - 1e-7),
                "predicted_kl": float(predicted_kl.item()),
                "actual_kl": float(actual_kl.item()),
                "policy_entropy_before": float(entropy_before.item()),
                "policy_entropy_after": float(entropy_after.item()),
                "actor_head_direction_max_abs": actor_zero_max,
            }
        )
    return diagnostics
