"""Official-style single-network PPG buffer and auxiliary utilities."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from phasic_ef_ggn import (
    categorical_entropy,
    categorical_policy_kl,
    explained_variance,
    flatten_named_tensors,
)


def encode_procgen_observations(observations: Tensor) -> Tensor:
    """Losslessly pack Procgen observations normalized to [-1, 1]."""
    detached = observations.detach()
    if float(detached.min().item()) < -1.0001 or float(detached.max().item()) > 1.0001:
        raise ValueError("uint8 auxiliary packing requires observations in [-1, 1]")
    pixels = torch.round((detached + 1.0) * 127.5).clamp_(0, 255)
    reconstructed = pixels / 127.5 - 1.0
    max_error = float((reconstructed - detached).abs().max().item())
    if max_error > 1e-5:
        raise ValueError(f"observation packing is not lossless; max_error={max_error}")
    return pixels.to(device="cpu", dtype=torch.uint8)


def decode_procgen_observations(
    packed_observations: Tensor, device
) -> Tensor:
    return packed_observations.to(device=device, dtype=torch.float32) / 127.5 - 1.0


@torch.no_grad()
def compute_reference_logits(
    model: nn.Module,
    packed_observations: Tensor,
    batch_size: int,
    device,
) -> Tensor:
    rows = []
    for start in range(0, packed_observations.shape[0], batch_size):
        observations = decode_procgen_observations(
            packed_observations[start : start + batch_size], device
        )
        rows.append(model.forward_pi(obs=observations).detach().cpu())
    return torch.cat(rows, dim=0)


@torch.no_grad()
def evaluate_auxiliary_buffer(
    model: nn.Module,
    packed_observations: Tensor,
    normalized_targets: Tensor,
    batch_size: int,
    device,
    reference_logits: Optional[Tensor] = None,
) -> Dict[str, float]:
    aux_predictions = []
    true_predictions = []
    logits_rows = []
    for start in range(0, packed_observations.shape[0], batch_size):
        observations = decode_procgen_observations(
            packed_observations[start : start + batch_size], device
        )
        features = model.backbone_net(observations)
        aux_predictions.append(model.forward_aux_v(latents=features).cpu())
        true_predictions.append(model.forward_v(latents=features).cpu())
        logits_rows.append(model.forward_pi(latents=features).cpu())
    aux_predictions = torch.cat(aux_predictions)
    true_predictions = torch.cat(true_predictions)
    logits = torch.cat(logits_rows)
    targets = normalized_targets.cpu()
    result = {
        "buffer_aux_mse": float(
            (0.5 * (aux_predictions - targets).square().mean()).item()
        ),
        "buffer_true_mse": float(
            (0.5 * (true_predictions - targets).square().mean()).item()
        ),
        "buffer_aux_ev": float(
            explained_variance(targets, aux_predictions).item()
        ),
        "buffer_true_ev": float(
            explained_variance(targets, true_predictions).item()
        ),
        "buffer_policy_entropy": float(categorical_entropy(logits).item()),
    }
    if reference_logits is not None:
        result["buffer_policy_kl"] = float(
            categorical_policy_kl(reference_logits.cpu(), logits).item()
        )
    return result


def compute_flat_full_buffer_aux_gradient(
    model: nn.Module,
    packed_observations: Tensor,
    normalized_targets: Tensor,
    aux_param_names: Sequence[str],
    batch_size: int,
    device,
) -> Tensor:
    """Exact mean auxiliary-value gradient accumulated over the full buffer."""
    named_params = OrderedDict(model.named_parameters())
    selected = [named_params[name] for name in aux_param_names]
    accumulator = OrderedDict(
        (name, torch.zeros_like(named_params[name])) for name in aux_param_names
    )
    total = float(packed_observations.shape[0])
    for start in range(0, packed_observations.shape[0], batch_size):
        end = min(start + batch_size, packed_observations.shape[0])
        observations = decode_procgen_observations(
            packed_observations[start:end], device
        )
        targets = normalized_targets[start:end].to(device)
        predictions = model.forward_aux_v(obs=observations)
        loss = 0.5 * (predictions - targets).square().sum() / total
        gradients = torch.autograd.grad(loss, selected, allow_unused=False)
        for name, gradient in zip(aux_param_names, gradients):
            accumulator[name].add_(gradient.detach())
    return flatten_named_tensors(accumulator, aux_param_names)


def run_official_ppg_auxiliary(
    model: nn.Module,
    packed_observations: Tensor,
    normalized_targets: Tensor,
    optimizer,
    *,
    epochs: int,
    batch_size: int,
    beta_clone: float,
    vf_true_weight: float,
    device,
) -> Dict[str, float]:
    """Official detach-architecture loss: vf_aux + vf_true + KL clone."""
    reference_logits = compute_reference_logits(
        model, packed_observations, batch_size, device
    )
    before = evaluate_auxiliary_buffer(
        model,
        packed_observations,
        normalized_targets,
        batch_size,
        device,
        reference_logits=reference_logits,
    )
    nrows = packed_observations.shape[0]
    loss_sums = {"vf_aux": 0.0, "vf_true": 0.0, "pol_distance": 0.0}
    steps = 0
    for _ in range(epochs):
        for indices in torch.randperm(nrows).split(batch_size):
            observations = decode_procgen_observations(
                packed_observations[indices], device
            )
            targets = normalized_targets[indices].to(device)
            old_logits = reference_logits[indices].to(device)
            features = model.backbone_net(observations)
            logits = model.forward_pi(latents=features)
            aux_values = model.forward_aux_v(latents=features)
            true_values = model.forward_v(latents=features.detach())
            vf_aux = 0.5 * (aux_values - targets).square().mean()
            vf_true = 0.5 * (true_values - targets).square().mean()
            pol_distance = categorical_policy_kl(old_logits, logits)
            loss = vf_aux + vf_true_weight * vf_true + beta_clone * pol_distance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sums["vf_aux"] += float(vf_aux.detach().item())
            loss_sums["vf_true"] += float(vf_true.detach().item())
            loss_sums["pol_distance"] += float(pol_distance.detach().item())
            steps += 1
    after = evaluate_auxiliary_buffer(
        model,
        packed_observations,
        normalized_targets,
        batch_size,
        device,
        reference_logits=reference_logits,
    )
    diagnostics = {
        "official_aux_steps": float(steps),
        "official_aux_epochs": float(epochs),
        "buffer_rows": float(nrows),
    }
    for key, value in loss_sums.items():
        diagnostics[f"mean_{key}"] = value / max(steps, 1)
    for key, value in before.items():
        diagnostics[f"{key}_before"] = value
    for key, value in after.items():
        diagnostics[f"{key}_after"] = value
    return diagnostics


def fit_true_value_head_on_full_buffer(
    model: nn.Module,
    packed_observations: Tensor,
    normalized_targets: Tensor,
    optimizer,
    *,
    epochs: int,
    feature_batch_size: int,
    head_batch_size: int,
    device,
) -> Dict[str, float]:
    """Fit only the true critic head after a manual shared/aux-head GGN step."""
    features = []
    with torch.no_grad():
        for start in range(0, packed_observations.shape[0], feature_batch_size):
            observations = decode_procgen_observations(
                packed_observations[start : start + feature_batch_size], device
            )
            features.append(model.backbone_net(observations).cpu())
    features = torch.cat(features)
    targets_cpu = normalized_targets.cpu()
    with torch.no_grad():
        predictions_before = []
        for start in range(0, features.shape[0], head_batch_size):
            predictions_before.append(
                model.forward_v(
                    latents=features[start : start + head_batch_size].to(device)
                ).cpu()
            )
        predictions_before = torch.cat(predictions_before)
    steps = 0
    for _ in range(epochs):
        for indices in torch.randperm(features.shape[0]).split(head_batch_size):
            predictions = model.forward_v(latents=features[indices].to(device))
            targets = targets_cpu[indices].to(device)
            loss = 0.5 * (predictions - targets).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
    with torch.no_grad():
        predictions_after = []
        for start in range(0, features.shape[0], head_batch_size):
            predictions_after.append(
                model.forward_v(
                    latents=features[start : start + head_batch_size].to(device)
                ).cpu()
            )
        predictions_after = torch.cat(predictions_after)
    return {
        "true_head_aux_steps": float(steps),
        "true_head_mse_before": float(
            (0.5 * (predictions_before - targets_cpu).square().mean()).item()
        ),
        "true_head_mse_after": float(
            (0.5 * (predictions_after - targets_cpu).square().mean()).item()
        ),
        "true_head_ev_before": float(
            explained_variance(targets_cpu, predictions_before).item()
        ),
        "true_head_ev_after": float(
            explained_variance(targets_cpu, predictions_after).item()
        ),
    }
