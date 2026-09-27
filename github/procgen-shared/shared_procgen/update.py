"""The two update rules. B=512 is the denominator, q=128 the reduced budget."""
import math
import torch
import torch.nn.functional as F

from .algebra import LayerwiseIO, named_add_fp32, named_scale_fp32
from .subcurvature import (select_anchors, named_from_gradients, assign_named_gradients,
                          solve_fullrhs_named_directions_exactfusion,
                          solve_dual_coefficients_exactfusion)
from .coefficient128 import solve_coeff128


def adaptive_lr(lr, behavior_kl):
    if not math.isfinite(behavior_kl):
        raise FloatingPointError("Nonfinite post-update behavior KL")
    if behavior_kl > 0.04:
        return max(lr / 1.5, 1e-4)
    if behavior_kl < 0.005:
        return min(lr * 1.5, 0.5)
    return lr


def update(model, optimizer, obs, actions, advantages, returns, old_logits, *,
           method="dual127p1", sampling="uniform", seed=0,
           rollout_index=0, epoch_index=0, minibatch_index=0, ray_scale=True):
    if len(obs) != 512:
        raise ValueError("This release freezes normal minibatch B=512")
    if method not in ("dual127p1", "coeff128", "fullrhs128"):
        raise ValueError(method)
    batch, q, damping = 512, 128, 0.5
    with LayerwiseIO(model) as layerwise:
        values, logits = model(obs)
    if (rollout_index, epoch_index, minibatch_index) == (0, 0, 0):
        layerwise.assert_parameter_coverage()
    log_probs = F.log_softmax(logits, -1)
    old_log_probs = F.log_softmax(old_logits.detach(), -1)
    logp = log_probs.gather(1, actions[:, None]).squeeze(1)
    log_ratio = (log_probs - old_log_probs).gather(1, actions[:, None]).squeeze(1)
    ratio = log_ratio.exp().clamp(0.1, 10.0)
    advantages = advantages - advantages.mean()
    advantages = advantages / (advantages.square().mean().sqrt().detach() + 1e-8)

    # One Gaussian draw per sample, shared by actor and critic curvature.
    # H_i = grad log pi(a_i|s_i) + 2*xi_i*grad V(s_i), xi_i ~ N(0,1).
    noise = torch.randn_like(values)
    factors = layerwise.factors(logp + 2.0 * noise.detach() * values, retain_graph=True)
    nanchors = 127 if method == "dual127p1" else 128
    sampling_seed = (seed * 1_000_003 + rollout_index * 10_007 + epoch_index * 101
                     + minibatch_index + q * 17) % (2**63 - 1)
    # Selection has a separate RNG stream; it cannot change rollout/noise draws.
    devices = [obs.device.index] if obs.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(sampling_seed)
        if obs.is_cuda:
            torch.cuda.manual_seed_all(sampling_seed)
        anchors, sampling_info = select_anchors(
            model, factors, ratio.detach(), nanchors, sampling,
            leverage_parameter_samples=2000, leverage_oversample=32,
            leverage_power_iterations=1, leverage_uniform_mixture=0.05)

    base_actor_loss = (-ratio * advantages).mean()  # /B, with actor ratio
    base_value_loss = (values - returns).square().mean()  # /B, no critic ratio
    if method == "fullrhs128":
        params = [p for p in model.parameters() if p.requires_grad]
        ga = torch.autograd.grad(base_actor_loss, params, retain_graph=True, allow_unused=True)
        gv = torch.autograd.grad(base_value_loss, params, retain_graph=True, allow_unused=True)
        directions, diagnostics = solve_fullrhs_named_directions_exactfusion(
            factors, [named_from_gradients(model, ga), named_from_gradients(model, gv)],
            [ratio.detach(), torch.ones_like(ratio)], anchors, damping, batch)
        # Separate actor/critic preconditioners, then ONE shared clipped update.
        direction = named_add_fp32(directions[0], named_scale_fp32(directions[1], 1.0))
        actor_loss, value_loss = base_actor_loss, base_value_loss
        optimizer.zero_grad()
        assign_named_gradients(model, direction)
        residual_key = "reduced_residual"
    else:
        solver = solve_coeff128 if method == "coeff128" else solve_dual_coefficients_exactfusion
        coefficients, diagnostics = solver(
            factors, torch.stack((advantages, torch.ones_like(advantages)), 1),
            torch.stack((ratio.detach(), torch.ones_like(ratio)), 1),
            anchors, damping, batch, **({"ray_scale": ray_scale} if method == "coeff128" else {}))
        # Coefficients are detached. Backprop uses REAL loss Jacobians, not H.
        actor_loss = (-ratio * coefficients[:, 0]).mean()  # /B
        value_loss = ((values - returns).square() * coefficients[:, 1]).mean()  # /B
        optimizer.zero_grad()
        (actor_loss + value_loss).backward()  # vf_coef=1; entropy coefficient=0
        residual_key = "reduced_residual" if method == "coeff128" else "combined_reduced_residual"

    lr_before = float(optimizer.param_groups[0]["lr"])
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        _, after_logits = model(obs)
        after_log_probs = F.log_softmax(after_logits, -1)
        kl = (old_log_probs.exp() * (old_log_probs - after_log_probs)).sum(-1).mean().item()
        step_kl = (log_probs.exp() * (log_probs - after_log_probs)).sum(-1).mean().item()
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean().item()
    lr_after = adaptive_lr(lr_before, kl)
    optimizer.param_groups[0]["lr"] = lr_after  # applies to the NEXT minibatch
    return dict(method=method, sampling=sampling, batch=batch, reduced_budget=q,
                anchors=nanchors, curvature_divisor=batch, loss_divisor=batch,
                critic_ratio=False, actor_loss=actor_loss.item(), value_loss=value_loss.item(),
                ordinary_value_mse=base_value_loss.item(), kl=kl, step_kl=step_kl,
                entropy=entropy, entropy_timing="preupdate", lr_before=lr_before, lr_after=lr_after,
                coefficient_ray_scale=method == "coeff128" and ray_scale,
                grad_norm=float(grad_norm), clip_scale=min(1.0, 0.5 / (float(grad_norm) + 1e-6)),
                actor_residual=diagnostics[0][residual_key], critic_residual=diagnostics[1][residual_key],
                solve=diagnostics, selection=sampling_info)
