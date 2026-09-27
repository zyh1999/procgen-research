"""Shared fixed-LR PPO: mean/B losses, one combined clip and Adam step."""
import math

import torch
from torch.nn import functional as F

LR = 3e-4
ADAM_EPS = 1e-5


def update_ppo(model, optimizer, obs, actions, advantages, returns, old_logits,
               *, chunk=128):
    """One logical minibatch; KL is diagnostic only, never an LR controller."""
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if any(group["lr"] != LR for group in optimizer.param_groups):
        raise ValueError("This PPO profile requires constant Adam LR=3e-4")
    device = next(model.parameters()).device
    batch = len(obs)
    if batch < 2:
        raise ValueError("At least two samples required for std normalization")
    actions = actions.detach().to(device)
    advantages = advantages.detach().to(device)
    returns = returns.detach().to(device)
    old_log_probs = F.log_softmax(old_logits.detach().to(device), dim=-1)
    advantages = advantages - advantages.mean()
    advantages = advantages / (advantages.std() + 1e-8)
    optimizer.zero_grad(set_to_none=True)
    policy_loss = value_mse = ratio_clip_fraction = 0.0
    for start in range(0, batch, chunk):
        sl = slice(start, min(start + chunk, batch))
        values, logits = model(obs[sl].to(device))
        log_probs = F.log_softmax(logits, dim=-1)
        action_index = actions[sl, None]
        ratio = (log_probs.gather(1, action_index).squeeze(1)
                 - old_log_probs[sl].gather(1, action_index).squeeze(1)).exp()
        pl = torch.maximum(-ratio * advantages[sl],
                           -ratio.clamp(.8, 1.2) * advantages[sl]).sum() / batch
        mse = (values - returns[sl]).square().sum() / batch
        (pl + .5 * mse).backward()
        policy_loss += float(pl.detach())
        value_mse += float(mse.detach())
        ratio_clip_fraction += float(((ratio.detach() - 1).abs() > .2).sum()) / batch
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
    optimizer.step()
    kl = entropy = 0.0
    with torch.no_grad():
        for start in range(0, batch, chunk):
            sl = slice(start, min(start + chunk, batch))
            _, logits = model(obs[sl].to(device))
            log_probs = F.log_softmax(logits, dim=-1)
            kl += float((old_log_probs[sl].exp() * (old_log_probs[sl] - log_probs)).sum() / batch)
            entropy += float(-(log_probs.exp() * log_probs).sum() / batch)
    metrics = dict(policy_loss=policy_loss, ordinary_value_mse=value_mse,
                   value_loss=.5 * value_mse, kl=kl, entropy=entropy,
                   grad_norm=float(norm), clip_scale=min(1., .5 / (float(norm) + 1e-6)),
                   ratio_clip_fraction=ratio_clip_fraction,
                   lr_before=LR, lr_after=LR, logical_minibatch=batch,
                   loss_divisor=batch, optimizer_steps=1,
                   controller="none_fixed_lr", entropy_timing="postupdate")
    if not all(math.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))):
        raise FloatingPointError("Nonfinite PPO update diagnostics")
    return metrics
