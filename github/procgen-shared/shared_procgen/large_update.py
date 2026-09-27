"""Shared B8192 update: the same B/B equations as the normal-batch methods."""
import torch
from .streamed_dual import solve_streamed
from .streamed_coefficient128 import solve_streamed_coeff128
from .update import adaptive_lr


def update_large(model, optimizer, obs, actions, advantages, returns, old_logits, *,
                 method, seed=0, rollout_index=0, epoch_index=0, minibatch_index=0,
                 batch_size=8192, chunk=128, ray_scale=True):
    # batch_size is explicit for numerical fixtures; public training fixes8192.
    batch = len(obs)
    if batch != batch_size or method not in ('dual127p1', 'coeff128'):
        raise ValueError('Invalid batch or shared method')
    device = actions.device
    oldlp = old_logits.detach().log_softmax(-1)
    adv = advantages - advantages.mean()
    adv = adv / (adv.square().mean().sqrt().detach() + 1e-8)
    ratios = []
    with torch.no_grad():
        for start in range(0, batch, chunk):
            sl = slice(start, min(start+chunk, batch))
            lp = model(obs[sl].to(device))[1].log_softmax(-1)
            ratios.append((lp-oldlp[sl]).gather(1, actions[sl, None]).squeeze(1).exp())
    ratio = torch.cat(ratios).clamp(.1, 10)
    nanchors = 127 if method == 'dual127p1' else 128
    sampling_seed = (seed*1_000_003 + rollout_index*10_007 + epoch_index*101
                     + minibatch_index + 128*17) % (2**63-1)
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    anchors = torch.randperm(batch, generator=generator, device=device)[:nanchors]
    noise = torch.randn(batch, device=device)
    targets = torch.stack((adv, torch.ones_like(adv)), 1)
    weights = torch.stack((ratio, torch.ones_like(ratio)), 1)
    if method == 'dual127p1':
        coefficients, diagnostics = solve_streamed(model, obs, actions, targets,
            weights, noise, anchors, True, batch, chunk=chunk)
        residual_key = 'residual'
    else:
        coefficients, diagnostics = solve_streamed_coeff128(model, obs, actions,
            targets, weights, noise, anchors, chunk=chunk, ray_scale=ray_scale)
        residual_key = 'reduced_residual'
    optimizer.zero_grad(set_to_none=True)
    actor_loss = value_loss = ordinary_mse = 0.0
    for start in range(0, batch, chunk):
        sl = slice(start, min(start+chunk, batch))
        values, logits = model(obs[sl].to(device))
        lp = logits.log_softmax(-1)
        r = (lp-oldlp[sl]).gather(1, actions[sl, None]).squeeze(1).exp().clamp(.1, 10)
        pi = -(r*coefficients[sl, 0].detach()).sum()/batch
        squared_error = (values-returns[sl]).square()
        vf = (squared_error*coefficients[sl, 1].detach()).sum()/batch
        (pi+vf).backward()
        actor_loss += float(pi.detach())
        value_loss += float(vf.detach())
        ordinary_mse += float(squared_error.detach().sum()/batch)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
    before = float(optimizer.param_groups[0]['lr'])
    optimizer.step()
    kl = entropy = 0.0
    with torch.no_grad():
        for start in range(0, batch, chunk):
            sl = slice(start, min(start+chunk, batch))
            lp = model(obs[sl].to(device))[1].log_softmax(-1)
            kl += float((oldlp[sl].exp()*(oldlp[sl]-lp)).sum()/batch)
            entropy += float(-(lp.exp()*lp).sum()/batch)
    after = adaptive_lr(before, kl)
    optimizer.param_groups[0]['lr'] = after
    return dict(method=method, sampling='uniform', batch=batch, reduced_budget=128,
                anchors=nanchors, curvature_divisor=batch, loss_divisor=batch,
                critic_ratio=False, actor_loss=actor_loss, value_loss=value_loss,
                ordinary_value_mse=ordinary_mse, kl=kl, entropy=entropy,
                entropy_timing='postupdate', lr_before=before, lr_after=after,
                grad_norm=float(norm), clip_scale=min(1., .5/(float(norm)+1e-6)),
                actor_residual=diagnostics[0][residual_key],
                critic_residual=diagnostics[1][residual_key], solve=diagnostics,
                coefficient_ray_scale=method == 'coeff128' and ray_scale, physical_chunk=chunk)
