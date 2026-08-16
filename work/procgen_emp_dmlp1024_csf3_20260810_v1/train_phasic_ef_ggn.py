from procgen import ProcgenEnv

import os
import json
import yaml
import time
import math
import types
import torch
import argparse
import numpy as np
from tqdm import trange
from datetime import datetime
from collections import OrderedDict, deque
import utils.logger as logger

from torch.nn import functional as F
from torch.func import vmap, grad, functional_call

# pytorch distributed training
import torch.multiprocessing as mp

from utils.runners import Runner
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from utils.utils import build_cnn, build_resnet, build_mlp
from utils.utils import SharedActorCritic, count_vars, safemean, set_seed, set_grads_from_flat
from vec_env import ( VecExtractDictObs, VecMonitor, VecNormalize)
from phasic_ef_ggn import (
    actor_fisher_vector_product,
    fisher_clip_scale,
    partition_named_parameters,
    policy_phase_critic_mse,
    run_auxiliary_critic_ggn_step,
    run_full_gradient_anchor_ggn_step,
)
from actor_ef_ablation import damped_empirical_fisher_inverse
from ppg_auxiliary import (
    compute_flat_full_buffer_aux_gradient,
    compute_reference_logits,
    decode_procgen_observations,
    encode_procgen_observations,
    evaluate_auxiliary_buffer,
    fit_true_value_head_on_full_buffer,
    run_official_ppg_auxiliary,
)


def adaptive_lr_update_value(
    current_lr,
    measured_kl,
    lower_kl,
    upper_kl,
    min_lr,
    max_lr,
):
    """Return the next LR for the Procgen divide/multiply-by-1.5 rule."""
    if measured_kl > upper_kl:
        return max(current_lr / 1.5, min_lr)
    if measured_kl < lower_kl:
        return min(current_lr * 1.5, max_lr)
    return current_lr

def learn(world_size, algo, actor_critic, writer, venv, device,
          total_timesteps, nsteps, algo_config, log_config, log_dir=None):

    gamma = .999
    lam = .95

    per_epoch_timesteps = nsteps * venv.num_envs
    epochs = total_timesteps // per_epoch_timesteps + 1

    minibatch_size = per_epoch_timesteps // algo_config.minibatches

    # Instantiate the runner object
    runner = Runner(env=venv, model=actor_critic, nsteps=nsteps, gamma=gamma, lam=lam, adv_type=algo_config.adv_type, device=device)
    epinfobuf = deque(maxlen=100)

    named_trainable_params = OrderedDict(
        (k, v) for k, v in actor_critic.named_parameters() if v.requires_grad
    )
    dict_params = {k: v.detach() for k, v in named_trainable_params.items()}
    dict_buffers = {k: v.detach() for k, v in actor_critic.named_buffers() if v.requires_grad}
    use_phasic_training = bool(getattr(algo_config, 'use_phasic_training', False))
    use_official_ppg_auxiliary = bool(getattr(
        algo_config, 'use_official_ppg_auxiliary', False
    ))
    use_critic_ggn_auxiliary = bool(getattr(
        algo_config, 'use_critic_ggn_auxiliary', False
    ))
    use_actor_fisher_aux_clip = bool(getattr(
        algo_config, 'use_actor_fisher_aux_clip', False
    ))
    use_full_gradient_anchor_curvature = bool(getattr(
        algo_config, 'use_full_gradient_anchor_curvature', False
    ))
    if bool(getattr(algo_config, 'use_joint_2b_auxiliary_system', False)):
        raise ValueError('joint 2B auxiliary systems are forbidden in the MVP')
    if bool(getattr(algo_config, 'use_gaussian_critic_sampling', False)):
        raise ValueError('Gaussian critic sampling is forbidden in the MVP')
    if bool(getattr(algo_config, 'use_kaczmarz', False)) or bool(
        getattr(algo_config, 'is_kaczmarz', False)
    ):
        raise ValueError('Kaczmarz is forbidden in the phasic MVP')
    if int(getattr(algo_config, 'auxiliary_steps_per_cycle', 1)) != 1:
        raise ValueError('The MVP requires auxiliary_steps_per_cycle=1')
    if use_critic_ggn_auxiliary and not use_phasic_training:
        raise ValueError('critic GGN auxiliary requires phasic separation')
    if use_official_ppg_auxiliary and not use_phasic_training:
        raise ValueError('official PPG auxiliary requires phasic separation')
    if use_official_ppg_auxiliary and use_critic_ggn_auxiliary:
        raise ValueError('official Adam auxiliary and GGN auxiliary are exclusive')
    if use_actor_fisher_aux_clip and not use_critic_ggn_auxiliary:
        raise ValueError('actor Fisher auxiliary clip requires critic GGN')
    if str(algo_config.optimizer).lower() != 'sgd':
        raise ValueError('The controlled actor EF/NPG baseline uses SGD')
    if float(getattr(algo_config, 'optimizer_momentum', 0.0)) != 0.0:
        raise ValueError('The phasic MVP forbids optimizer history/momentum')
    ent_coef = float(getattr(algo_config, 'ent_coef', 0.0))
    use_actor_entropy_natural_gradient = bool(getattr(
        algo_config, 'use_actor_entropy_natural_gradient', False
    ))
    if ent_coef < 0.0:
        raise ValueError('ent_coef must be non-negative')
    if (ent_coef != 0.0) != use_actor_entropy_natural_gradient:
        raise ValueError(
            'nonzero ent_coef and use_actor_entropy_natural_gradient=true '
            'must be enabled together'
        )
    actor_policy_target_kl = getattr(
        algo_config, 'actor_policy_target_kl', None
    )
    use_actor_policy_fisher_clip = bool(getattr(
        algo_config, 'use_actor_policy_fisher_clip', False
    ))
    if use_actor_policy_fisher_clip:
        if actor_policy_target_kl is None or float(actor_policy_target_kl) <= 0.0:
            raise ValueError(
                'actor policy Fisher clip requires positive actor_policy_target_kl'
            )
        if str(getattr(
            algo_config, 'actor_policy_kl_budget_mode', 'equal_split'
        )) != 'equal_split':
            raise ValueError('actor_policy_kl_budget_mode must be equal_split')
    elif actor_policy_target_kl is not None:
        raise ValueError(
            'actor_policy_target_kl requires use_actor_policy_fisher_clip=true'
        )
    if not bool(getattr(
        algo_config, 'separate_policy_critic_steps', False
    )):
        raise ValueError(
            'The phasic implementation requires separate_policy_critic_steps=true'
        )

    parameter_groups = partition_named_parameters(actor_critic)
    actor_policy_names = (
        parameter_groups['shared'] + parameter_groups['actor_head']
    )
    policy_critic_names = (
        parameter_groups['critic_head']
        if use_phasic_training
        else parameter_groups['shared'] + parameter_groups['critic_head']
    )
    actor_policy_name_set = set(actor_policy_names)
    actor_policy_params = [
        named_trainable_params[name] for name in actor_policy_names
    ]
    policy_critic_params = [
        named_trainable_params[name] for name in policy_critic_names
    ]
    if not actor_policy_params or not policy_critic_params:
        raise ValueError('actor and policy-phase critic parameter groups must be non-empty')
    if (
        use_official_ppg_auxiliary
        or (use_critic_ggn_auxiliary and use_full_gradient_anchor_curvature)
    ) and not (
        parameter_groups['aux_critic_head']
    ):
        raise ValueError('official-aligned auxiliary requires aux_vf_head')
    aux_buffer_observations = []
    aux_buffer_raw_targets = []
    last_aux_info = None

    adaptive_kl_mode = str(getattr(
        algo_config, 'adaptive_kl_mode', 'procgen_rollout'
    ))
    if adaptive_kl_mode != 'procgen_rollout':
        raise ValueError(
            'exact deterministic-GGN control requires '
            'adaptive_kl_mode=procgen_rollout'
        )
    adaptive_kl_lower = float(getattr(
        algo_config, 'adaptive_kl_lower', 0.005
    ))
    adaptive_kl_upper = float(getattr(
        algo_config, 'adaptive_kl_upper', 0.04
    ))
    adaptive_lr_min = float(getattr(
        algo_config, 'adaptive_lr_min', 1e-4
    ))
    adaptive_lr_max = float(getattr(
        algo_config, 'adaptive_lr_max', algo_config.lr
    ))
    if not (
        0.0 <= adaptive_kl_lower < adaptive_kl_upper
        and 0.0 < adaptive_lr_min <= float(algo_config.lr) <= adaptive_lr_max
    ):
        raise ValueError('invalid adaptive KL thresholds or LR bounds')

    actor_steps_per_rollout = int(algo_config.epochs) * int(
        algo_config.minibatches
    )
    if actor_steps_per_rollout <= 0:
        raise ValueError('actor steps per rollout must be positive')
    actor_policy_step_target_kl = (
        float(actor_policy_target_kl) / float(actor_steps_per_rollout)
        if use_actor_policy_fisher_clip
        else None
    )

    optimizer_momentum = float(getattr(
        algo_config, 'optimizer_momentum', 0.0
    ))
    if optimizer_momentum != 0.0:
        raise ValueError('optimizer_momentum must be zero in the phasic MVP')
    actor_optimizer = SGD(
        actor_policy_params,
        lr=algo_config.lr,
        momentum=0.0,
        dampening=0.0,
        nesterov=False,
    )
    policy_critic_optimizer = Adam(
        policy_critic_params,
        lr=float(getattr(algo_config, 'policy_critic_lr', 5e-4)),
        eps=float(getattr(algo_config, 'adam_eps', 1e-8)),
    )
    official_aux_optimizer = (
        Adam(
            actor_critic.parameters(),
            lr=float(getattr(algo_config, 'official_aux_lr', 5e-4)),
            eps=float(getattr(algo_config, 'adam_eps', 1e-8)),
        )
        if use_official_ppg_auxiliary
        else None
    )
    ggn_true_head_optimizer = (
        Adam(
            [
                named_trainable_params[name]
                for name in parameter_groups['critic_head']
            ],
            lr=float(getattr(algo_config, 'official_aux_lr', 5e-4)),
            eps=float(getattr(algo_config, 'adam_eps', 1e-8)),
        )
        if use_critic_ggn_auxiliary and use_full_gradient_anchor_curvature
        else None
    )

    if hasattr(algo_config, 'lr_decay') and algo_config.lr_decay == 'cosine':
        lr_scheduler = CosineAnnealingLR(actor_optimizer, T_max=epochs*algo_config.epochs*algo_config.minibatches, eta_min=0.001)
    else:
        lr_scheduler = None

    def adapt_learning_rate(measured_kl):
        before = float(actor_optimizer.param_groups[0]['lr'])
        after = adaptive_lr_update_value(
            before,
            measured_kl,
            adaptive_kl_lower,
            adaptive_kl_upper,
            adaptive_lr_min,
            adaptive_lr_max,
        )
        actor_optimizer.param_groups[0]['lr'] = after
        return before, after

    # Start total timer
    tfirststart = time.perf_counter()

    def Phasic_Advantage_Update(_obs, _act, _adv, _ret, _outputs_old):
        """Existing actor EF/NPG step, followed by an ordinary critic step.

        Baseline A permits the critic MSE to update the shared trunk.  B/C/D
        detach shared features so the same MSE updates only the critic head.
        Actor and critic gradients are clipped independently: a large critic
        head gradient must never rescale the actor EF/NPG direction.  The
        auxiliary preconditioned direction is never routed through this
        optimizer.
        """
        _vals, _outputs = actor_critic(_obs)
        if not actor_critic.is_discrete:
            raise NotImplementedError('Procgen phasic EF/NPG expects discrete actions')

        _logp_full = F.log_softmax(_outputs, dim=-1)
        _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
        _llr = torch.gather(
            _logp_full - _logp_full_old,
            dim=-1,
            index=_act.unsqueeze(-1),
        ).squeeze(1)
        _ratio = torch.exp(_llr)
        _entropy = -(
            torch.exp(_logp_full) * _logp_full
        ).sum(dim=-1).mean()

        def compute_pi_logp(params, buffers, batch_obs, batch_act):
            _, batch_outputs = functional_call(
                actor_critic,
                (params, buffers),
                (batch_obs.unsqueeze(0),),
            )
            batch_logp = F.log_softmax(batch_outputs, dim=-1)
            return torch.gather(
                batch_logp,
                dim=-1,
                index=batch_act.reshape(1, 1),
            ).reshape(())

        _adv = _adv - _adv.mean()
        if algo_config.clamp_ratio:
            _ratio = torch.clamp(
                _ratio, algo_config.min_ratio, algo_config.max_ratio
            )
        if algo_config.norm_obj == 'adv':
            normalizer = torch.sqrt(_adv.pow(2).mean()).detach()
        elif algo_config.norm_obj == 'obj':
            normalizer = torch.sqrt((_ratio * _adv).pow(2).mean()).detach()
        elif algo_config.norm_obj == 'ratio':
            normalizer = (
                _ratio.mean().detach()
                * torch.sqrt(_adv.pow(2).mean()).detach()
            )
        else:
            raise NotImplementedError
        normalized_advantage = _adv / (normalizer + 1e-8)

        per_sample_grad = vmap(
            grad(compute_pi_logp),
            in_dims=(None, None, 0, 0),
            randomness='different',
        )(dict_params, dict_buffers, _obs, _act)
        batch_size = _obs.shape[0]
        actor_jacobian = torch.cat(
            [
                value.contiguous().view(batch_size, -1)
                for value in per_sample_grad.values()
            ],
            dim=-1,
        )
        del per_sample_grad

        if use_actor_entropy_natural_gradient:
            entropy_gradients = torch.autograd.grad(
                _entropy,
                tuple(named_trainable_params.values()),
                allow_unused=True,
                retain_graph=False,
            )
            entropy_gradient = torch.cat([
                (
                    torch.zeros_like(parameter)
                    if gradient_value is None
                    else gradient_value
                ).contiguous().view(-1)
                for parameter, gradient_value in zip(
                    named_trainable_params.values(), entropy_gradients
                )
            ]).detach()
        else:
            entropy_gradient = torch.zeros(
                actor_jacobian.shape[1],
                device=actor_jacobian.device,
                dtype=actor_jacobian.dtype,
            )

        with torch.no_grad():
            kernel = actor_jacobian @ actor_jacobian.T / float(batch_size)
            weighted_kernel = kernel * _ratio.detach().unsqueeze(0)
            eye = torch.eye(
                batch_size, device=weighted_kernel.device, dtype=weighted_kernel.dtype
            )
            solve_dtype_name = str(
                getattr(algo_config, 'linear_solve_dtype', 'float64')
            ).lower()
            solve_dtype = (
                torch.float64
                if solve_dtype_name in {'float64', 'fp64', 'double'}
                else weighted_kernel.dtype
            )
            solve_matrix = (
                weighted_kernel + float(algo_config.cg_damping) * eye
            ).to(solve_dtype)
            solve_rhs = normalized_advantage.detach().to(solve_dtype)
            alpha_solve = torch.linalg.solve(solve_matrix, solve_rhs)
            actor_solve_residual = torch.linalg.vector_norm(
                solve_matrix @ alpha_solve - solve_rhs
            )
            alpha = alpha_solve.to(actor_jacobian.dtype)
            actor_direction = actor_jacobian.T @ (
                _ratio.detach().to(alpha.dtype) * alpha
            ) / float(batch_size)
            advantage_direction_l2 = torch.linalg.vector_norm(actor_direction)
            entropy_direction_l2 = actor_direction.new_zeros(())
            entropy_gradient_l2 = actor_direction.new_zeros(())
            entropy_inverse_solve_residual = actor_direction.new_zeros(())
            if use_actor_entropy_natural_gradient:
                entropy_direction, entropy_diagnostics = (
                    damped_empirical_fisher_inverse(
                        actor_jacobian,
                        _ratio.detach(),
                        entropy_gradient,
                        float(algo_config.cg_damping),
                        solve_dtype=solve_dtype,
                    )
                )
                actor_direction = (
                    actor_direction + ent_coef * entropy_direction
                )
                entropy_direction_l2 = actor_direction.new_tensor(
                    entropy_diagnostics['entropy_direction_l2']
                )
                entropy_gradient_l2 = actor_direction.new_tensor(
                    entropy_diagnostics['entropy_gradient_l2']
                )
                entropy_inverse_solve_residual = actor_direction.new_tensor(
                    entropy_diagnostics['entropy_inverse_solve_residual']
                )
            actor_direction_l2 = torch.linalg.vector_norm(actor_direction)

        # Optional actor-only safety ablation.  This is a pure categorical
        # Fisher/Hessian-KL clip with no damping.  The per-rollout budget is
        # divided evenly over the configured actor minibatch steps so that a
        # four-epoch run does not silently receive four times the trust budget.
        actor_direction_by_name = OrderedDict()
        direction_offset = 0
        actor_critic_direction_max_abs = 0.0
        for name, parameter in named_trainable_params.items():
            size = parameter.numel()
            piece = actor_direction[
                direction_offset:direction_offset + size
            ].view_as(parameter)
            if name in actor_policy_name_set:
                actor_direction_by_name[name] = piece
            else:
                actor_critic_direction_max_abs = max(
                    actor_critic_direction_max_abs,
                    float(piece.detach().abs().max().item()),
                )
            direction_offset += size
        if direction_offset != actor_direction.numel():
            raise RuntimeError('Actor direction/parameter flattening mismatch')

        actor_policy_fisher_quadratic = actor_direction.new_zeros(())
        actor_policy_fisher_clip_scale = actor_direction.new_ones(())
        actor_policy_predicted_kl_pre_clip = actor_direction.new_zeros(())
        if use_actor_policy_fisher_clip:
            _, actor_policy_fisher_quadratic = actor_fisher_vector_product(
                actor_critic,
                _obs,
                actor_direction_by_name,
                actor_policy_names,
            )
            current_actor_lr = float(actor_optimizer.param_groups[0]['lr'])
            actor_policy_fisher_clip_scale = fisher_clip_scale(
                actor_policy_fisher_quadratic,
                current_actor_lr,
                target_kl=actor_policy_step_target_kl,
                enabled=True,
            )
            actor_policy_predicted_kl_pre_clip = (
                0.5
                * current_actor_lr**2
                * actor_policy_fisher_quadratic
            )
            actor_direction = (
                actor_direction * actor_policy_fisher_clip_scale.detach()
            )

        _loss_pi = (-_ratio.detach() * normalized_advantage.detach()).mean()

        # Policy step.  SGD with zero momentum applies +lr*actor_direction
        # because the optimizer receives its negative as the gradient.  Clip
        # only shared+actor-head parameters, never the critic head.
        # Optimizers own overlapping/disjoint subsets across phases.  Clear
        # the complete model so stale actor gradients cannot masquerade as a
        # critic-gradient leak (and vice versa).
        actor_critic.zero_grad(set_to_none=True)
        offset = 0
        for name, parameter in named_trainable_params.items():
            size = parameter.numel()
            actor_piece = actor_direction[offset : offset + size].view_as(parameter)
            if name in actor_policy_name_set:
                parameter.grad = actor_piece.detach().clone().neg_()
            else:
                actor_critic_direction_max_abs = max(
                    actor_critic_direction_max_abs,
                    float(actor_piece.detach().abs().max().item()),
                )
            offset += size
        if offset != actor_direction.numel():
            raise RuntimeError('Actor direction/parameter flattening mismatch')
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_policy_params, algo_config.max_grad_norm
        )
        actor_clip_scale = min(
            1.0,
            float(algo_config.max_grad_norm)
            / (float(actor_grad_norm.item()) + 1e-12),
        )
        actor_policy_predicted_kl_post_clip = (
            actor_policy_predicted_kl_pre_clip
            * actor_policy_fisher_clip_scale.square()
            * actor_clip_scale**2
        )
        actor_optimizer.step()

        # Value step.  Recompute features after the actor update, as in the
        # conceptual PPG schedule.  In B/C/D the detach makes this exactly a
        # critic-head-only update.  A deliberately retains shared-trunk MSE.
        _loss_v, critic_values_for_loss = policy_phase_critic_mse(
            actor_critic.backbone_net,
            lambda features: actor_critic.forward_v(latents=features),
            _obs,
            _ret,
            detach_shared_features=use_phasic_training,
        )
        actor_critic.zero_grad(set_to_none=True)
        (float(algo_config.vf_coef) * _loss_v).backward()
        critic_grad_norm = torch.sqrt(sum(
            parameter.grad.detach().square().sum()
            for parameter in policy_critic_params
            if parameter.grad is not None
        ))
        policy_critic_max_grad_norm = getattr(
            algo_config, 'policy_critic_max_grad_norm', None
        )
        if policy_critic_max_grad_norm is None:
            critic_clip_scale = 1.0
        else:
            policy_critic_max_grad_norm = float(policy_critic_max_grad_norm)
            torch.nn.utils.clip_grad_norm_(
                policy_critic_params, policy_critic_max_grad_norm
            )
            critic_clip_scale = min(
                1.0,
                policy_critic_max_grad_norm
                / (float(critic_grad_norm.item()) + 1e-12),
            )
        critic_forbidden_grad_max_abs = max(
            [
                float(parameter.grad.detach().abs().max().item())
                for name, parameter in named_trainable_params.items()
                if name not in set(policy_critic_names)
                and parameter.grad is not None
            ]
            or [0.0]
        )
        if critic_forbidden_grad_max_abs != 0.0:
            raise RuntimeError(
                'Policy-phase critic gradient escaped its parameter group'
            )
        policy_critic_optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        _loss = _loss_pi + float(algo_config.vf_coef) * _loss_v

        with torch.no_grad():
            _, outputs_after = actor_critic(_obs)
            logp_after = F.log_softmax(outputs_after, dim=-1)
            current_kl = (
                torch.exp(_logp_full) * (_logp_full - logp_after)
            ).sum(dim=-1).mean()
            rollout_kl = (
                torch.exp(_logp_full_old) * (_logp_full_old - logp_after)
            ).sum(dim=-1).mean()
            pi_info = dict(
                kl=rollout_kl.item(),
                curr_kl=current_kl.item(),
                curr_lr=actor_optimizer.param_groups[0]['lr'],
                ent=_entropy.item(),
                cf=0.0,
                # Keep grad_norm as the actor pre-clip norm for compatibility;
                # the old combined actor+critic norm was mathematically unsafe.
                grad_norm=actor_grad_norm.item(),
                actor_grad_norm=actor_grad_norm.item(),
                actor_grad_clip_scale=actor_clip_scale,
                critic_grad_norm=critic_grad_norm.item(),
                critic_grad_clip_scale=critic_clip_scale,
                actor_critic_direction_max_abs=actor_critic_direction_max_abs,
                critic_forbidden_grad_max_abs=critic_forbidden_grad_max_abs,
                separate_policy_critic_steps=1.0,
                ratio_max=_ratio.max().item(),
                ratio_min=_ratio.min().item(),
                actor_rows=batch_size,
                actor_solve_residual=actor_solve_residual.item(),
                advantage_direction_l2=advantage_direction_l2.item(),
                actor_direction_l2=actor_direction_l2.item(),
                actor_entropy_coefficient=ent_coef,
                entropy_gradient_l2=entropy_gradient_l2.item(),
                entropy_direction_l2=entropy_direction_l2.item(),
                entropy_inverse_solve_residual=(
                    entropy_inverse_solve_residual.item()
                ),
                actor_policy_fisher_quadratic=(
                    actor_policy_fisher_quadratic.item()
                ),
                actor_policy_fisher_clip_scale=(
                    actor_policy_fisher_clip_scale.item()
                ),
                actor_policy_target_kl=(
                    float(actor_policy_target_kl)
                    if actor_policy_target_kl is not None
                    else 0.0
                ),
                actor_policy_step_target_kl=(
                    float(actor_policy_step_target_kl)
                    if actor_policy_step_target_kl is not None
                    else 0.0
                ),
                actor_policy_predicted_kl_pre_clip=(
                    actor_policy_predicted_kl_pre_clip.item()
                ),
                actor_policy_predicted_kl_post_clip=(
                    actor_policy_predicted_kl_post_clip.item()
                ),
                critic_head_only=float(use_phasic_training),
                policy_critic_mse=_loss_v.item(),
            )
        return _loss, _loss_pi, _loss_v, pi_info

    if algo != 'adv':
        raise ValueError("The phasic MVP exposes only algo=adv actor EF/NPG")
    update_actor_critic = Phasic_Advantage_Update

    tepochs = trange(epochs + 1, desc='Epoch starts', leave=True)
    inds = np.arange(per_epoch_timesteps)

    for epoch in tepochs:
        tstart = time.perf_counter()
        tepochs.set_description('Stepping environment...')
        actor_critic.eval()
        obs, ret, act, adv, outputs_old, epinfos = runner.run()
        raw_value_targets = ret.detach().clone()
        epinfobuf.extend(epinfos)
        tepochs.set_description('Minibatch training...')

        if actor_critic.with_popart:
            actor_critic.last_v_layer.update(ret)
            ret = actor_critic.last_v_layer.normalize(ret)
            adv = actor_critic.last_v_layer.normalize(adv)

        if actor_critic.obs_rms is not None:
            actor_critic.obs_rms.training = True
            obs = actor_critic.obs_rms(obs)
            actor_critic.obs_rms.training = False
            with torch.no_grad():
                outputs_old = actor_critic.forward_pi(obs)

        use_any_auxiliary = (
            use_official_ppg_auxiliary or use_critic_ggn_auxiliary
        )
        if use_any_auxiliary:
            # Official PPG retains the complete policy-phase rollout.  Procgen
            # pixels are packed losslessly to uint8 to keep the CPU buffer
            # practical; targets remain raw until the pre-aux PopArt snapshot.
            aux_buffer_observations.append(
                encode_procgen_observations(obs)
            )
            aux_buffer_raw_targets.append(
                raw_value_targets.detach().to('cpu')
            )

        actor_critic.train()
        for _ in range(algo_config.epochs):
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, per_epoch_timesteps, minibatch_size):
                end = start + minibatch_size
                mbinds = inds[start:end]
                mb_obs, mb_act, mb_adv, mb_ret, mb_outputs_old = obs[mbinds], act[mbinds], adv[mbinds], ret[mbinds], outputs_old[mbinds]
                mb_loss, mb_loss_pi, mb_loss_v, pi_info = update_actor_critic(mb_obs, mb_act, mb_adv, mb_ret, mb_outputs_old)

        policy_updates_per_cycle = int(getattr(
            algo_config, 'policy_updates_per_cycle', 16
        ))
        if policy_updates_per_cycle <= 0:
            raise ValueError('policy_updates_per_cycle must be positive')
        if (
            use_any_auxiliary
            and (epoch + 1) % policy_updates_per_cycle == 0
        ):
            all_aux_obs = torch.cat(aux_buffer_observations, dim=0)
            all_aux_raw_targets = torch.cat(aux_buffer_raw_targets, dim=0)
            if actor_critic.with_popart:
                all_aux_targets = actor_critic.last_v_layer.normalize(
                    all_aux_raw_targets.to(device)
                ).detach().cpu()
            else:
                all_aux_targets = all_aux_raw_targets
            was_training = actor_critic.training
            aux_minibatch_size = int(getattr(
                algo_config, 'official_aux_minibatch_size', 1024
            ))
            aux_epochs = int(getattr(algo_config, 'official_aux_epochs', 6))
            if use_official_ppg_auxiliary:
                actor_critic.train()
                last_aux_info = run_official_ppg_auxiliary(
                    actor_critic,
                    all_aux_obs,
                    all_aux_targets,
                    official_aux_optimizer,
                    epochs=aux_epochs,
                    batch_size=aux_minibatch_size,
                    beta_clone=float(getattr(algo_config, 'beta_clone', 1.0)),
                    vf_true_weight=float(getattr(
                        algo_config, 'vf_true_weight', 1.0
                    )),
                    device=device,
                )
                last_aux_info['auxiliary_mode_official_adam'] = 1.0
                last_aux_info['auxiliary_mode_anchor_ggn'] = 0.0
            else:
                actor_critic.eval()
                reference_logits = compute_reference_logits(
                    actor_critic,
                    all_aux_obs,
                    aux_minibatch_size,
                    device,
                )
                buffer_before = evaluate_auxiliary_buffer(
                    actor_critic,
                    all_aux_obs,
                    all_aux_targets,
                    aux_minibatch_size,
                    device,
                    reference_logits=reference_logits,
                    include_aux_head=use_full_gradient_anchor_curvature,
                )
                fisher_batch_size = int(getattr(
                    algo_config, 'actor_fisher_batch_size', 256
                ))
                permutation = torch.randperm(all_aux_obs.shape[0])
                common_step_kwargs = dict(
                    damping=float(getattr(
                        algo_config, 'critic_ggn_damping', 1e-2
                    )),
                    learning_rate=float(getattr(
                        algo_config, 'aux_learning_rate', 1e-3
                    )),
                    target_kl=getattr(algo_config, 'actor_target_kl', None),
                    fisher_radius=getattr(
                        algo_config, 'actor_fisher_radius', None
                    ),
                    use_actor_fisher_clip=use_actor_fisher_aux_clip,
                    jacobian_chunk_size=getattr(
                        algo_config, 'jacobian_chunk_size', 16
                    ),
                    cholesky_max_retries=int(getattr(
                        algo_config, 'cholesky_max_retries', 5
                    )),
                    cholesky_damping_multiplier=float(getattr(
                        algo_config, 'cholesky_damping_multiplier', 10.0
                    )),
                    linear_solve_dtype=torch.float64,
                )
                if use_full_gradient_anchor_curvature:
                    aux_names = (
                        parameter_groups['shared']
                        + parameter_groups['aux_critic_head']
                    )
                    flat_full_gradient = compute_flat_full_buffer_aux_gradient(
                        actor_critic,
                        all_aux_obs,
                        all_aux_targets,
                        aux_names,
                        aux_minibatch_size,
                        device,
                    )
                    anchor_size = int(getattr(
                        algo_config, 'curvature_anchor_size', 256
                    ))
                    diagnostic_size = int(getattr(
                        algo_config, 'aux_diagnostic_size', 2048
                    ))
                    needed = max(
                        anchor_size, fisher_batch_size, diagnostic_size
                    )
                    if all_aux_obs.shape[0] < needed:
                        raise RuntimeError(
                            f'auxiliary buffer has {all_aux_obs.shape[0]} rows, '
                            f'needs {needed}'
                        )
                    anchor_observations = decode_procgen_observations(
                        all_aux_obs[permutation[:anchor_size]], device
                    )
                    fisher_observations = decode_procgen_observations(
                        all_aux_obs[permutation[:fisher_batch_size]], device
                    )
                    diagnostic_indices = permutation[-diagnostic_size:]
                    diagnostic_observations = decode_procgen_observations(
                        all_aux_obs[diagnostic_indices], device
                    )
                    diagnostic_targets = all_aux_targets[
                        diagnostic_indices
                    ].to(device)
                    last_aux_info = run_full_gradient_anchor_ggn_step(
                        actor_critic,
                        flat_full_gradient,
                        anchor_observations,
                        diagnostic_observations,
                        diagnostic_targets,
                        fisher_observations,
                        **common_step_kwargs,
                    )
                    true_head_info = fit_true_value_head_on_full_buffer(
                        actor_critic,
                        all_aux_obs,
                        all_aux_targets,
                        ggn_true_head_optimizer,
                        epochs=aux_epochs,
                        feature_batch_size=aux_minibatch_size,
                        head_batch_size=aux_minibatch_size,
                        device=device,
                    )
                    last_aux_info.update(true_head_info)
                    last_aux_info['full_gradient_anchor_mode'] = 1.0
                    last_aux_info['same_batch_true_head_mode'] = 0.0
                else:
                    aux_batch_size = int(getattr(
                        algo_config, 'aux_batch_size', 128
                    ))
                    needed = aux_batch_size + fisher_batch_size
                    if all_aux_obs.shape[0] < needed:
                        raise RuntimeError(
                            f'auxiliary buffer has {all_aux_obs.shape[0]} rows, '
                            f'needs {needed}'
                        )
                    aux_indices = permutation[:aux_batch_size]
                    fisher_indices = permutation[
                        aux_batch_size:aux_batch_size + fisher_batch_size
                    ]
                    aux_observations = decode_procgen_observations(
                        all_aux_obs[aux_indices], device
                    )
                    aux_targets = all_aux_targets[aux_indices].to(device)
                    fisher_observations = decode_procgen_observations(
                        all_aux_obs[fisher_indices], device
                    )
                    last_aux_info = run_auxiliary_critic_ggn_step(
                        actor_critic,
                        aux_observations,
                        aux_targets,
                        fisher_observations,
                        **common_step_kwargs,
                    )
                    last_aux_info['aux_batch_rows'] = float(aux_batch_size)
                    last_aux_info['full_gradient_anchor_mode'] = 0.0
                    last_aux_info['same_batch_true_head_mode'] = 1.0
                buffer_after = evaluate_auxiliary_buffer(
                    actor_critic,
                    all_aux_obs,
                    all_aux_targets,
                    aux_minibatch_size,
                    device,
                    reference_logits=reference_logits,
                    include_aux_head=use_full_gradient_anchor_curvature,
                )
                for key, value in buffer_before.items():
                    last_aux_info[f'{key}_before'] = value
                for key, value in buffer_after.items():
                    last_aux_info[f'{key}_after'] = value
                last_aux_info['buffer_rows'] = float(all_aux_obs.shape[0])
                last_aux_info['official_aux_epochs'] = float(
                    aux_epochs if use_full_gradient_anchor_curvature else 0
                )
                last_aux_info['true_head_aux_steps'] = float(
                    last_aux_info.get('true_head_aux_steps', 0.0)
                )
                last_aux_info['auxiliary_mode_official_adam'] = 0.0
                last_aux_info['auxiliary_mode_anchor_ggn'] = float(
                    use_full_gradient_anchor_curvature
                )
                last_aux_info['auxiliary_mode_same_batch_ggn'] = float(
                    not use_full_gradient_anchor_curvature
                )
            actor_critic.train(was_training)
            last_aux_info['cycle_policy_update'] = float(epoch + 1)
            trace_payload = {
                f'aux/{key}': value for key, value in last_aux_info.items()
            }
            trace_line = json.dumps(
                trace_payload, sort_keys=True, allow_nan=True
            )
            print(f'[AUX_TRACE] {trace_line}', flush=True)
            if log_dir is not None:
                with open(
                    os.path.join(log_dir, 'aux_trace.jsonl'),
                    'a',
                    encoding='utf-8',
                ) as trace_file:
                    trace_file.write(trace_line + '\n')
            aux_buffer_observations.clear()
            aux_buffer_raw_targets.clear()

        adaptive_lr_updates = 0
        adaptive_lr_changes = 0
        scheduler_kl_last = float('nan')
        if algo_config.use_kl_adaptive_lr:
            # Match the current Procgen random-RAT control: make one LR
            # decision after all four epochs, measured against the fixed
            # rollout behavior policy.  The new LR affects the next rollout.
            scheduler_kl_last = float(pi_info['kl'])
            lr_before, lr_after = adapt_learning_rate(scheduler_kl_last)
            adaptive_lr_updates = 1
            adaptive_lr_changes = int(lr_before != lr_after)

        pi_info['adaptive_scheduler_kl'] = scheduler_kl_last
        pi_info['adaptive_lr_updates'] = adaptive_lr_updates
        pi_info['adaptive_lr_changes'] = adaptive_lr_changes
        pi_info['curr_lr'] = actor_optimizer.param_groups[0]['lr']

        tepochs.set_postfix(loss_pi=mb_loss_pi.item(), loss_v=mb_loss_v.item(), entropy=pi_info['ent'], kl=pi_info['kl'], cf=pi_info['cf'], lr=pi_info['curr_lr'])

        # clean GPU cache
        torch.cuda.empty_cache()

        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(per_epoch_timesteps / (tnow - tstart))

        if logger.get_dir() is not None and (epoch+1) % log_config.log_interval == 0:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            logger.logkv("misc/serial_timesteps", (epoch+1)*per_epoch_timesteps)
            logger.logkv("misc/nupdates", epoch)
            logger.logkv("misc/total_timesteps", (epoch+1)*per_epoch_timesteps*world_size)
            logger.logkv("fps", fps)
            logger.logkv("loss_pi", mb_loss_pi.item())
            logger.logkv("loss_v", mb_loss_v.item())
            logger.logkv("ret_max", ret.max().item())
            logger.logkv("ret_min", ret.min().item())
            logger.logkv("ret_avg", ret.mean().item())
            logger.logkv("ret_med", ret.median().item())
            logger.logkv("ret_var", ret.var().item())
            logger.logkv("action_max", act.max().item())
            logger.logkv("action_min", act.min().item())
            logger.logkv("adv_max", adv.max().item())
            logger.logkv("adv_min", adv.min().item())
            logger.logkv("adv_avg", adv.mean().item())
            logger.logkv("adv_med", adv.median().item())
            logger.logkv("adv_var", adv.var().item())
            logger.logkv("entropy", pi_info['ent'])
            logger.logkv("curr_lr", pi_info['curr_lr'])
            logger.logkv("kl", pi_info['kl'])
            if algo in {'true', 'empirical'}:
                logger.logkv("ent_kl", pi_info['ent_kl'])
                logger.logkv("kl_grad_norm", pi_info['kl_grad_norm'])
            logger.logkv("grad_norm", pi_info['grad_norm'])
            logger.logkv("lr", actor_optimizer.param_groups[0]['lr'])
            logger.logkv("clipfrac", pi_info['cf'])
            logger.logkv("ratio_max", pi_info['ratio_max'])
            logger.logkv("ratio_min", pi_info['ratio_min'])
            if algo in {'adv'}:
                logger.logkv("actor_rows", pi_info['actor_rows'])
                logger.logkv("actor_solve_residual", pi_info['actor_solve_residual'])
                logger.logkv("advantage_direction_l2", pi_info['advantage_direction_l2'])
                logger.logkv("actor_direction_l2", pi_info['actor_direction_l2'])
                logger.logkv("actor_entropy_coefficient", pi_info['actor_entropy_coefficient'])
                logger.logkv("entropy_gradient_l2", pi_info['entropy_gradient_l2'])
                logger.logkv("entropy_direction_l2", pi_info['entropy_direction_l2'])
                logger.logkv("entropy_inverse_solve_residual", pi_info['entropy_inverse_solve_residual'])
                logger.logkv("actor_policy_fisher_quadratic", pi_info['actor_policy_fisher_quadratic'])
                logger.logkv("actor_policy_fisher_clip_scale", pi_info['actor_policy_fisher_clip_scale'])
                logger.logkv("actor_policy_target_kl", pi_info['actor_policy_target_kl'])
                logger.logkv("actor_policy_step_target_kl", pi_info['actor_policy_step_target_kl'])
                logger.logkv("actor_policy_predicted_kl_pre_clip", pi_info['actor_policy_predicted_kl_pre_clip'])
                logger.logkv("actor_policy_predicted_kl_post_clip", pi_info['actor_policy_predicted_kl_post_clip'])
                logger.logkv("actor_grad_norm", pi_info['actor_grad_norm'])
                logger.logkv("actor_grad_clip_scale", pi_info['actor_grad_clip_scale'])
                logger.logkv("critic_grad_norm", pi_info['critic_grad_norm'])
                logger.logkv("critic_grad_clip_scale", pi_info['critic_grad_clip_scale'])
                logger.logkv("actor_critic_direction_max_abs", pi_info['actor_critic_direction_max_abs'])
                logger.logkv("critic_forbidden_grad_max_abs", pi_info['critic_forbidden_grad_max_abs'])
                logger.logkv("separate_policy_critic_steps", pi_info['separate_policy_critic_steps'])
                logger.logkv("critic_head_only", pi_info['critic_head_only'])
                logger.logkv("policy_critic_mse", pi_info['policy_critic_mse'])
                logger.logkv("variant_use_phasic_training", float(use_phasic_training))
                logger.logkv("variant_use_official_ppg_auxiliary", float(use_official_ppg_auxiliary))
                logger.logkv("variant_use_critic_ggn_auxiliary", float(use_critic_ggn_auxiliary))
                logger.logkv("variant_use_actor_fisher_aux_clip", float(use_actor_fisher_aux_clip))
                logger.logkv("variant_use_actor_entropy_natural_gradient", float(use_actor_entropy_natural_gradient))
                logger.logkv("variant_use_actor_policy_fisher_clip", float(use_actor_policy_fisher_clip))
                if last_aux_info is not None:
                    for key, value in last_aux_info.items():
                        logger.logkv(f"aux/{key}", value)
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('misc/time_elapsed', tnow - tfirststart)

            logger.dumpkvs()

        # Log changes from update
        # writer.add_scalar('train/rewards', rew.sum(), epoch)
        if writer is not None:
            writer.add_scalar('train/kl', pi_info['kl'], epoch)
            if algo in {'true', 'empirical'}:
                writer.add_scalar("ent_kl", pi_info['ent_kl'], epoch)
                writer.add_scalar("kl_grad_norm", pi_info['kl_grad_norm'], epoch)
            writer.add_scalar("grad_norm", pi_info['grad_norm'], epoch)
            writer.add_scalar('train/clipfrac', pi_info['cf'], epoch)
            writer.add_scalar('train/entropy', pi_info['ent'], epoch)
            writer.add_scalar('train/curr_lr', pi_info['curr_lr'], epoch)
            writer.add_scalar('train/ratio_max', pi_info['ratio_max'], epoch)
            writer.add_scalar('train/ratio_min', pi_info['ratio_min'], epoch)
            if algo in {'adv'}:
                writer.add_scalar('train/actor_solve_residual', pi_info['actor_solve_residual'], epoch)
                writer.add_scalar('train/advantage_direction_l2', pi_info['advantage_direction_l2'], epoch)
                writer.add_scalar('train/actor_direction_l2', pi_info['actor_direction_l2'], epoch)
                writer.add_scalar('train/actor_entropy_coefficient', pi_info['actor_entropy_coefficient'], epoch)
                writer.add_scalar('train/entropy_gradient_l2', pi_info['entropy_gradient_l2'], epoch)
                writer.add_scalar('train/entropy_direction_l2', pi_info['entropy_direction_l2'], epoch)
                writer.add_scalar('train/entropy_inverse_solve_residual', pi_info['entropy_inverse_solve_residual'], epoch)
                writer.add_scalar('train/actor_policy_fisher_quadratic', pi_info['actor_policy_fisher_quadratic'], epoch)
                writer.add_scalar('train/actor_policy_fisher_clip_scale', pi_info['actor_policy_fisher_clip_scale'], epoch)
                writer.add_scalar('train/actor_policy_predicted_kl_pre_clip', pi_info['actor_policy_predicted_kl_pre_clip'], epoch)
                writer.add_scalar('train/actor_policy_predicted_kl_post_clip', pi_info['actor_policy_predicted_kl_post_clip'], epoch)
                writer.add_scalar('train/actor_grad_norm', pi_info['actor_grad_norm'], epoch)
                writer.add_scalar('train/actor_grad_clip_scale', pi_info['actor_grad_clip_scale'], epoch)
                writer.add_scalar('train/critic_grad_norm', pi_info['critic_grad_norm'], epoch)
                writer.add_scalar('train/critic_grad_clip_scale', pi_info['critic_grad_clip_scale'], epoch)
                writer.add_scalar('train/actor_critic_direction_max_abs', pi_info['actor_critic_direction_max_abs'], epoch)
                writer.add_scalar('train/critic_forbidden_grad_max_abs', pi_info['critic_forbidden_grad_max_abs'], epoch)
                writer.add_scalar('train/critic_head_only', pi_info['critic_head_only'], epoch)
                writer.add_scalar('train/policy_critic_mse', pi_info['policy_critic_mse'], epoch)
                if last_aux_info is not None:
                    for key, value in last_aux_info.items():
                        writer.add_scalar(f'aux/{key}', value, epoch)
            writer.add_scalar('train/loss_pi', mb_loss_pi, epoch)
            writer.add_scalar('train/loss_v', mb_loss_v, epoch)
            writer.add_scalar('train/lr', actor_optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar("train/ret_max", ret.max().item(), epoch)
            writer.add_scalar("train/ret_min", ret.min().item(), epoch)
            writer.add_scalar("train/ret_avg", ret.mean().item(), epoch)
            writer.add_scalar("train/ret_med", ret.median().item(), epoch)
            writer.add_scalar("train/ret_var", ret.var().item(), epoch)
            writer.add_scalar("train/act_max", act.max().item(), epoch)
            writer.add_scalar("train/act_min", act.min().item(), epoch)
            writer.add_scalar("train/adv_max", adv.max().item(), epoch)
            writer.add_scalar("train/adv_min", adv.min().item(), epoch)
            writer.add_scalar("train/adv_avg", adv.mean().item(), epoch)
            writer.add_scalar("train/adv_med", adv.median().item(), epoch)
            writer.add_scalar("train/adv_var", adv.var().item(), epoch)
            writer.add_scalar('train/eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]), epoch)
            writer.add_scalar('train/eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]), epoch)
            writer.add_scalar('misc/time_elapsed', tnow - tfirststart, epoch)
            writer.add_scalar("misc/serial_timesteps", (epoch+1)*per_epoch_timesteps, epoch)
            writer.add_scalar("misc/nupdates", epoch)
            writer.add_scalar("misc/total_timesteps", (epoch+1)*per_epoch_timesteps*world_size, epoch)

    if log_dir is not None:
        # save checkpoints
        torch.save({'model_state_dict': actor_critic.state_dict(), }, f'{log_dir}/model.ckpt')

def train_fn(rank, world_size, algo, seed, algo_config, env_config, nets_config, log_config, device=-1):
    # Serialize data into file:
    time_now = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Random seed
    if seed is None:
        seed = np.random.randint(1e6) + 10000 * rank # different seeds for each process
    set_seed(seed, torch_deterministic=True)

    env_name = env_config.env_name
    num_envs = env_config.num_envs

    if env_name in ['cartpole', 'acrobot', 'mountaincar', 'lunarlander', 'carracing', 'hopper', 'invertedpendulum', 'inverteddoublependulum',
                    'halfcheetah', 'walker2d', 'humanoid', 'humanoidstandup', 'reacher', 'swimmer', 'ant']:
        timesteps_per_proc = env_config.timesteps_per_proc

    elif 'atari' not in env_name:
        env_name, distribution_mode, start_level, num_levels = env_name.split('-')
        start_level, num_levels = int(start_level), int(num_levels)

        if distribution_mode == 'easy':
            timesteps_per_proc = env_config.timesteps_per_proc_easy
        elif distribution_mode == 'hard':
            timesteps_per_proc = env_config.timesteps_per_proc_hard

    if rank==0:
        if env_name in {'cartpole', 'acrobot', 'mountaincar', 'lunarlander', 'hopper', 'invertedpendulum', 'inverteddoublependulum',
                        'halfcheetah', 'walker2d', 'humanoid', 'humanoidstandup', 'reacher', 'swimmer', 'ant'}:
            log_dir = f"logs/shared.{algo}.{nets_config.type}.a{nets_config.hidden_size}x{nets_config.num_layers}x{nets_config.dropout}e{algo_config.epochs}x{algo_config.minibatches}.clip_grad_{algo_config.max_grad_norm}.{algo_config.sigma_type}.damping_{algo_config.cg_damping}.lr_{algo_config.lr}/{env_name}.{time_now}_{seed}"
        else:
            variant_name = str(getattr(algo_config, 'variant_name', 'unnamed'))
            log_dir = (
                f"logs/phasic_ef_ggn.{variant_name}.solve_"
                f"{getattr(algo_config, 'linear_solve_dtype', 'float64')}."
                f"{nets_config.type}{'_bn' if nets_config.with_bn else ''}."
                f"dropout_{nets_config.dropout}.actor_damping_{algo_config.cg_damping}."
                f"actor_lr_{algo_config.lr}.aux_lr_"
                f"{getattr(algo_config, 'aux_learning_rate', 0.0)}/"
                f"{env_config.env_name}.{time_now}_{seed}"
            )

        format_strs = ['csv', 'stdout'] 
        logger.configure(dir=log_dir, format_strs=format_strs)
        writer = SummaryWriter(log_dir=log_dir)
    else:
        log_dir = None
        writer = None
    
    if rank==0:
        logger.info("creating environment")

    if 'atari' in env_name:
        from stable_baselines3.common.env_util import make_atari_env
        from stable_baselines3.common.vec_env import VecFrameStack
        env_name = env_name.split('.')[1]
        # use atari env with terminal on life loss for better value bootstrap
        # cannot use VecMonitor then: episodic return and length will be incorrect
        # venv = make_atari_env(env_name, n_envs=num_envs, monitor_dir=log_dir, wrapper_kwargs={'terminal_on_life_loss': True})
        venv = make_atari_env(env_name, n_envs=num_envs)
        venv = VecFrameStack(venv, n_stack=3) # set stack number to 3 (compatible with Procgen number of channels)
        timesteps_per_proc = env_config.timesteps_per_proc # 10M for atari envs # HARD coded
        distribution_mode = 'atari'

    elif env_name in ['cartpole', 'acrobot', 'mountaincar', 'lunarlander', 'carracing', 'invertedpendulum', 'inverteddoublependulum',
                      'hopper', 'halfcheetah', 'walker2d', 'humanoid', 'humanoidstandup', 'reacher', 'swimmer', 'ant']:
        from stable_baselines3.common.env_util import make_vec_env
        tag_name = {'cartpole': 'CartPole-v1', 'acrobot': 'Acrobot-v1', 'mountaincar': 'MountainCar-v0', 
                    'lunarlander': 'LunarLander-v2', 'carracing': 'CarRacing-v2', 'invertedpendulum': 'InvertedPendulum-v4',
                    'inverteddoublependulum': 'InvertedDoublePendulum-v4',
                    'hopper': 'Hopper-v4', 'halfcheetah': 'HalfCheetah-v4', 'walker2d': 'Walker2d-v4', 
                    'humanoid': 'Humanoid-v4', 'humanoidstandup': 'HumanoidStandup-v4', 'reacher': 'Reacher-v4', 
                    'swimmer': 'Swimmer-v3', 'ant': 'Ant-v4'}
        
        venv = make_vec_env(tag_name[env_name], n_envs=num_envs, env_kwargs={'continuous': False} if env_name == 'carracing' else {})

    else:
        venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode, rand_seed=seed)
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=log_dir)

    if device == -1:
        if torch.cuda.is_available(): 
            device_type = "cuda"
        else:
            device_type = "cpu"
        
        device = torch.device(device_type) # Select best available device
    else:
        assert device >= 0
        device = f"cuda:{device}"

    obs_space = venv.observation_space

    # Create actor-critic module
    if nets_config.type == 'resnet':
        # kwargs = {'with_bn': nets_config.with_bn, 'depths': [16, 32, 32], 'device': device}
        kwargs = {'with_bn': nets_config.with_bn, 'depths': [8, 16], 'device': device}
        fn_neural_nets, preprocess = build_resnet(obs_space.shape[0], nets_config.hidden_size, **kwargs)
        # now the obs_space becomes channel x height x width
        obs_shape = (obs_space.shape[2], obs_space.shape[0], obs_space.shape[1])

    elif nets_config.type == 'cnn':
        img_size = obs_space.shape[1]
        kwargs = {'with_bn': nets_config.with_bn, 'device': device}
        fn_neural_nets, preprocess = build_cnn(img_size, nets_config.hidden_size, **kwargs)
        # now the obs_space becomes channel x height x width
        obs_shape = (obs_space.shape[2], obs_space.shape[0], obs_space.shape[1])

    elif nets_config.type == 'mlp':
        kwargs = {'device': device, 'hidden_size': nets_config.hidden_size, 
                  'num_layers': nets_config.num_layers, 'p_dropout': nets_config.dropout}
        fn_neural_nets, preprocess = build_mlp(obs_space, **kwargs)
        obs_shape = obs_space.shape

    else: 
        raise NotImplementedError

    act_num, act_dim = None, None
    try:
        act_num = venv.action_space.n
    except AttributeError:
        act_dim = venv.action_space.shape[0]

    actor_critic = SharedActorCritic(fn_neural_nets, obs_shape, nets_config=nets_config, n_actions=act_num, 
                            dim_actions=act_dim, with_popart=algo_config.with_popart, 
                            sigma_type=algo_config.sigma_type, device=device).to(device)

    venv = VecNormalize(venv=venv, norm_ret=env_config.norm_ret, obs_preprocess=preprocess) # img transform and reward normalization

    if rank==0:
        logger.info(f'Running on device: {device}')
        logger.info(f"training...")

        # Count variables
        var_counts = count_vars(actor_critic)
        logger.log(f'\nNumber of parameters: {var_counts}\n')

        # yaml.dump(args, open( f"{log_dir}/args.yaml", 'w' ))
        config = {'algo_config': algo_config.__dict__, 
                'env_config': env_config.__dict__, 
                'nets_config': nets_config.__dict__, 
                'log_config': log_config.__dict__}

        yaml.dump(config, open( f"{log_dir}/config.yaml", 'w' ))

    learn(world_size, algo, actor_critic, writer, venv, device,
          total_timesteps=timesteps_per_proc, nsteps=env_config.nsteps, 
          algo_config=algo_config, log_config=log_config, log_dir=log_dir)

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--config', type=str, default='adv_mlp_shared.yaml')
    parser.add_argument('--device', type=int, default=-1) # -1: use any available device
    parser.add_argument('--env_name', type=str, default=None) # -1: use any available device
    parser.add_argument('--n_proc', type=int, default=1) # distributed training: number of processes
    parser.add_argument('--port_num', type=int, default=29500) # distributed training: number of processes
    parser.add_argument('--dropout', type=float, default=None) # distributed training: number of processes
    parser.add_argument('--hidden_size', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--num_layers', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--norm_obj', type=str, default=None) # distributed training: number of processes
    parser.add_argument('--optimizer', type=str, default=None) # distributed training: number of processes
    parser.add_argument('--sigma_type', type=str, default=None, choices=['vector', 'mu_shared', 'separate', 'linear']) 
    parser.add_argument('--cg_damping', type=float, default=None) # distributed training: number of processes
    parser.add_argument('--epochs', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--lr', type=float, default=None) # distributed training: number of processes
    parser.add_argument('--seed', type=int, default=None) 

    args = parser.parse_args()

    with open(f'configs/{args.config}') as fin:
        config = yaml.safe_load(fin)

    algo = config['algo']
    algo_config = types.SimpleNamespace(**config['algo_config'])
    env_config = types.SimpleNamespace(**config['env_config'])
    nets_config = types.SimpleNamespace(**config['nets_config'])
    log_config = types.SimpleNamespace(**config['log_config'])

    if args.env_name is not None:
        env_config.env_name = args.env_name

    if args.hidden_size is not None:
        nets_config.hidden_size = args.hidden_size
    if args.num_layers is not None:
        nets_config.num_layers = args.num_layers
    if args.dropout is not None:
        nets_config.dropout = args.dropout

    if args.optimizer is not None:
        algo_config.optimizer = args.optimizer

    if args.sigma_type is not None:
        algo_config.sigma_type = args.sigma_type

    if args.norm_obj is not None:
        algo_config.norm_obj = args.norm_obj

    if args.cg_damping is not None:
        algo_config.cg_damping = args.cg_damping

    if args.epochs is not None:
        algo_config.epochs = args.epochs

    if args.lr is not None:
        algo_config.lr = args.lr

    if args.n_proc > 1:
        # multiple nodes
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args.port_num)

        mp.spawn(train_fn, args=(args.n_proc, algo, args.seed, algo_config, env_config, nets_config, log_config, args.device),
                        nprocs=args.n_proc, # INFO: for TPU, either 1 or the maximum number of TPU chips
                        join=True)

    else:
        train_fn(0, args.n_proc, algo, args.seed, algo_config, env_config, nets_config, log_config, args.device)

if __name__ == '__main__':
    main()
