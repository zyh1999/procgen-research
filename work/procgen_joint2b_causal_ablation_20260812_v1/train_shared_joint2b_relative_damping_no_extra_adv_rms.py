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
from collections import deque
import utils.logger as logger

from torch.nn import functional as F
from torch.func import vmap, grad, functional_call

# pytorch distributed training
import torch.multiprocessing as mp

from utils.runners import Runner
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from utils.utils import build_cnn, build_resnet, build_mlp
from utils.utils import SharedActorCritic, count_vars, safemean, set_seed, set_grads_from_flat
from vec_env import ( VecExtractDictObs, VecMonitor, VecNormalize)


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

    metric_trace_file = None
    metric_trace_path = os.environ.get('PROCGEN_METRIC_TRACE_PATH')
    if metric_trace_path:
        metric_trace_dir = os.path.dirname(metric_trace_path)
        if metric_trace_dir:
            os.makedirs(metric_trace_dir, exist_ok=True)
        metric_trace_file = open(
            metric_trace_path, 'a', buffering=1024 * 1024
        )
    minibatch_global_step = 0

    # Instantiate the runner object
    runner = Runner(env=venv, model=actor_critic, nsteps=nsteps, gamma=gamma, lam=lam, adv_type=algo_config.adv_type, device=device)
    epinfobuf = deque(maxlen=100)

    dict_params = {k: v.detach() for k, v in actor_critic.named_parameters() if v.requires_grad}
    dict_buffers = {k: v.detach() for k, v in actor_critic.named_buffers() if v.requires_grad}
    trainable_params = [p for p in actor_critic.parameters() if p.requires_grad]
    param_names = list(dict_params.keys())
    param_numels = [dict_params[name].numel() for name in param_names]
    named_trainable_params = [
        (name, p) for name, p in actor_critic.named_parameters()
        if p.requires_grad
    ]
    if len(named_trainable_params) != len(trainable_params) or any(
        named_p is not flat_p
        for (_, named_p), flat_p in zip(
            named_trainable_params, trainable_params
        )
    ):
        raise RuntimeError(
            'named-parameter order does not match flattened gradient order'
        )
    critic_head_column_mask = torch.cat([
        torch.full(
            (numel,),
            name.startswith('last_v_layer.'),
            device=device,
            dtype=torch.bool,
        )
        for name, numel in zip(param_names, param_numels)
    ])
    critic_head_parameter_columns = int(
        critic_head_column_mask.sum().item()
    )
    total_parameter_columns = int(critic_head_column_mask.numel())
    if critic_head_parameter_columns <= 0:
        raise RuntimeError('value-head column mask selected no parameters')

    adaptive_kl_mode = str(getattr(
        algo_config, 'adaptive_kl_mode', 'procgen_rollout'
    ))
    if adaptive_kl_mode != 'procgen_rollout':
        raise ValueError(
            'Gaussian paired-RHS GGN control requires '
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

    optimizer_momentum = float(getattr(
        algo_config, 'optimizer_momentum', 0.0
    ))
    is_kaczmarz = bool(getattr(algo_config, 'is_kaczmarz', False))
    if optimizer_momentum not in (0.0, 0.9):
        raise ValueError(
            'this controlled branch permits optimizer_momentum in {0.0, 0.9}'
        )
    if is_kaczmarz and optimizer_momentum == 0.0:
        raise ValueError(
            'Kaczmarz history correction requires nonzero classic momentum'
        )

    if algo_config.optimizer == 'adam':
        ac_optimizer = Adam(actor_critic.parameters(), lr=algo_config.lr, weight_decay=algo_config.weight_decay)
    elif algo_config.optimizer == 'sgd': 
        # PyTorch's undamped SGD buffer is the requested classic recurrence:
        #   m_t = momentum * m_{t-1} + d_t
        # (there is no extra ``1-momentum`` multiplier on d_t).
        ac_optimizer = SGD(
            actor_critic.parameters(),
            lr=algo_config.lr,
            momentum=optimizer_momentum,
            dampening=0.0,
            nesterov=False,
        )
    elif algo_config.optimizer == 'rmsprop': 
        ac_optimizer = RMSprop(actor_critic.parameters(), lr=algo_config.lr, 
                               centered=True, weight_decay=algo_config.weight_decay)
    elif algo_config.optimizer == 'kfac':
        from kfac.kfac import KFACOptimizer
        ac_optimizer = KFACOptimizer(actor_critic, lr=algo_config.lr,
                                     weight_decay=algo_config.weight_decay)
    elif algo_config.optimizer == 'ekfac':
        from kfac.ekfac import EKFACOptimizer
        ac_optimizer = EKFACOptimizer(actor_critic, lr=algo_config.lr,
                                     weight_decay=algo_config.weight_decay)
    else:
        raise NotImplementedError

    if hasattr(algo_config, 'lr_decay') and algo_config.lr_decay == 'cosine':
        lr_scheduler = CosineAnnealingLR(ac_optimizer, T_max=epochs*algo_config.epochs*algo_config.minibatches, eta_min=0.001)
    else:
        lr_scheduler = None

    def adapt_learning_rate(measured_kl):
        before = float(ac_optimizer.param_groups[0]['lr'])
        after = adaptive_lr_update_value(
            before,
            measured_kl,
            adaptive_kl_lower,
            adaptive_kl_upper,
            adaptive_lr_min,
            adaptive_lr_max,
        )
        ac_optimizer.param_groups[0]['lr'] = after
        return before, after

    # Start total timer
    tfirststart = time.perf_counter()

    def PPO_Update(_obs, _act, _adv, _ret, _outputs_old):
        _vals, _outputs = actor_critic(_obs) # obtain new val estimate

        if actor_critic.is_discrete:
            _logp_full = F.log_softmax(_outputs, dim=-1)
            _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
            _logp = torch.gather(_logp_full, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _logp_old = torch.gather(_logp_full_old, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _llr = _logp - _logp_old
            _ratio = torch.exp(_llr)
            _p_log_p = torch.exp(_logp_full) * _logp_full
            _entropy = - _p_log_p.sum(-1).mean()
            _kl = (torch.exp(_logp_full_old) * (_logp_full_old - _logp_full)).sum(dim=-1).mean()

        else:
            _mu, _logstd = _outputs.chunk(2, dim=-1)
            _dist = torch.distributions.Normal(_mu, torch.exp(_logstd))
            _logp = _dist.log_prob(_act).sum(dim=-1)

            _mu_old, _logstd_old = _outputs_old.chunk(2, dim=-1)
            _dist_old = torch.distributions.Normal(_mu_old, torch.exp(_logstd_old))
            _logp_old = _dist_old.log_prob(_act).sum(dim=-1)

            _ratio = torch.exp(_logp - _logp_old)
            _entropy = _dist.entropy().sum(dim=-1).mean()
            _kl = (_logstd - _logstd_old + 0.5 * ( torch.exp(_logstd_old).pow(2) + (_mu_old - _mu).pow(2) ) / torch.exp(_logstd).pow(2) - 0.5).sum(dim=-1).mean()

        # advantage normalization
        if algo_config.norm_obj == 'adv':
            _adv_mean, _adv_std = _adv.mean(), _adv.std()
            _adv = (_adv - _adv_mean) / (_adv_std + 1e-8)
        else: 
            raise NotImplementedError

        _clip_adv = torch.clamp(_ratio, 1-algo_config.cliprange, 1+algo_config.cliprange) * _adv
        _losses_pi = torch.max(- _ratio * _adv, - _clip_adv)
        _loss_pi = _losses_pi.mean()

        # value loss
        _loss_v = F.mse_loss(_vals, _ret)

        # total loss
        _loss = _loss_pi - algo_config.ent_coef * _entropy + algo_config.vf_coef * _loss_v

        _loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), algo_config.max_grad_norm)
        ac_optimizer.step()

        # Useful extra info
        with torch.no_grad():
            approx_kl = _kl.item()
            ent = _entropy.item()
            clipped = _ratio.gt(1+algo_config.cliprange) | _ratio.lt(1-algo_config.cliprange)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac, curr_lr=ac_optimizer.param_groups[0]['lr'],
                           grad_norm=grad_norm.item(), ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())

        return _loss, _loss_pi, _loss_v, pi_info

    def Advantage_Update(_obs, _act, _adv, _ret, _outputs_old):
        """Gaussian paired-noise shared actor/regression-GGN joint solve.

        For one standard-normal draw per sample, the critic feature and RHS are
        paired as

        ``H_C = sqrt(lambda_C) diag(xi) J_V`` and
        ``b_C = c_C / sqrt(lambda_C) diag(xi) e_C``.

        Thus the realized critic curvature and RHS contain the same ``xi**2``
        weights, and their expectations are exactly the clean regression GGN
        and MSE gradient.  The complete 2B-by-2B dual retains all four blocks,
        actor rows alone carry rollout ratios, critic rows have unit ratio, and
        the update is reconstructed directly as ``H.T @ alpha``.
        """
        _vals, _outputs = actor_critic(_obs)

        if not actor_critic.is_discrete:
            raise NotImplementedError("This Procgen joint-GGN trainer expects a discrete policy")

        _logp_full = F.log_softmax(_outputs, dim=-1)
        _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
        _llr = torch.gather(
            _logp_full - _logp_full_old,
            dim=-1,
            index=_act.unsqueeze(-1),
        ).squeeze(1)
        _ratio = torch.exp(_llr)
        _p_log_p = torch.exp(_logp_full) * _logp_full
        _entropy = -_p_log_p.sum(-1).mean()
        categorical_fisher_trace = (
            1.0 - torch.exp(_logp_full).square().sum(dim=-1)
        ).mean()

        def compute_pi_logp(params, buffers, batch_obs, batch_act):
            batch_obs = batch_obs.unsqueeze(0)
            batch_act = batch_act.unsqueeze(0)
            _, batch_outs = functional_call(
                actor_critic, (params, buffers), (batch_obs,)
            )
            batch_logp_full = F.log_softmax(batch_outs, dim=-1)
            return torch.gather(
                batch_logp_full,
                dim=-1,
                index=batch_act.unsqueeze(-1),
            ).reshape(-1)[0]

        def compute_value(params, buffers, batch_obs):
            batch_obs = batch_obs.unsqueeze(0)
            batch_vals, _ = functional_call(
                actor_critic, (params, buffers), (batch_obs,)
            )
            return batch_vals.reshape(-1)[0]

        # Controlled combined intervention: retain rollout/PopArt
        # normalization and minibatch centering, but optionally leave the
        # centered advantage at its natural RMS. Relative kernel damping is
        # handled independently below.
        _adv = _adv - _adv.mean()
        if algo_config.clamp_ratio:
            _ratio = torch.clamp(
                _ratio, algo_config.min_ratio, algo_config.max_ratio
            )

        if algo_config.norm_obj == 'adv':
            _rms_sqrt = torch.sqrt(_adv.pow(2).mean()).detach()
        elif algo_config.norm_obj == 'obj':
            _rms_sqrt = torch.sqrt((_ratio * _adv).pow(2).mean()).detach()
        elif algo_config.norm_obj == 'ratio':
            _rms_sqrt = (
                _ratio.mean().detach()
                * torch.sqrt(_adv.pow(2).mean()).detach()
            )
        else:
            raise NotImplementedError
        normalize_advantage_rms = bool(getattr(
            algo_config, 'normalize_advantage_rms', True
        ))
        if normalize_advantage_rms:
            _adv = _adv / (_rms_sqrt + 1e-8)

        ft_compute_pi_grad = vmap(
            grad(compute_pi_logp),
            in_dims=(None, None, 0, 0),
            randomness='different',
        )
        ft_pi_grads = ft_compute_pi_grad(
            dict_params, dict_buffers, _obs, _act
        )

        num_sa = _obs.shape[0]
        H_pi = torch.cat(
            [v.contiguous().view(num_sa, -1) for v in ft_pi_grads.values()],
            dim=-1,
        )
        del ft_pi_grads

        ft_compute_v_grad = vmap(
            grad(compute_value),
            in_dims=(None, None, 0),
            randomness='different',
        )
        ft_v_grads = ft_compute_v_grad(dict_params, dict_buffers, _obs)
        J_v = torch.cat(
            [v.contiguous().view(num_sa, -1) for v in ft_v_grads.values()],
            dim=-1,
        )
        del ft_v_grads

        with torch.no_grad():
            critic_residual = (_ret - _vals).detach()

            # Controlled causal ablations.  The critic objective coefficient
            # and curvature coefficient remain mathematically independent.
            # For paired score modes, the same draw is used in the feature and
            # residual RHS:
            #   H_C = sqrt(lambda_C) diag(xi) J_C
            #   b_C = c_C / sqrt(lambda_C) diag(xi) e_C
            # Therefore E[H_C.T H_C] = lambda_C J_C.T J_C and
            # E[H_C.T b_C] = c_C J_C.T e_C for xi ~ N(0, 1).
            critic_curvature_coef = float(
                getattr(algo_config, 'joint_critic_curvature_coef', 1.0)
            )
            critic_objective_coef = float(
                getattr(algo_config, 'joint_critic_objective_coef', 1.0)
            )
            ablation_mode = str(getattr(
                algo_config, 'joint_ablation_mode', 'full_joint'
            ))
            if ablation_mode not in {'actor_only', 'curvature_only', 'full_joint'}:
                raise ValueError(
                    'joint_ablation_mode must be actor_only, curvature_only, '
                    'or full_joint'
                )
            if not math.isfinite(critic_curvature_coef):
                raise ValueError('joint_critic_curvature_coef must be finite')
            if ablation_mode == 'actor_only':
                critic_curvature_coef = 0.0
                critic_objective_coef = 0.0
            elif critic_curvature_coef <= 0.0:
                raise ValueError(
                    'joint_critic_curvature_coef must be positive when critic '
                    'rows are present'
                )
            if ablation_mode == 'curvature_only':
                critic_objective_coef = 0.0
            if not math.isfinite(critic_objective_coef):
                raise ValueError('joint_critic_objective_coef must be finite')
            critic_h_weight = math.sqrt(critic_curvature_coef)
            critic_rhs_weight = (
                critic_objective_coef / critic_h_weight
                if critic_h_weight > 0.0 else 0.0
            )

            configured_score_mode = str(getattr(
                algo_config, 'joint_critic_score_mode', ''
            ))
            configured_rhs_mode = str(getattr(
                algo_config, 'joint_critic_rhs_mode', ''
            ))
            if configured_score_mode not in {
                'clean', 'rademacher', 'gaussian_unit'
            }:
                raise ValueError(
                    'joint_critic_score_mode must be clean, rademacher, or '
                    'gaussian_unit'
                )
            if configured_rhs_mode != 'paired_score_residual':
                raise ValueError(
                    'causal ablation requires '
                    'joint_critic_rhs_mode=paired_score_residual'
                )
            critic_param_scope = str(getattr(
                algo_config, 'joint_critic_param_scope', 'all'
            ))
            if critic_param_scope not in {'all', 'head_only'}:
                raise ValueError(
                    'joint_critic_param_scope must be all or head_only'
                )
            critic_J = J_v
            if critic_param_scope == 'head_only':
                critic_J = J_v * critic_head_column_mask.to(J_v.dtype)
            critic_reconstruction_scope = str(getattr(
                algo_config, 'joint_critic_reconstruction_scope', 'all'
            ))
            if critic_reconstruction_scope not in {'all', 'head_only'}:
                raise ValueError(
                    'joint_critic_reconstruction_scope must be all or '
                    'head_only'
                )

            if configured_score_mode == 'clean':
                critic_score_noise = torch.ones(
                    num_sa, device=J_v.device, dtype=J_v.dtype
                )
            elif configured_score_mode == 'rademacher':
                critic_score_noise = torch.empty(
                    num_sa, device=J_v.device, dtype=J_v.dtype
                ).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            else:
                critic_score_noise = torch.randn(
                    num_sa, device=J_v.device, dtype=J_v.dtype
                )
            critic_H = (
                critic_h_weight
                * critic_score_noise.unsqueeze(1)
                * critic_J
            )
            critic_rhs = (
                critic_rhs_weight
                * critic_score_noise
                * critic_residual
            )

            # actor_only is the matched B-row control.  The other modes retain
            # a strict stacked 2B system and all sample-space cross blocks.
            critic_ratio = torch.ones_like(critic_residual)
            if ablation_mode == 'actor_only':
                joint_H = H_pi
                joint_rhs = _adv
                joint_ratio = _ratio.detach()
                critic_rows = 0
            else:
                joint_H = torch.cat([H_pi, critic_H], dim=0)
                joint_rhs = torch.cat([_adv, critic_rhs], dim=0)
                joint_ratio = torch.cat(
                    [_ratio.detach(), critic_ratio], dim=0
                )
                critic_rows = num_sa

            # PyTorch stores the momentum buffer in gradient sign.  We install
            # ``grad=-flat_dir`` below, so the actual ascent velocity after the
            # next optimizer step is
            #
            #   v_t = flat_dir - beta * buffer_{t-1}.
            #
            # If Kaczmarz correction is enabled, solve for ``flat_dir`` against
            # ``rhs + beta H buffer`` so that H v_t targets the clean RHS.
            previous_momentum_buffer = None
            previous_projection = torch.zeros_like(joint_rhs)
            if optimizer_momentum > 0.0:
                momentum_parts = []
                has_momentum_buffer = False
                for p in trainable_params:
                    buffer = ac_optimizer.state.get(p, {}).get(
                        'momentum_buffer', None
                    )
                    if buffer is None:
                        momentum_parts.append(torch.zeros_like(p).flatten())
                    else:
                        has_momentum_buffer = True
                        momentum_parts.append(buffer.detach().flatten())
                if has_momentum_buffer:
                    previous_momentum_buffer = torch.cat(momentum_parts, dim=0)
                    previous_projection = (
                        optimizer_momentum
                        * torch.mv(joint_H, previous_momentum_buffer)
                    )

            rhs_eff = joint_rhs
            if is_kaczmarz and previous_momentum_buffer is not None:
                rhs_eff = rhs_eff + previous_projection.to(rhs_eff.dtype)

            kernel_denom = float(num_sa)
            joint_K = torch.mm(joint_H, joint_H.t()) / kernel_denom
            weighted_K = joint_K * joint_ratio.to(joint_K.dtype).unsqueeze(0)
            kernel_diagonal = torch.diagonal(weighted_K)
            kernel_diag_min = kernel_diagonal.min()
            kernel_diag_median = kernel_diagonal.median()
            kernel_diag_max = kernel_diagonal.max()
            base_damping_value = torch.as_tensor(
                float(algo_config.cg_damping),
                device=kernel_diagonal.device,
                dtype=kernel_diagonal.dtype,
            )
            damping_to_median_floor = float(getattr(
                algo_config,
                'joint_damping_to_median_floor',
                0.0,
            ))
            if damping_to_median_floor < 0.0:
                raise ValueError(
                    'joint_damping_to_median_floor must be non-negative, got '
                    f'{damping_to_median_floor}'
                )
            # A fixed absolute damping becomes progressively negligible when
            # the learned joint sample-space kernel changes scale.  This
            # control preserves the requested 0.5 floor, while ensuring that
            # damping is at least a fixed fraction of the current median
            # weighted diagonal.  The matrix, RHS, 2B rows and cross blocks
            # are otherwise unchanged.
            damping_value = torch.maximum(
                base_damping_value,
                kernel_diag_median * damping_to_median_floor,
            )
            base_damping_to_median_diag = (
                base_damping_value / torch.clamp(
                    kernel_diag_median,
                    min=1e-30,
                )
            )
            damping_to_median_diag = damping_value / torch.clamp(
                kernel_diag_median,
                min=1e-30,
            )
            spectral_diagnostic_interval = int(getattr(
                algo_config,
                'spectral_diagnostic_interval',
                320,
            ))
            run_spectral_diagnostic = (
                spectral_diagnostic_interval > 0
                and minibatch_global_step % spectral_diagnostic_interval == 0
            )
            spectral_eigen_min = torch.zeros_like(kernel_diag_min)
            spectral_eigen_max = torch.zeros_like(kernel_diag_max)
            spectral_damped_condition = torch.zeros_like(kernel_diag_max)
            if run_spectral_diagnostic:
                # K D is similar to sqrt(D) K sqrt(D).  The symmetric
                # representative has the same eigenvalues and lets this
                # logging-only control compare the fixed damping against the
                # actual shared-network kernel scale.
                sqrt_ratio = torch.sqrt(joint_ratio.to(torch.float64))
                symmetric_metric = (
                    sqrt_ratio[:, None]
                    * joint_K.to(torch.float64)
                    * sqrt_ratio[None, :]
                )
                spectral_eigenvalues = torch.linalg.eigvalsh(
                    symmetric_metric
                )
                spectral_eigen_min = spectral_eigenvalues[0].to(
                    kernel_diag_min.dtype
                )
                spectral_eigen_max = spectral_eigenvalues[-1].to(
                    kernel_diag_max.dtype
                )
                spectral_damped_condition = (
                    spectral_eigen_max + damping_value
                ) / torch.clamp(
                    torch.clamp(spectral_eigen_min, min=0.0)
                    + damping_value,
                    min=1e-30,
                )
            eye = torch.eye(
                weighted_K.shape[0],
                device=device,
                dtype=weighted_K.dtype,
            )
            linear_solve_dtype_name = str(getattr(
                algo_config, 'linear_solve_dtype', 'float32'
            )).lower()
            if linear_solve_dtype_name not in {
                'float32', 'fp32', 'float64', 'fp64', 'double'
            }:
                raise ValueError(
                    'linear_solve_dtype must be float32 or float64, got '
                    f'{linear_solve_dtype_name!r}'
                )
            linear_solve_dtype = (
                torch.float64
                if linear_solve_dtype_name in {'float64', 'fp64', 'double'}
                else weighted_K.dtype
            )
            solve_system = (
                weighted_K + damping_value * eye
            ).to(linear_solve_dtype)
            solve_rhs = rhs_eff.to(linear_solve_dtype)
            joint_alpha_solve = torch.linalg.solve(
                solve_system,
                solve_rhs,
            )
            solve_residual = torch.linalg.vector_norm(
                torch.mv(solve_system, joint_alpha_solve) - solve_rhs
            )
            joint_alpha = joint_alpha_solve.to(weighted_K.dtype)
            applied_solve_residual = torch.linalg.vector_norm(
                torch.mv(
                    solve_system,
                    joint_alpha.to(linear_solve_dtype),
                ) - solve_rhs
            )
            weighted_alpha = joint_ratio.to(joint_alpha.dtype) * joint_alpha
            if critic_rows:
                actor_dual_coef = weighted_alpha[:num_sa]
                critic_dual_coef = weighted_alpha[num_sa:]
                dual_actor_coef_l2 = torch.linalg.vector_norm(
                    actor_dual_coef
                )
                dual_critic_coef_l2 = torch.linalg.vector_norm(
                    critic_dual_coef
                )
                dual_coef_cosine = torch.dot(
                    actor_dual_coef, critic_dual_coef
                ) / torch.clamp(
                    dual_actor_coef_l2 * dual_critic_coef_l2,
                    min=1e-30,
                )
                dual_difference_fraction = torch.linalg.vector_norm(
                    actor_dual_coef - critic_dual_coef
                ) / torch.clamp(
                    dual_actor_coef_l2 + dual_critic_coef_l2,
                    min=1e-30,
                )
            else:
                dual_actor_coef_l2 = torch.linalg.vector_norm(weighted_alpha)
                dual_critic_coef_l2 = torch.zeros((), device=device)
                dual_coef_cosine = torch.zeros((), device=device)
                dual_difference_fraction = torch.zeros((), device=device)
            zero_direction_metric = torch.zeros((), device=device)
            critic_reconstruction_trunk_l2 = zero_direction_metric
            critic_reconstruction_head_l2 = zero_direction_metric
            critic_reconstruction_trunk_fraction = zero_direction_metric
            actor_reconstruction_nonvalue_l2 = zero_direction_metric
            if (
                critic_rows
                and critic_reconstruction_scope == 'head_only'
            ):
                # Keep the exact full 2B x 2B solve, including both cross
                # blocks.  Only the critic-row contribution to H^T alpha is
                # prevented from writing into the shared trunk; the actor-row
                # contribution remains completely unmasked.
                actor_reconstruction = torch.mv(
                    H_pi.t(),
                    weighted_alpha[:num_sa].to(H_pi.dtype),
                )
                critic_reconstruction = torch.mv(
                    critic_H.t(),
                    weighted_alpha[num_sa:].to(critic_H.dtype),
                )
                head_mask = critic_head_column_mask.to(critic_H.dtype)
                trunk_mask = 1.0 - head_mask
                critic_reconstruction_trunk_l2 = torch.linalg.vector_norm(
                    critic_reconstruction * trunk_mask
                )
                critic_reconstruction_head_l2 = torch.linalg.vector_norm(
                    critic_reconstruction * head_mask
                )
                critic_reconstruction_trunk_fraction = (
                    critic_reconstruction_trunk_l2
                    / torch.clamp(
                        torch.linalg.vector_norm(critic_reconstruction),
                        min=1e-30,
                    )
                )
                actor_reconstruction_nonvalue_l2 = torch.linalg.vector_norm(
                    actor_reconstruction * trunk_mask
                )
                flat_dir = (
                    actor_reconstruction + critic_reconstruction * head_mask
                ) / kernel_denom
            else:
                flat_dir = torch.mv(
                    joint_H.t(), weighted_alpha.to(joint_H.dtype)
                ) / kernel_denom

            # Direct causal diagnostics, evaluated sparsely to avoid turning
            # every update into three extra linear solves.  The decomposition
            # uses the exact same joint metric as the applied update and
            # separates actor RHS from critic RHS.
            diagnostic_interval = int(getattr(
                algo_config, 'causal_diagnostic_interval', 32
            ))
            run_causal_diagnostic = (
                ablation_mode != 'actor_only'
                and diagnostic_interval > 0
                and minibatch_global_step % diagnostic_interval == 0
            )
            # Zero is an explicit placeholder when the sparse diagnostic did
            # not run; causal_diagnostic_ran is the validity mask.  Avoid NaN
            # placeholders because operational monitors correctly treat NaN
            # anywhere in a training trace as suspicious.
            zero_metric = torch.zeros((), device=device)
            metric_actor_critic_cosine = zero_metric
            metric_actor_critic_cosine_valid = zero_metric
            critic_induced_actor_quadratic = zero_metric
            actor_component_actor_quadratic = zero_metric
            actor_critic_component_cross_quadratic = zero_metric
            component_reconstructed_actor_quadratic = zero_metric
            component_actor_quadratic_relative_error = zero_metric
            critic_actor_quadratic_fraction = zero_metric
            actor_fullmetric_vs_actoronly_cosine = zero_metric
            actor_fullmetric_vs_actoronly_norm_ratio = zero_metric
            actor_fullmetric_delta_fraction = zero_metric
            actoronly_metric_direction_l2 = zero_metric
            actoronly_metric_actor_quadratic = zero_metric
            actor_gain_from_critic_rhs = zero_metric
            actor_rhs_self_response = zero_metric
            critic_rhs_self_response = zero_metric
            actor_component_l2 = zero_metric
            critic_component_l2 = zero_metric
            component_direction_cosine = zero_metric
            critic_vanilla_direction_l2 = zero_metric
            critic_ggn_vs_vanilla_cosine = zero_metric
            critic_ggn_vs_vanilla_norm_ratio = zero_metric
            critic_trunk_vanilla_direction_l2 = zero_metric
            critic_trunk_ggn_direction_l2 = zero_metric
            critic_trunk_ggn_vs_vanilla_cosine = zero_metric
            critic_trunk_ggn_vs_vanilla_norm_ratio = zero_metric
            if run_causal_diagnostic:
                actor_rhs_only = torch.cat([
                    _adv, torch.zeros_like(critic_rhs)
                ])
                critic_rhs_only = torch.cat([
                    torch.zeros_like(_adv), critic_rhs
                ])
                component_rhs = torch.stack(
                    [actor_rhs_only, critic_rhs_only], dim=1
                ).to(linear_solve_dtype)
                component_alpha = torch.linalg.solve(
                    solve_system, component_rhs
                ).to(weighted_K.dtype)
                weighted_component_alpha = (
                    joint_ratio.to(joint_H.dtype).unsqueeze(1)
                    * component_alpha.to(joint_H.dtype)
                )
                if critic_reconstruction_scope == 'head_only':
                    actor_component_reconstruction = torch.mm(
                        H_pi.t(), weighted_component_alpha[:num_sa]
                    )
                    critic_component_reconstruction = torch.mm(
                        critic_H.t(), weighted_component_alpha[num_sa:]
                    ) * critic_head_column_mask.to(critic_H.dtype).unsqueeze(1)
                    component_directions = (
                        actor_component_reconstruction
                        + critic_component_reconstruction
                    ) / kernel_denom
                else:
                    component_directions = torch.mm(
                        joint_H.t(), weighted_component_alpha
                    ) / kernel_denom
                actor_component = component_directions[:, 0]
                critic_component = component_directions[:, 1]
                actor_component_l2 = torch.linalg.vector_norm(
                    actor_component
                )
                critic_component_l2 = torch.linalg.vector_norm(
                    critic_component
                )
                component_direction_cosine = torch.dot(
                    actor_component, critic_component
                ) / torch.clamp(
                    actor_component_l2 * critic_component_l2,
                    min=1e-30,
                )
                # Exact same-minibatch actor-only counterfactual.  This
                # removes critic curvature rows and both cross blocks while
                # preserving H_pi, advantage, ratio, damping and solve dtype.
                # It isolates how the critic geometry changes the actor-RHS
                # preconditioner, independently of the critic RHS itself.
                actor_weighted_K = (
                    joint_K[:num_sa, :num_sa]
                    * _ratio.to(joint_K.dtype).unsqueeze(0)
                )
                actor_only_system = (
                    actor_weighted_K
                    + damping_value
                    * torch.eye(
                        num_sa,
                        device=device,
                        dtype=actor_weighted_K.dtype,
                    )
                ).to(linear_solve_dtype)
                actor_only_alpha = torch.linalg.solve(
                    actor_only_system,
                    _adv.to(linear_solve_dtype),
                ).to(H_pi.dtype)
                actor_only_metric_direction = torch.mv(
                    H_pi.t(),
                    _ratio.to(H_pi.dtype) * actor_only_alpha,
                ) / kernel_denom
                actoronly_metric_direction_l2 = torch.linalg.vector_norm(
                    actor_only_metric_direction
                )
                actor_fullmetric_vs_actoronly_cosine = torch.dot(
                    actor_component,
                    actor_only_metric_direction,
                ) / torch.clamp(
                    actor_component_l2 * actoronly_metric_direction_l2,
                    min=1e-30,
                )
                actor_fullmetric_vs_actoronly_norm_ratio = (
                    actor_component_l2
                    / torch.clamp(
                        actoronly_metric_direction_l2,
                        min=1e-30,
                    )
                )
                actor_fullmetric_delta_fraction = torch.linalg.vector_norm(
                    actor_component - actor_only_metric_direction
                ) / torch.clamp(
                    actoronly_metric_direction_l2,
                    min=1e-30,
                )
                actor_only_metric_projection = torch.mv(
                    H_pi, actor_only_metric_direction
                )
                actoronly_metric_actor_quadratic = (
                    _ratio.to(actor_only_metric_projection.dtype)
                    * actor_only_metric_projection.square()
                ).mean()
                actor_gradient = torch.mv(
                    H_pi.t(), _ratio.to(H_pi.dtype) * _adv
                ) / kernel_denom
                critic_gradient = torch.mv(
                    critic_H.t(), critic_rhs
                ) / kernel_denom
                # For the queued clean-score, lambda_C=c_C=1 control,
                # critic_gradient is exactly J_C^T e_C / B: the ordinary MSE
                # descent/update direction up to the conventional factor 2.
                # Compare it with the isolated critic contribution produced
                # by the full joint GGN solve.  These are logging-only tensor
                # operations and do not alter flat_dir or consume randomness.
                critic_vanilla_direction_l2 = torch.linalg.vector_norm(
                    critic_gradient
                )
                critic_ggn_vs_vanilla_cosine = torch.dot(
                    critic_component, critic_gradient
                ) / torch.clamp(
                    critic_component_l2 * critic_vanilla_direction_l2,
                    min=1e-30,
                )
                critic_ggn_vs_vanilla_norm_ratio = (
                    critic_component_l2
                    / torch.clamp(
                        critic_vanilla_direction_l2, min=1e-30
                    )
                )
                critic_trunk_mask = (
                    1.0 - critic_head_column_mask.to(critic_gradient.dtype)
                )
                critic_trunk_component = (
                    critic_component * critic_trunk_mask
                )
                critic_trunk_vanilla = (
                    critic_gradient * critic_trunk_mask
                )
                critic_trunk_ggn_direction_l2 = torch.linalg.vector_norm(
                    critic_trunk_component
                )
                critic_trunk_vanilla_direction_l2 = (
                    torch.linalg.vector_norm(critic_trunk_vanilla)
                )
                critic_trunk_ggn_vs_vanilla_cosine = torch.dot(
                    critic_trunk_component, critic_trunk_vanilla
                ) / torch.clamp(
                    critic_trunk_ggn_direction_l2
                    * critic_trunk_vanilla_direction_l2,
                    min=1e-30,
                )
                critic_trunk_ggn_vs_vanilla_norm_ratio = (
                    critic_trunk_ggn_direction_l2
                    / torch.clamp(
                        critic_trunk_vanilla_direction_l2, min=1e-30
                    )
                )
                actor_rhs_self_response = torch.dot(
                    actor_gradient, actor_component
                )
                critic_rhs_self_response = torch.dot(
                    critic_gradient, critic_component
                )
                actor_critic_response = torch.dot(
                    actor_gradient, critic_component
                )
                response_product = (
                    actor_rhs_self_response * critic_rhs_self_response
                )
                metric_actor_critic_cosine_valid = (
                    (actor_rhs_self_response > 0.0)
                    & (critic_rhs_self_response > 0.0)
                ).to(actor_critic_response.dtype)
                metric_actor_critic_cosine = torch.where(
                    metric_actor_critic_cosine_valid.bool(),
                    actor_critic_response / torch.sqrt(
                        torch.clamp(response_product, min=1e-30)
                    ),
                    torch.zeros_like(actor_critic_response),
                )
                actor_gain_from_critic_rhs = (
                    actor_rhs_self_response + actor_critic_response
                ) / torch.clamp(actor_rhs_self_response.abs(), min=1e-30)
                critic_actor_projection = torch.mv(
                    H_pi, critic_component
                )
                actor_component_projection = torch.mv(
                    H_pi, actor_component
                )
                actor_component_actor_quadratic = (
                    _ratio.to(actor_component_projection.dtype)
                    * actor_component_projection.square()
                ).mean()
                critic_induced_actor_quadratic = (
                    _ratio.to(critic_actor_projection.dtype)
                    * critic_actor_projection.square()
                ).mean()
                actor_critic_component_cross_quadratic = (
                    _ratio.to(actor_component_projection.dtype)
                    * actor_component_projection
                    * critic_actor_projection
                ).mean()
                component_reconstructed_actor_quadratic = (
                    actor_component_actor_quadratic
                    + 2.0 * actor_critic_component_cross_quadratic
                    + critic_induced_actor_quadratic
                )
                # This fraction answers the immediate causal question in the
                # policy Fisher metric: how large the isolated critic-RHS
                # direction is relative to the complete joint direction.  It
                # is a diagnostic ratio, not an additive attribution when the
                # cross term is nonzero.
                critic_actor_quadratic_fraction = (
                    critic_induced_actor_quadratic
                    / torch.clamp(
                        component_reconstructed_actor_quadratic.abs(),
                        min=1e-30,
                    )
                )
            direction_l2 = torch.linalg.vector_norm(flat_dir)
            actor_projection = torch.mv(H_pi, flat_dir)
            critic_projection = torch.mv(J_v, flat_dir)
            sampled_critic_projection = torch.mv(critic_H, flat_dir)
            actor_fisher_quadratic = actor_projection.pow(2).mean()
            actor_fisher_quadratic_ratio_weighted = (
                _ratio.to(actor_projection.dtype)
                * actor_projection.square()
            ).mean()
            if run_causal_diagnostic:
                component_actor_quadratic_relative_error = (
                    component_reconstructed_actor_quadratic
                    - actor_fisher_quadratic_ratio_weighted
                ).abs() / torch.clamp(
                    actor_fisher_quadratic_ratio_weighted.abs(), min=1e-30
                )
            critic_ggn_quadratic = critic_projection.pow(2).mean()
            sampled_critic_quadratic = sampled_critic_projection.pow(2).mean()
            critic_noise_weights = critic_score_noise.square()
            critic_noise_ess = (
                critic_noise_weights.sum().square()
                / torch.clamp(
                    critic_noise_weights.square().sum(), min=1e-30
                )
            )
            actor_block_fro = torch.linalg.matrix_norm(
                joint_K[:num_sa, :num_sa]
            )
            if critic_rows:
                critic_block_fro = torch.linalg.matrix_norm(
                    joint_K[num_sa:, num_sa:]
                )
                cross_block_fro = torch.linalg.matrix_norm(
                    joint_K[:num_sa, num_sa:]
                )
                normalized_cross_block = cross_block_fro / torch.sqrt(
                    torch.clamp(actor_block_fro * critic_block_fro, min=1e-30)
                )
            else:
                critic_block_fro = torch.zeros_like(actor_block_fro)
                cross_block_fro = torch.zeros_like(actor_block_fro)
                normalized_cross_block = torch.zeros_like(actor_block_fro)
            clip_scale = torch.clamp(
                torch.as_tensor(
                    float(algo_config.max_grad_norm),
                    device=direction_l2.device,
                    dtype=direction_l2.dtype,
                ) / (direction_l2 + 1e-12),
                max=1.0,
            )
            actor_alone_clip_scale = zero_metric
            joint_clip_extra_actor_attenuation = zero_metric
            if run_causal_diagnostic:
                actor_alone_clip_scale = torch.clamp(
                    torch.as_tensor(
                        float(algo_config.max_grad_norm),
                        device=actor_component_l2.device,
                        dtype=actor_component_l2.dtype,
                    ) / (actor_component_l2 + 1e-12),
                    max=1.0,
                )
                # Values below one mean the combined actor+critic direction
                # attenuates the actor more than clipping the actor component
                # by itself would have done.
                joint_clip_extra_actor_attenuation = (
                    clip_scale
                    / torch.clamp(actor_alone_clip_scale, min=1e-30)
                )
            step_scale = float(ac_optimizer.param_groups[0]['lr']) * clip_scale
        # These losses are logging quantities. The update direction is the
        # joint RAT/GGN direction constructed above.
        _loss_pi = (-_ratio.detach() * _adv.detach()).mean()
        _loss_v = F.mse_loss(_vals, _ret)
        _loss = (
            _loss_pi
            - algo_config.ent_coef * _entropy
            + algo_config.vf_coef * _loss_v
        )

        if float(algo_config.ent_coef) != 0.0:
            raise ValueError(
                "joint-GGN Procgen trainer requires ent_coef=0.0"
            )

        ac_optimizer.zero_grad(set_to_none=False)
        for p in trainable_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        set_grads_from_flat(trainable_params, -flat_dir)
        # Preserve the paper Procgen clipping semantics: one global Euclidean
        # norm clip over all shared actor-critic parameters.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, algo_config.max_grad_norm
        )
        ac_optimizer.step()

        with torch.no_grad():
            if optimizer_momentum > 0.0:
                current_buffer = torch.cat([
                    ac_optimizer.state[p]['momentum_buffer'].detach().flatten()
                    for p in trainable_params
                ], dim=0)
                # The optimizer buffer is in descent-gradient sign.
                effective_ascent_direction = -current_buffer
            else:
                effective_ascent_direction = flat_dir * clip_scale
            effective_direction_l2 = torch.linalg.vector_norm(
                effective_ascent_direction
            )
            effective_actor_projection = torch.mv(
                H_pi, effective_ascent_direction
            )
            effective_actor_fisher_quadratic = (
                effective_actor_projection.pow(2).mean()
            )
            predicted_step_kl = (
                0.5
                * float(ac_optimizer.param_groups[0]['lr']) ** 2
                * effective_actor_fisher_quadratic
            )

        if lr_scheduler is not None:
            lr_scheduler.step()

        with torch.no_grad():
            _, _outputs_after = actor_critic(_obs)
            _logp_full_after = F.log_softmax(_outputs_after, dim=-1)
            _curr_kl = (
                torch.exp(_logp_full)
                * (_logp_full - _logp_full_after)
            ).sum(dim=-1).mean()
            _real_kl = (
                torch.exp(_logp_full_old)
                * (_logp_full_old - _logp_full_after)
            ).sum(dim=-1).mean()

            pi_info = dict(
                kl=_real_kl.item(),
                curr_kl=_curr_kl.item(),
                curr_lr=ac_optimizer.param_groups[0]['lr'],
                ent=_entropy.item(),
                cf=0.0,
                grad_norm=grad_norm.item(),
                ratio_max=_ratio.max().item(),
                ratio_min=_ratio.min().item(),
                critic_ratio_min=critic_ratio.min().item(),
                critic_ratio_max=critic_ratio.max().item(),
                actor_rows=num_sa,
                critic_rows=critic_rows,
                joint_system_rows=num_sa + critic_rows,
                joint_kernel_mode=(
                    f'{ablation_mode}_{configured_score_mode}_'
                    f'{critic_param_scope}'
                ),
                joint_ablation_mode=ablation_mode,
                joint_critic_param_scope=critic_param_scope,
                joint_critic_reconstruction_scope=(
                    critic_reconstruction_scope
                ),
                total_parameter_columns=total_parameter_columns,
                critic_head_parameter_columns=(
                    critic_head_parameter_columns
                ),
                kernel_diag_min=kernel_diag_min.item(),
                kernel_diag_median=kernel_diag_median.item(),
                kernel_diag_max=kernel_diag_max.item(),
                base_damping_value=base_damping_value.item(),
                effective_damping_value=damping_value.item(),
                damping_to_median_floor=damping_to_median_floor,
                base_damping_to_median_diag=(
                    base_damping_to_median_diag.item()
                ),
                damping_to_median_diag=damping_to_median_diag.item(),
                spectral_diagnostic_ran=float(run_spectral_diagnostic),
                spectral_eigen_min=spectral_eigen_min.item(),
                spectral_eigen_max=spectral_eigen_max.item(),
                spectral_damped_condition=(
                    spectral_damped_condition.item()
                ),
                dual_actor_coef_l2=dual_actor_coef_l2.item(),
                dual_critic_coef_l2=dual_critic_coef_l2.item(),
                dual_coef_cosine=dual_coef_cosine.item(),
                dual_difference_fraction=dual_difference_fraction.item(),
                critic_reconstruction_trunk_l2=(
                    critic_reconstruction_trunk_l2.item()
                ),
                critic_reconstruction_head_l2=(
                    critic_reconstruction_head_l2.item()
                ),
                critic_reconstruction_trunk_fraction=(
                    critic_reconstruction_trunk_fraction.item()
                ),
                actor_reconstruction_nonvalue_l2=(
                    actor_reconstruction_nonvalue_l2.item()
                ),
                categorical_fisher_trace=categorical_fisher_trace.item(),
                critic_noise_ess=critic_noise_ess.item(),
                actor_block_fro=actor_block_fro.item(),
                critic_block_fro=critic_block_fro.item(),
                cross_block_fro=cross_block_fro.item(),
                normalized_cross_block=normalized_cross_block.item(),
                causal_diagnostic_ran=float(run_causal_diagnostic),
                metric_actor_critic_cosine_valid=(
                    metric_actor_critic_cosine_valid.item()
                ),
                metric_actor_critic_cosine=metric_actor_critic_cosine.item(),
                critic_induced_actor_quadratic=(
                    critic_induced_actor_quadratic.item()
                ),
                actor_component_actor_quadratic=(
                    actor_component_actor_quadratic.item()
                ),
                actor_critic_component_cross_quadratic=(
                    actor_critic_component_cross_quadratic.item()
                ),
                component_reconstructed_actor_quadratic=(
                    component_reconstructed_actor_quadratic.item()
                ),
                component_actor_quadratic_relative_error=(
                    component_actor_quadratic_relative_error.item()
                ),
                critic_actor_quadratic_fraction=(
                    critic_actor_quadratic_fraction.item()
                ),
                actor_fullmetric_vs_actoronly_cosine=(
                    actor_fullmetric_vs_actoronly_cosine.item()
                ),
                actor_fullmetric_vs_actoronly_norm_ratio=(
                    actor_fullmetric_vs_actoronly_norm_ratio.item()
                ),
                actor_fullmetric_delta_fraction=(
                    actor_fullmetric_delta_fraction.item()
                ),
                actoronly_metric_direction_l2=(
                    actoronly_metric_direction_l2.item()
                ),
                actoronly_metric_actor_quadratic=(
                    actoronly_metric_actor_quadratic.item()
                ),
                actor_gain_from_critic_rhs=actor_gain_from_critic_rhs.item(),
                actor_rhs_self_response=actor_rhs_self_response.item(),
                critic_rhs_self_response=critic_rhs_self_response.item(),
                actor_component_l2=actor_component_l2.item(),
                critic_component_l2=critic_component_l2.item(),
                component_direction_cosine=(
                    component_direction_cosine.item()
                ),
                critic_vanilla_direction_l2=(
                    critic_vanilla_direction_l2.item()
                ),
                critic_ggn_vs_vanilla_cosine=(
                    critic_ggn_vs_vanilla_cosine.item()
                ),
                critic_ggn_vs_vanilla_norm_ratio=(
                    critic_ggn_vs_vanilla_norm_ratio.item()
                ),
                critic_trunk_vanilla_direction_l2=(
                    critic_trunk_vanilla_direction_l2.item()
                ),
                critic_trunk_ggn_direction_l2=(
                    critic_trunk_ggn_direction_l2.item()
                ),
                critic_trunk_ggn_vs_vanilla_cosine=(
                    critic_trunk_ggn_vs_vanilla_cosine.item()
                ),
                critic_trunk_ggn_vs_vanilla_norm_ratio=(
                    critic_trunk_ggn_vs_vanilla_norm_ratio.item()
                ),
                actor_alone_clip_scale=actor_alone_clip_scale.item(),
                joint_clip_extra_actor_attenuation=(
                    joint_clip_extra_actor_attenuation.item()
                ),
                optimizer_momentum=optimizer_momentum,
                optimizer_momentum_semantics='classic_beta_m_plus_d',
                optimizer_history_correction=float(is_kaczmarz),
                kaczmarz_rhs_semantics='rhs_plus_beta_H_buffer',
                kaczmarz_previous_projection_l2=(
                    torch.linalg.vector_norm(previous_projection).item()
                ),
                joint_solve_residual=solve_residual.item(),
                joint_applied_solve_residual=applied_solve_residual.item(),
                joint_linear_solve_dtype=str(linear_solve_dtype),
                joint_critic_curvature_coef=critic_curvature_coef,
                joint_critic_objective_coef=critic_objective_coef,
                joint_direction_l2=direction_l2.item(),
                joint_effective_direction_l2=effective_direction_l2.item(),
                joint_clip_scale=clip_scale.item(),
                joint_actor_fisher_quadratic=actor_fisher_quadratic.item(),
                joint_actor_fisher_quadratic_ratio_weighted=(
                    actor_fisher_quadratic_ratio_weighted.item()
                ),
                joint_critic_ggn_quadratic=critic_ggn_quadratic.item(),
                joint_sampled_critic_quadratic=(
                    sampled_critic_quadratic.item()
                ),
                joint_predicted_step_kl=predicted_step_kl.item(),
                joint_effective_actor_fisher_quadratic=(
                    effective_actor_fisher_quadratic.item()
                ),
                joint_system_ratio_mode='actor_ratio_critic_unit',
                joint_reconstruction_mode=(
                    'direct_Ht_alpha_critic_head_only'
                    if critic_reconstruction_scope == 'head_only'
                    else 'direct_Ht_alpha'
                ),
                joint_rhs_mode='paired_score_residual',
                joint_rhs_columns=1,
                joint_critic_score_mode=configured_score_mode,
                joint_critic_score_scale=critic_h_weight,
                critic_score_noise_mean=critic_score_noise.mean().item(),
                critic_score_noise_std=(
                    critic_score_noise.std(unbiased=False).item()
                ),
                critic_score_noise_second_moment=(
                    critic_score_noise.square().mean().item()
                ),
                critic_score_noise_min=critic_score_noise.min().item(),
                critic_score_noise_max=critic_score_noise.max().item(),
                actor_rhs_l2=torch.linalg.vector_norm(_adv).item(),
                advantage_rms_before_optional_normalization=(
                    _rms_sqrt.item()
                ),
                normalize_advantage_rms=float(normalize_advantage_rms),
                critic_rhs_l2=torch.linalg.vector_norm(critic_rhs).item(),
            )

        return _loss, _loss_pi, _loss_v, pi_info

    def KFAC_Update(_obs, _act, _adv, _ret, _outputs_old):
        _vals, _outputs = actor_critic(_obs)

        if actor_critic.is_discrete:
            _logp_full = F.log_softmax(_outputs, dim=-1)
            _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
            _logp = torch.gather(_logp_full, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _llr = torch.gather(_logp_full - _logp_full_old, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _ratio = torch.exp(_llr)
            _p_log_p = torch.exp(_logp_full) * _logp_full
            _entropy = - _p_log_p.sum(-1).mean()
            _kl = (torch.exp(_logp_full_old) * (_logp_full_old - _logp_full)).sum(dim=-1).mean()

        else:
            _mu, _logstd = _outputs.chunk(2, dim=-1)
            _dist = torch.distributions.Normal(_mu, torch.exp(_logstd))
            _logp = _dist.log_prob(_act).sum(dim=-1) 

            _mu_old, _logstd_old = _outputs_old.chunk(2, dim=-1)
            _dist_old = torch.distributions.Normal(_mu_old, torch.exp(_logstd_old))
            _logp_old = _dist_old.log_prob(_act).sum(dim=-1)

            _llr = _logp - _logp_old
            _ratio = torch.exp(_llr)

            _entropy = _dist.entropy().sum(dim=-1).mean()
            _kl = (_logstd - _logstd_old + 0.5 * ( torch.exp(_logstd_old).pow(2) + (_mu_old - _mu).pow(2) ) / torch.exp(_logstd).pow(2) - 0.5).sum(dim=-1).mean()

        if ac_optimizer.steps % ac_optimizer.TInv == 0:
            # Compute fisher, see Martens 2014
            actor_critic.zero_grad()
            _pg_likelihood = _logp.mean() # likelihood for policy
            _val_noise = torch.randn(_vals.size(), device=device)
            sample_values = _vals + _val_noise
            _vf_likelihood = -(_vals - sample_values.detach()).pow(2).mean() # likelihood for value function
            _likelihood = _pg_likelihood + _vf_likelihood
            ac_optimizer.acc_stats = True
            _likelihood.backward(retain_graph=True)
            ac_optimizer.acc_stats = False

        # zero mean of advantage
        _adv = _adv - _adv.mean() 
        
        # clamp the ratio
        if algo_config.clamp_ratio:
            _ratio = torch.clamp(_ratio, algo_config.min_ratio, algo_config.max_ratio)
        _loss_pi = (- _ratio * _adv).mean() 

        if algo_config.norm_obj == 'adv':
            _rms_sqrt = torch.sqrt( _adv.pow(2).mean() ).detach()
        elif algo_config.norm_obj == 'obj':
            _rms_sqrt = torch.sqrt( (_ratio * _adv).pow(2).mean() ).detach() # might related to variance reduction in importance sampling
        elif algo_config.norm_obj == 'ratio':
            _rms_sqrt = _ratio.mean().detach() * torch.sqrt( _adv.pow(2).mean() ).detach()
        else: 
            raise NotImplementedError

        # normalize the loss to stabilize the training
        _loss_pi = _loss_pi / (_rms_sqrt + 1e-8)
        # value loss
        _loss_v = F.mse_loss(_vals, _ret)

        _loss = _loss_pi - algo_config.ent_coef * _entropy + algo_config.vf_coef * _loss_v

        _loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), algo_config.max_grad_norm)
        ac_optimizer.step()

        # Useful extra info
        with torch.no_grad():
            clipfrac = 0.0
            approx_kl = _kl.item()
            ent = _entropy.item()
            pi_info = dict(kl=approx_kl, ent=ent, curr_lr=ac_optimizer.param_groups[0]['lr'], cf=clipfrac,
                           grad_norm=grad_norm.item(), ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())

        return _loss, _loss_pi, _loss_v, pi_info

    # choose the policy update rule
    if algo in {'ppo'}: 
        update_actor_critic = PPO_Update
    elif algo in {'adv'}:
        update_actor_critic = Advantage_Update
    elif algo in {'kfac', 'ekfac'}:
        update_actor_critic = KFAC_Update
    else: 
        raise NotImplementedError

    tepochs = trange(epochs+1, desc='Epoch starts', leave=True)

    # Main loop: collect experience in env and update/log each epoch
    inds = np.arange(per_epoch_timesteps)

    for epoch in tepochs:
        tstart = time.perf_counter()

        tepochs.set_description('Stepping environment...')

        actor_critic.eval() # set to eval mode for PPO
        obs, ret, act, adv, outputs_old, epinfos = runner.run() #pylint: disable=E0632

        epinfobuf.extend(epinfos)
        tepochs.set_description('Minibatch training...')

        # pop art
        if actor_critic.with_popart:
            actor_critic.last_v_layer.update(ret) # update the mean/var
            ret = actor_critic.last_v_layer.normalize(ret)
            adv = actor_critic.last_v_layer.normalize(adv)

        if actor_critic.obs_rms is not None:
            actor_critic.obs_rms.training = True
            obs = actor_critic.obs_rms(obs) # norm obs for training
            actor_critic.obs_rms.training = False
            # recalculate outputs_old with normalized obs
            with torch.no_grad():
                outputs_old = actor_critic.forward_pi(obs)

        actor_critic.train()  # set to train mode
        for optimizer_epoch in range(algo_config.epochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, per_epoch_timesteps, minibatch_size):
                end = start + minibatch_size
                mbinds = inds[start:end]
                mb_obs, mb_act, mb_adv, mb_ret, mb_outputs_old = obs[mbinds], act[mbinds], adv[mbinds], ret[mbinds], outputs_old[mbinds]
                ac_optimizer.zero_grad()
                mb_loss, mb_loss_pi, mb_loss_v, pi_info = update_actor_critic(mb_obs, mb_act, mb_adv, mb_ret, mb_outputs_old)

                minibatch_global_step += 1
                if metric_trace_file is not None:
                    # Logging-only scale audit. PopArt preserves unnormalized
                    # predictions by rescaling the normalized value head, so
                    # its running standard deviation can change J_v and the
                    # joint sample-space kernel even when task-space behavior
                    # is smooth.
                    if actor_critic.with_popart:
                        with torch.no_grad():
                            popart_mean_tensor, popart_var_tensor = (
                                actor_critic.last_v_layer.debiased_mean_var()
                            )
                            popart_mean_value = float(
                                popart_mean_tensor.mean().item()
                            )
                            popart_std_value = float(
                                torch.sqrt(popart_var_tensor).mean().item()
                            )
                            value_head_weight_l2 = float(
                                torch.linalg.vector_norm(
                                    actor_critic.last_v_layer.weight
                                ).item()
                            )
                    else:
                        popart_mean_value = float('nan')
                        popart_std_value = float('nan')
                        value_head_weight_l2 = float(
                            torch.linalg.vector_norm(
                                actor_critic.last_v_layer.weight
                            ).item()
                        )
                    trace_record = {
                        'minibatch_global_step': minibatch_global_step,
                        'rollout_update': epoch,
                        'optimizer_epoch': optimizer_epoch,
                        'minibatch_index': start // minibatch_size,
                        'environment_transitions': (
                            (epoch + 1) * per_epoch_timesteps * world_size
                        ),
                        'eprewmean': float(safemean([
                            epinfo['r'] for epinfo in epinfobuf
                        ])),
                        # Pre-step normalized value regression diagnostics for
                        # this exact minibatch.  These are logging-only and
                        # let the reconstruction ablation distinguish policy
                        # recovery from simple critic under-fitting.
                        'value_mse_pre_step': float(mb_loss_v.item()),
                        'minibatch_return_variance': float(
                            mb_ret.var().item()
                        ),
                        'minibatch_advantage_variance': float(
                            mb_adv.var().item()
                        ),
                        'popart_mean': popart_mean_value,
                        'popart_std': popart_std_value,
                        'value_head_weight_l2': value_head_weight_l2,
                        'lr_used': float(pi_info['curr_lr']),
                        'behavior_kl_after_step': float(pi_info['kl']),
                        'current_step_kl': float(pi_info['curr_kl']),
                        'entropy': float(pi_info['ent']),
                        'ratio_min': float(pi_info['ratio_min']),
                        'ratio_max': float(pi_info['ratio_max']),
                        'critic_ratio_min': float(
                            pi_info['critic_ratio_min']
                        ),
                        'critic_ratio_max': float(
                            pi_info['critic_ratio_max']
                        ),
                        'joint_solve_residual': float(
                            pi_info['joint_solve_residual']
                        ),
                        'joint_system_rows': int(
                            pi_info['joint_system_rows']
                        ),
                        'joint_kernel_mode': pi_info['joint_kernel_mode'],
                        'joint_ablation_mode': pi_info['joint_ablation_mode'],
                        'joint_critic_param_scope': (
                            pi_info['joint_critic_param_scope']
                        ),
                        'joint_critic_reconstruction_scope': (
                            pi_info['joint_critic_reconstruction_scope']
                        ),
                        'total_parameter_columns': int(
                            pi_info['total_parameter_columns']
                        ),
                        'critic_head_parameter_columns': int(
                            pi_info['critic_head_parameter_columns']
                        ),
                        'kernel_diag_min': float(
                            pi_info['kernel_diag_min']
                        ),
                        'kernel_diag_median': float(
                            pi_info['kernel_diag_median']
                        ),
                        'kernel_diag_max': float(
                            pi_info['kernel_diag_max']
                        ),
                        'base_damping_value': float(
                            pi_info['base_damping_value']
                        ),
                        'effective_damping_value': float(
                            pi_info['effective_damping_value']
                        ),
                        'damping_to_median_floor': float(
                            pi_info['damping_to_median_floor']
                        ),
                        'base_damping_to_median_diag': float(
                            pi_info['base_damping_to_median_diag']
                        ),
                        'damping_to_median_diag': float(
                            pi_info['damping_to_median_diag']
                        ),
                        'spectral_diagnostic_ran': float(
                            pi_info['spectral_diagnostic_ran']
                        ),
                        'spectral_eigen_min': float(
                            pi_info['spectral_eigen_min']
                        ),
                        'spectral_eigen_max': float(
                            pi_info['spectral_eigen_max']
                        ),
                        'spectral_damped_condition': float(
                            pi_info['spectral_damped_condition']
                        ),
                        'dual_actor_coef_l2': float(
                            pi_info['dual_actor_coef_l2']
                        ),
                        'dual_critic_coef_l2': float(
                            pi_info['dual_critic_coef_l2']
                        ),
                        'dual_coef_cosine': float(
                            pi_info['dual_coef_cosine']
                        ),
                        'dual_difference_fraction': float(
                            pi_info['dual_difference_fraction']
                        ),
                        'critic_reconstruction_trunk_l2': float(
                            pi_info['critic_reconstruction_trunk_l2']
                        ),
                        'critic_reconstruction_head_l2': float(
                            pi_info['critic_reconstruction_head_l2']
                        ),
                        'critic_reconstruction_trunk_fraction': float(
                            pi_info[
                                'critic_reconstruction_trunk_fraction'
                            ]
                        ),
                        'actor_reconstruction_nonvalue_l2': float(
                            pi_info['actor_reconstruction_nonvalue_l2']
                        ),
                        'categorical_fisher_trace': float(
                            pi_info['categorical_fisher_trace']
                        ),
                        'critic_noise_ess': float(
                            pi_info['critic_noise_ess']
                        ),
                        'actor_block_fro': float(pi_info['actor_block_fro']),
                        'critic_block_fro': float(pi_info['critic_block_fro']),
                        'cross_block_fro': float(pi_info['cross_block_fro']),
                        'normalized_cross_block': float(
                            pi_info['normalized_cross_block']
                        ),
                        'causal_diagnostic_ran': float(
                            pi_info['causal_diagnostic_ran']
                        ),
                        'metric_actor_critic_cosine_valid': float(
                            pi_info['metric_actor_critic_cosine_valid']
                        ),
                        'metric_actor_critic_cosine': float(
                            pi_info['metric_actor_critic_cosine']
                        ),
                        'critic_induced_actor_quadratic': float(
                            pi_info['critic_induced_actor_quadratic']
                        ),
                        'actor_component_actor_quadratic': float(
                            pi_info['actor_component_actor_quadratic']
                        ),
                        'actor_critic_component_cross_quadratic': float(
                            pi_info[
                                'actor_critic_component_cross_quadratic'
                            ]
                        ),
                        'component_reconstructed_actor_quadratic': float(
                            pi_info[
                                'component_reconstructed_actor_quadratic'
                            ]
                        ),
                        'component_actor_quadratic_relative_error': float(
                            pi_info[
                                'component_actor_quadratic_relative_error'
                            ]
                        ),
                        'critic_actor_quadratic_fraction': float(
                            pi_info['critic_actor_quadratic_fraction']
                        ),
                        'actor_fullmetric_vs_actoronly_cosine': float(
                            pi_info[
                                'actor_fullmetric_vs_actoronly_cosine'
                            ]
                        ),
                        'actor_fullmetric_vs_actoronly_norm_ratio': float(
                            pi_info[
                                'actor_fullmetric_vs_actoronly_norm_ratio'
                            ]
                        ),
                        'actor_fullmetric_delta_fraction': float(
                            pi_info['actor_fullmetric_delta_fraction']
                        ),
                        'actoronly_metric_direction_l2': float(
                            pi_info['actoronly_metric_direction_l2']
                        ),
                        'actoronly_metric_actor_quadratic': float(
                            pi_info['actoronly_metric_actor_quadratic']
                        ),
                        'actor_gain_from_critic_rhs': float(
                            pi_info['actor_gain_from_critic_rhs']
                        ),
                        'actor_rhs_self_response': float(
                            pi_info['actor_rhs_self_response']
                        ),
                        'critic_rhs_self_response': float(
                            pi_info['critic_rhs_self_response']
                        ),
                        'actor_component_l2': float(
                            pi_info['actor_component_l2']
                        ),
                        'critic_component_l2': float(
                            pi_info['critic_component_l2']
                        ),
                        'component_direction_cosine': float(
                            pi_info['component_direction_cosine']
                        ),
                        'critic_vanilla_direction_l2': float(
                            pi_info['critic_vanilla_direction_l2']
                        ),
                        'critic_ggn_vs_vanilla_cosine': float(
                            pi_info['critic_ggn_vs_vanilla_cosine']
                        ),
                        'critic_ggn_vs_vanilla_norm_ratio': float(
                            pi_info['critic_ggn_vs_vanilla_norm_ratio']
                        ),
                        'critic_trunk_vanilla_direction_l2': float(
                            pi_info['critic_trunk_vanilla_direction_l2']
                        ),
                        'critic_trunk_ggn_direction_l2': float(
                            pi_info['critic_trunk_ggn_direction_l2']
                        ),
                        'critic_trunk_ggn_vs_vanilla_cosine': float(
                            pi_info[
                                'critic_trunk_ggn_vs_vanilla_cosine'
                            ]
                        ),
                        'critic_trunk_ggn_vs_vanilla_norm_ratio': float(
                            pi_info[
                                'critic_trunk_ggn_vs_vanilla_norm_ratio'
                            ]
                        ),
                        'actor_alone_clip_scale': float(
                            pi_info['actor_alone_clip_scale']
                        ),
                        'joint_clip_scale': float(
                            pi_info['joint_clip_scale']
                        ),
                        'joint_clip_extra_actor_attenuation': float(
                            pi_info[
                                'joint_clip_extra_actor_attenuation'
                            ]
                        ),
                        'joint_system_ratio_mode': (
                            pi_info['joint_system_ratio_mode']
                        ),
                        'joint_reconstruction_mode': (
                            pi_info['joint_reconstruction_mode']
                        ),
                        'joint_rhs_mode': pi_info['joint_rhs_mode'],
                        'joint_rhs_columns': int(
                            pi_info['joint_rhs_columns']
                        ),
                        'joint_critic_score_mode': (
                            pi_info['joint_critic_score_mode']
                        ),
                        'joint_critic_score_scale': float(
                            pi_info['joint_critic_score_scale']
                        ),
                        'critic_score_noise_mean': float(
                            pi_info['critic_score_noise_mean']
                        ),
                        'critic_score_noise_std': float(
                            pi_info['critic_score_noise_std']
                        ),
                        'critic_score_noise_second_moment': float(
                            pi_info['critic_score_noise_second_moment']
                        ),
                        'critic_score_noise_min': float(
                            pi_info['critic_score_noise_min']
                        ),
                        'critic_score_noise_max': float(
                            pi_info['critic_score_noise_max']
                        ),
                        'actor_rhs_l2': float(pi_info['actor_rhs_l2']),
                        'advantage_rms_before_optional_normalization': float(
                            pi_info[
                                'advantage_rms_before_optional_normalization'
                            ]
                        ),
                        'normalize_advantage_rms': float(
                            pi_info['normalize_advantage_rms']
                        ),
                        'critic_rhs_l2': float(pi_info['critic_rhs_l2']),
                        'joint_direction_l2': float(
                            pi_info['joint_direction_l2']
                        ),
                        'joint_effective_direction_l2': float(
                            pi_info['joint_effective_direction_l2']
                        ),
                        'optimizer_momentum': float(
                            pi_info['optimizer_momentum']
                        ),
                        'optimizer_momentum_semantics': (
                            pi_info['optimizer_momentum_semantics']
                        ),
                        'optimizer_history_correction': float(
                            pi_info['optimizer_history_correction']
                        ),
                        'kaczmarz_rhs_semantics': (
                            pi_info['kaczmarz_rhs_semantics']
                        ),
                        'kaczmarz_previous_projection_l2': float(
                            pi_info['kaczmarz_previous_projection_l2']
                        ),
                        'joint_actor_fisher_quadratic': float(
                            pi_info['joint_actor_fisher_quadratic']
                        ),
                        'joint_actor_fisher_quadratic_ratio_weighted': float(
                            pi_info[
                                'joint_actor_fisher_quadratic_ratio_weighted'
                            ]
                        ),
                        'joint_effective_actor_fisher_quadratic': float(
                            pi_info['joint_effective_actor_fisher_quadratic']
                        ),
                        'joint_critic_ggn_quadratic': float(
                            pi_info['joint_critic_ggn_quadratic']
                        ),
                        'joint_sampled_critic_quadratic': float(
                            pi_info['joint_sampled_critic_quadratic']
                        ),
                    }
                    metric_trace_file.write(json.dumps(trace_record) + '\n')
                    if minibatch_global_step % 32 == 0:
                        metric_trace_file.flush()

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
        pi_info['curr_lr'] = ac_optimizer.param_groups[0]['lr']

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
            logger.logkv("lr", ac_optimizer.param_groups[0]['lr'])
            logger.logkv("clipfrac", pi_info['cf'])
            logger.logkv("ratio_max", pi_info['ratio_max'])
            logger.logkv("ratio_min", pi_info['ratio_min'])
            if algo in {'adv'}:
                logger.logkv("critic_ratio_min", pi_info['critic_ratio_min'])
                logger.logkv("critic_ratio_max", pi_info['critic_ratio_max'])
                logger.logkv("actor_rows", pi_info['actor_rows'])
                logger.logkv("critic_rows", pi_info['critic_rows'])
                logger.logkv("joint_system_rows", pi_info['joint_system_rows'])
                logger.logkv("joint_kernel_mode", pi_info['joint_kernel_mode'])
                logger.logkv("joint_ablation_mode", pi_info['joint_ablation_mode'])
                logger.logkv("joint_critic_param_scope", pi_info['joint_critic_param_scope'])
                logger.logkv(
                    "joint_critic_reconstruction_scope",
                    pi_info['joint_critic_reconstruction_scope'],
                )
                logger.logkv(
                    "dual_actor_coef_l2", pi_info['dual_actor_coef_l2']
                )
                logger.logkv(
                    "dual_critic_coef_l2", pi_info['dual_critic_coef_l2']
                )
                logger.logkv(
                    "dual_coef_cosine", pi_info['dual_coef_cosine']
                )
                logger.logkv(
                    "dual_difference_fraction",
                    pi_info['dual_difference_fraction'],
                )
                logger.logkv(
                    "critic_reconstruction_trunk_l2",
                    pi_info['critic_reconstruction_trunk_l2'],
                )
                logger.logkv(
                    "critic_reconstruction_head_l2",
                    pi_info['critic_reconstruction_head_l2'],
                )
                logger.logkv(
                    "critic_reconstruction_trunk_fraction",
                    pi_info['critic_reconstruction_trunk_fraction'],
                )
                logger.logkv(
                    "actor_reconstruction_nonvalue_l2",
                    pi_info['actor_reconstruction_nonvalue_l2'],
                )
                logger.logkv("categorical_fisher_trace", pi_info['categorical_fisher_trace'])
                logger.logkv("critic_noise_ess", pi_info['critic_noise_ess'])
                logger.logkv("normalized_cross_block", pi_info['normalized_cross_block'])
                logger.logkv("causal_diagnostic_ran", pi_info['causal_diagnostic_ran'])
                logger.logkv("metric_actor_critic_cosine", pi_info['metric_actor_critic_cosine'])
                logger.logkv("critic_induced_actor_quadratic", pi_info['critic_induced_actor_quadratic'])
                logger.logkv("actor_gain_from_critic_rhs", pi_info['actor_gain_from_critic_rhs'])
                logger.logkv("optimizer_momentum", pi_info['optimizer_momentum'])
                logger.logkv("optimizer_history_correction", pi_info['optimizer_history_correction'])
                logger.logkv("kaczmarz_previous_projection_l2", pi_info['kaczmarz_previous_projection_l2'])
                logger.logkv("joint_solve_residual", pi_info['joint_solve_residual'])
                logger.logkv("joint_applied_solve_residual", pi_info['joint_applied_solve_residual'])
                logger.logkv("joint_linear_solve_dtype", pi_info['joint_linear_solve_dtype'])
                logger.logkv("joint_critic_curvature_coef", pi_info['joint_critic_curvature_coef'])
                logger.logkv("joint_critic_objective_coef", pi_info['joint_critic_objective_coef'])
                logger.logkv("joint_direction_l2", pi_info['joint_direction_l2'])
                logger.logkv("joint_effective_direction_l2", pi_info['joint_effective_direction_l2'])
                logger.logkv("joint_clip_scale", pi_info['joint_clip_scale'])
                logger.logkv("joint_actor_fisher_quadratic", pi_info['joint_actor_fisher_quadratic'])
                logger.logkv("joint_effective_actor_fisher_quadratic", pi_info['joint_effective_actor_fisher_quadratic'])
                logger.logkv("joint_critic_ggn_quadratic", pi_info['joint_critic_ggn_quadratic'])
                logger.logkv("joint_sampled_critic_quadratic", pi_info['joint_sampled_critic_quadratic'])
                logger.logkv("joint_predicted_step_kl", pi_info['joint_predicted_step_kl'])
                logger.logkv("joint_rhs_mode", pi_info['joint_rhs_mode'])
                logger.logkv("joint_reconstruction_mode", pi_info['joint_reconstruction_mode'])
                logger.logkv("joint_critic_score_scale", pi_info['joint_critic_score_scale'])
                logger.logkv("critic_score_noise_mean", pi_info['critic_score_noise_mean'])
                logger.logkv("critic_score_noise_std", pi_info['critic_score_noise_std'])
                logger.logkv("critic_score_noise_second_moment", pi_info['critic_score_noise_second_moment'])
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
                writer.add_scalar('train/joint_critic_curvature_coef', pi_info['joint_critic_curvature_coef'], epoch)
                writer.add_scalar('train/joint_critic_objective_coef', pi_info['joint_critic_objective_coef'], epoch)
                writer.add_scalar('train/joint_direction_l2', pi_info['joint_direction_l2'], epoch)
                writer.add_scalar('train/joint_clip_scale', pi_info['joint_clip_scale'], epoch)
                writer.add_scalar('train/joint_actor_fisher_quadratic', pi_info['joint_actor_fisher_quadratic'], epoch)
                writer.add_scalar('train/joint_critic_ggn_quadratic', pi_info['joint_critic_ggn_quadratic'], epoch)
                writer.add_scalar('train/joint_predicted_step_kl', pi_info['joint_predicted_step_kl'], epoch)
                writer.add_scalar('train/categorical_fisher_trace', pi_info['categorical_fisher_trace'], epoch)
                writer.add_scalar('train/critic_noise_ess', pi_info['critic_noise_ess'], epoch)
                writer.add_scalar('train/normalized_cross_block', pi_info['normalized_cross_block'], epoch)
                if pi_info['causal_diagnostic_ran']:
                    writer.add_scalar('train/metric_actor_critic_cosine', pi_info['metric_actor_critic_cosine'], epoch)
                    writer.add_scalar('train/critic_induced_actor_quadratic', pi_info['critic_induced_actor_quadratic'], epoch)
                    writer.add_scalar('train/actor_gain_from_critic_rhs', pi_info['actor_gain_from_critic_rhs'], epoch)
            writer.add_scalar('train/loss_pi', mb_loss_pi, epoch)
            writer.add_scalar('train/loss_v', mb_loss_v, epoch)
            writer.add_scalar('train/lr', ac_optimizer.param_groups[0]['lr'], epoch)
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
        if metric_trace_file is not None:
            metric_trace_file.flush()
            metric_trace_file.close()
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
            log_dir = f"logs/shared.{algo}.causal_{getattr(algo_config, 'joint_ablation_mode', 'full_joint')}.{getattr(algo_config, 'joint_critic_score_mode', 'clean')}.{getattr(algo_config, 'joint_critic_param_scope', 'all')}.dmlp{getattr(nets_config, 'decision_hidden_size', 0)}.solve_{getattr(algo_config, 'linear_solve_dtype', 'float32')}.{nets_config.type}{'_bn' if nets_config.with_bn else ''}.dropout_{nets_config.dropout}.lambdac_{getattr(algo_config, 'joint_critic_curvature_coef', 1.0)}.cC_{getattr(algo_config, 'joint_critic_objective_coef', 1.0)}.klmode_{getattr(algo_config, 'adaptive_kl_mode', 'procgen_rollout')}.damping_{algo_config.cg_damping}.lr_{algo_config.lr}/{env_config.env_name}.{time_now}_{seed}"

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
    parser.add_argument('--joint_critic_curvature_coef', type=float, default=None)
    parser.add_argument('--joint_critic_objective_coef', type=float, default=None)
    parser.add_argument('--joint_ablation_mode', type=str, default=None,
                        choices=['actor_only', 'curvature_only', 'full_joint'])
    parser.add_argument('--joint_critic_score_mode', type=str, default=None,
                        choices=['clean', 'rademacher', 'gaussian_unit'])
    parser.add_argument('--joint_critic_param_scope', type=str, default=None,
                        choices=['all', 'head_only'])
    parser.add_argument(
        '--joint_critic_reconstruction_scope',
        type=str,
        default=None,
        choices=['all', 'head_only'],
    )
    parser.add_argument('--adaptive_lr_max', type=float, default=None)
    parser.add_argument('--adaptive_kl_upper', type=float, default=None)
    parser.add_argument('--total_timesteps', type=int, default=None)
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

    if args.joint_critic_curvature_coef is not None:
        algo_config.joint_critic_curvature_coef = args.joint_critic_curvature_coef
    if args.joint_critic_objective_coef is not None:
        algo_config.joint_critic_objective_coef = args.joint_critic_objective_coef
    if args.joint_ablation_mode is not None:
        algo_config.joint_ablation_mode = args.joint_ablation_mode
    if args.joint_critic_score_mode is not None:
        algo_config.joint_critic_score_mode = args.joint_critic_score_mode
    if args.joint_critic_param_scope is not None:
        algo_config.joint_critic_param_scope = args.joint_critic_param_scope
    if args.joint_critic_reconstruction_scope is not None:
        algo_config.joint_critic_reconstruction_scope = (
            args.joint_critic_reconstruction_scope
        )
    if args.adaptive_lr_max is not None:
        algo_config.adaptive_lr_max = args.adaptive_lr_max
    if args.adaptive_kl_upper is not None:
        algo_config.adaptive_kl_upper = args.adaptive_kl_upper
    if args.total_timesteps is not None:
        env_config.timesteps_per_proc_easy = args.total_timesteps

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
