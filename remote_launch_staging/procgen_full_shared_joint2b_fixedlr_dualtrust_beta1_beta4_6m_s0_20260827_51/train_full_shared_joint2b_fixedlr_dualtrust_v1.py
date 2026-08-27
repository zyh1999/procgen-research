from procgen import ProcgenEnv

import os
import yaml
import time
import math
import types
import json
import hashlib
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

def chunked_gram_fp64(rows, denominator, chunk_cols):
    """Form rows @ rows.T in FP64 without materializing a full FP64 row matrix."""
    if rows.ndim != 2:
        raise ValueError(f'rows must be rank-2, got shape={tuple(rows.shape)}')
    if chunk_cols <= 0:
        raise ValueError(f'chunk_cols must be positive, got {chunk_cols}')
    result = torch.zeros(
        (rows.shape[0], rows.shape[0]),
        device=rows.device,
        dtype=torch.float64,
    )
    alpha = 1.0 / float(denominator)
    for start in range(0, rows.shape[1], chunk_cols):
        stop = min(start + chunk_cols, rows.shape[1])
        block = rows[:, start:stop].to(dtype=torch.float64)
        result.addmm_(block, block.t(), beta=1.0, alpha=alpha)
        del block
    return result


def chunked_transpose_mv_fp64(rows, coefficients, denominator, chunk_cols):
    """Compute rows.T @ coefficients in FP64, returning one FP32 parameter vector."""
    if rows.ndim != 2 or coefficients.ndim != 1:
        raise ValueError('rows must be rank-2 and coefficients rank-1')
    if rows.shape[0] != coefficients.shape[0]:
        raise ValueError('row/coefficient size mismatch')
    output_chunks = []
    scale = 1.0 / float(denominator)
    coefficients64 = coefficients.to(dtype=torch.float64)
    for start in range(0, rows.shape[1], chunk_cols):
        stop = min(start + chunk_cols, rows.shape[1])
        block = rows[:, start:stop].to(dtype=torch.float64)
        direction_block = torch.mv(block.t(), coefficients64) * scale
        output_chunks.append(direction_block.to(dtype=rows.dtype))
        del block, direction_block
    return torch.cat(output_chunks, dim=0)


def dualtrust_update_value(current_eta, measured_divergence, lower, upper,
                           eta_min, eta_max, multiplier):
    """Update one positive metric multiplier exactly once per rollout."""
    if measured_divergence > upper:
        return min(current_eta * multiplier, eta_max), 'ABOVE_0P04_MULTIPLY_1P5'
    if measured_divergence < lower:
        return max(current_eta / multiplier, eta_min), 'BELOW_0P005_DIVIDE_1P5'
    return current_eta, 'IN_BAND_UNCHANGED'

def validate_paper_matched_config(algo_config):
    """Reject actor-side P1 controls and enforce the frozen Paper identity."""
    forbidden = ('adaptive_kl_mode', 'optimizer_momentum', 'is_kaczmarz')
    present = [name for name in forbidden if hasattr(algo_config, name)]
    if present:
        raise ValueError(f'forbidden P1 actor fields: {present}')
    required = {
        'optimizer': 'sgd', 'lr': 0.004, 'epochs': 4, 'minibatches': 8,
        'use_kl_adaptive_lr': True, 'cg_damping': 0.5,
        'max_grad_norm': 0.5, 'joint_critic_objective_coef': 1.0,
    }
    for name, expected in required.items():
        actual = getattr(algo_config, name, None)
        if actual != expected:
            raise ValueError(f'{name} must be {expected!r}, got {actual!r}')
    ppo_required = {
        'ppo_warmup_transitions': 503_808,
        'ppo_optimizer': 'adam',
        'ppo_lr': 0.001,
        'ppo_cliprange': 0.2,
        'ppo_epochs': 4,
        'ppo_minibatches': 8,
        'ppo_vf_coef': 1.0,
        'ppo_ent_coef': 0.0,
        'ppo_max_grad_norm': 0.5,
        'dualtrust_lower': 0.005,
        'dualtrust_upper': 0.04,
        'dualtrust_eta_min': 1.0 / 64.0,
        'dualtrust_eta_max': 64.0,
        'dualtrust_multiplier': 1.5,
        'eta_pi_init': 1.0,
        'eta_v_init': 1.0,
    }
    if getattr(algo_config, 'gate_mode', False):
        ppo_required['ppo_warmup_transitions'] = 4_096
    for name, expected in ppo_required.items():
        actual = getattr(algo_config, name, None)
        if actual != expected:
            raise ValueError(f'{name} must be {expected!r}, got {actual!r}')
    beta_v = float(getattr(algo_config, 'beta_v', float('nan')))
    if beta_v not in (1.0, 4.0):
        raise ValueError(f'beta_v must be exactly 1 or 4, got {beta_v!r}')

def learn(world_size, algo, actor_critic, writer, venv, device,
          total_timesteps, nsteps, algo_config, log_config, log_dir=None):

    gamma = .999
    lam = .95
    validate_paper_matched_config(algo_config)

    per_epoch_timesteps = nsteps * venv.num_envs
    epochs = total_timesteps // per_epoch_timesteps + 1

    minibatch_size = per_epoch_timesteps // algo_config.minibatches

    # Instantiate the runner object
    runner = Runner(env=venv, model=actor_critic, nsteps=nsteps, gamma=gamma, lam=lam, adv_type=algo_config.adv_type, device=device)
    epinfobuf = deque(maxlen=100)

    dict_params = {k: v.detach() for k, v in actor_critic.named_parameters() if v.requires_grad}
    dict_buffers = {k: v.detach() for k, v in actor_critic.named_buffers() if v.requires_grad}
    trainable_named_params = [
        (name, p) for name, p in actor_critic.named_parameters()
        if p.requires_grad
    ]
    trainable_params = [p for _, p in trainable_named_params]

    if algo_config.optimizer != 'sgd':
        raise ValueError('Task51 Joint-2B path must use the frozen SGD optimizer')

    # Task51 creates the Joint optimizer only at the fixed phase boundary. It
    # cannot inherit PPO Adam moments or Task50's rollout LR scheduler.
    ac_optimizer = None

    # The PPO warmup owns independent Adam state.  The Joint-2B SGD optimizer
    # does not exist until the one fixed rollout-boundary switch.
    ppo_optimizer = Adam(
        actor_critic.parameters(),
        lr=algo_config.ppo_lr,
        weight_decay=0.0,
    )

    lr_scheduler = None

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

        _clip_adv = torch.clamp(
            _ratio,
            1-algo_config.ppo_cliprange,
            1+algo_config.ppo_cliprange,
        ) * _adv
        _losses_pi = torch.max(- _ratio * _adv, - _clip_adv)
        _loss_pi = _losses_pi.mean()

        # value loss
        _loss_v = F.mse_loss(_vals, _ret)

        # total loss
        _loss = (
            _loss_pi
            - algo_config.ppo_ent_coef * _entropy
            + algo_config.ppo_vf_coef * _loss_v
        )

        _loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_critic.parameters(), algo_config.ppo_max_grad_norm
        )
        ppo_optimizer.step()

        # Useful extra info
        with torch.no_grad():
            approx_kl = _kl.item()
            ent = _entropy.item()
            clipped = (
                _ratio.gt(1+algo_config.ppo_cliprange)
                | _ratio.lt(1-algo_config.ppo_cliprange)
            )
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac, curr_lr=ppo_optimizer.param_groups[0]['lr'],
                           grad_norm=grad_norm.item(), ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())

        return _loss, _loss_pi, _loss_v, pi_info

    def Advantage_Update(_obs, _act, _adv, _ret, _outputs_old):
        """Exact deterministic shared actor/regression-GGN joint solve.

        The stacked feature matrix is ``H=[A; sqrt(lambda_C) J_V]``.  Therefore
        ``H.T H`` is exactly ``A.T A + lambda_C J_V.T J_V``: it is the
        conditional expectation of the sampled critic-score curvature before
        inversion.  The 2B-by-2B dual contains AC/CA blocks because the actor
        and critic share parameters; these blocks do not introduce cross terms
        in the primal curvature. Actor rows use rollout ratios and critic rows
        use unit weights. The dual is solved through the algebraically exact
        symmetric similarity transform sqrt(D) K sqrt(D), with FP64 Gram,
        damping, equilibration, solve, and direction reconstruction.
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

        # Keep the paper Procgen advantage normalization.
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
        parameter_columns = sum(parameter.numel() for parameter in trainable_params)
        if num_sa != 512 or H_pi.shape != (512, parameter_columns) or J_v.shape != (512, parameter_columns):
            raise RuntimeError(
                f'Task51 strict Joint-2B shape drift: actor={tuple(H_pi.shape)}, '
                f'critic={tuple(J_v.shape)}, P={parameter_columns}'
            )
        if parameter_columns != 938_976:
            raise RuntimeError(f'Task51 production parameter coverage drift: {parameter_columns}')

        with torch.no_grad():
            critic_residual = (_ret - _vals).detach()

            # Keep the critic objective coefficient and critic curvature
            # coefficient mathematically independent:
            #   H_C = sqrt(lambda_C) J_C
            #   b_C = c_C / sqrt(lambda_C) e_C
            # This sweep changes lambda_C only. It does not change c_C, the
            # ordinary value-loss coefficient, or the Tikhonov damping.
            beta_v = float(algo_config.beta_v)
            lambda_pi = float(eta_pi)
            lambda_v = beta_v * float(eta_v)
            critic_objective_coef = float(
                getattr(algo_config, 'joint_critic_objective_coef', 1.0)
            )
            if not all(math.isfinite(value) and value > 0.0 for value in (
                beta_v, lambda_pi, lambda_v,
            )):
                raise ValueError('dual-trust metric multipliers must be finite and positive')
            if not math.isfinite(critic_objective_coef):
                raise ValueError('joint_critic_objective_coef must be finite')

            actor_h_weight = math.sqrt(lambda_pi)
            actor_rhs_weight = 1.0 / actor_h_weight
            critic_h_weight = math.sqrt(lambda_v)
            critic_rhs_weight = critic_objective_coef / critic_h_weight

            # Stacked joint kernel. The critic block uses unit weights; only
            # actor columns carry the rollout importance ratio.
            actor_raw_rows = H_pi
            critic_raw_rows = J_v
            joint_H = torch.cat([
                actor_h_weight * actor_raw_rows,
                critic_h_weight * critic_raw_rows,
            ], dim=0)
            joint_rhs = torch.cat(
                [actor_rhs_weight * _adv, critic_rhs_weight * critic_residual],
                dim=0,
            )
            critic_ratio = torch.ones_like(critic_residual)
            joint_ratio = torch.cat([_ratio.detach(), critic_ratio], dim=0)

            previous_momentum_buffer = None
            previous_projection = torch.zeros_like(joint_rhs)
            history_correction_applied = False
            if True:
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
                    previous_projection = torch.mv(
                        joint_H, previous_momentum_buffer
                    )

            # Exact original Paper history correction, now applied to the
            # stacked actor and deterministic-critic rows.
            rhs_eff = joint_rhs
            if previous_momentum_buffer is not None:
                rhs_eff = rhs_eff - previous_projection.to(rhs_eff.dtype)
                history_correction_applied = True

            kernel_denom = float(num_sa)
            fp64_chunk_cols = int(getattr(
                algo_config, 'fp64_gram_chunk_cols', 32768
            ))
            if fp64_chunk_cols <= 0:
                raise ValueError('fp64_gram_chunk_cols must be positive')

            # The old "fp64 solve" first formed K and added damping in FP32.
            # Here every numerically sensitive operation starts in FP64.  The
            # chunked implementation avoids a full 2B-by-P FP64 copy.
            joint_K64 = chunked_gram_fp64(
                joint_H, kernel_denom, fp64_chunk_cols
            )
            actor_raw_scale = actor_raw_rows.pow(2).sum() / kernel_denom
            critic_raw_scale = critic_raw_rows.pow(2).sum() / kernel_denom
            actor_rhs_norm = torch.linalg.vector_norm(_adv)
            critic_rhs_norm = torch.linalg.vector_norm(critic_residual)
            raw_cross64 = chunked_gram_fp64(
                torch.cat([actor_raw_rows, critic_raw_rows], dim=0),
                kernel_denom,
                fp64_chunk_cols,
            )[:num_sa, num_sa:]
            actor_critic_cross_frobenius = torch.linalg.matrix_norm(
                raw_cross64, ord='fro'
            )
            critic_actor_cross_frobenius = torch.linalg.matrix_norm(
                raw_cross64.t(), ord='fro'
            )
            if not (
                actor_critic_cross_frobenius > 0
                and critic_actor_cross_frobenius > 0
            ):
                raise RuntimeError('Task51 natural actor-critic cross blocks vanished')
            ratio64 = joint_ratio.to(dtype=torch.float64)
            if not torch.all(torch.isfinite(ratio64)) or torch.any(ratio64 <= 0):
                raise FloatingPointError('joint_ratio must be finite and positive')
            sqrt_ratio64 = torch.sqrt(ratio64)
            rhs64 = rhs_eff.to(dtype=torch.float64)
            damping64 = torch.as_tensor(
                float(algo_config.cg_damping),
                device=device,
                dtype=torch.float64,
            )
            eye64 = torch.eye(
                joint_K64.shape[0], device=device, dtype=torch.float64
            )

            # (K D + mu I) alpha = b is similar to the symmetric system
            # (sqrt(D) K sqrt(D) + mu I) beta = sqrt(D) b,
            # with beta=sqrt(D) alpha. This changes neither alpha nor the
            # reconstructed primal direction in exact arithmetic.
            symmetric_system64 = (
                sqrt_ratio64[:, None]
                * joint_K64
                * sqrt_ratio64[None, :]
                + damping64 * eye64
            )
            symmetric_rhs64 = sqrt_ratio64 * rhs64

            # Symmetric Jacobi congruence is another exact change of variables:
            # solve S A S y = S r, then beta = S y. It only improves scaling.
            jacobi_eps = float(getattr(
                algo_config, 'dual_jacobi_eps', 1e-18
            ))
            if jacobi_eps <= 0.0:
                raise ValueError('dual_jacobi_eps must be positive')
            jacobi_scale64 = torch.rsqrt(
                torch.diagonal(symmetric_system64).clamp_min(jacobi_eps)
            )
            equilibrated_system64 = (
                jacobi_scale64[:, None]
                * symmetric_system64
                * jacobi_scale64[None, :]
            )
            equilibrated_rhs64 = jacobi_scale64 * symmetric_rhs64
            chol64, chol_info = torch.linalg.cholesky_ex(
                equilibrated_system64, check_errors=False
            )
            if torch.any(chol_info != 0):
                raise torch.linalg.LinAlgError(
                    'equilibrated deterministic 2B system is not SPD; '
                    f'cholesky_info_max={int(chol_info.max().item())}'
                )
            y64 = torch.cholesky_solve(
                equilibrated_rhs64.unsqueeze(1), chol64
            ).squeeze(1)
            beta64 = jacobi_scale64 * y64
            alpha64 = beta64 / sqrt_ratio64

            solve_residual = torch.linalg.vector_norm(
                torch.mv(symmetric_system64, beta64) - symmetric_rhs64
            )
            original_residual64 = torch.linalg.vector_norm(
                torch.mv(joint_K64, ratio64 * alpha64)
                + damping64 * alpha64
                - rhs64
            )
            relative_solve_residual = original_residual64 / (
                torch.linalg.vector_norm(rhs64) + 1e-30
            )
            weighted_alpha64 = ratio64 * alpha64
            flat_dir = chunked_transpose_mv_fp64(
                joint_H,
                weighted_alpha64,
                kernel_denom,
                fp64_chunk_cols,
            )
            applied_solve_residual = original_residual64
            direction_l2 = torch.linalg.vector_norm(flat_dir)
            actor_projection = torch.mv(actor_raw_rows, flat_dir)
            critic_projection = torch.mv(critic_raw_rows, flat_dir)
            actor_fisher_quadratic = actor_projection.pow(2).mean()
            critic_ggn_quadratic = critic_projection.pow(2).mean()
            clip_scale = torch.clamp(
                torch.as_tensor(
                    float(algo_config.max_grad_norm),
                    device=direction_l2.device,
                    dtype=direction_l2.dtype,
                ) / (direction_l2 + 1e-12),
                max=1.0,
            )
            step_scale = float(ac_optimizer.param_groups[0]['lr']) * clip_scale
            predicted_step_kl = (
                0.5 * step_scale.pow(2) * actor_fisher_quadratic
            )
            predicted_critic_divergence = (
                0.5 * beta_v * step_scale.pow(2) * critic_ggn_quadratic
            )
            finite_scan = all(torch.all(torch.isfinite(value)).item() for value in (
                actor_raw_scale,
                critic_raw_scale,
                actor_critic_cross_frobenius,
                critic_actor_cross_frobenius,
                solve_residual,
                relative_solve_residual,
                direction_l2,
                actor_fisher_quadratic,
                critic_ggn_quadratic,
                predicted_step_kl,
                predicted_critic_divergence,
            ))
            if not finite_scan:
                raise FloatingPointError('Task51 Joint-2B finite scan failed')
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

        params_before = [p.detach().clone() for p in trainable_params]
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

        if lr_scheduler is not None:
            lr_scheduler.step()

        with torch.no_grad():
            _vals_after, _outputs_after = actor_critic(_obs)
            _logp_full_after = F.log_softmax(_outputs_after, dim=-1)
            _curr_kl = (
                torch.exp(_logp_full)
                * (_logp_full - _logp_full_after)
            ).sum(dim=-1).mean()
            _real_kl = (
                torch.exp(_logp_full_old)
                * (_logp_full_old - _logp_full_after)
            ).sum(dim=-1).mean()
            parameter_deltas = [
                (name, (p.detach() - before).reshape(-1))
                for (name, p), before in zip(trainable_named_params, params_before)
            ]
            def _block_delta_norm(predicate):
                pieces = [delta for name, delta in parameter_deltas if predicate(name)]
                return torch.linalg.vector_norm(torch.cat(pieces)).item()
            policy_delta = _block_delta_norm(lambda name: name.startswith('pi_head.'))
            value_delta = _block_delta_norm(lambda name: name.startswith('last_v_layer.'))
            shared_delta = _block_delta_norm(
                lambda name: not name.startswith(('pi_head.', 'last_v_layer.'))
            )
            normalized_value_prediction_change_rms = torch.sqrt(
                (_vals_after - _vals).pow(2).mean()
            )
            if actor_critic.with_popart:
                _, popart_var = actor_critic.last_v_layer.debiased_mean_var()
                popart_std = torch.sqrt(popart_var)
            else:
                popart_std = torch.ones((), device=_vals.device)
            raw_value_prediction_change_rms = (
                normalized_value_prediction_change_rms * popart_std.reshape(-1)[0]
            )

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
                critic_rows=num_sa,
                joint_system_rows=2 * num_sa,
                joint_parameter_columns=parameter_columns,
                joint_kernel_mode='exact_deterministic_ggn_2b',
                optimizer_momentum=1e-6,
                optimizer_momentum_semantics='paper_sgd_momentum_1e-6',
                optimizer_history_correction=float(history_correction_applied),
                kaczmarz_rhs_semantics='paper_rhs_minus_H_buffer',
                kaczmarz_previous_projection_l2=(
                    torch.linalg.vector_norm(previous_projection).item()
                ),
                joint_solve_residual=solve_residual.item(),
                joint_applied_solve_residual=applied_solve_residual.item(),
                joint_relative_solve_residual=relative_solve_residual.item(),
                joint_solver_mode='symmetrized_jacobi_fullfp64_chunked',
                joint_cholesky_info_max=chol_info.max().item(),
                joint_jacobi_scale_min=jacobi_scale64.min().item(),
                joint_jacobi_scale_max=jacobi_scale64.max().item(),
                joint_linear_solve_dtype=str(torch.float64),
                beta_v=beta_v,
                eta_pi=eta_pi,
                eta_v=eta_v,
                lambda_pi=lambda_pi,
                lambda_v=lambda_v,
                joint_critic_objective_coef=critic_objective_coef,
                joint_direction_l2=direction_l2.item(),
                joint_clip_scale=clip_scale.item(),
                joint_actor_fisher_quadratic=actor_fisher_quadratic.item(),
                joint_critic_ggn_quadratic=critic_ggn_quadratic.item(),
                joint_predicted_step_kl=predicted_step_kl.item(),
                joint_predicted_critic_divergence=predicted_critic_divergence.item(),
                joint_actor_raw_scale=actor_raw_scale.item(),
                joint_critic_raw_scale=critic_raw_scale.item(),
                joint_actor_rhs_norm=actor_rhs_norm.item(),
                joint_critic_rhs_norm=critic_rhs_norm.item(),
                joint_actor_critic_cross_frobenius=actor_critic_cross_frobenius.item(),
                joint_critic_actor_cross_frobenius=critic_actor_cross_frobenius.item(),
                joint_cross_block_frobenius=actor_critic_cross_frobenius.item(),
                joint_policy_delta_l2=policy_delta,
                joint_shared_delta_l2=shared_delta,
                joint_value_delta_l2=value_delta,
                joint_normalized_value_prediction_change_rms=normalized_value_prediction_change_rms.item(),
                joint_raw_value_prediction_change_rms=raw_value_prediction_change_rms.item(),
                joint_popart_std=popart_std.reshape(-1)[0].item(),
                joint_finite_scan_pass=float(finite_scan),
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

    if algo != 'adv':
        raise ValueError('Task51 must preserve the Task50/Task06 Joint-2B parent algo')

    tepochs = trange(epochs+1, desc='Epoch starts', leave=True)

    # Main loop: collect experience in env and update/log each epoch
    inds = np.arange(per_epoch_timesteps)

    rollout_scheduler_update_count = 0
    actor_trust_update_count = 0
    critic_trust_update_count = 0
    eta_pi = float(algo_config.eta_pi_init)
    eta_v = float(algo_config.eta_v_init)
    minibatch_update_count = 0
    phase_switch_count = 0
    previous_phase = None

    for epoch in tepochs:
        tstart = time.perf_counter()
        rollout_transitions = (epoch + 1) * per_epoch_timesteps
        warmup_active = rollout_transitions <= algo_config.ppo_warmup_transitions
        phase = 'ppo_warmup' if warmup_active else 'joint2b'
        if previous_phase is None:
            previous_phase = phase
        elif phase != previous_phase:
            if previous_phase != 'ppo_warmup' or phase != 'joint2b':
                raise RuntimeError(f'invalid Task51 phase transition {previous_phase}->{phase}')
            phase_switch_count += 1
            if ac_optimizer is not None:
                raise RuntimeError('Joint-2B optimizer existed before the fixed switch')
            ac_optimizer = SGD(
                actor_critic.parameters(),
                lr=algo_config.lr,
                momentum=1e-6,
            )
            joint_state_entries = sum(len(state) for state in ac_optimizer.state.values())
            if joint_state_entries != 0:
                raise RuntimeError('Joint-2B optimizer state was not clean at switch')
            if log_dir is not None:
                switch_row = {
                    'event': 'PPO_WARMUP_TO_JOINT2B_SWITCH',
                    'switch_count': phase_switch_count,
                    'previous_phase': previous_phase,
                    'phase': phase,
                    'last_ppo_transition': algo_config.ppo_warmup_transitions,
                    'first_joint_rollout_transition': rollout_transitions,
                    'joint_optimizer_state_entries_before_first_step': joint_state_entries,
                    'joint_lr_at_switch': ac_optimizer.param_groups[0]['lr'],
                    'ppo_optimizer_state_tensors': len(ppo_optimizer.state),
                }
                with open(os.path.join(log_dir, 'phase_switch.jsonl'), 'a') as switch_file:
                    switch_file.write(json.dumps(switch_row, sort_keys=True) + '\n')
            previous_phase = phase
        update_actor_critic = PPO_Update if warmup_active else Advantage_Update
        active_optimizer = ppo_optimizer if warmup_active else ac_optimizer

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

        joint_behavior_logits = None
        joint_behavior_values = None
        joint_behavior_hash = None
        joint_behavior_value_hash = None
        joint_rollout_lr_before = None
        eta_pi_before = None
        eta_v_before = None
        popart_state_before = None
        popart_mean = None
        popart_std = None
        joint_minibatch_lrs = []
        if not warmup_active:
            if ac_optimizer is None:
                raise RuntimeError('Joint-2B optimizer missing after phase switch')
            actor_critic.eval()
            with torch.no_grad():
                joint_behavior_values, joint_behavior_logits = actor_critic(obs)
                joint_behavior_values = joint_behavior_values.detach().clone()
                joint_behavior_logits = joint_behavior_logits.detach().clone()
            joint_behavior_hash = hashlib.sha256(
                joint_behavior_logits.contiguous().cpu().numpy().tobytes()
            ).hexdigest()
            joint_behavior_value_hash = hashlib.sha256(
                joint_behavior_values.contiguous().cpu().numpy().tobytes()
            ).hexdigest()
            joint_rollout_lr_before = float(ac_optimizer.param_groups[0]['lr'])
            if joint_rollout_lr_before != 0.004:
                raise RuntimeError('Task51 Joint parameter LR must remain exactly .004')
            eta_pi_before = float(eta_pi)
            eta_v_before = float(eta_v)
            if actor_critic.with_popart:
                popart_state_before = tuple(
                    value.detach().clone() for value in (
                        actor_critic.last_v_layer.mean,
                        actor_critic.last_v_layer.mean_sq,
                        actor_critic.last_v_layer.debiasing_term,
                    )
                )
                popart_mean_tensor, popart_var_tensor = (
                    actor_critic.last_v_layer.debiased_mean_var()
                )
                popart_mean = float(popart_mean_tensor.reshape(-1)[0].item())
                popart_std = float(torch.sqrt(popart_var_tensor).reshape(-1)[0].item())

        actor_critic.train()  # set to train mode
        update_epochs = algo_config.ppo_epochs if warmup_active else algo_config.epochs
        if update_epochs != 4:
            raise RuntimeError(f'phase epochs drifted: {update_epochs}')
        for _ in range(update_epochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, per_epoch_timesteps, minibatch_size):
                end = start + minibatch_size
                mbinds = inds[start:end]
                mb_obs, mb_act, mb_adv, mb_ret, mb_outputs_old = obs[mbinds], act[mbinds], adv[mbinds], ret[mbinds], outputs_old[mbinds]
                active_optimizer.zero_grad()
                if not warmup_active:
                    mb_lr = float(ac_optimizer.param_groups[0]['lr'])
                    if mb_lr != 0.004 or mb_lr != joint_rollout_lr_before:
                        raise RuntimeError('Joint LR changed inside a rollout')
                    joint_minibatch_lrs.append(mb_lr)
                mb_loss, mb_loss_pi, mb_loss_v, pi_info = update_actor_critic(mb_obs, mb_act, mb_adv, mb_ret, mb_outputs_old)
                minibatch_update_count += 1
                pi_info['training_phase'] = phase
                pi_info['ppo_warmup_active'] = float(warmup_active)
                pi_info['ppo_warmup_boundary'] = algo_config.ppo_warmup_transitions
                pi_info['phase_switch_count'] = phase_switch_count
                pi_info['joint_optimizer_state_entries'] = (
                    0 if ac_optimizer is None else
                    sum(len(state) for state in ac_optimizer.state.values())
                )
                pi_info['ppo_optimizer_state_tensors'] = len(ppo_optimizer.state)
                pi_info['adaptive_kl_update_count'] = 0
                pi_info['minibatch_update_count'] = minibatch_update_count
                if log_dir is not None:
                    _trace_row = dict(pi_info)
                    _trace_row.update(transitions=(epoch+1)*per_epoch_timesteps, loss_pi=float(mb_loss_pi.item()), loss_v=float(mb_loss_v.item()))
                    with open(os.path.join(log_dir, 'metric_trace.jsonl'), 'a') as _trace_file:
                        _trace_file.write(json.dumps(_trace_row, sort_keys=True) + '\n')

        if not warmup_active:
            unique_lrs = sorted(set(joint_minibatch_lrs))
            if unique_lrs != [0.004] or joint_rollout_lr_before != 0.004:
                raise RuntimeError(f'Joint rollout LR was not constant: {unique_lrs}')
            actor_critic.eval()
            with torch.no_grad():
                final_values, final_logits = actor_critic(obs)
                behavior_logp = F.log_softmax(joint_behavior_logits, dim=-1)
                final_logp = F.log_softmax(final_logits, dim=-1)
                d_pi = (
                    torch.exp(behavior_logp) * (behavior_logp - final_logp)
                ).sum(dim=-1).mean()
                normalized_value_change = final_values - joint_behavior_values
                d_v = 0.5 * float(algo_config.beta_v) * normalized_value_change.pow(2).mean()
                normalized_value_change_rms = torch.sqrt(
                    normalized_value_change.pow(2).mean()
                )
                raw_value_change_rms = normalized_value_change_rms * popart_std
            if not torch.isfinite(d_pi) or not torch.isfinite(d_v):
                raise RuntimeError('nonfinite Task51 rollout trust measurement')
            if actor_critic.with_popart:
                popart_state_after = (
                    actor_critic.last_v_layer.mean,
                    actor_critic.last_v_layer.mean_sq,
                    actor_critic.last_v_layer.debiasing_term,
                )
                if any(not torch.equal(before, after) for before, after in zip(
                    popart_state_before, popart_state_after
                )):
                    raise RuntimeError('PopArt statistics changed inside Joint rollout')
            d_pi_value = float(d_pi.item())
            d_v_value = float(d_v.item())
            eta_pi_after, actor_reason = dualtrust_update_value(
                eta_pi_before, d_pi_value,
                algo_config.dualtrust_lower, algo_config.dualtrust_upper,
                algo_config.dualtrust_eta_min, algo_config.dualtrust_eta_max,
                algo_config.dualtrust_multiplier,
            )
            eta_v_after, critic_reason = dualtrust_update_value(
                eta_v_before, d_v_value,
                algo_config.dualtrust_lower, algo_config.dualtrust_upper,
                algo_config.dualtrust_eta_min, algo_config.dualtrust_eta_max,
                algo_config.dualtrust_multiplier,
            )
            if (
                (d_pi_value > algo_config.dualtrust_upper and eta_pi_after < eta_pi_before)
                or (d_pi_value < algo_config.dualtrust_lower and eta_pi_after > eta_pi_before)
                or (d_v_value > algo_config.dualtrust_upper and eta_v_after < eta_v_before)
                or (d_v_value < algo_config.dualtrust_lower and eta_v_after > eta_v_before)
            ):
                raise RuntimeError('Task51 dual-trust update direction is inverted')
            eta_pi = eta_pi_after
            eta_v = eta_v_after
            actor_trust_update_count += 1
            critic_trust_update_count += 1
            rollout_scheduler_update_count += 1
            if float(ac_optimizer.param_groups[0]['lr']) != 0.004:
                raise RuntimeError('Task51 Joint parameter LR changed between rollouts')
            scheduler_row = {
                'event': 'JOINT_ROLLOUT_DUALTRUST_SCHEDULER',
                'training_phase': phase,
                'rollout_index': epoch + 1,
                'rollout_start_transition': epoch * per_epoch_timesteps,
                'rollout_end_transition': rollout_transitions,
                'behavior_logits_sha256': joint_behavior_hash,
                'behavior_values_sha256': joint_behavior_value_hash,
                'fixed_parameter_lr': 0.004,
                'minibatch_lr_unique_values': unique_lrs,
                'minibatch_lr_change_count': 0,
                'between_rollout_lr_change_count': 0,
                'beta_v': float(algo_config.beta_v),
                'eta_pi_before': eta_pi_before,
                'eta_pi_after': eta_pi_after,
                'eta_v_before': eta_v_before,
                'eta_v_after': eta_v_after,
                'lambda_pi': eta_pi_before,
                'lambda_v': float(algo_config.beta_v) * eta_v_before,
                'd_pi': d_pi_value,
                'd_v': d_v_value,
                'actor_trust_reason': actor_reason,
                'critic_trust_reason': critic_reason,
                'actor_trust_update_count_this_rollout': 1,
                'critic_trust_update_count_this_rollout': 1,
                'actor_trust_update_count_total': actor_trust_update_count,
                'critic_trust_update_count_total': critic_trust_update_count,
                'rollout_scheduler_update_count_total': rollout_scheduler_update_count,
                'popart_mean': popart_mean,
                'popart_std': popart_std,
                'normalized_value_prediction_change_rms': float(normalized_value_change_rms.item()),
                'raw_value_prediction_change_rms': float(raw_value_change_rms.item()),
            }
            for key, value in pi_info.items():
                if key.startswith(('joint_', 'critic_', 'optimizer_')):
                    scheduler_row[key] = value
            pi_info.update(scheduler_row)
            if log_dir is not None:
                with open(os.path.join(log_dir, 'rollout_scheduler.jsonl'), 'a') as scheduler_file:
                    scheduler_file.write(json.dumps(scheduler_row, sort_keys=True) + '\n')
                with open(os.path.join(log_dir, 'metric_trace.jsonl'), 'a') as trace_file:
                    trace_file.write(json.dumps(scheduler_row, sort_keys=True) + '\n')

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
            logger.logkv("lr", active_optimizer.param_groups[0]['lr'])
            logger.logkv("clipfrac", pi_info['cf'])
            logger.logkv("ratio_max", pi_info['ratio_max'])
            logger.logkv("ratio_min", pi_info['ratio_min'])
            for _key, _value in pi_info.items():
                if _key.startswith(('joint_', 'critic_', 'optimizer_')):
                    logger.logkv(_key, _value)
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
            writer.add_scalar('train/loss_pi', mb_loss_pi, epoch)
            writer.add_scalar('train/loss_v', mb_loss_v, epoch)
            writer.add_scalar('train/lr', active_optimizer.param_groups[0]['lr'], epoch)
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

        # The sole production gate contains exactly one real PPO rollout and
        # one complete Joint rollout for whichever frozen beta arm is invoked.
        if getattr(algo_config, 'gate_mode', False) and not warmup_active:
            break

    if total_timesteps > algo_config.ppo_warmup_transitions and phase_switch_count != 1:
        raise RuntimeError(f'expected exactly one phase switch, got {phase_switch_count}')

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
            log_dir = f"logs/shared.{algo}.{nets_config.type}{'_bn' if nets_config.with_bn else ''}.dropout_{nets_config.dropout}.damping_{algo_config.cg_damping}.lr_{algo_config.lr}/{env_config.env_name}.{time_now}_{seed}"

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
    parser.add_argument('--gate_mode', action='store_true')

    args = parser.parse_args()

    with open(f'configs/{args.config}') as fin:
        config = yaml.safe_load(fin)

    algo = config['algo']
    algo_config = types.SimpleNamespace(**config['algo_config'])
    env_config = types.SimpleNamespace(**config['env_config'])
    nets_config = types.SimpleNamespace(**config['nets_config'])
    log_config = types.SimpleNamespace(**config['log_config'])

    if args.gate_mode:
        algo_config.gate_mode = True
        algo_config.ppo_warmup_transitions = 4_096
        env_config.timesteps_per_proc_easy = 4_096
        env_config.timesteps_per_proc_hard = 4_096
        log_config.log_interval = 1

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
