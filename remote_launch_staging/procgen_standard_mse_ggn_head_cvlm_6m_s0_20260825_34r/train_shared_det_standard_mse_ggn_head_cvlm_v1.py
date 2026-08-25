from procgen import ProcgenEnv

import os
import copy
import yaml
import time
import math
import types
import json
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

from utils.runners import Runner, sf01
from utils.utils import model_step
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from utils.utils import build_cnn, build_resnet, build_mlp
from utils.utils import SharedActorCritic, count_vars, safemean, set_seed
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

def solve_head_critic_b_fp64(rows, rhs, denominator, damping, chunk_cols, jacobi_eps):
    """Solve the independent critic-exclusive value-head B-by-B system."""
    if rows.shape[0] != rhs.shape[0]:
        raise ValueError('head critic row/RHS mismatch')
    kernel64 = chunked_gram_fp64(rows, denominator, chunk_cols)
    rhs64 = rhs.to(dtype=torch.float64)
    damping64 = torch.as_tensor(float(damping), device=rows.device, dtype=torch.float64)
    system64 = kernel64 + damping64 * torch.eye(
        rows.shape[0], device=rows.device, dtype=torch.float64
    )
    jacobi64 = torch.rsqrt(torch.diagonal(system64).clamp_min(float(jacobi_eps)))
    equilibrated64 = jacobi64[:, None] * system64 * jacobi64[None, :]
    equilibrated_rhs64 = jacobi64 * rhs64
    chol64, info = torch.linalg.cholesky_ex(equilibrated64, check_errors=False)
    if torch.any(info != 0):
        raise torch.linalg.LinAlgError(
            f'head critic B system is not SPD; info={int(info.max().item())}'
        )
    y64 = torch.cholesky_solve(equilibrated_rhs64[:, None], chol64).squeeze(1)
    alpha64 = jacobi64 * y64
    residual64 = torch.mv(system64, alpha64) - rhs64
    residual_norm64 = torch.linalg.vector_norm(residual64)
    relative64 = residual_norm64 / (torch.linalg.vector_norm(rhs64) + 1e-30)
    return alpha64, kernel64, system64, jacobi64, info, residual_norm64, relative64


def solve_standard_mse_head_fp64(J, error, damping, jacobi_eps):
    """Solve the standard D=I, W=I value-head MSE GGN in FP64."""
    if J.ndim != 2 or error.ndim != 1 or J.shape[0] != error.shape[0]:
        raise ValueError('standard MSE head J/error shape mismatch')
    if J.shape[0] != 512 or J.shape[1] != 257:
        raise ValueError(f'standard MSE head requires 512x257 J, got {tuple(J.shape)}')
    J64 = J.to(torch.float64)
    error64 = error.to(torch.float64)
    denominator = float(J.shape[0])
    g64 = J64.t() @ error64 / denominator
    G64 = J64.t() @ J64 / denominator
    system64 = G64 + float(damping) * torch.eye(
        J.shape[1], device=J.device, dtype=torch.float64
    )
    rhs64 = -g64
    jacobi64 = torch.rsqrt(torch.diagonal(system64).clamp_min(float(jacobi_eps)))
    equilibrated64 = jacobi64[:, None] * system64 * jacobi64[None, :]
    equilibrated_rhs64 = jacobi64 * rhs64
    chol64, info = torch.linalg.cholesky_ex(equilibrated64, check_errors=False)
    if torch.any(info != 0):
        raise torch.linalg.LinAlgError(
            f'standard MSE head primal system is not SPD; info={int(info.max().item())}'
        )
    solved64 = torch.cholesky_solve(equilibrated_rhs64[:, None], chol64).squeeze(1)
    direction64 = jacobi64 * solved64
    residual64 = system64 @ direction64 - rhs64
    residual_norm64 = torch.linalg.vector_norm(residual64)
    relative64 = residual_norm64 / (torch.linalg.vector_norm(rhs64) + 1e-30)
    return direction64, G64, g64, system64, rhs64, jacobi64, info, residual_norm64, relative64


def capture_trial_state(actor_critic, optimizer):
    """Capture every mutable training item required by deterministic LM rollback."""
    return {
        'model': {name: value.detach().clone() for name, value in actor_critic.state_dict().items()},
        'grads': {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in actor_critic.named_parameters()
        },
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'torch_rng': torch.get_rng_state().clone(),
        'cuda_rng': [state.clone() for state in torch.cuda.get_rng_state_all()],
        'numpy_rng': copy.deepcopy(np.random.get_state()),
    }


def restore_trial_state(actor_critic, optimizer, snapshot):
    actor_critic.load_state_dict(snapshot['model'], strict=True)
    for name, parameter in actor_critic.named_parameters():
        expected = snapshot['grads'][name]
        parameter.grad = None if expected is None else expected.detach().clone()
    optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
    torch.set_rng_state(snapshot['torch_rng'])
    torch.cuda.set_rng_state_all(snapshot['cuda_rng'])
    np.random.set_state(snapshot['numpy_rng'])


def assert_trial_state_equal(actor_critic, optimizer, snapshot):
    current = actor_critic.state_dict()
    for name, expected in snapshot['model'].items():
        if not torch.equal(current[name], expected):
            raise AssertionError(f'LM rollback changed model state: {name}')
    for name, parameter in actor_critic.named_parameters():
        expected = snapshot['grads'][name]
        if expected is None:
            if parameter.grad is not None:
                raise AssertionError(f'LM rollback created gradient: {name}')
        elif parameter.grad is None or not torch.equal(parameter.grad, expected):
            raise AssertionError(f'LM rollback changed gradient: {name}')
    restored_optimizer = optimizer.state_dict()
    if restored_optimizer['param_groups'] != snapshot['optimizer']['param_groups']:
        raise AssertionError('LM rollback changed optimizer parameter groups')
    for key, expected_state in snapshot['optimizer']['state'].items():
        actual_state = restored_optimizer['state'][key]
        if set(actual_state) != set(expected_state):
            raise AssertionError('LM rollback changed optimizer state keys')
        for field, expected in expected_state.items():
            actual = actual_state[field]
            if torch.is_tensor(expected):
                if not torch.equal(actual, expected):
                    raise AssertionError(f'LM rollback changed optimizer tensor: {key}/{field}')
            elif actual != expected:
                raise AssertionError(f'LM rollback changed optimizer scalar: {key}/{field}')
    if not torch.equal(torch.get_rng_state(), snapshot['torch_rng']):
        raise AssertionError('LM rollback changed CPU RNG')
    for actual, expected in zip(torch.cuda.get_rng_state_all(), snapshot['cuda_rng']):
        if not torch.equal(actual, expected):
            raise AssertionError('LM rollback changed CUDA RNG')
    actual_numpy = np.random.get_state()
    expected_numpy = snapshot['numpy_rng']
    if actual_numpy[0] != expected_numpy[0] or not np.array_equal(actual_numpy[1], expected_numpy[1]) or actual_numpy[2:] != expected_numpy[2:]:
        raise AssertionError('LM rollback changed NumPy RNG')


class GAEMetadataRunner(Runner):
    """Byte-semantic Runner copy that additionally retains the exact GAE masks."""
    def run(self):
        mb_obs, mb_rewards, mb_values, mb_actions, mb_dones, mb_logits = [], [], [], [], [], []
        epinfos = []
        if self.model.obs_rms is not None:
            self.model.obs_rms.training = False
        for _ in range(self.nsteps):
            actions, values, logits = model_step(self.model, self.obs, deterministic=self.test_mode)
            mb_obs.append(self.obs.clone())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_logits.append(logits)
            mb_dones.append(self.dones)
            clipped_actions = actions
            if not self.model.is_discrete:
                clipped_actions = torch.tanh(actions) * self.env.action_space.high[0]
            self.obs, rewards, self.dones, infos = self.env.step(clipped_actions.cpu().numpy())
            for index, done in enumerate(self.dones):
                if done and infos[index].get('TimeLimit.truncated', False):
                    terminal_obs = infos[index]['terminal_observation'].unsqueeze(0)
                    rewards[index] += self.gamma * model_step(self.model, terminal_obs)[1][0]
            for info in infos:
                maybe_epinfo = info.get('episode')
                if maybe_epinfo:
                    epinfos.append(maybe_epinfo)
            mb_rewards.append(rewards)
        if self.test_mode:
            return epinfos
        mb_obs = torch.stack(mb_obs, dim=0)
        mb_rewards = torch.from_numpy(np.asarray(mb_rewards)).to(self.device)
        mb_actions = torch.stack(mb_actions, dim=0)
        mb_values = torch.stack(mb_values, dim=0)
        mb_logits = torch.stack(mb_logits, dim=0)
        mb_dones = torch.from_numpy(np.asarray(mb_dones).astype(np.float32)).to(self.device)
        last_values = model_step(self.model, self.obs)[1]
        mb_returns = torch.zeros_like(mb_rewards)
        mb_advs = torch.zeros_like(mb_rewards)
        mb_td_res = torch.zeros_like(mb_rewards)
        mb_next_nonterminal = torch.zeros_like(mb_rewards)
        lastgaelam = 0
        for time_index in reversed(range(self.nsteps)):
            if time_index == self.nsteps - 1:
                nextnonterminal = 1.0 - torch.from_numpy(
                    self.dones.astype(np.float32)
                ).to(self.device)
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[time_index + 1]
                nextvalues = mb_values[time_index + 1]
            mb_next_nonterminal[time_index] = nextnonterminal
            mb_td_res[time_index] = (
                mb_rewards[time_index] + self.gamma * nextvalues * nextnonterminal
                - mb_values[time_index]
            )
            lastgaelam = (
                mb_td_res[time_index]
                + self.gamma * self.lam * nextnonterminal * lastgaelam
            )
            mb_advs[time_index] = lastgaelam
        mb_returns = mb_advs + mb_values
        if self.adv_type == 'td':
            mb_advs = mb_td_res
        else:
            assert self.adv_type == 'gae'
        self.last_gae_next_nonterminal = sf01(mb_next_nonterminal).detach()
        self.last_gae_td_residual = sf01(mb_td_res).detach()
        return (*map(sf01, (mb_obs, mb_returns, mb_actions, mb_advs, mb_logits)), epinfos)


def parameter_partition(actor_critic):
    """Partition by module ownership, preserving global named-parameter order."""
    named = [(name, p) for name, p in actor_critic.named_parameters() if p.requires_grad]
    policy_ids = {id(p) for p in actor_critic.pi_head.parameters() if p.requires_grad}
    if getattr(actor_critic, 'shared_sigma', None) is not None:
        policy_ids.add(id(actor_critic.shared_sigma))
    critic_ids = {id(p) for p in actor_critic.last_v_layer.parameters() if p.requires_grad}
    if policy_ids & critic_ids:
        raise AssertionError('policy and critic module ownership overlap')
    groups = {'POLICY_EXCLUSIVE': [], 'SHARED': [], 'CRITIC_EXCLUSIVE': []}
    for name, p in named:
        if id(p) in policy_ids:
            groups['POLICY_EXCLUSIVE'].append((name, p))
        elif id(p) in critic_ids:
            groups['CRITIC_EXCLUSIVE'].append((name, p))
        else:
            groups['SHARED'].append((name, p))
    flattened = [id(p) for entries in groups.values() for _, p in entries]
    if len(flattened) != len(named) or len(set(flattened)) != len(named):
        raise AssertionError('parameter partition is not exhaustive and exclusive')
    if not all(groups.values()):
        raise AssertionError('all three parameter groups must be nonempty')
    return groups


def partition_manifest(actor_critic, probe_obs):
    """Prove module ownership with policy/value autograd connectivity tests."""
    groups = parameter_partition(actor_critic)
    values, logits = actor_critic(probe_obs)
    policy_cotangent = torch.arange(
        1, logits.numel() + 1, device=logits.device, dtype=logits.dtype
    ).reshape_as(logits)
    value_cotangent = torch.arange(
        1, values.numel() + 1, device=values.device, dtype=values.dtype
    ).reshape_as(values)
    all_entries = [entry for entries in groups.values() for entry in entries]
    all_params = [p for _, p in all_entries]
    policy_grads = torch.autograd.grad(
        logits, all_params, grad_outputs=policy_cotangent,
        retain_graph=True, allow_unused=True
    )
    value_grads = torch.autograd.grad(
        values, all_params, grad_outputs=value_cotangent,
        retain_graph=False, allow_unused=True
    )
    connectivity = {
        name: dict(
            policy_connected=pg is not None,
            value_connected=vg is not None,
            policy_jacobian_probe_l2=0.0 if pg is None else float(torch.linalg.vector_norm(pg).item()),
            value_jacobian_probe_l2=0.0 if vg is None else float(torch.linalg.vector_norm(vg).item()),
        )
        for (name, _), pg, vg in zip(all_entries, policy_grads, value_grads)
    }
    for name, _ in groups['CRITIC_EXCLUSIVE']:
        if connectivity[name]['policy_connected']:
            raise AssertionError(f'critic-exclusive parameter reaches policy logits: {name}')
        if not connectivity[name]['value_connected']:
            raise AssertionError(f'critic-exclusive parameter misses value output: {name}')
    for name, _ in groups['POLICY_EXCLUSIVE']:
        if not connectivity[name]['policy_connected'] or connectivity[name]['value_connected']:
            raise AssertionError(f'policy-exclusive connectivity mismatch: {name}')
    for name, _ in groups['SHARED']:
        if not connectivity[name]['policy_connected'] or not connectivity[name]['value_connected']:
            raise AssertionError(f'shared connectivity mismatch: {name}')
    manifest = {}
    for group, entries in groups.items():
        manifest[group] = dict(
            names=[name for name, _ in entries],
            tensors=len(entries),
            numel=sum(p.numel() for _, p in entries),
            connectivity={name: connectivity[name] for name, _ in entries},
        )
    frozen_stats = [
        name for name, p in actor_critic.last_v_layer.named_parameters()
        if not p.requires_grad
    ]
    manifest['POPART_NONCURVATURE_STATE'] = frozen_stats
    return groups, manifest


def validate_standard_mse_cvlm_config(algo_config):
    """Reject every mechanism outside the fixed standard-MSE CVLM head replacement."""
    forbidden_exact = {'adaptive_kl_mode', 'optimizer_momentum', 'is_kaczmarz'}
    forbidden = [name for name in vars(algo_config) if (
        name in forbidden_exact or name.startswith('joint_') or
        'low_fisher' in name or 'cross' in name or 'kaczmarz' in name or
        'projection' in name or 'shared_ggn' in name
    )]
    if forbidden:
        raise ValueError(f'forbidden coupled/P1/guard/projection fields: {sorted(forbidden)}')
    required = {
        'optimizer': 'sgd', 'lr': 0.5, 'epochs': 4, 'minibatches': 8,
        'use_kl_adaptive_lr': True, 'cg_damping': 0.5,
        'max_grad_norm': 0.5, 'ent_coef': 0.0,
        'cvlm_alpha_init': 1.0, 'cvlm_alpha_min': 2.0 ** -20,
        'cvlm_alpha_max': 2.0 ** 20, 'cvlm_max_trials': 4,
        'cvlm_accept_lower': 0.25, 'cvlm_accept_upper': 0.75,
    }
    for name, expected in required.items():
        actual = getattr(algo_config, name, None)
        if actual != expected:
            raise ValueError(f'{name} must be {expected!r}, got {actual!r}')


def flat_group(entries, tensors):
    return torch.cat([
        (tensor if tensor is not None else torch.zeros_like(parameter)).reshape(-1)
        for (_, parameter), tensor in zip(entries, tensors)
    ])


def categorical_kl_from_logits(before, after):
    log_before = F.log_softmax(before, dim=-1)
    log_after = F.log_softmax(after, dim=-1)
    return (torch.exp(log_before) * (log_before - log_after)).sum(dim=-1).mean()

def learn(world_size, algo, actor_critic, writer, venv, device,
          total_timesteps, nsteps, algo_config, log_config, log_dir=None):

    gamma = .999
    lam = .95
    validate_standard_mse_cvlm_config(algo_config)

    per_epoch_timesteps = nsteps * venv.num_envs
    epochs = total_timesteps // per_epoch_timesteps + 1

    minibatch_size = per_epoch_timesteps // algo_config.minibatches

    # Instantiate the runner object
    runner = GAEMetadataRunner(env=venv, model=actor_critic, nsteps=nsteps, gamma=gamma, lam=lam, adv_type=algo_config.adv_type, device=device)
    epinfobuf = deque(maxlen=100)

    dict_params = {k: v.detach() for k, v in actor_critic.named_parameters() if v.requires_grad}
    dict_buffers = {k: v.detach() for k, v in actor_critic.named_buffers() if v.requires_grad}
    trainable_named = [(name, p) for name, p in actor_critic.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in trainable_named]
    parameter_groups = parameter_partition(actor_critic)
    policy_names = [name for name, _ in parameter_groups['POLICY_EXCLUSIVE']]
    shared_names = [name for name, _ in parameter_groups['SHARED']]
    head_names = [name for name, _ in parameter_groups['CRITIC_EXCLUSIVE']]
    name_to_index = {name: i for i, (name, _) in enumerate(trainable_named)}
    head_params_dict = {name: dict_params[name] for name in head_names}
    frozen_nonhead_params = {name: value for name, value in dict_params.items() if name not in head_names}
    current_rollout_obs = None
    current_rollout_ret = None
    current_mbinds = None
    current_cvinds = None
    current_cv_obs = None
    current_cv_ret = None
    cvlm_alpha = float(algo_config.cvlm_alpha_init)
    if log_dir is not None:
        _probe = torch.zeros((2,) + tuple(runner.obs.shape[1:]), device=device)
        _, _partition_evidence = partition_manifest(actor_critic, _probe)
        with open(os.path.join(log_dir, 'parameter_partition.json'), 'w') as _manifest_file:
            json.dump(_partition_evidence, _manifest_file, indent=2, sort_keys=True)

    if algo_config.optimizer == 'adam':
        ac_optimizer = Adam(actor_critic.parameters(), lr=algo_config.lr, weight_decay=algo_config.weight_decay)
    elif algo_config.optimizer == 'sgd':
        # momentum is enabled to facilitate the implementation of adv-normalized SGD
        # gradient update does not use momentum
        ac_optimizer = SGD(actor_critic.parameters(), lr=algo_config.lr, momentum=1e-6)
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
        nonlocal cvlm_alpha
        raw_gae_adv = _adv.detach().clone()
        _vals, _outputs = actor_critic(_obs)

        if actor_critic.is_discrete:
            _logp_full = F.log_softmax(_outputs, dim=-1)
            _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
            _llr = torch.gather(_logp_full - _logp_full_old, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _ratio = torch.exp(_llr)
            _p_log_p = torch.exp(_logp_full) * _logp_full
            _entropy = - _p_log_p.sum(-1).mean()
            _logp = torch.gather(_logp_full, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)

            def compute_logp(params, buffers, batch_obs, batch_act):
                batch_obs, batch_act = batch_obs.unsqueeze(0), batch_act.unsqueeze(0)
                batch_vals, batch_outs = functional_call(actor_critic, (params, buffers), (batch_obs,) )
                batch_logp_full = F.log_softmax(batch_outs, dim=-1)
                pi_logp = torch.gather(batch_logp_full, dim=-1, index=batch_act.unsqueeze(-1)).squeeze(1).squeeze(0)

                batch_vals_noise = torch.randn(batch_vals.size(), device=device)
                sample_vals = batch_vals + batch_vals_noise
                vf_logp = -(batch_vals - sample_vals.detach()).pow(2).squeeze(0) # likelihood for value function

                all_logp = pi_logp + vf_logp
                return all_logp

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

            def compute_logp(params, buffers, batch_obs, batch_act):
                batch_obs, batch_act = batch_obs.unsqueeze(0), batch_act.unsqueeze(0)
                batch_vals, batch_outs = functional_call(actor_critic, (params, buffers), (batch_obs,) )
                batch_mu, batch_logstd = batch_outs.chunk(2, dim=-1)

                var = torch.exp(batch_logstd)**2
                batch_logp = (
                    -((batch_act - batch_mu) ** 2) / (2 * var)
                    - batch_logstd
                    - math.log(math.sqrt(2 * math.pi))
                )
                pi_logp = batch_logp.sum(dim=-1).squeeze(0)

                batch_vals_noise = torch.randn(batch_vals.size(), device=device)
                sample_vals = batch_vals + batch_vals_noise
                vf_logp = -(batch_vals - sample_vals.detach()).pow(2).squeeze(0) # likelihood for value function
                all_logp = pi_logp + vf_logp
                return all_logp

        # zero mean of advantage
        _adv = _adv - _adv.mean()

        # clamp the ratio
        if algo_config.clamp_ratio:
            _ratio = torch.clamp(_ratio, algo_config.min_ratio, algo_config.max_ratio)

        if algo_config.norm_obj == 'adv':
            _rms_sqrt = torch.sqrt( _adv.pow(2).mean() ).detach()
        elif algo_config.norm_obj == 'obj':
            _rms_sqrt = torch.sqrt( (_ratio * _adv).pow(2).mean() ).detach() # might related to variance reduction in importance sampling
        elif algo_config.norm_obj == 'ratio':
            _rms_sqrt = _ratio.mean().detach() * torch.sqrt( _adv.pow(2).mean() ).detach()
        else:
            raise NotImplementedError
        _adv = _adv / (_rms_sqrt + 1e-8)

        ac_optimizer.zero_grad()
        ft_compute_sample_grad = vmap(grad(compute_logp), in_dims=(None, None, 0, 0), randomness='different')
        ft_per_sample_grads = ft_compute_sample_grad(dict_params, dict_buffers, _obs, _act) # num_samples x param_shape

        if current_rollout_obs is None or current_rollout_ret is None:
            raise RuntimeError('frozen lambda-return rollout context is unavailable')
        if any(item is None for item in (current_mbinds, current_cvinds, current_cv_obs, current_cv_ret)):
            raise RuntimeError('cross-minibatch LM context is unavailable')
        if _obs.shape[0] != 512 or current_cv_obs.shape[0] != 512:
            raise RuntimeError('CVLM requires two complete 512-row minibatches')
        if torch.isin(current_mbinds, current_cvinds).any():
            raise RuntimeError('CVLM train and validation minibatches overlap')

        with torch.no_grad():
            num_sa = _obs.shape[0]
            # Exact original Paper combined sampled policy/value score matrix.
            H = torch.cat([v.view(num_sa, -1) for v in ft_per_sample_grads.values()], dim=-1)
            HHT = H @ H.t() / num_sa
            _pseudo_adv = torch.ones_like(_adv)

            gk_list = [v['momentum_buffer'].flatten() for v in ac_optimizer.state.values() if v['momentum_buffer'] is not None]
            history_correction_applied = False
            if len(gk_list) > 0:
                g_k = torch.cat(gk_list, dim=0)
                _adv = _adv - torch.mv(H, g_k)
                _pseudo_adv = _pseudo_adv - torch.mv(H, g_k)
                history_correction_applied = True

            # Exact original Paper actor and sampled-critic dual directions.
            actor_system = HHT @ torch.diag(_ratio) + algo_config.cg_damping * torch.eye(num_sa, device=device)
            paper_critic_system = HHT + algo_config.cg_damping * torch.eye(num_sa, device=device)
            _png_adv = torch.mv(torch.inverse(actor_system), _adv)
            _critic_adv = torch.mv(torch.inverse(paper_critic_system), _pseudo_adv)
            actor_relative_residual = torch.linalg.vector_norm(actor_system @ _png_adv - _adv) / (torch.linalg.vector_norm(_adv) + 1e-30)
            paper_critic_relative_residual = torch.linalg.vector_norm(paper_critic_system @ _critic_adv - _pseudo_adv) / (torch.linalg.vector_norm(_pseudo_adv) + 1e-30)

            # Sole replacement: ordinary frozen lambda-return MSE on the
            # critic-exclusive linear value head, with D=I, W=I and K=J.
            train_latents = actor_critic.backbone_net(_obs).detach()
            validation_latents = actor_critic.backbone_net(current_cv_obs).detach()
            train_values = actor_critic.forward_v(latents=train_latents).detach()
            validation_values = actor_critic.forward_v(latents=validation_latents).detach()
            train_J = torch.cat((
                train_latents,
                torch.ones((num_sa, 1), device=device, dtype=train_latents.dtype),
            ), dim=-1)
            validation_J = torch.cat((
                validation_latents,
                torch.ones((num_sa, 1), device=device, dtype=validation_latents.dtype),
            ), dim=-1)
            if train_J.shape != (512, 257) or validation_J.shape != (512, 257):
                raise RuntimeError('value-head Jacobian must have exactly 257 columns')
            train_error = (train_values - _ret).detach()
            validation_error = (validation_values - current_cv_ret).detach()
            train_J64 = train_J.to(torch.float64)
            validation_J64 = validation_J.to(torch.float64)
            train_error64 = train_error.to(torch.float64)
            validation_error64 = validation_error.to(torch.float64)
            train_G64 = train_J64.t() @ train_J64 / float(num_sa)
            validation_G64 = validation_J64.t() @ validation_J64 / float(num_sa)
            train_g64 = train_J64.t() @ train_error64 / float(num_sa)
            validation_g64 = validation_J64.t() @ validation_error64 / float(num_sa)
            train_loss_before64 = 0.5 * train_error64.pow(2).mean()
            validation_loss_before64 = 0.5 * validation_error64.pow(2).mean()
            head_momentum = []
            head_history_present = False
            for _, parameter in parameter_groups['CRITIC_EXCLUSIVE']:
                state = ac_optimizer.state.get(parameter, {})
                head_history_present = head_history_present or ('momentum_buffer' in state)
                head_momentum.append(state.get('momentum_buffer', torch.zeros_like(parameter)).flatten())
            head_momentum_flat = torch.cat(head_momentum)
            head_momentum_flat = head_momentum_flat.to(torch.float64)
            head_history_projection = torch.mv(train_J64, head_momentum_flat)
            spectrum = torch.linalg.eigvalsh(train_G64).clamp_min(0.0)
            spectrum_max = spectrum.max()
            spectrum_positive = spectrum[spectrum > spectrum_max * 1e-10]
            spectrum_min_positive = spectrum_positive.min() if spectrum_positive.numel() else torch.zeros((), device=device, dtype=torch.float64)
            effective_rank = (spectrum.sum().pow(2) / (spectrum.pow(2).sum() + 1e-30))
            head_trace_mean = torch.trace(train_G64) / 257.0

        # Compose exact Paper actor and sampled critic first.
        _loss_pi = (- _ratio * _png_adv).mean()
        _loss_v = ((_vals - _ret).pow(2) * _critic_adv).mean()
        _actor_objective = _loss_pi - algo_config.ent_coef * _entropy
        _paper_critic_objective = algo_config.vf_coef * _loss_v
        actor_grads = torch.autograd.grad(
            _actor_objective, trainable_params, retain_graph=True, allow_unused=True
        )
        paper_critic_grads = torch.autograd.grad(
            _paper_critic_objective, trainable_params, retain_graph=False, allow_unused=True
        )
        paper_full_grads = [
            (ag if ag is not None else torch.zeros_like(p)) +
            (cg if cg is not None else torch.zeros_like(p))
            for p, ag, cg in zip(trainable_params, actor_grads, paper_critic_grads)
        ]
        actor_direction_l2 = torch.linalg.vector_norm(torch.cat([
            (g if g is not None else torch.zeros_like(p)).flatten()
            for p, g in zip(trainable_params, actor_grads)
        ]))
        paper_shared_critic_l2 = torch.linalg.vector_norm(torch.cat([
            (paper_critic_grads[name_to_index[name]] if paper_critic_grads[name_to_index[name]] is not None else torch.zeros_like(p)).flatten()
            for name, p in parameter_groups['SHARED']
        ]))

        before_params = {name: p.detach().clone() for name, p in trainable_named}
        ac_optimizer.zero_grad()
        for parameter, gradient in zip(trainable_params, paper_full_grads):
            parameter.grad = gradient.detach().clone()
        # Preserve Paper's global clipping coefficient for every unchanged
        # policy/shared delta. Only afterward replace the critic-head gradient.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_critic.parameters(), algo_config.max_grad_norm
        )
        paper_clip_scale = torch.clamp(
            torch.as_tensor(float(algo_config.max_grad_norm), device=grad_norm.device, dtype=grad_norm.dtype) /
            (grad_norm + 1e-6), max=1.0
        )
        # Deterministic cross-minibatch LM.  The complete train minibatch is
        # the sole source of G, g and every proposal.  The disjoint next
        # minibatch only accepts/rejects that already-formed actual delta.
        cvlm_trials = []
        accepted = False
        chosen_direction64 = torch.zeros(257, device=device, dtype=torch.float64)
        chosen_delta64 = torch.zeros_like(chosen_direction64)
        chosen_pred64 = torch.zeros((), device=device, dtype=torch.float64)
        chosen_ared_train64 = torch.zeros_like(chosen_pred64)
        chosen_ared_validation64 = torch.zeros_like(chosen_pred64)
        chosen_rho64 = torch.full_like(chosen_pred64, float('-inf'))
        alpha_before_minibatch = cvlm_alpha
        for trial_index in range(int(algo_config.cvlm_max_trials)):
            alpha_trial = cvlm_alpha
            mu = alpha_trial * max(
                float(head_trace_mean.item()), float(torch.finfo(torch.float64).eps)
            )
            (
                candidate_direction64, candidate_G64, candidate_g64,
                head_system64, head_primal_rhs64, head_jacobi64,
                head_chol_info, head_residual_norm64, head_relative64,
            ) = solve_standard_mse_head_fp64(
                train_J, train_error, mu, algo_config.dual_jacobi_eps
            )
            current_lr = float(ac_optimizer.param_groups[0]['lr'])
            candidate_delta64 = current_lr * (
                paper_clip_scale.to(torch.float64) * candidate_direction64
                - 1e-6 * head_momentum_flat
            )

            snapshot = capture_trial_state(actor_critic, ac_optimizer)
            offset = 0
            with torch.no_grad():
                for _, parameter in parameter_groups['CRITIC_EXCLUSIVE']:
                    count = parameter.numel()
                    parameter.add_(candidate_delta64[offset:offset + count].view_as(parameter).to(parameter.dtype))
                    offset += count
                if offset != candidate_delta64.numel():
                    raise ValueError('CVLM candidate head delta length mismatch')
                applied_delta64 = torch.cat([
                    (
                        parameter.detach()
                        - snapshot['model'][name].to(parameter.device)
                    ).reshape(-1).to(torch.float64)
                    for name, parameter in parameter_groups['CRITIC_EXCLUSIVE']
                ])
                pred_train64 = -(
                    candidate_g64 @ applied_delta64
                    + 0.5 * applied_delta64 @ candidate_G64 @ applied_delta64
                )
                # Evaluate the temporarily applied linear-head change in
                # exact FP64 frozen-feature coordinates.
                trial_train_error64 = train_error64 + train_J64 @ applied_delta64
                trial_validation_error64 = validation_error64 + validation_J64 @ applied_delta64
                trial_train_loss64 = 0.5 * trial_train_error64.pow(2).mean()
                trial_validation_loss64 = 0.5 * trial_validation_error64.pow(2).mean()
                ared_train64 = train_loss_before64 - trial_train_loss64
                ared_validation64 = validation_loss_before64 - trial_validation_loss64
            restore_trial_state(actor_critic, ac_optimizer, snapshot)
            assert_trial_state_equal(actor_critic, ac_optimizer, snapshot)
            candidate_delta64 = applied_delta64

            train_identity_error64 = torch.abs(ared_train64 - pred_train64)
            train_identity_tolerance64 = 5e-11 * (
                1.0 + torch.abs(ared_train64) + torch.abs(pred_train64)
            )
            if train_identity_error64 > train_identity_tolerance64:
                raise AssertionError('same-minibatch MSE ared/pred identity failed')
            finite_pred = bool(torch.isfinite(pred_train64).item())
            rho64 = ared_validation64 / pred_train64 if finite_pred and pred_train64 > 0 else torch.full_like(pred_train64, float('-inf'))
            finite_rho = bool(torch.isfinite(rho64).item())
            # Preserve the last bounded trial telemetry even when all four
            # proposals reject.  The committed head delta remains exactly
            # zero in that case; avoid emitting NaN/Inf into scientific logs.
            chosen_pred64 = pred_train64 if finite_pred else torch.zeros_like(pred_train64)
            chosen_ared_train64 = ared_train64 if torch.isfinite(ared_train64) else torch.zeros_like(ared_train64)
            chosen_ared_validation64 = ared_validation64 if torch.isfinite(ared_validation64) else torch.zeros_like(ared_validation64)
            chosen_rho64 = rho64 if finite_rho else torch.full_like(rho64, -1e300)
            accept_trial = bool(
                finite_pred and pred_train64 > 0 and finite_rho
                and rho64 >= float(algo_config.cvlm_accept_lower)
            )
            cvlm_trials.append({
                'trial': trial_index + 1,
                'alpha': alpha_trial,
                'mu': mu,
                'pred_train': float(pred_train64.item()),
                'ared_train': float(ared_train64.item()),
                'ared_validation': float(ared_validation64.item()),
                'rho_cv': float(rho64.item()) if finite_rho else -1e300,
                'accepted': accept_trial,
                'rollback_bit_identical': True,
            })
            if accept_trial:
                accepted = True
                chosen_direction64 = candidate_direction64
                chosen_delta64 = candidate_delta64
                if rho64 > float(algo_config.cvlm_accept_upper):
                    cvlm_alpha = alpha_trial / 2.0
                break
            cvlm_alpha = min(
                float(algo_config.cvlm_alpha_max),
                max(float(algo_config.cvlm_alpha_min), alpha_trial * 4.0),
            )
        cvlm_alpha = min(
            float(algo_config.cvlm_alpha_max),
            max(float(algo_config.cvlm_alpha_min), cvlm_alpha),
        )
        head_direction64 = chosen_direction64
        head_direction = head_direction64.to(train_J.dtype)
        head_direction_l2 = torch.linalg.vector_norm(head_direction64)
        head_projection64 = train_J64 @ chosen_delta64
        validation_prediction_change64 = validation_J64 @ chosen_delta64
        head_ggn_quadratic = chosen_delta64 @ train_G64 @ chosen_delta64
        condition_number = spectrum_max / (spectrum_min_positive + 1e-30)
        offset = 0
        for _, parameter in parameter_groups['CRITIC_EXCLUSIVE']:
            count = parameter.numel()
            state = ac_optimizer.state.get(parameter, {})
            old_momentum = state.get('momentum_buffer', torch.zeros_like(parameter))
            if accepted:
                # Let the unchanged SGD momentum/history chain produce the
                # exact candidate delta validated above.
                parameter.grad = (
                    -head_direction[offset:offset + count].view_as(parameter)
                    * paper_clip_scale
                ).to(parameter.dtype)
            else:
                # Four rejected trials commit the actor/shared control update
                # once while making the critic-head delta exactly zero.
                parameter.grad = (-1e-6 * old_momentum).to(parameter.dtype)
            offset += count
        if offset != head_direction.numel():
            raise ValueError('head direction length mismatch')

        _loss = _actor_objective + _paper_critic_objective
        ac_optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Useful extra info
        with torch.no_grad():
            _, _outputs_after = actor_critic(_obs)

            if actor_critic.is_discrete:
                _logp_full_after = F.log_softmax(_outputs_after, dim=-1)
                _curr_kl = (torch.exp(_logp_full) * (_logp_full - _logp_full_after)).sum(dim=-1).mean()
                _real_kl = (torch.exp(_logp_full_old) * (_logp_full_old - _logp_full_after)).sum(dim=-1).mean()
            else:
                _mu_after, _logstd_after = _outputs_after.chunk(2, dim=-1)
                _curr_kl = (_logstd_after - _logstd + 0.5 * ( torch.exp(_logstd).pow(2) + (_mu - _mu_after).pow(2) ) / torch.exp(_logstd_after).pow(2) - 0.5).sum(dim=-1).mean()
                _real_kl = (_logstd_after - _logstd_old + 0.5 * ( torch.exp(_logstd_old).pow(2) + (_mu_old - _mu_after).pow(2) ) / torch.exp(_logstd_after).pow(2) - 0.5).sum(dim=-1).mean()

            clipfrac = 0.0
            pi_info = dict(kl=_real_kl.item(), curr_kl=_curr_kl.item(), curr_lr=ac_optimizer.param_groups[0]['lr'], ent=_entropy.item(), cf=clipfrac,
                           grad_norm=grad_norm.item(), ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())
            after_params = {name: p.detach().clone() for name, p in trainable_named}
            policy_delta = torch.cat([(after_params[name] - before_params[name]).flatten() for name in policy_names])
            shared_delta = torch.cat([(after_params[name] - before_params[name]).flatten() for name in shared_names])
            head_delta = torch.cat([(after_params[name] - before_params[name]).flatten() for name in head_names])
            # Head-old and head-new policy logits must be exactly equal because
            # the changed head has zero policy-output Jacobian.
            post_shared_params = dict(after_params)
            for name in head_names:
                post_shared_params[name] = before_params[name]
            _, post_shared_logits = functional_call(actor_critic, (post_shared_params, dict_buffers), (_obs,))
            _, post_head_logits = functional_call(actor_critic, (after_params, dict_buffers), (_obs,))
            head_policy_logit_max_abs = (post_shared_logits - post_head_logits).abs().max()
            head_policy_kl = categorical_kl_from_logits(post_shared_logits, post_head_logits) if actor_critic.is_discrete else torch.zeros((), device=device)
            realized_head_delta64 = head_delta.to(torch.float64)
            head_delta_match_error64 = torch.linalg.vector_norm(realized_head_delta64 - chosen_delta64)
            popart_mean, popart_var = actor_critic.last_v_layer.debiased_mean_var()
            pi_info.update(
                parameter_partition='policy_exclusive_shared_critic_exclusive',
                actor_system_rows=num_sa,
                actor_kernel_mode='paper_sampled_policy_value_score',
                actor_relative_solve_residual=actor_relative_residual.item(),
                actor_direction_l2=actor_direction_l2.item(),
                paper_shared_critic_direction_l2=paper_shared_critic_l2.item(),
                paper_critic_relative_solve_residual=paper_critic_relative_residual.item(),
                head_critic_system_rows=num_sa,
                head_critic_kernel_mode='standard_mse_D_I_W_I_K_J_primal_cvlm',
                head_gaussian_precision=1.0,
                head_train_rows=int(train_J.shape[0]),
                head_validation_rows=int(validation_J.shape[0]),
                head_train_validation_overlap=0,
                gae_mean=raw_gae_adv.mean().item(),
                gae_variance=raw_gae_adv.var(unbiased=False).item(),
                gae_rms=torch.sqrt(raw_gae_adv.pow(2).mean()).item(),
                td_residual_mean=runner.last_gae_td_residual[current_mbinds].mean().item(),
                td_residual_rms=torch.sqrt(runner.last_gae_td_residual[current_mbinds].pow(2).mean()).item(),
                return_error_mean=train_error.mean().item(),
                return_error_rms=torch.sqrt(train_error.pow(2).mean()).item(),
                standard_mse_train_before=train_loss_before64.item(),
                standard_mse_validation_before=validation_loss_before64.item(),
                head_critic_linear_solve_dtype=str(torch.float64),
                head_critic_solver_mode='primal_symmetric_jacobi_cholesky_fp64',
                head_critic_cholesky_info_max=head_chol_info.max().item(),
                head_critic_jacobi_scale_min=head_jacobi64.min().item(),
                head_critic_jacobi_scale_max=head_jacobi64.max().item(),
                head_critic_solve_residual=head_residual_norm64.item(),
                head_critic_relative_solve_residual=head_relative64.item(),
                head_critic_direction_l2=head_direction_l2.item(),
                head_critic_ggn_quadratic=head_ggn_quadratic.item(),
                head_ggn_spectrum_min=spectrum.min().item(),
                head_ggn_spectrum_max=spectrum_max.item(),
                head_ggn_trace=torch.trace(train_G64).item(),
                head_ggn_trace_per_parameter=head_trace_mean.item(),
                head_ggn_effective_rank=effective_rank.item(),
                head_ggn_condition_number=condition_number.item(),
                head_prediction_change_l2=torch.linalg.vector_norm(head_projection64).item(),
                head_validation_prediction_change_l2=torch.linalg.vector_norm(validation_prediction_change64).item(),
                head_cvlm_alpha_before=alpha_before_minibatch,
                head_cvlm_alpha_after=cvlm_alpha,
                head_cvlm_mu=cvlm_trials[-1]['mu'],
                head_cvlm_trials=len(cvlm_trials),
                head_cvlm_accepted=float(accepted),
                head_cvlm_decision='ACCEPT' if accepted else 'REJECT_ZERO_HEAD_DELTA',
                head_cvlm_pred_train=chosen_pred64.item(),
                head_cvlm_ared_train=chosen_ared_train64.item(),
                head_cvlm_ared_validation=chosen_ared_validation64.item(),
                head_cvlm_rho_cv=chosen_rho64.item(),
                head_cvlm_same_minibatch_identity_error=(chosen_ared_train64 - chosen_pred64).abs().item(),
                head_cvlm_trial_ledger=json.dumps(cvlm_trials, sort_keys=True),
                paper_clip_scale=paper_clip_scale.item(),
                policy_parameter_delta_l2=torch.linalg.vector_norm(policy_delta).item(),
                shared_parameter_delta_l2=torch.linalg.vector_norm(shared_delta).item(),
                head_parameter_delta_l2=torch.linalg.vector_norm(head_delta).item(),
                head_parameter_delta_match_error=head_delta_match_error64.item(),
                shared_to_actor_update_ratio=(torch.linalg.vector_norm(shared_delta) / (actor_direction_l2 + 1e-30)).item(),
                head_to_actor_update_ratio=(torch.linalg.vector_norm(head_delta) / (actor_direction_l2 + 1e-30)).item(),
                post_shared_to_post_head_policy_logit_max_abs=head_policy_logit_max_abs.item(),
                post_shared_to_post_head_policy_kl=head_policy_kl.item(),
                optimizer_momentum=1e-6,
                optimizer_history_correction=float(head_history_present),
                head_history_projection_l2=torch.linalg.vector_norm(head_history_projection).item(),
                adaptive_kl_timing='per_minibatch',
                popart_mean=popart_mean.item(),
                popart_std=torch.sqrt(popart_var).item(),
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

    adaptive_kl_update_count = 0
    minibatch_update_count = 0

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

        current_rollout_obs = obs
        current_rollout_ret = ret

        actor_critic.train()  # set to train mode
        for _ in range(algo_config.epochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            minibatch_blocks = [
                inds[start:start + minibatch_size].copy()
                for start in range(0, per_epoch_timesteps, minibatch_size)
            ]
            if len(minibatch_blocks) != 8 or any(len(block) != 512 for block in minibatch_blocks):
                raise RuntimeError('CVLM requires exactly eight complete 512-row minibatches')
            if np.unique(np.concatenate(minibatch_blocks)).size != per_epoch_timesteps:
                raise RuntimeError('CVLM shuffled schedule is not exhaustive and disjoint')
            for block_index, mbinds in enumerate(minibatch_blocks):
                cvinds = minibatch_blocks[(block_index + 1) % len(minibatch_blocks)]
                current_mbinds = torch.as_tensor(mbinds, device=device, dtype=torch.long)
                current_cvinds = torch.as_tensor(cvinds, device=device, dtype=torch.long)
                current_cv_obs = obs[cvinds]
                current_cv_ret = ret[cvinds]
                mb_obs, mb_act, mb_adv, mb_ret, mb_outputs_old = obs[mbinds], act[mbinds], adv[mbinds], ret[mbinds], outputs_old[mbinds]
                ac_optimizer.zero_grad()
                mb_loss, mb_loss_pi, mb_loss_v, pi_info = update_actor_critic(mb_obs, mb_act, mb_adv, mb_ret, mb_outputs_old)

                # kl adaptive lr adjustment
                if algo_config.use_kl_adaptive_lr:
                    curr_kl = pi_info['kl']
                    if curr_kl > 0.02 * 2:
                        ac_optimizer.param_groups[0]['lr'] = max(ac_optimizer.param_groups[0]['lr'] / 1.5, 1e-4)
                    elif curr_kl < 0.01 / 2:
                        ac_optimizer.param_groups[0]['lr'] = min(ac_optimizer.param_groups[0]['lr'] * 1.5, algo_config.lr)
                minibatch_update_count += 1
                if algo_config.use_kl_adaptive_lr:
                    adaptive_kl_update_count += 1
                pi_info['adaptive_kl_update_count'] = adaptive_kl_update_count
                pi_info['minibatch_update_count'] = minibatch_update_count
                if log_dir is not None:
                    _trace_row = dict(pi_info)
                    _trace_row.update(transitions=(epoch+1)*per_epoch_timesteps, loss_pi=float(mb_loss_pi.item()), loss_v=float(mb_loss_v.item()))
                    with open(os.path.join(log_dir, 'metric_trace.jsonl'), 'a') as _trace_file:
                        _trace_file.write(json.dumps(_trace_row, sort_keys=True) + '\n')

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
            for _key, _value in pi_info.items():
                if _key.startswith(('actor_', 'paper_', 'head_', 'policy_', 'shared_', 'post_', 'optimizer_')):
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
