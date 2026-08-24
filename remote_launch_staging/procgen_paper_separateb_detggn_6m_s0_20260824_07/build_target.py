#!/usr/bin/env python3
"""Build the separate-B candidate deterministically from exact Paper RAT."""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT / "work/procgen_paper_2b5affd_6m_bede_20260722/source/train_shared.py"
P1 = ROOT / "deterministic_2b_symfp64_20260807/train_shared_rat_exact_deterministic_ggn_symfp64.py"
OUT = HERE / "train_shared_paper_separateb_detggn_v1.py"
PAPER_SHA = "cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7"
P1_SHA = "2b50f8cc26bb8c85f91e0b394acc7535f5564ac70036403597d3738d3dab9c1b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(PAPER) == PAPER_SHA, (PAPER, sha(PAPER))
assert sha(P1) == P1_SHA, (P1, sha(P1))
paper = PAPER.read_text()
p1 = P1.read_text()

helper_start = p1.index("def chunked_gram_fp64")
helper_end = p1.index("def learn(", helper_start)
helpers = p1[helper_start:helper_end]
helpers += '''def solve_separate_critic_b_fp64(rows, rhs, denominator, damping, chunk_cols, jacobi_eps):
    """Solve one independent deterministic critic B-by-B dual system."""
    if rows.shape[0] != rhs.shape[0]:
        raise ValueError('critic row/RHS mismatch')
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
            f'separate critic B system is not SPD; info={int(info.max().item())}'
        )
    y64 = torch.cholesky_solve(equilibrated_rhs64[:, None], chol64).squeeze(1)
    alpha64 = jacobi64 * y64
    residual64 = torch.mv(system64, alpha64) - rhs64
    residual_norm64 = torch.linalg.vector_norm(residual64)
    relative64 = residual_norm64 / (torch.linalg.vector_norm(rhs64) + 1e-30)
    return alpha64, kernel64, system64, jacobi64, info, residual_norm64, relative64


def add_flat_to_grads(parameters, flat_delta):
    """Add one flat vector to existing parameter gradients without reordering."""
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        delta = flat_delta[offset:offset + count].view_as(parameter)
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        parameter.grad.add_(delta)
        offset += count
    if offset != flat_delta.numel():
        raise ValueError('flat gradient length mismatch')


def validate_separateb_config(algo_config):
    """Enforce exact Paper actor identity and reject coupled/experimental fields."""
    forbidden_exact = {'adaptive_kl_mode', 'optimizer_momentum', 'is_kaczmarz'}
    forbidden = [name for name in vars(algo_config) if (
        name in forbidden_exact or name.startswith('joint_') or
        'low_fisher' in name or 'cross' in name or 'kaczmarz' in name
    )]
    if forbidden:
        raise ValueError(f'forbidden coupled/P1/guard fields: {sorted(forbidden)}')
    required = {
        'optimizer': 'sgd', 'lr': 0.5, 'epochs': 4, 'minibatches': 8,
        'use_kl_adaptive_lr': True, 'cg_damping': 0.5,
        'max_grad_norm': 0.5, 'critic_curvature_coef': 0.1,
        'critic_objective_coef': 1.0, 'ent_coef': 0.0,
    }
    for name, expected in required.items():
        actual = getattr(algo_config, name, None)
        if actual != expected:
            raise ValueError(f'{name} must be {expected!r}, got {actual!r}')

'''
paper = paper.replace(
    "from vec_env import ( VecExtractDictObs, VecMonitor, VecNormalize)\n\n",
    "from vec_env import ( VecExtractDictObs, VecMonitor, VecNormalize)\n\n" + helpers,
    1,
)
paper = paper.replace("import types\n", "import types\nimport json\n", 1)

learn_anchor = "    gamma = .999\n    lam = .95\n"
paper = paper.replace(
    learn_anchor,
    learn_anchor + "    validate_separateb_config(algo_config)\n",
    1,
)
anchor = "    dict_buffers = {k: v.detach() for k, v in actor_critic.named_buffers() if v.requires_grad}\n"
paper = paper.replace(
    anchor,
    anchor + "    trainable_params = [p for p in actor_critic.parameters() if p.requires_grad]\n",
    1,
)

block_start = paper.index("    def Advantage_Update(")
block_end = paper.index("    def KFAC_Update(", block_start)
block = paper[block_start:block_end]

sample_anchor = "        ft_per_sample_grads = ft_compute_sample_grad(dict_params, dict_buffers, _obs, _act) # num_samples x param_shape\n\n"
value_rows = sample_anchor + '''        def compute_value(params, buffers, batch_obs):
            batch_obs = batch_obs.unsqueeze(0)
            batch_vals, _ = functional_call(
                actor_critic, (params, buffers), (batch_obs,)
            )
            return batch_vals.reshape(-1)[0]

        ft_compute_value_grad = vmap(
            grad(compute_value), in_dims=(None, None, 0), randomness='different'
        )
        ft_value_grads = ft_compute_value_grad(dict_params, dict_buffers, _obs)

'''
assert sample_anchor in block
block = block.replace(sample_anchor, value_rows, 1)

core_start = block.index("        with torch.no_grad():\n            num_sa")
core_end = block.index("        # udpate actor", core_start)
core = '''        with torch.no_grad():
            num_sa = _obs.shape[0]
            # Literal Paper actor sampled-score rows and actor B-by-B kernel.
            H = torch.cat([v.view(num_sa, -1) for v in ft_per_sample_grads.values()], dim=-1)  # num_samples x num_params
            HHT = H @ H.t() / num_sa # num_samples x num_samples
            _pseudo_adv = torch.ones_like(_adv)

            # The only replacement: deterministic value Jacobian/residual in
            # one independent critic B-by-B system. No stacking or cross block.
            J_v = torch.cat(
                [v.contiguous().view(num_sa, -1) for v in ft_value_grads.values()],
                dim=-1,
            )
            critic_curvature_coef = float(algo_config.critic_curvature_coef)
            critic_objective_coef = float(algo_config.critic_objective_coef)
            critic_h_weight = math.sqrt(critic_curvature_coef)
            critic_rows = critic_h_weight * J_v
            critic_residual = (_ret - _vals).detach()
            critic_rhs = (
                critic_objective_coef / critic_h_weight
            ) * critic_residual

            gk_list = [ v['momentum_buffer'].flatten() for v in ac_optimizer.state.values() if v['momentum_buffer'] is not None ]
            history_correction_applied = False
            actor_history_projection = torch.zeros_like(_adv)
            critic_history_projection = torch.zeros_like(critic_rhs)
            if len(gk_list) > 0:
                g_k = torch.cat(gk_list, dim=0)
                actor_history_projection = torch.mv(H, g_k)
                critic_history_projection = torch.mv(critic_rows, g_k)
                _adv = _adv - torch.mv(H, g_k)
                critic_rhs = critic_rhs - critic_history_projection
                history_correction_applied = True

            _png_adv = torch.mv( torch.inverse(HHT @ torch.diag(_ratio) + algo_config.cg_damping * torch.eye(num_sa, device=device)), _adv)

            actor_system = (
                HHT @ torch.diag(_ratio)
                + algo_config.cg_damping * torch.eye(num_sa, device=device)
            )
            actor_relative_residual = torch.linalg.vector_norm(
                torch.mv(actor_system, _png_adv) - _adv
            ) / (torch.linalg.vector_norm(_adv) + 1e-30)

            fp64_chunk_cols = int(algo_config.fp64_gram_chunk_cols)
            critic_alpha64, critic_kernel64, critic_system64, critic_jacobi64, critic_chol_info, critic_residual_norm64, critic_relative64 = solve_separate_critic_b_fp64(
                critic_rows,
                critic_rhs,
                num_sa,
                algo_config.cg_damping,
                fp64_chunk_cols,
                algo_config.dual_jacobi_eps,
            )
            critic_direction = chunked_transpose_mv_fp64(
                critic_rows, critic_alpha64, num_sa, fp64_chunk_cols
            )
            critic_direction_l2 = torch.linalg.vector_norm(critic_direction)
            critic_projection = torch.mv(J_v, critic_direction)
            critic_ggn_quadratic = critic_projection.pow(2).mean()

'''
block = block[:core_start] + core + block[core_end:]

update_start = block.index("        # udpate actor")
update_end = block.index("        ac_optimizer.step()", update_start) + len("        ac_optimizer.step()\n")
update = '''        # Literal Paper actor objective; only the critic contribution is
        # replaced by the independent deterministic-GGN direction above.
        _loss_pi = (- _ratio * _png_adv).mean()
        _loss_v = F.mse_loss(_vals, _ret)
        _loss = _loss_pi - algo_config.ent_coef * _entropy + algo_config.vf_coef * _loss_v

        ac_optimizer.zero_grad()
        _loss_pi.backward()
        actor_grad_flat = torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for p in trainable_params
        ])
        actor_direction_l2 = torch.linalg.vector_norm(actor_grad_flat)
        # SGD subtracts gradients, so add the negative desired critic update.
        add_flat_to_grads(trainable_params, -critic_direction)
        grad_norm = torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), algo_config.max_grad_norm)
        combined_clip_scale = torch.clamp(
            torch.as_tensor(
                float(algo_config.max_grad_norm),
                device=grad_norm.device,
                dtype=grad_norm.dtype,
            ) / (grad_norm + 1e-12),
            max=1.0,
        )
        ac_optimizer.step()
'''
block = block[:update_start] + update + block[update_end:]

info_anchor = "            pi_info = dict(kl=_real_kl.item(), curr_kl=_curr_kl.item(), curr_lr=ac_optimizer.param_groups[0]['lr'], ent=_entropy.item(), cf=clipfrac, \n                           grad_norm=grad_norm.item(), ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())\n"
assert info_anchor in block
info = info_anchor + '''            pi_info.update(
                actor_system_rows=num_sa,
                actor_kernel_mode='paper_sampled_score_separate_b',
                actor_relative_solve_residual=actor_relative_residual.item(),
                actor_direction_l2=actor_direction_l2.item(),
                critic_system_rows=num_sa,
                critic_kernel_mode='deterministic_value_jacobian_separate_b',
                critic_cross_blocks=0,
                critic_curvature_coef=critic_curvature_coef,
                critic_objective_coef=critic_objective_coef,
                critic_linear_solve_dtype=str(torch.float64),
                critic_solver_mode='symmetric_jacobi_cholesky_fp64',
                critic_cholesky_info_max=critic_chol_info.max().item(),
                critic_jacobi_scale_min=critic_jacobi64.min().item(),
                critic_jacobi_scale_max=critic_jacobi64.max().item(),
                critic_solve_residual=critic_residual_norm64.item(),
                critic_relative_solve_residual=critic_relative64.item(),
                critic_direction_l2=critic_direction_l2.item(),
                critic_ggn_quadratic=critic_ggn_quadratic.item(),
                combined_clip_scale=combined_clip_scale.item(),
                optimizer_momentum=1e-6,
                optimizer_history_correction=float(history_correction_applied),
                actor_history_projection_l2=torch.linalg.vector_norm(actor_history_projection).item(),
                critic_history_projection_l2=torch.linalg.vector_norm(critic_history_projection).item(),
                adaptive_kl_timing='per_minibatch',
            )
'''
block = block.replace(info_anchor, info, 1)
paper = paper[:block_start] + block + paper[block_end:]

# Non-invasive telemetry at the unchanged Paper logging cadence and immediately
# after the unchanged per-minibatch adaptive-KL branch.
logger_anchor = '            logger.logkv("ratio_min", pi_info[\'ratio_min\'])\n'
paper = paper.replace(
    logger_anchor,
    logger_anchor
    + "            for _key, _value in pi_info.items():\n"
    + "                if _key.startswith(('actor_', 'critic_', 'optimizer_', 'combined_')):\n"
    + "                    logger.logkv(_key, _value)\n",
    1,
)
paper = paper.replace(
    "    for epoch in tepochs:\n",
    "    adaptive_kl_update_count = 0\n"
    "    minibatch_update_count = 0\n\n"
    "    for epoch in tepochs:\n",
    1,
)
kl_tail = (
    "                    elif curr_kl < 0.01 / 2:\n"
    "                        ac_optimizer.param_groups[0]['lr'] = min(ac_optimizer.param_groups[0]['lr'] * 1.5, algo_config.lr)\n"
)
telemetry = kl_tail + (
    "                minibatch_update_count += 1\n"
    "                if algo_config.use_kl_adaptive_lr:\n"
    "                    adaptive_kl_update_count += 1\n"
    "                pi_info['adaptive_kl_update_count'] = adaptive_kl_update_count\n"
    "                pi_info['minibatch_update_count'] = minibatch_update_count\n"
    "                if log_dir is not None:\n"
    "                    _trace_row = dict(pi_info)\n"
    "                    _trace_row.update(transitions=(epoch+1)*per_epoch_timesteps, "
    "loss_pi=float(mb_loss_pi.item()), loss_v=float(mb_loss_v.item()))\n"
    "                    with open(os.path.join(log_dir, 'metric_trace.jsonl'), 'a') as _trace_file:\n"
    "                        _trace_file.write(json.dumps(_trace_row, sort_keys=True) + '\\n')\n"
)
assert kl_tail in paper
paper = paper.replace(kl_tail, telemetry, 1)

OUT.write_text(paper)
print(sha(OUT))
