#!/usr/bin/env python3
"""Build the hybrid head-only candidate deterministically from exact Paper RAT."""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT / "work/procgen_paper_2b5affd_6m_bede_20260722/source/train_shared.py"
P1 = ROOT / "deterministic_2b_symfp64_20260807/train_shared_rat_exact_deterministic_ggn_symfp64.py"
OUT = HERE / "train_shared_paper_hybrid_head_detggn_v1.py"
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
helpers += '''def solve_head_critic_b_fp64(rows, rhs, denominator, damping, chunk_cols, jacobi_eps):
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


def validate_hybrid_head_config(algo_config):
    """Reject every mechanism outside the one frozen critic-head replacement."""
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
        'max_grad_norm': 0.5, 'critic_curvature_coef': 0.1,
        'critic_objective_coef': 1.0, 'ent_coef': 0.0,
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
    learn_anchor + "    validate_hybrid_head_config(algo_config)\n",
    1,
)
anchor = "    dict_buffers = {k: v.detach() for k, v in actor_critic.named_buffers() if v.requires_grad}\n"
partition_setup = anchor + '''    trainable_named = [(name, p) for name, p in actor_critic.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in trainable_named]
    parameter_groups = parameter_partition(actor_critic)
    policy_names = [name for name, _ in parameter_groups['POLICY_EXCLUSIVE']]
    shared_names = [name for name, _ in parameter_groups['SHARED']]
    head_names = [name for name, _ in parameter_groups['CRITIC_EXCLUSIVE']]
    name_to_index = {name: i for i, (name, _) in enumerate(trainable_named)}
    head_params_dict = {name: dict_params[name] for name in head_names}
    frozen_nonhead_params = {name: value for name, value in dict_params.items() if name not in head_names}
    if log_dir is not None:
        _probe = torch.zeros((2,) + tuple(runner.obs.shape[1:]), device=device)
        _, _partition_evidence = partition_manifest(actor_critic, _probe)
        with open(os.path.join(log_dir, 'parameter_partition.json'), 'w') as _manifest_file:
            json.dump(_partition_evidence, _manifest_file, indent=2, sort_keys=True)
'''
paper = paper.replace(anchor, partition_setup, 1)

block_start = paper.index("    def Advantage_Update(")
block_end = paper.index("    def KFAC_Update(", block_start)
block = paper[block_start:block_end]

sample_anchor = "        ft_per_sample_grads = ft_compute_sample_grad(dict_params, dict_buffers, _obs, _act) # num_samples x param_shape\n\n"
head_rows = sample_anchor + '''        def compute_head_value(head_params, buffers, batch_obs):
            batch_obs = batch_obs.unsqueeze(0)
            combined_params = dict(frozen_nonhead_params)
            combined_params.update(head_params)
            batch_vals, _ = functional_call(
                actor_critic, (combined_params, buffers), (batch_obs,)
            )
            return batch_vals.reshape(-1)[0]

        ft_compute_head_value_grad = vmap(
            grad(compute_head_value), in_dims=(None, None, 0), randomness='different'
        )
        ft_head_value_grads = ft_compute_head_value_grad(
            head_params_dict, dict_buffers, _obs
        )

'''
if sample_anchor not in block:
    raise AssertionError('Paper sampled-score anchor missing')
block = block.replace(sample_anchor, head_rows, 1)

core_start = block.index("        with torch.no_grad():\n            num_sa")
core_end = block.index("        # udpate actor", core_start)
core = '''        with torch.no_grad():
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

            # The sole replacement: critic-exclusive value-head J_v/residual GGN.
            J_head = torch.cat(
                [ft_head_value_grads[name].contiguous().view(num_sa, -1) for name in head_names],
                dim=-1,
            )
            critic_curvature_coef = float(algo_config.critic_curvature_coef)
            critic_objective_coef = float(algo_config.critic_objective_coef)
            critic_h_weight = math.sqrt(critic_curvature_coef)
            head_rows = critic_h_weight * J_head
            head_rhs = (critic_objective_coef / critic_h_weight) * (_ret - _vals).detach()
            head_momentum = []
            for _, parameter in parameter_groups['CRITIC_EXCLUSIVE']:
                state = ac_optimizer.state.get(parameter, {})
                head_momentum.append(state.get('momentum_buffer', torch.zeros_like(parameter)).flatten())
            head_momentum_flat = torch.cat(head_momentum)
            head_history_projection = torch.mv(head_rows, head_momentum_flat)
            if history_correction_applied:
                head_rhs = head_rhs - head_history_projection

            fp64_chunk_cols = int(algo_config.fp64_gram_chunk_cols)
            head_alpha64, head_kernel64, head_system64, head_jacobi64, head_chol_info, head_residual_norm64, head_relative64 = solve_head_critic_b_fp64(
                head_rows, head_rhs, num_sa, algo_config.cg_damping,
                fp64_chunk_cols, algo_config.dual_jacobi_eps,
            )
            head_direction = chunked_transpose_mv_fp64(
                head_rows, head_alpha64, num_sa, fp64_chunk_cols
            )
            head_direction_l2 = torch.linalg.vector_norm(head_direction)
            head_projection = torch.mv(J_head, head_direction)
            head_ggn_quadratic = head_projection.pow(2).mean()

'''
block = block[:core_start] + core + block[core_end:]

update_start = block.index("        # udpate actor")
update_end = block.index("        ac_optimizer.step()", update_start) + len("        ac_optimizer.step()\n")
update = '''        # Compose exact Paper actor and sampled critic first.
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
        offset = 0
        for _, parameter in parameter_groups['CRITIC_EXCLUSIVE']:
            count = parameter.numel()
            parameter.grad = (-head_direction[offset:offset + count].view_as(parameter) * paper_clip_scale).to(parameter.dtype)
            offset += count
        if offset != head_direction.numel():
            raise ValueError('head direction length mismatch')

        _loss = _actor_objective + _paper_critic_objective
        ac_optimizer.step()
'''
block = block[:update_start] + update + block[update_end:]

info_anchor = "            pi_info = dict(kl=_real_kl.item(), curr_kl=_curr_kl.item(), curr_lr=ac_optimizer.param_groups[0]['lr'], ent=_entropy.item(), cf=clipfrac, \n                           grad_norm=grad_norm.item(), ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())\n"
if info_anchor not in block:
    raise AssertionError('Paper telemetry anchor missing')
info = info_anchor + '''            after_params = {name: p.detach().clone() for name, p in trainable_named}
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
            pi_info.update(
                parameter_partition='policy_exclusive_shared_critic_exclusive',
                actor_system_rows=num_sa,
                actor_kernel_mode='paper_sampled_policy_value_score',
                actor_relative_solve_residual=actor_relative_residual.item(),
                actor_direction_l2=actor_direction_l2.item(),
                paper_shared_critic_direction_l2=paper_shared_critic_l2.item(),
                paper_critic_relative_solve_residual=paper_critic_relative_residual.item(),
                head_critic_system_rows=num_sa,
                head_critic_kernel_mode='deterministic_value_head_jacobian_b',
                head_critic_curvature_coef=critic_curvature_coef,
                head_critic_objective_coef=critic_objective_coef,
                head_critic_linear_solve_dtype=str(torch.float64),
                head_critic_solver_mode='symmetric_jacobi_cholesky_fp64',
                head_critic_cholesky_info_max=head_chol_info.max().item(),
                head_critic_jacobi_scale_min=head_jacobi64.min().item(),
                head_critic_jacobi_scale_max=head_jacobi64.max().item(),
                head_critic_solve_residual=head_residual_norm64.item(),
                head_critic_relative_solve_residual=head_relative64.item(),
                head_critic_direction_l2=head_direction_l2.item(),
                head_critic_ggn_quadratic=head_ggn_quadratic.item(),
                paper_clip_scale=paper_clip_scale.item(),
                policy_parameter_delta_l2=torch.linalg.vector_norm(policy_delta).item(),
                shared_parameter_delta_l2=torch.linalg.vector_norm(shared_delta).item(),
                head_parameter_delta_l2=torch.linalg.vector_norm(head_delta).item(),
                shared_to_actor_update_ratio=(torch.linalg.vector_norm(shared_delta) / (actor_direction_l2 + 1e-30)).item(),
                head_to_actor_update_ratio=(torch.linalg.vector_norm(head_delta) / (actor_direction_l2 + 1e-30)).item(),
                post_shared_to_post_head_policy_logit_max_abs=head_policy_logit_max_abs.item(),
                post_shared_to_post_head_policy_kl=head_policy_kl.item(),
                optimizer_momentum=1e-6,
                optimizer_history_correction=float(history_correction_applied),
                head_history_projection_l2=torch.linalg.vector_norm(head_history_projection).item(),
                adaptive_kl_timing='per_minibatch',
            )
'''
block = block.replace(info_anchor, info, 1)
paper = paper[:block_start] + block + paper[block_end:]

logger_anchor = '            logger.logkv("ratio_min", pi_info[\'ratio_min\'])\n'
paper = paper.replace(
    logger_anchor,
    logger_anchor
    + "            for _key, _value in pi_info.items():\n"
    + "                if _key.startswith(('actor_', 'paper_', 'head_', 'policy_', 'shared_', 'post_', 'optimizer_')):\n"
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
    "                    _trace_row.update(transitions=(epoch+1)*per_epoch_timesteps, loss_pi=float(mb_loss_pi.item()), loss_v=float(mb_loss_v.item()))\n"
    "                    with open(os.path.join(log_dir, 'metric_trace.jsonl'), 'a') as _trace_file:\n"
    "                        _trace_file.write(json.dumps(_trace_row, sort_keys=True) + '\\n')\n"
)
if kl_tail not in paper:
    raise AssertionError('Paper adaptive-KL anchor missing')
paper = paper.replace(kl_tail, telemetry, 1)

OUT.write_text(paper)
print(sha(OUT))
