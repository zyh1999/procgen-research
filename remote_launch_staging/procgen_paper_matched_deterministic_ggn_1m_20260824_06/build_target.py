#!/usr/bin/env python3
"""Deterministically build the target trainer from frozen Paper and P1 inputs."""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT / "work/procgen_paper_2b5affd_6m_bede_20260722/source/train_shared.py"
P1 = ROOT / "deterministic_2b_symfp64_20260807/train_shared_rat_exact_deterministic_ggn_symfp64.py"
OUT = HERE / "train_shared_paper_matched_deterministic_ggn_v1.py"
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
helpers += '''def validate_paper_matched_config(algo_config):
    """Reject actor-side P1 controls and enforce the frozen Paper identity."""
    forbidden = ('adaptive_kl_mode', 'optimizer_momentum', 'is_kaczmarz')
    present = [name for name in forbidden if hasattr(algo_config, name)]
    if present:
        raise ValueError(f'forbidden P1 actor fields: {present}')
    required = {
        'optimizer': 'sgd', 'lr': 0.5, 'epochs': 4, 'minibatches': 8,
        'use_kl_adaptive_lr': True, 'cg_damping': 0.5,
        'max_grad_norm': 0.5, 'joint_critic_curvature_coef': 0.1,
        'joint_critic_objective_coef': 1.0,
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

learn_anchor = "    gamma = .999\n    lam = .95\n"
paper = paper.replace(
    learn_anchor,
    learn_anchor + "    validate_paper_matched_config(algo_config)\n",
    1,
)

paper_start = paper.index("    def Advantage_Update(")
paper_end = paper.index("    def KFAC_Update(", paper_start)
p1_start = p1.index("    def Advantage_Update(")
p1_end = p1.index("    def KFAC_Update(", p1_start)
target_block = p1[p1_start:p1_end]

# P1's actor experiments are deliberately not migrated. History correction is
# the literal Paper rule: once SGD owns buffers, subtract H @ buffer from both
# actor and deterministic-critic rows before the direct solve.
target_block = target_block.replace(
    "            previous_momentum_buffer = None\n"
    "            previous_projection = torch.zeros_like(joint_rhs)\n"
    "            if optimizer_momentum > 0.0:\n",
    "            previous_momentum_buffer = None\n"
    "            previous_projection = torch.zeros_like(joint_rhs)\n"
    "            history_correction_applied = False\n"
    "            if True:\n",
    1,
)
target_block = target_block.replace(
    "                    previous_projection = (\n"
    "                        optimizer_momentum\n"
    "                        * torch.mv(joint_H, previous_momentum_buffer)\n"
    "                    )\n",
    "                    previous_projection = torch.mv(\n"
    "                        joint_H, previous_momentum_buffer\n"
    "                    )\n",
    1,
)
rhs_start = target_block.index("            # The SGD buffer has descent-gradient sign.")
rhs_end = target_block.index("\n            kernel_denom =", rhs_start)
target_block = (
    target_block[:rhs_start]
    + "            # Exact original Paper history correction, now applied to the\n"
      "            # stacked actor and deterministic-critic rows.\n"
      "            rhs_eff = joint_rhs\n"
      "            if previous_momentum_buffer is not None:\n"
      "                rhs_eff = rhs_eff - previous_projection.to(rhs_eff.dtype)\n"
      "                history_correction_applied = True\n"
    + target_block[rhs_end:]
)
target_block = target_block.replace(
    "                optimizer_momentum=optimizer_momentum,\n"
    "                optimizer_momentum_semantics='classic_beta_m_plus_d',\n"
    "                optimizer_history_correction=float(is_kaczmarz),\n"
    "                kaczmarz_rhs_semantics='rhs_plus_beta_H_buffer',\n",
    "                optimizer_momentum=1e-6,\n"
    "                optimizer_momentum_semantics='paper_sgd_momentum_1e-6',\n"
    "                optimizer_history_correction=float(history_correction_applied),\n"
    "                kaczmarz_rhs_semantics='paper_rhs_minus_H_buffer',\n",
    1,
)
paper = paper[:paper_start] + target_block + paper[paper_end:]
paper = paper.replace("import types\n", "import types\nimport json\n", 1)
paper = paper.replace(
    "from utils.utils import SharedActorCritic, count_vars, safemean, set_seed\n",
    "from utils.utils import SharedActorCritic, count_vars, safemean, set_seed, set_grads_from_flat\n",
    1,
)

anchor = "    dict_buffers = {k: v.detach() for k, v in actor_critic.named_buffers() if v.requires_grad}\n"
paper = paper.replace(anchor, anchor + "    trainable_params = [p for p in actor_critic.parameters() if p.requires_grad]\n", 1)

# Preserve Paper's loop. Only expose solver health fields at the same logger
# cadence; this does not change actor, data, evaluation, or checkpoint paths.
anchor = '            logger.logkv("ratio_min", pi_info[\'ratio_min\'])\n'
addition = (
    anchor
    + "            for _key, _value in pi_info.items():\n"
    + "                if _key.startswith(('joint_', 'critic_', 'optimizer_')):\n"
    + "                    logger.logkv(_key, _value)\n"
)
paper = paper.replace(anchor, addition, 1)

# Required non-invasive per-minibatch health trace. It is appended strictly
# after the unchanged Paper adaptive-KL block.
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
paper = paper.replace(kl_tail, telemetry, 1)
OUT.write_text(paper)
print(sha(OUT))
