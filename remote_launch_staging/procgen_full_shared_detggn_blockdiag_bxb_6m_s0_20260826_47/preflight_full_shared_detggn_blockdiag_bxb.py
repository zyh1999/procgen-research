#!/usr/bin/env python3
"""One no-training production construction and two-BxB finite solve check."""
import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from torch.func import functional_call, grad, vmap


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location("task47_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser()
parser.add_argument("--trainer", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--env-name", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

trainer_path = Path(args.trainer).resolve()
sys.path.insert(0, str(trainer_path.parent))
trainer = load_module(trainer_path)
import yaml
from utils.utils import SharedActorCritic, build_resnet

cfg = yaml.safe_load(Path(args.config).read_text())
acfg = types.SimpleNamespace(**cfg["algo_config"])
ncfg = types.SimpleNamespace(**cfg["nets_config"])
ecfg = types.SimpleNamespace(**cfg["env_config"])
trainer.validate_paper_matched_config(acfg)

torch.manual_seed(470026)
device = torch.device("cuda:0")
base_env, distribution_mode, start_level, num_levels = args.env_name.split("-")
venv = trainer.ProcgenEnv(
    num_envs=ecfg.num_envs,
    env_name=base_env,
    num_levels=int(num_levels),
    start_level=int(start_level),
    distribution_mode=distribution_mode,
    rand_seed=0,
)
venv = trainer.VecExtractDictObs(venv, "rgb")
venv = trainer.VecMonitor(venv=venv, filename=None)
observation_shape = tuple(int(value) for value in venv.observation_space.shape)
if len(observation_shape) != 3 or observation_shape[2] != 3:
    raise RuntimeError(f"unexpected production observation shape {observation_shape}")
image_size = observation_shape[0]
model_input_shape = (
    observation_shape[2], observation_shape[0], observation_shape[1]
)
action_count = int(venv.action_space.n)
fn, _ = build_resnet(
    image_size,
    ncfg.hidden_size,
    with_bn=ncfg.with_bn,
    depths=[8, 16],
    device=device,
)
model = SharedActorCritic(
    fn,
    model_input_shape,
    nets_config=ncfg,
    n_actions=action_count,
    dim_actions=None,
    with_popart=acfg.with_popart,
    sigma_type=acfg.sigma_type,
    device=device,
).to(device).eval()
venv.close()

named = [
    (name, parameter)
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
]
names = [name for name, _ in named]
params = {name: parameter.detach() for name, parameter in named}
buffers = {
    name: buffer.detach()
    for name, buffer in model.named_buffers()
    if buffer.requires_grad
}
parameter_count = sum(parameter.numel() for _, parameter in named)
if parameter_count != 938976:
    raise RuntimeError(f"unexpected trainable parameter count {parameter_count}")

batch = 512
observations = torch.randint(
    0,
    256,
    (batch,) + model_input_shape,
    device=device,
    dtype=torch.uint8,
).float()
actions = torch.arange(batch, device=device, dtype=torch.long) % action_count


def actor_one(ps, bs, observation, action):
    _, logits = functional_call(model, (ps, bs), (observation[None],))
    logp = torch.log_softmax(logits, dim=-1)
    return torch.gather(
        logp, dim=-1, index=action.reshape(1, 1)
    ).reshape(())


def critic_one(ps, bs, observation):
    value, _ = functional_call(model, (ps, bs), (observation[None],))
    return value.reshape(())


actor_tree = vmap(
    grad(actor_one),
    in_dims=(None, None, 0, 0),
    randomness="different",
)(params, buffers, observations, actions)
critic_tree = vmap(
    grad(critic_one),
    in_dims=(None, None, 0),
    randomness="different",
)(params, buffers, observations)
actor_rows = torch.cat(
    [actor_tree[name].reshape(batch, -1) for name in names], dim=1
)
critic_rows = torch.cat(
    [critic_tree[name].reshape(batch, -1) for name in names], dim=1
)
if actor_rows.shape != critic_rows.shape or actor_rows.shape != (512, 938976):
    raise RuntimeError(
        f"actor/critic row drift {actor_rows.shape} {critic_rows.shape}"
    )

offset = 0
role_columns = {"shared": [], "policy": [], "value": []}
for name, parameter in named:
    stop = offset + parameter.numel()
    role = (
        "policy" if name.startswith("pi_head.")
        else "value" if name.startswith("last_v_layer.")
        else "shared"
    )
    role_columns[role].append(torch.arange(offset, stop, device=device))
    offset = stop
role_columns = {
    role: torch.cat(columns) for role, columns in role_columns.items()
}
if role_columns["policy"].numel() != 3855:
    raise RuntimeError("policy-head column count drift")
if role_columns["value"].numel() != 257:
    raise RuntimeError("value-head column count drift")
if torch.count_nonzero(actor_rows[:, role_columns["value"]]) != 0:
    raise RuntimeError("actor value-head columns are not structural zero")
if torch.count_nonzero(critic_rows[:, role_columns["policy"]]) != 0:
    raise RuntimeError("critic policy-head columns are not structural zero")
for rows, role in (
    (actor_rows, "shared"),
    (actor_rows, "policy"),
    (critic_rows, "shared"),
    (critic_rows, "value"),
):
    if torch.linalg.vector_norm(rows[:, role_columns[role]]) <= 0:
        raise RuntimeError(f"missing connected {role} columns")

torch.manual_seed(470027)
actor_rhs = torch.randn(batch, device=device)
critic_residual = torch.randn(batch, device=device)
actor_ratio = torch.exp(0.05 * torch.randn(batch, device=device))
critic_h_weight = float(acfg.joint_critic_curvature_coef) ** 0.5
critic_rhs_weight = (
    float(acfg.joint_critic_objective_coef) / critic_h_weight
)
critic_weighted_rows = critic_h_weight * critic_rows
actor_direction, actor_solve = trainer.solve_raw_weighted_bxb_fp64(
    actor_rows,
    actor_rhs,
    actor_ratio,
    acfg.cg_damping,
    batch,
    acfg.fp64_gram_chunk_cols,
    acfg.dual_jacobi_eps,
)
critic_direction, critic_solve = trainer.solve_raw_weighted_bxb_fp64(
    critic_weighted_rows,
    critic_rhs_weight * critic_residual,
    torch.ones_like(critic_residual),
    acfg.cg_damping,
    batch,
    acfg.fp64_gram_chunk_cols,
    acfg.dual_jacobi_eps,
)
summed_direction = actor_direction + critic_direction
if int(actor_solve["cholesky_info"]) != 0:
    raise RuntimeError("actor BxB Cholesky failed")
if int(critic_solve["cholesky_info"]) != 0:
    raise RuntimeError("critic BxB Cholesky failed")
if not all(
    torch.all(torch.isfinite(value))
    for value in (actor_direction, critic_direction, summed_direction)
):
    raise RuntimeError("nonfinite block-diagonal direction")
if not all(
    torch.isfinite(solve["relative_residual"])
    for solve in (actor_solve, critic_solve)
):
    raise RuntimeError("nonfinite block-diagonal residual")

coverage = {}
for role, columns in role_columns.items():
    coverage[role] = {
        "actor_norm": float(torch.linalg.vector_norm(actor_direction[columns])),
        "critic_norm": float(torch.linalg.vector_norm(critic_direction[columns])),
        "sum_norm": float(torch.linalg.vector_norm(summed_direction[columns])),
    }
if coverage["shared"]["actor_norm"] <= 0 or coverage["shared"]["critic_norm"] <= 0:
    raise RuntimeError("shared direction lacks an actor or critic contribution")
if coverage["policy"]["sum_norm"] <= 0 or coverage["value"]["sum_norm"] <= 0:
    raise RuntimeError("policy/value direction coverage missing")

result = {
    "status": "PRECHECK_PASS",
    "marker": "GPUH_FULL_SHARED_DETGGN_BLOCKDIAG_BXB_PASS",
    "method": "FULL_SHARED_DETGGN_BLOCKDIAG_BXB_V1",
    "trainer_sha256": sha256(args.trainer),
    "config_sha256": sha256(args.config),
    "environment": args.env_name,
    "observation_shape_hwc": observation_shape,
    "model_input_shape_chw": model_input_shape,
    "image_size": image_size,
    "action_count": action_count,
    "parameter_tensors": len(named),
    "parameter_count": parameter_count,
    "actor_rows": list(actor_rows.shape),
    "critic_rows": list(critic_rows.shape),
    "actor_value_head_zero_columns": 257,
    "critic_policy_head_zero_columns": 3855,
    "two_independent_system_shapes": [[512, 512], [512, 512]],
    "dual_cross_blocks_assembled_or_solved": False,
    "actor_cholesky_info": int(actor_solve["cholesky_info"]),
    "critic_cholesky_info": int(critic_solve["cholesky_info"]),
    "actor_relative_residual": float(actor_solve["relative_residual"]),
    "critic_relative_residual": float(critic_solve["relative_residual"]),
    "actor_direction_norm": float(torch.linalg.vector_norm(actor_direction)),
    "critic_direction_norm": float(torch.linalg.vector_norm(critic_direction)),
    "summed_direction_norm": float(torch.linalg.vector_norm(summed_direction)),
    "direction_coverage": coverage,
    "finite": True,
}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print("GPUH_FULL_SHARED_DETGGN_BLOCKDIAG_BXB_PASS")
print(json.dumps(result, sort_keys=True))
