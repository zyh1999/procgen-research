#!/usr/bin/env python3
"""Production actor/critic 512-row structural-zero equivalence gate."""
import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from torch.func import functional_call, grad, vmap

from structural_zero import materialize_structural_zeros


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser()
parser.add_argument("--task41-preflight", required=True)
parser.add_argument("--task42-dir", required=True)
parser.add_argument("--trainer", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--oracle", required=True)
parser.add_argument("--oracle-sha256", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

task41_dir = Path(args.task41_preflight).resolve().parent
sys.path.insert(0, str(task41_dir))
sys.path.insert(0, str(Path(args.task42_dir).resolve()))
from actor_gather import selected_logp_gather
task41 = load_module("task41_preflight_reused_for_task43", args.task41_preflight)
oracle_module = load_module("task41_oracle_reused_for_task43", task41_dir / "manifest_oracle.py")
oracle_bytes = Path(args.oracle).read_bytes()
assert hashlib.sha256(oracle_bytes).hexdigest() == args.oracle_sha256
oracle = json.loads(oracle_bytes)

construct_args = types.SimpleNamespace(
    trainer=args.trainer, config=args.config, env_name="bigfish-easy-0-10")
_, model, optimizer, manifest, construction = task41.construct_production(construct_args)
oracle_module.compare_manifest(manifest, oracle)
named = [(name, parameter) for name, parameter in model.named_parameters()
         if parameter.requires_grad]
names = [name for name, _ in named]
entries = [entry for entry in oracle["parameters"] if entry["requires_grad"]]
roles = [entry["role"] for entry in entries]
assert names == [entry["name"] for entry in entries]
assert names == oracle["ordered_joint2b_column_names"]
assert sum(parameter.numel() for _, parameter in named) == 938976
for (_, parameter), optimizer_parameter in zip(
        named, [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]):
    assert parameter is optimizer_parameter

params = {name: parameter.detach() for name, parameter in named}
buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}
batch = 512
torch.manual_seed(430026)
observations = torch.randint(
    0, 256, (batch,) + tuple(construction["model_input_shape"]),
    device="cuda:0", dtype=torch.uint8).float()
actions = torch.arange(batch, device="cuda:0", dtype=torch.long) % construction["action_count"]
observations_before = observations.clone()
actions_before = actions.clone()
model_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
optimizer_state_before = len(optimizer.state)
cpu_rng_before = torch.get_rng_state().clone()
cuda_rng_before = torch.cuda.get_rng_state_all()

def actor_one(ps, bs, observation, action):
    _, logits = functional_call(model, (ps, bs), (observation[None],))
    return selected_logp_gather(logits, action).reshape(())

def critic_one(ps, bs, observation):
    value, _ = functional_call(model, (ps, bs), (observation[None],))
    return value.reshape(())

actor_tree = vmap(grad(actor_one), in_dims=(None, None, 0, 0), randomness="different")(
    params, buffers, observations, actions)
critic_tree = vmap(grad(critic_one), in_dims=(None, None, 0), randomness="different")(
    params, buffers, observations)
assert list(actor_tree) == names and list(critic_tree) == names
for name, parameter in named:
    assert actor_tree[name].shape == critic_tree[name].shape == (batch,) + parameter.shape
    assert actor_tree[name].dtype == critic_tree[name].dtype == parameter.dtype
    assert actor_tree[name].device == critic_tree[name].device == parameter.device

actor_max_abs = 0.0
actor_max_rel = 0.0
critic_max_abs = 0.0
critic_max_rel = 0.0
actor_none_total = {"CRITIC_EXCLUSIVE": 0}
critic_none_total = {"POLICY_EXCLUSIVE": 0}
for row in range(batch):
    _, logits = model(observations[row:row + 1])
    actor_scalar = torch.log_softmax(logits, dim=-1)[0, int(actions[row].item())]
    actor_raw = torch.autograd.grad(
        actor_scalar, tuple(parameter for _, parameter in named), allow_unused=True)
    actor_reference, actor_stats = materialize_structural_zeros(
        actor_raw, named, roles, {"CRITIC_EXCLUSIVE"})
    value, _ = model(observations[row:row + 1])
    critic_scalar = value.reshape(())
    critic_raw = torch.autograd.grad(
        critic_scalar, tuple(parameter for _, parameter in named), allow_unused=True)
    critic_reference, critic_stats = materialize_structural_zeros(
        critic_raw, named, roles, {"POLICY_EXCLUSIVE"})
    if actor_stats["none_by_role"] != {"CRITIC_EXCLUSIVE": 2}:
        raise AssertionError(f"actor structural-None role/count drift at row {row}")
    if critic_stats["none_by_role"] != {"POLICY_EXCLUSIVE": 2}:
        raise AssertionError(f"critic structural-None role/count drift at row {row}")
    actor_none_total["CRITIC_EXCLUSIVE"] += 2
    critic_none_total["POLICY_EXCLUSIVE"] += 2
    for (name, _), actor_ref, critic_ref in zip(named, actor_reference, critic_reference):
        actor_candidate = actor_tree[name][row]
        critic_candidate = critic_tree[name][row]
        actor_error = torch.max(torch.abs(actor_candidate - actor_ref)).item()
        critic_error = torch.max(torch.abs(critic_candidate - critic_ref)).item()
        actor_max_abs = max(actor_max_abs, actor_error)
        critic_max_abs = max(critic_max_abs, critic_error)
        actor_max_rel = max(actor_max_rel, actor_error / (torch.max(torch.abs(actor_ref)).item() + 1e-30))
        critic_max_rel = max(critic_max_rel, critic_error / (torch.max(torch.abs(critic_ref)).item() + 1e-30))
        torch.testing.assert_close(actor_candidate, actor_ref, rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(critic_candidate, critic_ref, rtol=3e-5, atol=2e-6)

actor_rows = torch.cat([actor_tree[name].reshape(batch, -1) for name in names], dim=1)
critic_rows = torch.cat([critic_tree[name].reshape(batch, -1) for name in names], dim=1)
assert actor_rows.shape == critic_rows.shape == (512, 938976)
offsets = [0]
for _, parameter in named:
    offsets.append(offsets[-1] + parameter.numel())
role_columns = {}
for role in set(roles):
    role_columns[role] = torch.cat([
        torch.arange(offsets[index], offsets[index + 1], device="cuda:0")
        for index, parameter_role in enumerate(roles) if parameter_role == role])
assert role_columns["CRITIC_EXCLUSIVE"].numel() == 257
assert role_columns["POLICY_EXCLUSIVE"].numel() == 3855
assert torch.count_nonzero(actor_rows[:, role_columns["CRITIC_EXCLUSIVE"]]) == 0
assert torch.count_nonzero(critic_rows[:, role_columns["POLICY_EXCLUSIVE"]]) == 0
assert torch.linalg.vector_norm(actor_rows[:, role_columns["POLICY_EXCLUSIVE"]]) > 0
assert torch.linalg.vector_norm(critic_rows[:, role_columns["CRITIC_EXCLUSIVE"]]) > 0
assert torch.linalg.vector_norm(actor_rows[:, role_columns["SHARED"]]) > 0
assert torch.linalg.vector_norm(critic_rows[:, role_columns["SHARED"]]) > 0

assert torch.equal(observations, observations_before) and torch.equal(actions, actions_before)
assert len(optimizer.state) == optimizer_state_before == 0
for name, parameter in model.named_parameters():
    assert torch.equal(parameter, model_before[name])
assert torch.equal(torch.get_rng_state(), cpu_rng_before)
assert all(torch.equal(before, after) for before, after in
           zip(cuda_rng_before, torch.cuda.get_rng_state_all()))

result = {
    "status": "LOCAL_EQUIVALENCE_PASS",
    "marker": "TASK43_PRODUCTION_STRUCTURAL_ZERO_512ROW_EQUIVALENCE_PASS",
    "oracle_sha256": args.oracle_sha256,
    "rows": {"actor": 512, "critic": 512, "joint": 1024},
    "columns": 938976,
    "parameter_tensors": 26,
    "actor_structural_none_total": actor_none_total,
    "critic_structural_none_total": critic_none_total,
    "actor_value_head_zero_columns": 257,
    "critic_policy_head_zero_columns": 3855,
    "actor_max_absolute_error": actor_max_abs,
    "actor_max_relative_error": actor_max_rel,
    "critic_max_absolute_error": critic_max_abs,
    "critic_max_relative_error": critic_max_rel,
    "ordered_names_shapes_dtypes_devices_match_oracle": True,
    "shared_connected_actor_and_critic": True,
    "input_model_rng_optimizer_popart_bit_identical": True,
}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print("TASK43_PRODUCTION_STRUCTURAL_ZERO_512ROW_EQUIVALENCE_PASS")
print(json.dumps(result, sort_keys=True))
