#!/usr/bin/env python3
"""Required production 512-row gather/Jacobian equivalence gate for Task42."""
import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
from torch.func import functional_call, grad, vmap

from actor_gather import selected_logp_gather


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser()
parser.add_argument("--task41-preflight", required=True)
parser.add_argument("--trainer", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--oracle", required=True)
parser.add_argument("--oracle-sha256", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

task41_dir = Path(args.task41_preflight).resolve().parent
sys.path.insert(0, str(task41_dir))
task41 = load_module("task41_frozen_preflight_for_gather_gate", args.task41_preflight)
oracle_module = load_module("task41_frozen_manifest_oracle", task41_dir / "manifest_oracle.py")
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
assert names == oracle["ordered_joint2b_column_names"]
optimizer_trainable = [parameter for group in optimizer.param_groups
                       for parameter in group["params"] if parameter.requires_grad]
for (_, parameter), optimizer_parameter in zip(named, optimizer_trainable):
    assert parameter is optimizer_parameter

params = {name: parameter.detach() for name, parameter in named}
buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}
batch = 512
torch.manual_seed(420026)
observations = torch.randint(
    0, 256, (batch,) + tuple(construction["model_input_shape"]),
    device="cuda:0", dtype=torch.uint8).float()
actions = torch.arange(batch, device="cuda:0", dtype=torch.long) % construction["action_count"]
assert 0 in actions and construction["action_count"] - 1 in actions and 7 in actions
observations_before = observations.clone()
actions_before = actions.clone()
cpu_rng_before = torch.get_rng_state().clone()
cuda_rng_before = torch.cuda.get_rng_state_all()

def gather_logp_one(ps, bs, observation, action):
    _, logits = functional_call(model, (ps, bs), (observation[None],))
    return selected_logp_gather(logits, action).reshape(())

actor_tree = vmap(
    grad(gather_logp_one), in_dims=(None, None, 0, 0), randomness="different")(
        params, buffers, observations, actions)
assert list(actor_tree) == names
for name, parameter in named:
    assert actor_tree[name].shape == (batch,) + parameter.shape
    assert actor_tree[name].dtype == parameter.dtype

max_abs_error = 0.0
max_relative_error = 0.0
for row in range(batch):
    _, logits = model(observations[row:row + 1])
    explicit = torch.log_softmax(logits, dim=-1)[0, int(actions[row].item())]
    explicit_grads = torch.autograd.grad(explicit, tuple(parameter for _, parameter in named))
    for (name, _), explicit_grad in zip(named, explicit_grads):
        candidate = actor_tree[name][row]
        error = torch.max(torch.abs(candidate - explicit_grad)).item()
        relative = error / (torch.max(torch.abs(explicit_grad)).item() + 1e-30)
        max_abs_error = max(max_abs_error, error)
        max_relative_error = max(max_relative_error, relative)
        torch.testing.assert_close(candidate, explicit_grad, rtol=3e-5, atol=2e-6)

assert torch.equal(observations, observations_before)
assert torch.equal(actions, actions_before)
assert torch.equal(torch.get_rng_state(), cpu_rng_before)
cuda_rng_after = torch.cuda.get_rng_state_all()
assert len(cuda_rng_after) == len(cuda_rng_before)
assert all(torch.equal(before, after) for before, after in zip(cuda_rng_before, cuda_rng_after))
try:
    assert list(reversed(names)) == oracle["ordered_joint2b_column_names"]
except AssertionError:
    parameter_reorder_rejected = True
else:
    raise AssertionError("parameter reorder negative case accepted")

result = {
    "status": "LOCAL_EQUIVALENCE_PASS",
    "marker": "TASK42_PRODUCTION_ACTOR_GATHER_512ROW_EQUIVALENCE_PASS",
    "oracle_sha256": args.oracle_sha256,
    "rows": batch,
    "columns": sum(parameter.numel() for _, parameter in named),
    "parameter_tensors": len(named),
    "ordered_names_match_oracle": True,
    "boundary_actions": [0, 7, construction["action_count"] - 1],
    "max_actor_parameter_jacobian_absolute_error": max_abs_error,
    "max_actor_parameter_jacobian_relative_error": max_relative_error,
    "actor_row_shape": [batch, sum(parameter.numel() for _, parameter in named)],
    "actor_row_dtype": str(next(iter(actor_tree.values())).dtype),
    "input_bit_identical_after": True,
    "rng_bit_identical_after": True,
    "parameter_reorder_negative_rejected": parameter_reorder_rejected,
}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print("TASK42_PRODUCTION_ACTOR_GATHER_512ROW_EQUIVALENCE_PASS")
print(json.dumps(result, sort_keys=True))
