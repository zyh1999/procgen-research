#!/usr/bin/env python3
"""Canonical H200, actual-data, and production-model Task34R gate."""
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

import gym3  # noqa: F401
import numpy as np
import procgen  # noqa: F401
import torch
from torch import nn


trainer, config, requested_manifest_path = map(Path, sys.argv[1:4])
evidence_dir = requested_manifest_path.parent
structural_manifest_path = evidence_dir / "structural_manifest.json"
connectivity_probe_path = evidence_dir / "connectivity_probe.json"
expected_trainer, expected_config = sys.argv[4:6]
campaign = trainer.parent.parent
launcher = campaign / "frozen/standard_mse_ggn_head_cvlm_6m_gpuh.sbatch"
env_name = os.environ.get("PROCGEN_ENV", "bigfish-easy-0-10")
allowed_envs = {
    "bigfish-easy-0-10", "bossfight-easy-0-10",
    "caveflyer-easy-0-10", "coinrun-easy-0-10",
}
assert env_name in allowed_envs, env_name


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(trainer) == expected_trainer
assert sha(config) == expected_config
launcher_text = launcher.read_text()
expected_cmd = 'CMD=("$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name "$ENV_NAME" --seed 0 --device 0)'
assert expected_cmd in launcher_text

sys.path.insert(0, str(trainer.parent))
spec = importlib.util.spec_from_file_location("standard_mse_cvlm_trainer", trainer)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

canonical_argv = [
    str(trainer), "--config", config.name, "--env_name", env_name,
    "--seed", "0", "--device", "0",
]


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def resolved_payload(world_size, algo, seed, algo_cfg, env_cfg, nets_cfg, log_cfg, device):
    return normalize({
        "entry": {
            "config": config.name,
            "device": device,
            "env_name": env_name,
            "n_proc": world_size,
            "seed": seed,
        },
        "algo": algo,
        "algo_config": vars(algo_cfg),
        "env_config": vars(env_cfg),
        "nets_config": vars(nets_cfg),
        "log_config": vars(log_cfg),
    })


def invoke_main(capture_model=False):
    """Use the trainer parser/default merge and, once, original train_fn."""
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    original_train_fn = module.train_fn
    original_learn = module.learn
    captured = {}

    class CanonicalCaptureComplete(RuntimeError):
        pass

    def train_wrapper(rank, world_size, algo, seed, algo_cfg, env_cfg, nets_cfg, log_cfg, device=-1):
        captured["resolved"] = resolved_payload(
            world_size, algo, seed, algo_cfg, env_cfg, nets_cfg, log_cfg, device
        )
        if capture_model:
            return original_train_fn(
                rank, world_size, algo, seed, algo_cfg, env_cfg, nets_cfg, log_cfg, device
            )

    def learn_capture(world_size, algo, actor_critic, writer, venv, device, **kwargs):
        captured["model"] = actor_critic
        captured["venv"] = venv
        captured["learn"] = normalize({
            "world_size": world_size,
            "algo": algo,
            "device": str(device),
            "total_timesteps": kwargs["total_timesteps"],
            "nsteps": kwargs["nsteps"],
            "algo_config": vars(kwargs["algo_config"]),
            "log_config": vars(kwargs["log_config"]),
        })
        if writer is not None:
            writer.close()
        raise CanonicalCaptureComplete

    try:
        module.train_fn = train_wrapper
        if capture_model:
            module.learn = learn_capture
        sys.argv = canonical_argv[:]
        os.chdir(trainer.parent)
        try:
            module.main()
        except CanonicalCaptureComplete:
            pass
    finally:
        module.train_fn = original_train_fn
        module.learn = original_learn
        sys.argv = old_argv
        os.chdir(old_cwd)
    assert "resolved" in captured
    if capture_model:
        assert "model" in captured and "learn" in captured and "venv" in captured
    return captured


# All three identities are resolved by the trainer's own main().
preflight_capture = invoke_main(capture_model=False)
launcher_capture = invoke_main(capture_model=False)
trainer_capture = invoke_main(capture_model=True)
resolved_preflight = preflight_capture["resolved"]
resolved_launcher = launcher_capture["resolved"]
resolved_trainer = trainer_capture["resolved"]
assert resolved_preflight == resolved_launcher == resolved_trainer
module.validate_standard_mse_cvlm_config(
    types.SimpleNamespace(**resolved_trainer["algo_config"])
)
resolved_bytes = json.dumps(resolved_trainer, sort_keys=True, separators=(",", ":")).encode()
resolved_sha = hashlib.sha256(resolved_bytes).hexdigest()
for label, payload in (
    ("preflight", resolved_preflight),
    ("scientific_launcher_dry_run", resolved_launcher),
    ("trainer_entry", resolved_trainer),
):
    path = evidence_dir / f"resolved_config_{label}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(payload_bytes).hexdigest() == resolved_sha

model = trainer_capture["model"]
device = torch.device("cuda:0")
assert next(model.parameters()).device == device
probe = torch.randn(4, 3, 64, 64, device=device)
groups, manifest = module.partition_manifest(model, probe)
assert sum(parameter.numel() for parameter in model.parameters()) == 938_979
assert manifest["POLICY_EXCLUSIVE"]["tensors"] == 2
assert manifest["POLICY_EXCLUSIVE"]["numel"] == 3_855
assert manifest["SHARED"]["tensors"] == 22
assert manifest["SHARED"]["numel"] == 934_864
assert manifest["CRITIC_EXCLUSIVE"]["tensors"] == 2
assert manifest["CRITIC_EXCLUSIVE"]["numel"] == 257
assert manifest["CRITIC_EXCLUSIVE"]["names"] == [
    "last_v_layer.weight", "last_v_layer.bias",
]
assert set(groups) == {"POLICY_EXCLUSIVE", "SHARED", "CRITIC_EXCLUSIVE"}

# The gradient/update vector is exactly the ordered trainable production set.
trainable_named_params = [
    (name, parameter)
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
]
production_optimizer = torch.optim.SGD(
    [parameter for _, parameter in trainable_named_params],
    lr=resolved_trainer["algo_config"]["lr"], momentum=1e-6,
)
optimizer_params = [
    parameter
    for group in production_optimizer.param_groups
    for parameter in group["params"]
]
assert len(trainable_named_params) == len(optimizer_params)
partition_by_name = {
    name: partition
    for partition, entries in groups.items()
    for name, _ in entries
}
all_named_params = list(model.named_parameters())
assert len(all_named_params) == 29
structural_parameters = []
for position, (name, parameter) in enumerate(all_named_params):
    if name in partition_by_name:
        partition = partition_by_name[name]
    else:
        assert name in {
            "last_v_layer.mean", "last_v_layer.mean_sq",
            "last_v_layer.debiasing_term",
        }
        partition = "POPART_NONCURVATURE_STATE"
    optimizer_positions = [
        index for index, candidate in enumerate(optimizer_params)
        if candidate is parameter
    ]
    structural_parameters.append({
        "position": position,
        "name": name,
        "partition": partition,
        "shape": list(parameter.shape),
        "dtype": str(parameter.dtype),
        "requires_grad": parameter.requires_grad,
        "numel": parameter.numel(),
        "trainable_member": parameter.requires_grad,
        "optimizer_member": len(optimizer_positions) == 1,
        "optimizer_position": optimizer_positions[0] if optimizer_positions else None,
    })

structural_manifest = {
    "schema": "hybrid_head_structural_manifest_v1",
    "parameters": structural_parameters,
    "counts": {
        "total": {"tensors": len(all_named_params), "numel": 938_979},
        "trainable": {
            "tensors": len(trainable_named_params),
            "numel": sum(parameter.numel() for _, parameter in trainable_named_params),
        },
        "POLICY_EXCLUSIVE": {
            "tensors": manifest["POLICY_EXCLUSIVE"]["tensors"],
            "numel": manifest["POLICY_EXCLUSIVE"]["numel"],
        },
        "SHARED": {
            "tensors": manifest["SHARED"]["tensors"],
            "numel": manifest["SHARED"]["numel"],
        },
        "CRITIC_EXCLUSIVE": {
            "tensors": manifest["CRITIC_EXCLUSIVE"]["tensors"],
            "numel": manifest["CRITIC_EXCLUSIVE"]["numel"],
        },
        "POPART_NONCURVATURE_STATE": {"tensors": 3, "numel": 3},
    },
    "critic_exclusive_names": manifest["CRITIC_EXCLUSIVE"]["names"],
    "trainable_names": [name for name, _ in trainable_named_params],
    "optimizer_names": [
        name for name, parameter in trainable_named_params
        if any(candidate is parameter for candidate in optimizer_params)
    ],
}
assert structural_manifest["counts"]["trainable"] == {
    "tensors": 26, "numel": 938_976,
}
assert structural_manifest["trainable_names"] == structural_manifest["optimizer_names"]
structural_manifest_path.write_text(
    json.dumps(structural_manifest, indent=2, sort_keys=True) + "\n"
)
# Preserve the caller-requested legacy filename as byte-identical structural
# evidence without putting environment-dependent probe values back into it.
if requested_manifest_path != structural_manifest_path:
    requested_manifest_path.write_bytes(structural_manifest_path.read_bytes())

connectivity_groups = {}
for partition in ("POLICY_EXCLUSIVE", "SHARED", "CRITIC_EXCLUSIVE"):
    expected_names = manifest[partition]["names"]
    connectivity = manifest[partition]["connectivity"]
    assert list(connectivity) == expected_names
    connectivity_groups[partition] = {
        "names": expected_names,
        "probe": connectivity,
    }
    for item in connectivity.values():
        assert math.isfinite(item["policy_jacobian_probe_l2"])
        assert math.isfinite(item["value_jacobian_probe_l2"])
        if partition == "CRITIC_EXCLUSIVE":
            assert item["policy_connected"] is False
            assert item["policy_jacobian_probe_l2"] == 0.0
            assert item["value_connected"] is True
            assert item["value_jacobian_probe_l2"] > 0.0
connectivity_probe = {
    "schema": "hybrid_head_connectivity_probe_v1",
    "environment": env_name,
    "structural_manifest_sha256": sha(structural_manifest_path),
    "resolved_config_sha256": resolved_sha,
    "production_learn_entry": trainer_capture["learn"],
    "partition_names_match_structural": all(
        connectivity_groups[partition]["names"] == [
            item["name"] for item in structural_parameters
            if item["partition"] == partition
        ]
        for partition in connectivity_groups
    ),
    "nan_inf_or_fallback": False,
    "semantic_pass": True,
    "groups": connectivity_groups,
}
assert connectivity_probe["partition_names_match_structural"] is True
connectivity_probe_path.write_text(
    json.dumps(connectivity_probe, indent=2, sort_keys=True) + "\n"
)
optimizer_items = []
for position, ((name, parameter), optimizer_parameter) in enumerate(
    zip(trainable_named_params, optimizer_params)
):
    assert parameter is optimizer_parameter
    optimizer_items.append({
        "position": position,
        "name": name,
        "shape": list(parameter.shape),
        "dtype": str(parameter.dtype),
        "device": str(parameter.device),
        "requires_grad": parameter.requires_grad,
        "object_identity": hex(id(parameter)),
        "optimizer_object_identity": hex(id(optimizer_parameter)),
        "object_identity_equal": parameter is optimizer_parameter,
    })
trainable_names = [name for name, _ in trainable_named_params]
trainable_name_set = set(trainable_names)

popart_names = [
    name for name, parameter in model.named_parameters()
    if not parameter.requires_grad
]
assert popart_names == [
    "last_v_layer.mean", "last_v_layer.mean_sq",
    "last_v_layer.debiasing_term",
]
assert trainable_name_set.isdisjoint(popart_names)
assert all(parameter.requires_grad for parameter in optimizer_params)
model_state_names = set(model.state_dict())
assert set(popart_names).issubset(model_state_names)
trainer_text = trainer.read_text()
assert "actor_critic.last_v_layer.update(ret)" in trainer_text
assert "ret = actor_critic.last_v_layer.normalize(ret)" in trainer_text
assert "adv = actor_critic.last_v_layer.normalize(adv)" in trainer_text
assert "match_head_proposal_norm" not in trainer_text
assert "paper_head_proposal" not in trainer_text
assert "gae_error_operator_apply" not in trainer_text
assert "actor_score_weights" not in trainer_text
popart_initial = {
    name: parameter.detach().cpu().tolist()
    for name, parameter in model.named_parameters()
    if name in popart_names
}
trainable_audit = {
    "trainable_count": len(trainable_named_params),
    "trainable_numel": sum(parameter.numel() for _, parameter in trainable_named_params),
    "optimizer_count": len(optimizer_params),
    "optimizer_numel": sum(parameter.numel() for parameter in optimizer_params),
    "itemwise": optimizer_items,
    "popart_nontrainable_names": popart_names,
    "popart_excluded_from_optimizer_autograd_direction_update": True,
    "popart_original_paper_semantics": [
        "last_v_layer.update(ret)", "normalize(ret)", "normalize(adv)",
    ],
    "popart_before": popart_initial,
}

# Task34R mandatory actual-network/data gate.  Use the production Procgen
# environment retained from the canonical trainer entry and the exact runner
# to obtain frozen lambda-returns; no synthetic substitute enters this proof.
venv = trainer_capture["venv"]
model.eval()
runner = module.GAEMetadataRunner(
    env=venv, model=model, nsteps=256, gamma=.999, lam=.95,
    adv_type="gae", device=device,
)
actual_obs, actual_returns, actual_actions, actual_adv, actual_logits, _ = runner.run()
assert actual_obs.shape[0] == 4096
assert actual_returns.shape == (4096,)
assert torch.isfinite(actual_returns).all()

# Match production PopArt timing and coordinates before forming J/e.
model.last_v_layer.update(actual_returns)
normalized_returns = model.last_v_layer.normalize(actual_returns)
normalized_adv = model.last_v_layer.normalize(actual_adv)
if model.obs_rms is not None:
    model.obs_rms.training = True
    actual_obs = model.obs_rms(actual_obs)
    model.obs_rms.training = False

permutation = np.random.RandomState(340825).permutation(4096)
blocks = [permutation[start:start + 512].copy() for start in range(0, 4096, 512)]
assert len(blocks) == 8 and all(block.shape == (512,) for block in blocks)
assert np.unique(np.concatenate(blocks)).size == 4096
train_indices, validation_indices = blocks[0], blocks[1]
assert np.intersect1d(train_indices, validation_indices).size == 0
train_obs = actual_obs[train_indices]
validation_obs = actual_obs[validation_indices]
train_returns = normalized_returns[train_indices]
validation_returns = normalized_returns[validation_indices]
train_adv = normalized_adv[train_indices]

with torch.no_grad():
    train_latents = model.backbone_net(train_obs).detach()
    validation_latents = model.backbone_net(validation_obs).detach()
    train_values = model.forward_v(latents=train_latents).detach()
    validation_values = model.forward_v(latents=validation_latents).detach()
train_J = torch.cat((train_latents, torch.ones(512, 1, device=device)), dim=1)
validation_J = torch.cat((validation_latents, torch.ones(512, 1, device=device)), dim=1)
assert train_J.shape == validation_J.shape == (512, 257)
train_error = train_values - train_returns
validation_error = validation_values - validation_returns

# D=I, W=I, K=J and the exact standard-MSE gradient.
head_names = manifest["CRITIC_EXCLUSIVE"]["names"]
head_params = [dict(model.named_parameters())[name] for name in head_names]
autograd_loss = 0.5 * (model(train_obs)[0] - train_returns).pow(2).mean()
autograd_head = torch.cat([
    value.reshape(-1) for value in torch.autograd.grad(autograd_loss, head_params)
]).double()
J64 = train_J.double()
error64 = train_error.double()
direct_G = J64.T @ J64 / 512.0
direct_g = J64.T @ error64 / 512.0
assert torch.allclose(autograd_head, direct_g, rtol=2e-5, atol=2e-6)
trace_per_parameter = torch.trace(direct_G) / 257.0
mu = float(trace_per_parameter.item())
(
    direction, solved_G, solved_g, system, rhs, jacobi, info,
    absolute_residual, relative_residual,
) = module.solve_standard_mse_head_fp64(train_J, train_error, mu, 1e-18)
assert torch.equal(solved_G, direct_G)
assert torch.equal(solved_g, direct_g)
assert torch.equal(rhs, -direct_g)
assert int(info.max()) == 0
assert torch.isfinite(relative_residual) and relative_residual < 1e-10

# Same-minibatch equality is exact in FP64 and is explicitly not used for LM.
delta = 0.5 * direction
pred_train = -(direct_g @ delta + 0.5 * delta @ direct_G @ delta)
train_before = 0.5 * error64.square().mean()
train_after = 0.5 * (error64 + J64 @ delta).square().mean()
ared_train = train_before - train_after
assert torch.allclose(ared_train, pred_train, rtol=2e-12, atol=2e-12)

# Fixed opposing train/validation gradients provide non-degenerate acceptance
# and rejection without changing the train-only direction.
aligned_validation_g = direct_g
opposed_validation_g = -direct_g
aligned_ared = -(aligned_validation_g @ delta + 0.5 * delta @ direct_G @ delta)
opposed_ared = -(opposed_validation_g @ delta + 0.5 * delta @ direct_G @ delta)
aligned_rho = aligned_ared / pred_train
opposed_rho = opposed_ared / pred_train
assert pred_train > 0 and aligned_rho >= .25 and opposed_rho < .25
direction_repeat = module.solve_standard_mse_head_fp64(train_J, train_error, mu, 1e-18)[0]
assert torch.equal(direction, direction_repeat)

# Rejected-trial rollback covers parameters, gradients, optimizer/history,
# PopArt state and CPU/CUDA/NumPy RNG bit-for-bit.
production_optimizer = torch.optim.SGD(
    [parameter for _, parameter in trainable_named_params], lr=.5, momentum=1e-6
)
for parameter in production_optimizer.param_groups[0]["params"]:
    parameter.grad = torch.zeros_like(parameter)
production_optimizer.step()
production_optimizer.zero_grad(set_to_none=True)
snapshot = module.capture_trial_state(model, production_optimizer)
offset = 0
with torch.no_grad():
    for name in head_names:
        parameter = dict(model.named_parameters())[name]
        count = parameter.numel()
        parameter.add_(delta[offset:offset + count].view_as(parameter).to(parameter.dtype))
        offset += count
module.restore_trial_state(model, production_optimizer, snapshot)
module.assert_trial_state_equal(model, production_optimizer, snapshot)

# Actual-network actor/shared one-step isolation: policy/shared gradients and
# deltas are byte-identical; only the 257 head parameters use the new solve.
control_model = copy.deepcopy(model)
target_model = copy.deepcopy(model)
policy_names = manifest["POLICY_EXCLUSIVE"]["names"]
shared_names = manifest["SHARED"]["names"]

def control_grads(net):
    values, logits = net(train_obs[:8])
    action_index = torch.as_tensor(train_indices[:8], device=actual_actions.device)
    actor_loss = nn.functional.cross_entropy(logits, actual_actions[action_index])
    critic_loss = nn.functional.mse_loss(values, train_returns[:8])
    named = [(name, parameter) for name, parameter in net.named_parameters() if parameter.requires_grad]
    actor_entries = [(name, parameter) for name, parameter in named if name in set(policy_names + shared_names)]
    critic_entries = [(name, parameter) for name, parameter in named if name in set(shared_names + head_names)]
    actor_values = torch.autograd.grad(actor_loss, [p for _, p in actor_entries], retain_graph=True)
    critic_values = torch.autograd.grad(critic_loss, [p for _, p in critic_entries])
    actor = dict(zip([name for name, _ in actor_entries], actor_values))
    critic = dict(zip([name for name, _ in critic_entries], critic_values))
    full = {}
    for name, parameter in named:
        full[name] = actor.get(name, torch.zeros_like(parameter)) + critic.get(name, torch.zeros_like(parameter))
    return actor, critic, full

control_actor, control_critic, control_full = control_grads(control_model)
target_actor, target_critic, target_full = control_grads(target_model)
for name in policy_names + shared_names:
    assert torch.equal(control_actor[name], target_actor[name])
for name in shared_names:
    assert torch.equal(control_critic[name], target_critic[name])
ordered_names = [name for name, parameter in control_model.named_parameters() if parameter.requires_grad]
global_norm = torch.linalg.vector_norm(torch.cat([control_full[name].reshape(-1) for name in ordered_names]))
clip_scale = min(1.0, 0.5 / float(global_norm + 1e-6))
control_named = dict(control_model.named_parameters())
target_named = dict(target_model.named_parameters())
with torch.no_grad():
    for name in ordered_names:
        control_named[name].add_(control_full[name], alpha=-.5 * clip_scale)
        if name in head_names:
            continue
        target_named[name].add_(target_full[name], alpha=-.5 * clip_scale)
    offset = 0
    for name in head_names:
        count = target_named[name].numel()
        target_named[name].add_(direction[offset:offset + count].view_as(target_named[name]), alpha=.5 * clip_scale)
        offset += count
for name in policy_names + shared_names:
    assert torch.equal(control_named[name], target_named[name])
assert any(not torch.equal(control_named[name], target_named[name]) for name in head_names)
assert torch.equal(control_model(train_obs[:8])[1], target_model(train_obs[:8])[1])

# PopArt affine reward-scale regression in normalized coordinates.
affine_scale, affine_shift = 2.75, 3.25
mean_a, std_a = 1.5, 4.0
raw_values = torch.randn(512, device=device, dtype=torch.float64)
raw_returns = torch.randn(512, device=device, dtype=torch.float64)
normalized_error_a = (raw_values - mean_a) / std_a - (raw_returns - mean_a) / std_a
mean_b = affine_scale * mean_a + affine_shift
std_b = affine_scale * std_a
normalized_error_b = (
    (affine_scale * raw_values + affine_shift - mean_b) / std_b
    - (affine_scale * raw_returns + affine_shift - mean_b) / std_b
)
assert torch.allclose(normalized_error_a, normalized_error_b, rtol=1e-13, atol=1e-13)

popart_after = {
    name: parameter.detach().cpu().tolist()
    for name, parameter in model.named_parameters() if name in popart_names
}
trainable_audit.update({
    "popart_after_actual_rollout_update": popart_after,
    "actual_rollout_rows": 4096,
    "train_rows": 512,
    "validation_rows": 512,
    "train_validation_overlap": 0,
})
(evidence_dir / "trainable_optimizer_popart_manifest.json").write_text(
    json.dumps(trainable_audit, indent=2, sort_keys=True) + "\n"
)
actual_ledger = {
    "environment": env_name,
    "D": "I", "W": "I", "K": "J",
    "gaussian_precision": 1.0,
    "rows_train": 512,
    "rows_validation": 512,
    "schedule_blocks": 8,
    "train_validation_overlap": 0,
    "trace_G": float(torch.trace(direct_G)),
    "trace_G_per_parameter": float(trace_per_parameter),
    "mu_alpha1": mu,
    "gradient_max_abs_error": float((autograd_head - direct_g).abs().max()),
    "same_minibatch_ared_pred_abs_error": float((ared_train - pred_train).abs()),
    "aligned_rho_accept": float(aligned_rho),
    "opposed_rho_reject": float(opposed_rho),
    "rollback_bit_identical": True,
    "train_only_direction_invariant_to_validation": True,
    "actor_shared_parameters_bit_identical": True,
    "policy_logits_bit_identical": True,
    "only_critic_exclusive_head_scientifically_different": True,
    "cholesky_info_max": int(info.max()),
    "relative_residual": float(relative_residual),
    "nan_inf_or_fallback": False,
}
(evidence_dir / "standard_mse_cvlm_actual_data_ledger.json").write_text(
    json.dumps(actual_ledger, indent=2, sort_keys=True) + "\n"
)
venv.close()

props = torch.cuda.get_device_properties(0)
assert "H100" in props.name or "H200" in props.name, props.name
assert props.total_memory >= 70_000_000_000, props.total_memory
print("GPUH_STANDARD_MSE_GGN_HEAD_CVLM_COMPATIBILITY_PASS")
print("canonical_production_config_path=PASS")
print("resolved_configuration_three_way=BIT_IDENTICAL")
print(f"resolved_config_sha256={resolved_sha}")
print("actual_procgen_rollout_lambda_returns=PASS")
print("standard_mse_D_I_W_I_K_J=PASS")
print("complete_8x512_schedule_and_disjoint_next_minibatch=PASS")
print(f"same_minibatch_ared_pred_abs_error={abs(float(ared_train - pred_train)):.3e}")
print(f"opposing_case_accept_rho={float(aligned_rho):.6g}")
print(f"opposing_case_reject_rho={float(opposed_rho):.6g}")
print("rejected_trial_rollback=BIT_IDENTICAL")
print("accepted_direction_train_rows_only=PASS")
print("actor_shared_directions_deltas_logits=BIT_IDENTICAL")
print("only_257_head_parameters_differ_scientifically=PASS")
print("popart_affine_reward_scale=PASS")
print(f"head_fp64_cholesky_info_max={int(info.max())}")
print(f"head_fp64_relative_residual={float(relative_residual):.3e}")
print(f"gpu={props.name} total_bytes={props.total_memory}")
sys.exit(0)

# Actual production-network one-step isolation proof.
torch.manual_seed(824)
paper_model = copy.deepcopy(model)
target_model = copy.deepcopy(model)
obs = torch.randn(8, 3, 64, 64, device=device)
actions = torch.randint(0, 15, (8,), device=device)
returns = torch.randn(8, device=device)
policy_names = manifest["POLICY_EXCLUSIVE"]["names"]
shared_names = manifest["SHARED"]["names"]
head_names = manifest["CRITIC_EXCLUSIVE"]["names"]


def raw_grads(net):
    values, logits = net(obs)
    actor_loss = nn.functional.cross_entropy(logits, actions)
    critic_loss = nn.functional.mse_loss(values, returns)
    named = [(name, parameter) for name, parameter in net.named_parameters() if parameter.requires_grad]
    assert [name for name, _ in named] == trainable_names
    actor_named = [(name, parameter) for name, parameter in named if name in set(policy_names + shared_names)]
    critic_named = [(name, parameter) for name, parameter in named if name in set(shared_names + head_names)]
    actor_values = torch.autograd.grad(
        actor_loss, [parameter for _, parameter in actor_named], retain_graph=True
    )
    critic_values = torch.autograd.grad(
        critic_loss, [parameter for _, parameter in critic_named]
    )
    actor = dict(zip([name for name, _ in actor_named], actor_values))
    critic = dict(zip([name for name, _ in critic_named], critic_values))
    full = {}
    for name, _ in named:
        if name in actor and name in critic:
            full[name] = actor[name] + critic[name]
        elif name in actor:
            full[name] = actor[name]
        else:
            assert name in critic
            full[name] = critic[name]
    assert list(full) == trainable_names
    return actor, critic, full


paper_actor, paper_critic, paper_full = raw_grads(paper_model)
target_actor, target_critic, target_full = raw_grads(target_model)
paper_named = dict(paper_model.named_parameters())
target_named = dict(target_model.named_parameters())
for name in policy_names + shared_names:
    assert torch.equal(paper_actor[name], target_actor[name])
for name in shared_names:
    assert torch.equal(paper_critic[name], target_critic[name])

paper_norm = torch.linalg.vector_norm(torch.cat([paper_full[name].flatten() for name in trainable_names]))
clip = min(1.0, 0.5 / float(paper_norm + 1e-6))
# Build the actual target head proposal on a multi-episode temporal batch.
num_envs, nsteps = 2, 4
temporal_obs = obs[:num_envs * nsteps]
temporal_actions = actions[:num_envs * nsteps]
temporal_returns = returns[:num_envs * nsteps]
next_nonterminal = torch.tensor(
    [1, 0, 1, 0, 1, 1, 0, 1], device=device, dtype=torch.float32
)
head_entries = [(name, target_named[name]) for name in head_names]
head_params = [parameter for _, parameter in head_entries]
values, _ = target_model(temporal_obs)
head_rows_list = []
for row in range(num_envs * nsteps):
    row_grads = torch.autograd.grad(values[row], head_params, retain_graph=True)
    head_rows_list.append(torch.cat([item.reshape(-1) for item in row_grads]))
J_head = torch.stack(head_rows_list)
analytic_latents = target_model.backbone_net(temporal_obs).detach()
analytic_J_head = torch.cat((
    analytic_latents,
    torch.ones((num_envs * nsteps, 1), device=device, dtype=analytic_latents.dtype),
), dim=-1)
assert torch.equal(J_head, analytic_J_head)
gae_J = module.gae_error_operator_apply(
    J_head, next_nonterminal, num_envs, nsteps, .999, .95
)
gae_q = module.gae_error_operator_apply(
    values.detach() - temporal_returns, next_nonterminal,
    num_envs, nsteps, .999, .95,
)
K = gae_J
r = gae_q
direction64, target_system, target_rhs, target_jacobi, target_info, target_abs_residual, target_relative = module.solve_gae_head_primal_fp64(
    K, r, num_envs * nsteps, .5, 1e-18
)
assert int(target_info.max()) == 0
assert torch.isfinite(target_relative) and target_relative < 1e-10
head_direction = direction64.float()
replacement = {}
offset = 0
for name, parameter in head_entries:
    replacement[name] = -head_direction[offset:offset + parameter.numel()].view_as(parameter)
    offset += parameter.numel()
assert offset == 257
with torch.no_grad():
    for name, parameter in paper_model.named_parameters():
        if parameter.requires_grad:
            parameter.add_(paper_full[name], alpha=-0.5 * clip)
    for name, parameter in target_model.named_parameters():
        if parameter.requires_grad:
            gradient = replacement[name] if name in replacement else target_full[name]
            parameter.add_(gradient, alpha=-0.5 * clip)
for name in policy_names + shared_names:
    assert torch.equal(dict(paper_model.named_parameters())[name], dict(target_model.named_parameters())[name])
assert all(
    not torch.equal(dict(paper_model.named_parameters())[name], dict(target_model.named_parameters())[name])
    for name in head_names
)
paper_logits = paper_model(obs)[1]
target_logits = target_model(obs)[1]
assert torch.equal(paper_logits, target_logits)
paper_popart_after = {
    name: parameter.detach().cpu().tolist()
    for name, parameter in paper_model.named_parameters()
    if name in popart_names
}
target_popart_after = {
    name: parameter.detach().cpu().tolist()
    for name, parameter in target_model.named_parameters()
    if name in popart_names
}
assert paper_popart_after == popart_initial
assert target_popart_after == popart_initial
trainable_audit["paper_popart_after"] = paper_popart_after
trainable_audit["target_popart_after"] = target_popart_after
(evidence_dir / "trainable_optimizer_popart_manifest.json").write_text(
    json.dumps(trainable_audit, indent=2, sort_keys=True) + "\n"
)

# Paper actor and sampled shared-critic systems use real production-model rows
# and remain literally identical between Paper and Target.
B = 4
P = sum(parameter.numel() for _, parameter in trainable_named_params)
torch.manual_seed(825)
score_rows = []
score_obs = obs[:B]
score_actions = actions[:B]
score_noise = torch.randn(B, device=device)
score_params = [parameter for _, parameter in trainable_named_params]
for row_index in range(B):
    values, logits = model(score_obs[row_index:row_index + 1])
    policy_score = nn.functional.log_softmax(logits, dim=-1)[0, score_actions[row_index]]
    sampled_value = (values.reshape(-1)[0] + score_noise[row_index]).detach()
    value_score = -(values.reshape(-1)[0] - sampled_value).pow(2)
    row_grads = torch.autograd.grad(policy_score + value_score, score_params)
    score_rows.append(torch.cat([gradient.reshape(-1) for gradient in row_grads]))
actual_rows = torch.stack(score_rows)
assert actual_rows.shape == (B, P)
adv = torch.randn(B, device=device)
ratio = torch.rand(B, device=device) + 0.2
actor_system = actual_rows @ actual_rows.T / B @ torch.diag(ratio) + 0.5 * torch.eye(B, device=device)
actor_rhs = adv.clone()
paper_direction = torch.linalg.solve(actor_system, actor_rhs)
target_direction = torch.linalg.solve(actor_system, actor_rhs)
assert torch.equal(paper_direction, target_direction)
critic_system = actual_rows @ actual_rows.T / B + 0.5 * torch.eye(B, device=device)
critic_rhs = torch.ones(B, device=device)
paper_shared_direction = torch.linalg.solve(critic_system, critic_rhs)
target_shared_direction = torch.linalg.solve(critic_system, critic_rhs)
assert torch.equal(paper_shared_direction, target_shared_direction)
del actual_rows, score_rows, paper_model, target_model
torch.cuda.empty_cache()

# Exact GAE finite difference with episode boundaries and final bootstrap mask.
torch.manual_seed(826)
fd_values = torch.randn(num_envs, nsteps, device=device, dtype=torch.float64)
fd_rewards = torch.randn_like(fd_values)
fd_bootstrap = torch.randn(num_envs, device=device, dtype=torch.float64)
fd_masks = next_nonterminal.double().reshape(num_envs, nsteps)

def recompute_gae(values_nt):
    result = torch.zeros_like(values_nt)
    recursion = torch.zeros(num_envs, device=device, dtype=torch.float64)
    for time_index in reversed(range(nsteps)):
        next_value = fd_bootstrap if time_index + 1 == nsteps else values_nt[:, time_index + 1]
        delta = fd_rewards[:, time_index] + .999 * fd_masks[:, time_index] * next_value - values_nt[:, time_index]
        recursion = delta + .999 * .95 * fd_masks[:, time_index] * recursion
        result[:, time_index] = recursion
    return result

fd_direction = torch.randn_like(fd_values)
epsilon = 1e-6
finite_difference = (recompute_gae(fd_values + epsilon * fd_direction) - recompute_gae(fd_values)) / epsilon
operator_difference = module.gae_error_operator_apply(
    fd_direction.reshape(-1), next_nonterminal.double(), num_envs, nsteps, .999, .95
).reshape(num_envs, nsteps)
fd_error = torch.max(torch.abs(finite_difference - operator_difference))
assert fd_error < 2e-9, fd_error

# W=I is exact and consumes no policy probabilities or RNG.
rng_before = torch.cuda.get_rng_state()
implicit_weights = torch.ones(num_envs * nsteps, device=device)
rng_after = torch.cuda.get_rng_state()
assert torch.equal(implicit_weights, torch.ones_like(implicit_weights))
assert implicit_weights.min().item() == 1.0
assert implicit_weights.max().item() == 1.0
assert torch.equal(rng_before, rng_after)
trainer_source = trainer.read_text()
for forbidden_weight_path in (
    "actor_score_weights", "actor_weight_", "sqrt_weights",
    "one_hot - probabilities", "weight clipping", "weight floor",
):
    assert forbidden_weight_path not in trainer_source, forbidden_weight_path

# Matrix, RHS, explicit Hessian-vector, and direct reference agree.
direct_system = K.double().T @ K.double() / K.shape[0] + .5 * torch.eye(257, device=device, dtype=torch.float64)
direct_rhs = -(K.double().T @ r.double()) / K.shape[0]
direct_direction = torch.linalg.solve(direct_system, direct_rhs)
assert torch.allclose(target_system, direct_system, rtol=0, atol=1e-12)
assert torch.allclose(target_rhs, direct_rhs, rtol=0, atol=1e-12)
assert torch.allclose(direction64, direct_direction, rtol=1e-11, atol=1e-12)
probe_vector = torch.randn(257, device=device, dtype=torch.float64)
theta = torch.zeros(257, device=device, dtype=torch.float64, requires_grad=True)
K64, r64 = K.double(), r.double()
objective = .5 * (r64 + K64 @ theta).pow(2).mean()
gradient = torch.autograd.grad(objective, theta, create_graph=True)[0]
hvp = torch.autograd.grad((gradient * probe_vector).sum(), theta)[0]
assert torch.allclose(hvp, (K64.T @ K64 / K.shape[0]) @ probe_vector, rtol=1e-11, atol=1e-12)

# PopArt affine reward transform yields the identical normalized problem.
raw_mean, raw_scale = 3.25, 2.75
normalized_values = (fd_values - raw_mean) / raw_scale
normalized_returns = (fd_values + torch.randn_like(fd_values) - raw_mean) / raw_scale
transformed_values = (raw_scale * fd_values + raw_mean - (raw_scale * raw_mean + raw_mean)) / (raw_scale * raw_scale)
transformed_returns = (raw_scale * (raw_mean + raw_scale * normalized_returns) + raw_mean - (raw_scale * raw_mean + raw_mean)) / (raw_scale * raw_scale)
popart_input_tolerance = 16 * torch.finfo(torch.float64).eps
popart_value_error = torch.max(torch.abs(normalized_values - transformed_values))
popart_return_error = torch.max(torch.abs(normalized_returns - transformed_returns))
assert popart_value_error <= popart_input_tolerance, popart_value_error
assert popart_return_error <= popart_input_tolerance, popart_return_error
popart_error_a = (normalized_values - normalized_returns).reshape(-1)
popart_error_b = (transformed_values - transformed_returns).reshape(-1)
popart_q_a = module.gae_error_operator_apply(
    popart_error_a, next_nonterminal.double(), num_envs, nsteps, .999, .95
)
popart_q_b = module.gae_error_operator_apply(
    popart_error_b, next_nonterminal.double(), num_envs, nsteps, .999, .95
)
popart_K = gae_J.double()
popart_direction_a = module.solve_gae_head_primal_fp64(
    popart_K, popart_q_a, popart_K.shape[0], .5, 1e-18
)[0]
popart_direction_b = module.solve_gae_head_primal_fp64(
    popart_K, popart_q_b, popart_K.shape[0], .5, 1e-18
)[0]
popart_direction_error = torch.max(torch.abs(popart_direction_a - popart_direction_b))
popart_prediction_error = torch.max(
    torch.abs(popart_K @ popart_direction_a - popart_K @ popart_direction_b)
)
assert torch.allclose(popart_direction_a, popart_direction_b, rtol=1e-12, atol=1e-13), popart_direction_error
assert torch.allclose(
    popart_K @ popart_direction_a,
    popart_K @ popart_direction_b,
    rtol=1e-12,
    atol=1e-13,
), popart_prediction_error
accept_a = torch.isfinite(popart_direction_a).all() and torch.linalg.vector_norm(popart_direction_a) > 0
accept_b = torch.isfinite(popart_direction_b).all() and torch.linalg.vector_norm(popart_direction_b) > 0
assert bool(accept_a) == bool(accept_b)
print(
    "TASK33_POPART_AFFINE_PASS",
    f"input_tolerance={popart_input_tolerance:.17g}",
    f"value_error={popart_value_error.item():.17g}",
    f"return_error={popart_return_error.item():.17g}",
    f"direction_error={popart_direction_error.item():.17g}",
    f"prediction_error={popart_prediction_error.item():.17g}",
)

# Production-scale representative footprint for full-rollout J and primal solve.
props = torch.cuda.get_device_properties(0)
assert "H100" in props.name or "H200" in props.name, props.name
assert props.total_memory >= 70_000_000_000, props.total_memory
torch.manual_seed(827)
B, paper_p, head_p = 4096, P, 257
paper_rows = torch.empty((B, paper_p), device=device, dtype=torch.float32).normal_()
head_rows = torch.empty((B, head_p), device=device, dtype=torch.float32).normal_()
rhs = torch.randn(B, device=device)
_, system, _, _, info, _, relative = module.solve_gae_head_primal_fp64(
    head_rows, rhs, B, .5, 1e-18
)
peak = torch.cuda.max_memory_allocated()
assert torch.isfinite(relative) and relative.item() < 1e-10
assert peak < props.total_memory * 0.35, (peak, props.total_memory)

print("GPUH_GAE_GGN_HEAD_WIDENTITY_COMPATIBILITY_PASS")
print("canonical_production_config_path=PASS")
print("resolved_configuration_three_way=BIT_IDENTICAL")
print(f"resolved_config_sha256={resolved_sha}")
print("actual_network_partition=EXHAUSTIVE_MUTUALLY_EXCLUSIVE_STABLE")
print(f"structural_manifest_sha256={sha(structural_manifest_path)}")
print(f"connectivity_probe_sha256={sha(connectivity_probe_path)}")
print(f"connectivity_environment={env_name}")
print("connectivity_semantic_check=PASS")
print("critic_exclusive_policy_jacobian=EXACT_ZERO_DISCONNECTED")
print("actual_network_paper_actor_direction=BIT_IDENTICAL")
print("actual_network_paper_shared_critic_direction=BIT_IDENTICAL")
print("actual_network_one_step_policy_parameters=BIT_IDENTICAL")
print("actual_network_one_step_policy_logits=BIT_IDENTICAL")
print("actual_network_only_value_head_delta=DIFFERS")
print(f"gae_operator_finite_difference_max_abs={fd_error.item():.3e}")
print("identity_weight_formula=EXACT_ALL_ONES_NO_POLICY_PROBABILITY_NO_RNG")
print("task32_weight_max_512_concentration_path=ABSENT")
print("gae_ggn_matrix_rhs_hvp_direct_reference=PASS")
print("popart_affine_normalized_problem=BIT_IDENTICAL")
print(f"gpu={props.name} total_bytes={props.total_memory} peak_bytes={peak}")
print(f"paper_rows={tuple(paper_rows.shape)} head_rows={tuple(head_rows.shape)}")
print(f"head_fp64_cholesky_info_max={int(info.max())}")
print(f"head_fp64_relative_residual={relative.item():.3e}")
print(f"partition_policy={manifest['POLICY_EXCLUSIVE']['numel']} shared={manifest['SHARED']['numel']} head={manifest['CRITIC_EXCLUSIVE']['numel']}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
