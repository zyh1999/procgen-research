#!/usr/bin/env python3
"""Canonical non-training H200 and production-model compatibility gate."""
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import gym3  # noqa: F401
import procgen  # noqa: F401
import torch
from torch import nn


trainer, config, requested_manifest_path = map(Path, sys.argv[1:4])
evidence_dir = requested_manifest_path.parent
structural_manifest_path = evidence_dir / "structural_manifest.json"
connectivity_probe_path = evidence_dir / "connectivity_probe.json"
expected_trainer, expected_config = sys.argv[4:6]
campaign = trainer.parent.parent
launcher = campaign / "frozen/normmatch_v2_6m_gpuh.sbatch"
expected_launcher = "85e12886ce5cf81fd98647aa5163319a50174a39210cbeea1ccfde015aaf9d19"
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
assert sha(launcher) == expected_launcher
launcher_text = launcher.read_text()
expected_cmd = 'CMD=("$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name "$ENV_NAME" --seed 0 --device 0)'
assert expected_cmd in launcher_text

sys.path.insert(0, str(trainer.parent))
spec = importlib.util.spec_from_file_location("hybrid_trainer", trainer)
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
        if hasattr(venv, "close"):
            venv.close()
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
        assert "model" in captured and "learn" in captured
    return captured


# All three identities are resolved by the trainer's own main().
preflight_capture = invoke_main(capture_model=False)
launcher_capture = invoke_main(capture_model=False)
trainer_capture = invoke_main(capture_model=True)
resolved_preflight = preflight_capture["resolved"]
resolved_launcher = launcher_capture["resolved"]
resolved_trainer = trainer_capture["resolved"]
assert resolved_preflight == resolved_launcher == resolved_trainer
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
assert "match_head_proposal_norm(head_direction, paper_head_proposal)" in trainer_text
assert "head_target_proposal" in trainer_text
assert "paper_global_preclip_norm" in trainer_text
assert "target_global_preclip_norm" in trainer_text
assert "normmatch_identity='paper_head_norm_exact_no_cap_floor_ema_guard'" in trainer_text
reference_v1 = campaign / "frozen/reference_hybrid_v1.py"
assert sha(reference_v1) == "7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54"
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

# Actual production-network one-step norm-match isolation proof.
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
initial_paper_head = {name: paper_named[name].detach().clone() for name in head_names}
initial_target_head = {name: target_named[name].detach().clone() for name in head_names}
for name in policy_names + shared_names:
    assert torch.equal(paper_actor[name], target_actor[name])
for name in shared_names:
    assert torch.equal(paper_critic[name], target_critic[name])

head_parameters = [target_named[name] for name in head_names]
head_rows = []
target_values = target_model(obs)[0]
for row_index in range(obs.shape[0]):
    value = target_model(obs[row_index:row_index + 1])[0].reshape(-1)[0]
    row = torch.autograd.grad(value, head_parameters)
    head_rows.append(torch.cat([gradient.reshape(-1) for gradient in row]))
head_jacobian = torch.stack(head_rows)
head_weight = math.sqrt(0.1)
weighted_rows = head_weight * head_jacobian
head_rhs = (returns - target_values).detach() / head_weight
head_solution = module.solve_head_critic_b_fp64(
    weighted_rows, head_rhs, obs.shape[0], 0.5, 257, 1e-18,
)
det_proposal = module.chunked_transpose_mv_fp64(
    weighted_rows, head_solution[0], obs.shape[0], 257,
)
paper_head_proposal = -torch.cat([
    paper_full[name].reshape(-1) for name in head_names
])
rng_before_match = torch.cuda.get_rng_state(device).clone()
matched = module.match_head_proposal_norm(det_proposal, paper_head_proposal)
target_head_proposal, det_norm, paper_head_norm, normmatch_scale, target_head_norm, proposal_cosine = matched
rng_after_match = torch.cuda.get_rng_state(device)
assert torch.equal(rng_before_match, rng_after_match)
assert torch.allclose(target_head_norm, paper_head_norm, rtol=2e-6, atol=2e-8)

replacement = {}
offset = 0
for name in head_names:
    count = target_named[name].numel()
    replacement[name] = -target_head_proposal[offset:offset + count].view_as(target_named[name])
    offset += count
assert offset == target_head_proposal.numel()
target_preclip = {
    name: replacement[name] if name in replacement else target_full[name]
    for name in trainable_names
}
paper_norm = torch.linalg.vector_norm(torch.cat([
    paper_full[name].flatten() for name in trainable_names
]))
target_norm = torch.linalg.vector_norm(torch.cat([
    target_preclip[name].flatten() for name in trainable_names
]))
assert torch.allclose(target_norm, paper_norm, rtol=2e-6, atol=2e-8)
paper_clip = min(1.0, 0.5 / float(paper_norm + 1e-6))
target_clip = paper_clip
assert target_clip == paper_clip
with torch.no_grad():
    for name, parameter in paper_model.named_parameters():
        if parameter.requires_grad:
            parameter.add_(paper_full[name], alpha=-0.5 * paper_clip)
    for name, parameter in target_model.named_parameters():
        if parameter.requires_grad:
            gradient = replacement[name] if name in replacement else target_full[name]
            parameter.add_(gradient, alpha=-0.5 * target_clip)
for name in policy_names + shared_names:
    assert torch.equal(dict(paper_model.named_parameters())[name], dict(target_model.named_parameters())[name])
assert all(
    not torch.equal(dict(paper_model.named_parameters())[name], dict(target_model.named_parameters())[name])
    for name in head_names
)
paper_head_delta = torch.cat([
    (dict(paper_model.named_parameters())[name] - initial_paper_head[name]).reshape(-1)
    for name in head_names
])
target_head_delta = torch.cat([
    (dict(target_model.named_parameters())[name] - initial_target_head[name]).reshape(-1)
    for name in head_names
])
assert torch.allclose(
    torch.linalg.vector_norm(target_head_delta),
    torch.linalg.vector_norm(paper_head_delta),
    rtol=2e-6, atol=2e-8,
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

# Production-scale representative Paper/head footprint and exact head solver.
props = torch.cuda.get_device_properties(0)
assert "H100" in props.name or "H200" in props.name, props.name
assert props.total_memory >= 70_000_000_000, props.total_memory
torch.manual_seed(826)
B, paper_p, head_p = 512, P, 257
paper_rows = torch.empty((B, paper_p), device=device, dtype=torch.float32).normal_()
head_rows = torch.empty((B, head_p), device=device, dtype=torch.float32).normal_()
head_kernel = head_rows.double() @ head_rows.double().T / B
system = head_kernel + 0.5 * torch.eye(B, device=device, dtype=torch.float64)
jacobi = torch.diagonal(system).rsqrt()
chol, info = torch.linalg.cholesky_ex(jacobi[:, None] * system * jacobi[None, :])
assert int(info.max()) == 0
rhs = torch.randn(B, device=device, dtype=torch.float64)
alpha = jacobi * torch.cholesky_solve((jacobi * rhs)[:, None], chol).squeeze(1)
relative = torch.linalg.vector_norm(system @ alpha - rhs) / torch.linalg.vector_norm(rhs)
peak = torch.cuda.max_memory_allocated()
assert torch.isfinite(relative) and relative.item() < 1e-10
assert peak < props.total_memory * 0.35, (peak, props.total_memory)

print("GPUH_HYBRID_HEAD_COMPATIBILITY_PASS")
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
print("actual_network_head_proposal_normmatch=PASS")
print("actual_network_global_preclip_norm=COUNTERFACTUAL_PAPER_MATCH")
print("actual_network_global_clip_scale=BIT_IDENTICAL_REUSED")
print("actual_network_rng_state_across_normmatch=BIT_IDENTICAL")
print(f"head_det_proposal_l2={det_norm.item():.9e}")
print(f"head_paper_proposal_l2={paper_head_norm.item():.9e}")
print(f"head_normmatch_scale={normmatch_scale.item():.9e}")
print(f"head_target_proposal_l2={target_head_norm.item():.9e}")
print(f"head_det_paper_proposal_cosine={proposal_cosine.item():.9e}")
print(f"gpu={props.name} total_bytes={props.total_memory} peak_bytes={peak}")
print(f"paper_rows={tuple(paper_rows.shape)} head_rows={tuple(head_rows.shape)}")
print(f"head_fp64_cholesky_info_max={int(info.max())}")
print(f"head_fp64_relative_residual={relative.item():.3e}")
print(f"partition_policy={manifest['POLICY_EXCLUSIVE']['numel']} shared={manifest['SHARED']['numel']} head={manifest['CRITIC_EXCLUSIVE']['numel']}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
