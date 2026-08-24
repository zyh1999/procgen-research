#!/usr/bin/env python3
"""Canonical non-training H200 and production-model compatibility gate."""
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import gym3  # noqa: F401
import procgen  # noqa: F401
import torch
from torch import nn


trainer, config, manifest_path = map(Path, sys.argv[1:4])
expected_trainer, expected_config = sys.argv[4:6]
campaign = trainer.parent.parent
launcher = campaign / "frozen/hybrid_head_detggn_6m_gpuh.sbatch"
expected_launcher = "ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e"
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
    path = manifest_path.parent / f"resolved_config_{label}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(payload_bytes).hexdigest() == resolved_sha

model = trainer_capture["model"]
device = torch.device("cuda:0")
assert next(model.parameters()).device == device
probe = torch.randn(4, 3, 64, 64, device=device)
groups, manifest = module.partition_manifest(model, probe)
manifest["resolved_config_sha256"] = resolved_sha
manifest["production_learn_entry"] = trainer_capture["learn"]
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
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
assert sha(manifest_path) == "b45298be8fc5bdccfa36ce653c7dbc0c41f2d013f23d4f4db0a4a580035f3087"
assert set(groups) == {"POLICY_EXCLUSIVE", "SHARED", "CRITIC_EXCLUSIVE"}
for item in manifest["CRITIC_EXCLUSIVE"]["connectivity"].values():
    assert item["policy_connected"] is False
    assert item["policy_jacobian_probe_l2"] == 0.0

# Actual production-network one-step isolation proof.
torch.manual_seed(824)
paper_model = copy.deepcopy(model)
target_model = copy.deepcopy(model)
obs = torch.randn(8, 3, 64, 64, device=device)
actions = torch.randint(0, 15, (8,), device=device)
returns = torch.randn(8, device=device)


def raw_grads(net):
    values, logits = net(obs)
    actor_loss = nn.functional.cross_entropy(logits, actions)
    critic_loss = nn.functional.mse_loss(values, returns)
    params = list(net.parameters())
    actor = torch.autograd.grad(actor_loss, params, retain_graph=True, allow_unused=True)
    critic = torch.autograd.grad(critic_loss, params, allow_unused=True)
    full = [
        (a if a is not None else torch.zeros_like(p))
        + (c if c is not None else torch.zeros_like(p))
        for p, a, c in zip(params, actor, critic)
    ]
    return actor, critic, full


paper_actor, paper_critic, paper_full = raw_grads(paper_model)
target_actor, target_critic, target_full = raw_grads(target_model)
paper_named = dict(paper_model.named_parameters())
target_named = dict(target_model.named_parameters())
ordered_names = list(paper_named)
index = {name: i for i, name in enumerate(ordered_names)}
policy_names = manifest["POLICY_EXCLUSIVE"]["names"]
shared_names = manifest["SHARED"]["names"]
head_names = manifest["CRITIC_EXCLUSIVE"]["names"]
for name in policy_names + shared_names:
    i = index[name]
    if paper_actor[i] is not None or target_actor[i] is not None:
        assert torch.equal(paper_actor[i], target_actor[i])
for name in shared_names:
    i = index[name]
    assert torch.equal(paper_critic[i], target_critic[i])

paper_norm = torch.linalg.vector_norm(torch.cat([value.flatten() for value in paper_full]))
clip = min(1.0, 0.5 / float(paper_norm + 1e-6))
replacement = {name: torch.randn_like(target_named[name]) for name in head_names}
with torch.no_grad():
    for parameter, gradient in zip(paper_model.parameters(), paper_full):
        parameter.add_(gradient, alpha=-0.5 * clip)
    for name, parameter in target_model.named_parameters():
        gradient = replacement[name] if name in replacement else target_full[index[name]]
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

# Paper actor and sampled shared-critic systems use real production-model rows
# and remain literally identical between Paper and Target.
B = 4
P = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
torch.manual_seed(825)
score_rows = []
score_obs = obs[:B]
score_actions = actions[:B]
score_noise = torch.randn(B, device=device)
score_params = list(model.parameters())
for row_index in range(B):
    values, logits = model(score_obs[row_index:row_index + 1])
    policy_score = nn.functional.log_softmax(logits, dim=-1)[0, score_actions[row_index]]
    sampled_value = (values.reshape(-1)[0] + score_noise[row_index]).detach()
    value_score = -(values.reshape(-1)[0] - sampled_value).pow(2)
    row_grads = torch.autograd.grad(
        policy_score + value_score, score_params, allow_unused=True
    )
    score_rows.append(torch.cat([
        (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
        for parameter, gradient in zip(score_params, row_grads)
    ]))
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
print("critic_exclusive_policy_jacobian=EXACT_ZERO_DISCONNECTED")
print("actual_network_paper_actor_direction=BIT_IDENTICAL")
print("actual_network_paper_shared_critic_direction=BIT_IDENTICAL")
print("actual_network_one_step_policy_parameters=BIT_IDENTICAL")
print("actual_network_one_step_policy_logits=BIT_IDENTICAL")
print("actual_network_only_value_head_delta=DIFFERS")
print(f"gpu={props.name} total_bytes={props.total_memory} peak_bytes={peak}")
print(f"paper_rows={tuple(paper_rows.shape)} head_rows={tuple(head_rows.shape)}")
print(f"head_fp64_cholesky_info_max={int(info.max())}")
print(f"head_fp64_relative_residual={relative.item():.3e}")
print(f"partition_policy={manifest['POLICY_EXCLUSIVE']['numel']} shared={manifest['SHARED']['numel']} head={manifest['CRITIC_EXCLUSIVE']['numel']}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
