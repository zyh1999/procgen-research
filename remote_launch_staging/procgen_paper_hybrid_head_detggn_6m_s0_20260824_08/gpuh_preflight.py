#!/usr/bin/env python3
"""Non-training H200 compatibility and actual-network partition gate."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import procgen  # noqa: F401
import gym3  # noqa: F401

trainer, config, manifest_path = map(Path, sys.argv[1:4])
expected_trainer, expected_config = sys.argv[4:6]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(trainer) == expected_trainer
assert sha(config) == expected_config

spec = importlib.util.spec_from_file_location("hybrid_trainer", trainer)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from utils.utils import build_resnet, SharedActorCritic

props = torch.cuda.get_device_properties(0)
assert "H100" in props.name or "H200" in props.name, props.name
assert props.total_memory >= 70_000_000_000, props.total_memory
device = torch.device("cuda:0")
fn_net, _ = build_resnet(64, 256, depths=[8, 16], with_bn=False, device=device)
nets = SimpleNamespace(hidden_size=256, dropout=0.0)
model = SharedActorCritic(
    fn_net, (3, 64, 64), nets_config=nets, n_actions=15,
    with_popart=True, sigma_type="vector", device=device,
).to(device)
probe = torch.randn(4, 3, 64, 64, device=device)
groups, manifest = module.partition_manifest(model, probe)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
assert manifest["CRITIC_EXCLUSIVE"]["numel"] == 257
assert manifest["POLICY_EXCLUSIVE"]["numel"] == 3855
assert manifest["SHARED"]["numel"] > 1_000_000

torch.manual_seed(824)
B, paper_p, head_p = 512, sum(p.numel() for p in model.parameters() if p.requires_grad), 257
paper_rows = torch.empty((B, paper_p), device=device, dtype=torch.float32).normal_()
head_rows = torch.empty((B, head_p), device=device, dtype=torch.float32).normal_()
paper_kernel = paper_rows.double() @ paper_rows.double().T / B
head_kernel = head_rows.double() @ head_rows.double().T / B
system = head_kernel + .5 * torch.eye(B, device=device, dtype=torch.float64)
jacobi = torch.diagonal(system).rsqrt()
chol, info = torch.linalg.cholesky_ex(jacobi[:, None] * system * jacobi[None, :])
assert int(info.max()) == 0
rhs = torch.randn(B, device=device, dtype=torch.float64)
alpha = jacobi * torch.cholesky_solve((jacobi * rhs)[:, None], chol).squeeze(1)
relative = torch.linalg.vector_norm(system @ alpha - rhs) / torch.linalg.vector_norm(rhs)
peak = torch.cuda.max_memory_allocated()
assert relative.item() < 1e-10
assert peak < props.total_memory * .35, (peak, props.total_memory)
print("GPUH_HYBRID_HEAD_COMPATIBILITY_PASS")
print(f"gpu={props.name} total_bytes={props.total_memory} peak_bytes={peak}")
print(f"paper_rows={tuple(paper_rows.shape)} head_rows={tuple(head_rows.shape)}")
print(f"head_fp64_relative_residual={relative.item():.3e}")
print(f"partition_policy={manifest['POLICY_EXCLUSIVE']['numel']} shared={manifest['SHARED']['numel']} head={manifest['CRITIC_EXCLUSIVE']['numel']}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
