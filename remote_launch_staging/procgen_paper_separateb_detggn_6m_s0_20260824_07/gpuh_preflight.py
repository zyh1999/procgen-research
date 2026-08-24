#!/usr/bin/env python3
"""Non-training H200 compatibility gate for one frozen separate-B cell."""
import hashlib, os, sys
from pathlib import Path
import torch
import procgen, gym3

trainer, config = map(Path, sys.argv[1:3])
expected_trainer, expected_config = sys.argv[3:5]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(trainer) == expected_trainer
assert sha(config) == expected_config
props = torch.cuda.get_device_properties(0)
assert "H100" in props.name or "H200" in props.name, props.name
assert props.total_memory >= 70_000_000_000, props.total_memory
torch.manual_seed(23)
B, P = 512, 1_464_544
rows = torch.empty((B, P), device="cuda", dtype=torch.float32).normal_()
kernel = rows.double() @ rows.double().T / B
system = kernel + .5 * torch.eye(B, device="cuda", dtype=torch.float64)
j = torch.diagonal(system).rsqrt()
chol, info = torch.linalg.cholesky_ex(j[:, None] * system * j[None, :])
assert int(info.max()) == 0
rhs = torch.randn(B, device="cuda", dtype=torch.float64)
alpha = j * torch.cholesky_solve((j * rhs)[:, None], chol).squeeze(1)
rel = torch.linalg.vector_norm(system @ alpha - rhs) / torch.linalg.vector_norm(rhs)
peak = torch.cuda.max_memory_allocated()
assert rel.item() < 1e-10
assert peak < props.total_memory * .35, (peak, props.total_memory)
print("GPUH_SEPARATEB_COMPATIBILITY_PASS")
print(f"gpu={props.name} total_bytes={props.total_memory} peak_bytes={peak}")
print(f"shape={tuple(rows.shape)} fp64_relative_residual={rel.item():.3e}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
