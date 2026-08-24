#!/usr/bin/env python3
"""Non-scientific H200 proof for one eight-process Procgen bundle."""
import hashlib
import subprocess
import torch
import procgen  # noqa: F401
import yaml

C = "/scratch/h99859yz/procgen_paper_matched_deterministic_ggn_1m_20260824_06"
TRAINER = f"{C}/code/train_shared_paper_matched_deterministic_ggn_v1.py"
CONFIG = f"{C}/code/configs/adv_resnet_shared_paper_matched_deterministic_ggn_v1_1m.yaml"

def sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()

assert sha(TRAINER) == "41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a"
assert sha(CONFIG) == "69d12937debb8ef8b4531e79b8f9613185b26e4e9056a51e67754129b269391d"
with open(CONFIG) as handle:
    config = yaml.safe_load(handle)
assert config["algo_config"]["lr"] == .5
assert config["algo_config"]["joint_critic_curvature_coef"] == .1
assert config["env_config"]["timesteps_per_proc_easy"] == 1_000_000
subprocess.run(
    ["/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python", TRAINER, "--help"],
    cwd=f"{C}/code", stdout=subprocess.DEVNULL, check=True,
)

assert torch.cuda.is_available()
device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print(f"detected_gpu={props.name} detected_total_bytes={props.total_memory}", flush=True)
assert "H200" in props.name
assert props.total_memory >= 135_000_000_000, props.total_memory

# The exact single-process 2B preflight peaked at 14,996,930,560 bytes on the
# L40S. Reserve 15.5 GB decimal per child for eight children simultaneously,
# including 0.5 GB extra allocator margin per process, then require 15 GB of
# physical headroom beyond that conservative 124 GB aggregate reservation.
children = 8
reserve_per_child = 15_500_000_000
aggregate_reserve = children * reserve_per_child
assert props.total_memory - aggregate_reserve >= 15_000_000_000
torch.cuda.reset_peak_memory_stats(device)
reservation = torch.empty(aggregate_reserve, dtype=torch.uint8, device=device)
assert torch.cuda.memory_allocated(device) >= aggregate_reserve
peak = torch.cuda.max_memory_allocated(device)
del reservation
torch.cuda.empty_cache()

# Native H200 FP64/Jacobi/Cholesky support, representative 2B system.
B = 512
rows = 2 * B
probe = torch.randn((rows, 1024), device=device, dtype=torch.float64) * 1e-3
k64 = probe @ probe.T / B
ratio64 = torch.linspace(.2, 2.0, rows, device=device, dtype=torch.float64)
sqrt_ratio64 = ratio64.sqrt()
system64 = sqrt_ratio64[:, None] * k64 * sqrt_ratio64[None, :]
system64.diagonal().add_(.5)
rhs64 = torch.ones(rows, device=device, dtype=torch.float64)
jacobi = system64.diagonal().rsqrt()
equilibrated = jacobi[:, None] * system64 * jacobi[None, :]
chol, info = torch.linalg.cholesky_ex(equilibrated)
assert int(info.max()) == 0
beta = jacobi * torch.cholesky_solve((jacobi * rhs64)[:, None], chol).squeeze(1)
relative = torch.linalg.vector_norm(system64 @ beta - rhs64) / torch.linalg.vector_norm(rhs64)
assert float(relative) < 1e-10

print("GPUH_BUNDLE_COMPATIBILITY_PASS")
print(f"children={children} reserve_per_child_bytes={reserve_per_child}")
print(f"aggregate_reserved_bytes={aggregate_reserve} peak_allocated_bytes={peak}")
print(f"post_reservation_headroom_bytes={props.total_memory - aggregate_reserve}")
print(f"fp64_jacobi_cholesky_relative_residual={float(relative):.3e}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print("driver=" + subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True,
).strip())
