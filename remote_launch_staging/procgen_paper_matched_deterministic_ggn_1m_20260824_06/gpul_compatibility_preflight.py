#!/usr/bin/env python3
"""Non-training L40S proof for the frozen 2B FP64/Jacobi solver."""
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
    [
        "/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python",
        TRAINER,
        "--help",
    ],
    cwd=f"{C}/code",
    stdout=subprocess.DEVNULL,
    check=True,
)
assert torch.cuda.is_available()
device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print(f"detected_gpu={props.name} detected_total_bytes={props.total_memory}", flush=True)
assert "L40S" in props.name
# NVIDIA's L40S is marketed as 48 GB decimal; CUDA reports roughly 44.7 GiB.
# Use a hardware-accurate decimal threshold, not the erroneous 45 GiB gate.
assert props.total_memory >= 47_000_000_000, props.total_memory

# Exact 2B-by-P footprint, conservatively retaining H_pi, J_v and joint_H.
B, P = 512, 1_464_544
torch.cuda.reset_peak_memory_stats(device)
h_pi = torch.empty((B, P), dtype=torch.float32, device=device)
j_v = torch.empty((B, P), dtype=torch.float32, device=device)
joint_h = torch.cat((h_pi, (0.1 ** 0.5) * j_v), dim=0)
assert joint_h.shape == (2 * B, P)

# Exercise native FP64 Gram, symmetric similarity, Jacobi and Cholesky solve.
probe = joint_h[:, :1024].double()
probe.normal_(0.0, 1e-3)
k64 = probe @ probe.T / B
ratio64 = torch.linspace(.2, 2.0, 2 * B, device=device, dtype=torch.float64)
sqrt_ratio64 = ratio64.sqrt()
system64 = sqrt_ratio64[:, None] * k64 * sqrt_ratio64[None, :]
system64.diagonal().add_(.5)
rhs64 = torch.ones(2 * B, device=device, dtype=torch.float64)
jacobi = system64.diagonal().rsqrt()
equilibrated = jacobi[:, None] * system64 * jacobi[None, :]
chol, info = torch.linalg.cholesky_ex(equilibrated)
assert int(info.max()) == 0
beta = jacobi * torch.cholesky_solve((jacobi * rhs64)[:, None], chol).squeeze(1)
relative = torch.linalg.vector_norm(system64 @ beta - rhs64) / torch.linalg.vector_norm(rhs64)
assert float(relative) < 1e-10

free, total = torch.cuda.mem_get_info(device)
peak = torch.cuda.max_memory_allocated(device)
assert peak < 18 * 1024**3
assert total - peak > 25 * 1024**3
print("GPUL_COMPATIBILITY_PASS")
print(f"gpu={props.name} capability={props.major}.{props.minor}")
print(f"total_bytes={total} free_after_probe_bytes={free}")
print(f"peak_allocated_bytes={peak} conservative_headroom_bytes={total - peak}")
print(f"exact_joint_shape={tuple(joint_h.shape)} parameter_columns={P}")
print(f"fp64_jacobi_cholesky_relative_residual={float(relative):.3e}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print("driver=" + subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    text=True,
).strip())
