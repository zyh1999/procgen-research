#!/usr/bin/env python3
import json

import gym3
import gymnasium
import numpy
import scipy
import stable_baselines3
import torch
from procgen import ProcgenEnv


env = ProcgenEnv(
    num_envs=2,
    env_name="coinrun",
    num_levels=10,
    start_level=0,
    distribution_mode="easy",
    rand_seed=0,
)
observation = env.reset()
assert observation["rgb"].shape == (2, 64, 64, 3)
observation, reward, done, info = env.step(
    numpy.zeros(2, dtype=numpy.int32)
)
assert observation["rgb"].shape == (2, 64, 64, 3)
assert reward.shape == (2,)
assert done.shape == (2,)
env.close()

assert torch.cuda.is_available()
x = torch.randn(1024, 1024, device="cuda")
checksum = float((x @ x.T).mean().item())
assert numpy.isfinite(checksum)

print(
    json.dumps(
        {
            "status": "PASS",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "procgen_observation": list(observation["rgb"].shape),
            "gym3": getattr(gym3, "__version__", "unknown"),
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "matmul_checksum": checksum,
        },
        sort_keys=True,
    )
)
