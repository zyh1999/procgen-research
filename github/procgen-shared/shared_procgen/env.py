"""Small Procgen adapter: RGB to [-1,1], raw rewards and episode accounting."""
import numpy as np
import torch


class ProcgenAdapter:
    def __init__(self, name, seed, device):
        # Keep Procgen optional for algebra tests and --help on non-Linux hosts.
        from procgen import ProcgenEnv
        self.venv = ProcgenEnv(num_envs=16, env_name=name, num_levels=10,
                              start_level=0, distribution_mode="easy", rand_seed=seed)
        self.num_envs = 16
        self.action_space = self.venv.action_space
        self.device = device
        self.episode_returns = np.zeros(16, dtype=np.float32)
        self.episode_lengths = np.zeros(16, dtype=np.int32)

    def preprocess(self, obs):
        image = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        image = (image / 255.0 - 0.5) / 0.5
        return image.permute(0, 3, 1, 2) if image.ndim == 4 else image.permute(2, 0, 1)

    def reset(self):
        self.episode_returns.fill(0)
        self.episode_lengths.fill(0)
        return self.preprocess(self.venv.reset()["rgb"])

    def step(self, actions):
        obs, rewards, dones, infos = self.venv.step(actions)
        self.episode_returns += rewards
        self.episode_lengths += 1
        infos = [dict(info) for info in infos]
        for i, done in enumerate(dones):
            if "terminal_observation" in infos[i]:
                terminal = infos[i]["terminal_observation"]
                if isinstance(terminal, dict):
                    terminal = terminal["rgb"]
                infos[i]["terminal_observation"] = self.preprocess(terminal)
            if done:
                infos[i]["episode"] = {"r": float(self.episode_returns[i]),
                                        "l": int(self.episode_lengths[i])}
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return self.preprocess(obs["rgb"]), rewards, dones, infos

    def close(self):
        self.venv.close()
