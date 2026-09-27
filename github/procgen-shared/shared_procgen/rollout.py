"""Frozen rollout/GAE convention (including env-major flattening)."""
from abc import ABC, abstractmethod
import numpy as np
import torch
from .model import model_step

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.obs = env.reset()
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam, adv_type, device, test_mode=False,
                 store_obs_on_cpu=False):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.adv_type = adv_type
        self.device = device
        self.test_mode = test_mode
        self.store_obs_on_cpu = store_obs_on_cpu

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_values, mb_actions, mb_dones, mb_logits = [],[],[],[],[],[]
        epinfos = []

        if self.model.obs_rms is not None:
            self.model.obs_rms.training = False

        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            # actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            actions, values, logits = model_step(self.model, self.obs, deterministic=self.test_mode)

            mb_obs.append(self.obs.detach().cpu().clone() if self.store_obs_on_cpu
                          else self.obs.clone())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_logits.append(logits)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            if self.model.is_discrete:
                clipped_actions = actions # no clipping for discrete actions
            else:
                # clipped_actions = torch.clamp(actions, -1.0, 1.0) # clip actions to be in the range [-1, 1]
                act_lim = self.env.action_space.high[0]
                clipped_actions = torch.tanh(actions) * act_lim # squash actions to be in the range (-1, 1)

            self.obs, rewards, self.dones, infos = self.env.step(clipped_actions.cpu().numpy()) # done==true: a new episode just started

            for i, d in enumerate(self.dones):
                if d and infos[i].get('TimeLimit.truncated', False):
                    terminal_obs = infos[i]['terminal_observation'].unsqueeze(0)
                    bootstrap_val = model_step(self.model, terminal_obs)[1]
                    rewards[i] += self.gamma * bootstrap_val[0]

            for idx, info in enumerate(infos):
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            mb_rewards.append(rewards)

        if self.test_mode:
            return epinfos

        #batch of steps to batch of rollouts
        mb_obs = torch.stack(mb_obs, dim=0)
        mb_rewards = torch.from_numpy(np.asarray(mb_rewards)).to(self.device)
        mb_actions = torch.stack(mb_actions, dim=0)
        mb_values = torch.stack(mb_values, dim=0)
        mb_logits = torch.stack(mb_logits, dim=0)
        mb_dones = torch.from_numpy(np.asarray(mb_dones).astype(np.float32)).to(self.device)

        last_values = model_step(self.model, self.obs)[1]

        # discount/bootstrap off value fn
        mb_returns = torch.zeros_like(mb_rewards)
        mb_advs = torch.zeros_like(mb_rewards)
        mb_td_res = torch.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - torch.from_numpy(self.dones.astype(np.float32)).to(self.device)
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            mb_td_res[t] = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = mb_td_res[t] + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        if self.adv_type == 'td':
            mb_advs = mb_td_res
        else:
            assert self.adv_type == 'gae'

        # mb_obs = sf01(mb_obs)
        # if self.model.obs_rms is not None:
        #     self.model.obs_rms.training = True
        #     mb_obs = self.model.obs_rms(mb_obs)
        #     self.model.obs_rms.training = False
        # return (mb_obs, *map(sf01, (mb_returns, mb_actions, mb_advs, mb_logits)), epinfos)

        return (*map(sf01, (mb_obs, mb_returns, mb_actions, mb_advs, mb_logits)), epinfos)


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])
