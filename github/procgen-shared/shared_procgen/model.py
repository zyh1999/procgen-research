"""One shared ResNet trunk, a categorical actor and a PopArt value head."""
import torch
from torch import nn
from torch.distributions import Categorical

from .resnet import ResNetImpala
from .popart import PopArt


class SharedActorCritic(nn.Module):
    def __init__(self, n_actions, img_size=64):
        super().__init__()
        # Names, construction order and initialization match the frozen model.
        self.obs_rms = None
        self.is_discrete = True
        self.with_popart = True
        self.backbone_net = ResNetImpala(img_size, 256, depths=[8, 16], with_bn=False)
        self.pi_head = nn.Linear(256, n_actions)
        self.last_v_layer = PopArt(256, 1, norm_axes=0)

    def forward(self, obs):
        latent = self.backbone_net(obs)
        return self.last_v_layer(latent).squeeze(-1), self.pi_head(latent)


@torch.no_grad()
def model_step(model, obs, deterministic=False):
    values, logits = model(obs)
    values = model.last_v_layer.unnormalize(values)
    actions = logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()
    return actions, values, logits
