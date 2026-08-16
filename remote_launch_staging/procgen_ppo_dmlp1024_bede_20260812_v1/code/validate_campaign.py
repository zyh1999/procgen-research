#!/usr/bin/env python3
from types import SimpleNamespace

import torch
import yaml

from utils.utils import SharedActorCritic, build_resnet, count_vars


EXPECTED_PARAMETERS = 1_464_547


def main():
    with open("configs/ppo_resnet_shared_dmlp1024.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    algo = config["algo_config"]
    env = config["env_config"]
    nets = SimpleNamespace(**config["nets_config"])
    assert config["algo"] == "ppo"
    assert algo["optimizer"] == "adam"
    assert algo["lr"] == 0.001
    assert algo["cliprange"] == 0.2
    assert algo["epochs"] == 4
    assert algo["minibatches"] == 8
    assert algo["ent_coef"] == 0.0
    assert algo["use_kl_adaptive_lr"] is False
    assert env["num_envs"] == 16
    assert env["nsteps"] == 256
    assert env["timesteps_per_proc_easy"] == 6_000_000
    assert nets.hidden_size == 256
    assert nets.decision_hidden_size == 1024

    factory, _ = build_resnet(
        64, 256, with_bn=False, depths=[8, 16], device="cpu"
    )
    model = SharedActorCritic(
        factory,
        (3, 64, 64),
        nets_config=nets,
        n_actions=15,
        with_popart=True,
        device="cpu",
    )
    total = int(count_vars(model))
    assert total == EXPECTED_PARAMETERS, (total, EXPECTED_PARAMETERS)
    assert not hasattr(model, "aux_vf_head")
    assert model.decision_hidden_size == 1024
    assert isinstance(model.backbone_net[1], torch.nn.Linear)
    assert model.backbone_net[1].in_features == 256
    assert model.backbone_net[1].out_features == 1024
    assert isinstance(model.backbone_net[3], torch.nn.Linear)
    assert model.backbone_net[3].in_features == 1024
    assert model.backbone_net[3].out_features == 256
    values, logits = model(torch.zeros(2, 3, 64, 64))
    assert tuple(values.shape) == (2,)
    assert tuple(logits.shape) == (2, 15)
    print(
        {
            "status": "PASS",
            "algorithm": "pure_ppo",
            "total_parameters": total,
            "decision_mlp": "256-1024-256",
            "active_auxiliary_head": False,
        }
    )


if __name__ == "__main__":
    main()
