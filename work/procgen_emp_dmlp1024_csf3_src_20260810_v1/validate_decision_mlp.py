#!/usr/bin/env python3
from types import SimpleNamespace

import torch

from phasic_ef_ggn import partition_named_parameters
from utils.utils import SharedActorCritic, build_resnet, count_vars


EXPECTED_PARAMS = 1_464_804


def main():
    nets = SimpleNamespace(
        norm_obs=False,
        dropout=0.0,
        hidden_size=256,
        decision_hidden_size=1024,
    )
    factory, _ = build_resnet(
        64,
        256,
        with_bn=False,
        depths=[8, 16],
        device="cpu",
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
    assert total == EXPECTED_PARAMS, (total, EXPECTED_PARAMS)

    values, logits = model(torch.zeros(2, 3, 64, 64))
    assert tuple(values.shape) == (2,)
    assert tuple(logits.shape) == (2, 15)

    groups = partition_named_parameters(model)
    decision_names = [
        name
        for name, _ in model.named_parameters()
        if name.startswith("backbone_net.1.") or name.startswith("backbone_net.3.")
    ]
    assert len(decision_names) == 4, decision_names
    assert set(decision_names).issubset(groups["shared"])
    print(
        {
            "status": "PASS",
            "total_parameters": total,
            "decision_mlp": "256-1024-256",
            "decision_parameter_names": decision_names,
        }
    )


if __name__ == "__main__":
    main()
