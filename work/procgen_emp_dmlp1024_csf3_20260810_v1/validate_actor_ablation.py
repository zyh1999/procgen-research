"""Validate that formal configs differ from V3-B only on actor controls."""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "configs"
BASE = yaml.safe_load((CONFIG_ROOT / "phasic_B_bigfish_formal.yaml").read_text())
FILES = {
    "G": "actor_G_entropy_official_ppg_formal.yaml",
    "H": "actor_H_policykl_official_ppg_formal.yaml",
    "I": "actor_I_epoch1_official_ppg_formal.yaml",
    "J": "actor_J_combined_official_ppg_formal.yaml",
}
ALLOWED_ACTOR_KEYS = {
    "variant_name",
    "ent_coef",
    "epochs",
    "use_actor_entropy_natural_gradient",
    "use_actor_policy_fisher_clip",
    "actor_policy_target_kl",
    "actor_policy_kl_budget_mode",
}
EXPECTED = {
    "G": dict(ent=0.01, entropy=True, clip=False, epochs=4),
    "H": dict(ent=0.0, entropy=False, clip=True, epochs=4),
    "I": dict(ent=0.0, entropy=False, clip=False, epochs=1),
    "J": dict(ent=0.01, entropy=True, clip=True, epochs=1),
}


def stripped_actor(config):
    result = dict(config["algo_config"])
    for key in ALLOWED_ACTOR_KEYS:
        result.pop(key, None)
    return result


summary = {}
for variant, filename in FILES.items():
    config = yaml.safe_load((CONFIG_ROOT / filename).read_text())
    assert config["algo"] == BASE["algo"] == "adv"
    assert config["env_config"] == BASE["env_config"]
    assert config["nets_config"] == BASE["nets_config"]
    assert config["log_config"] == BASE["log_config"]
    assert stripped_actor(config) == stripped_actor(BASE)
    algo = config["algo_config"]
    expected = EXPECTED[variant]
    assert float(algo["ent_coef"]) == expected["ent"]
    assert bool(algo["use_actor_entropy_natural_gradient"]) is expected["entropy"]
    assert bool(algo["use_actor_policy_fisher_clip"]) is expected["clip"]
    assert int(algo["epochs"]) == expected["epochs"]
    assert algo["use_official_ppg_auxiliary"] is True
    assert algo["use_critic_ggn_auxiliary"] is False
    assert algo["separate_policy_critic_steps"] is True
    assert int(algo["policy_updates_per_cycle"]) == 16
    assert int(algo["official_aux_epochs"]) == 6
    assert int(algo["official_aux_minibatch_size"]) == 1024
    assert float(algo["official_aux_lr"]) == 5e-4
    assert float(algo["beta_clone"]) == 1.0
    if expected["clip"]:
        assert float(algo["actor_policy_target_kl"]) == 0.0025
        assert algo["actor_policy_kl_budget_mode"] == "equal_split"
    summary[variant] = {
        "config": filename,
        "entropy": expected["entropy"],
        "policy_fisher_clip": expected["clip"],
        "epochs": expected["epochs"],
        "critic": "unchanged_official_ppg_adam",
    }

print(json.dumps(summary, sort_keys=True))
