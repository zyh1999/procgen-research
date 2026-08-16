#!/usr/bin/env python3
"""Prove that K differs from V3-B only in the actor LR working point."""

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
BASELINE_PATH = ROOT / "configs" / "phasic_B_bigfish_formal.yaml"
CONTROL_PATH = (
    ROOT
    / "configs"
    / "actor_K_exactrat_adaptivekl_official_ppg_formal.yaml"
)
EXACT_CONFIG_PATH = (
    ROOT
    / "reference_exactrat"
    / "adv_resnet_shared_procgen_maincfg_pklbranch.yaml"
)
EXACT_TRAINER_PATH = (
    ROOT
    / "reference_exactrat"
    / "train_shared_procgen_maincfg_pklbranch.py"
)


def load(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


baseline = load(BASELINE_PATH)
control = load(CONTROL_PATH)
exact = load(EXACT_CONFIG_PATH)

baseline_comparable = deepcopy(baseline)
control_comparable = deepcopy(control)
baseline_comparable["algo_config"].pop("variant_name")
control_comparable["algo_config"].pop("variant_name")

assert baseline_comparable["algo_config"].pop("lr") == 0.004
assert control_comparable["algo_config"].pop("lr") == 0.5
assert baseline_comparable == control_comparable

algo = control["algo_config"]
assert algo["adaptive_kl_mode"] == "procgen_rollout"
assert algo["adaptive_kl_lower"] == 0.005
assert algo["adaptive_kl_upper"] == 0.04
assert algo["adaptive_lr_min"] == 0.0001
assert algo["adaptive_lr_max"] == 0.5
assert algo["epochs"] == 4
assert algo["minibatches"] == 8
assert algo["optimizer_momentum"] == 0.0
assert algo["use_official_ppg_auxiliary"] is True
assert algo.get("use_actor_entropy_natural_gradient", False) is False
assert algo.get("use_actor_policy_fisher_clip", False) is False

exact_algo = exact["algo_config"]
assert exact_algo["lr"] == 0.5
assert exact_algo["use_kl_adaptive_lr"] is True
assert exact_algo["use_procgen_kl_thresholds"] is True

trainer = EXACT_TRAINER_PATH.read_text(encoding="utf-8")
required_fragments = (
    "curr_kl = pi_info['kl']",
    "kl_upper = 0.02 * 2",
    "elif curr_kl < 0.01 / 2",
    "param_groups[0]['lr'] / 1.5",
    "param_groups[0]['lr'] * 1.5",
    "_real_kl = (torch.exp(_logp_full_old)",
)
for fragment in required_fragments:
    assert fragment in trainer, fragment

print("K exact-RAT adaptive-KL controlled-difference assertions: PASS")
