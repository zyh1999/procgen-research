#!/usr/bin/env python3
"""Static exact-Paper diff and historical-distinctness audit."""
from pathlib import Path
import hashlib
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT / "work/procgen_paper_2b5affd_6m_bede_20260722/source/train_shared.py"
PCONFIG = ROOT / "work/procgen_paper_2b5affd_6m_bede_20260722/source/configs/adv_resnet_shared.yaml"
P1 = ROOT / "deterministic_2b_symfp64_20260807/train_shared_rat_exact_deterministic_ggn_symfp64.py"
JOINT = ROOT / "remote_launch_staging/procgen_paper_matched_deterministic_ggn_1m_20260824_06/train_shared_paper_matched_deterministic_ggn_v1.py"
SEPARATE = ROOT / "remote_launch_staging/procgen_paper_separateb_detggn_6m_s0_20260824_07/train_shared_paper_separateb_detggn_v1.py"
TARGET = HERE / "train_shared_paper_hybrid_head_detggn_v1.py"
TCONFIG = HERE / "adv_resnet_shared_paper_hybrid_head_detggn_v1_6m.yaml"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def section(text, start, end):
    a = text.index(start); b = text.index(end, a); return text[a:b]

assert sha(PAPER) == "cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7"
assert sha(PCONFIG) == "1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea"
assert sha(P1) == "2b50f8cc26bb8c85f91e0b394acc7535f5564ac70036403597d3738d3dab9c1b"
assert sha(JOINT) == "41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a"
assert sha(SEPARATE) == "b0dad110c36dbab4c601aa9128ba51eb437bfc6a3e9cadf87be8fd2172f3729a"
before = sha(TARGET)
subprocess.run([sys.executable, str(HERE / "build_target.py")], check=True, stdout=subprocess.DEVNULL)
assert sha(TARGET) == before

paper, target = PAPER.read_text(), TARGET.read_text()
for start, end, label in [
    ("    def PPO_Update(", "    def Advantage_Update(", "PPO"),
    ("    def KFAC_Update(", "    # choose the policy update rule", "KFAC"),
    ("def train_fn(", "def main(", "network/env/evaluation"),
    ("def main(", "if __name__ == '__main__':", "config propagation"),
]:
    assert section(paper, start, end) == section(target, start, end), label

for literal in [
    "all_logp = pi_logp + vf_logp",
    "HHT = H @ H.t() / num_sa",
    "_adv = _adv - torch.mv(H, g_k)",
    "_pseudo_adv = _pseudo_adv - torch.mv(H, g_k)",
    "_png_adv = torch.mv(torch.inverse(actor_system), _adv)",
    "_critic_adv = torch.mv(torch.inverse(paper_critic_system), _pseudo_adv)",
    "_loss_pi = (- _ratio * _png_adv).mean()",
    "_loss_v = ((_vals - _ret).pow(2) * _critic_adv).mean()",
    "SGD(actor_critic.parameters(), lr=algo_config.lr, momentum=1e-6)",
    "if curr_kl > 0.02 * 2:",
    "elif curr_kl < 0.01 / 2:",
]:
    assert literal in target, literal

for literal in [
    "parameter_groups['CRITIC_EXCLUSIVE']",
    "head_rows = critic_h_weight * J_head",
    "solve_head_critic_b_fp64",
    "post_shared_to_post_head_policy_logit_max_abs",
    "paper_shared_critic_direction_l2",
]:
    assert literal in target, literal
for forbidden in ["joint_H =", "2 * num_sa", "deterministic GGN on shared"]:
    assert forbidden not in target, forbidden

pconf = PCONFIG.read_text().splitlines()
tconf = TCONFIG.read_text().splitlines()
extras = {
    "  critic_curvature_coef: 0.1", "  critic_objective_coef: 1.0",
    "  fp64_gram_chunk_cols: 32768", "  dual_jacobi_eps: 1.0e-18",
}
assert [line for line in tconf if line not in extras] == pconf

historical = {
    "csf3_blocktrace_18669377": "1881bf7c3fe3f8d29ded23e25976810ab9127d9bc125d9c89332aa39c1ab61dc",
    "csf3_expected_18669454_18669615": "c976c0e563eb3aedb2d306c450d60b44af0c595d0f4a499cf32c65bcec9933d3",
    "bede_expected_1072337_1072344_46_49_50": "0514703d9fb6ca17cc68febabb012defb279ab5a54f57cf95365422164848934",
}
assert sha(TARGET) not in set(historical.values())

print("HYBRID_HEAD_AUDIT_PASS")
print(f"paper_trainer_sha256={sha(PAPER)}")
print(f"paper_config_sha256={sha(PCONFIG)}")
print(f"p1_donor_sha256={sha(P1)}")
print(f"joint_v1_sha256={sha(JOINT)}")
print(f"separate_b_v1_sha256={sha(SEPARATE)}")
print(f"target_trainer_sha256={sha(TARGET)}")
print(f"target_config_sha256={sha(TCONFIG)}")
for key, value in historical.items(): print(f"{key}_sha256={value}")
print("historical_formula_distinctness=DISTINCT_FORMULA_PASS")
print("allowed_diff=critic_exclusive_value_head_raw_direction_and_telemetry_only")
print("paper_actor_and_sampled_shared_critic=STRICT_LITERAL_PRESERVED")
