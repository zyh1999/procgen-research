#!/usr/bin/env python3
"""Machine-auditable strict Paper actor and allowed critic-only diff gate."""
from pathlib import Path
import hashlib
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT / "work/procgen_paper_2b5affd_6m_bede_20260722/source/train_shared.py"
PCONFIG = ROOT / "work/procgen_paper_2b5affd_6m_bede_20260722/source/configs/adv_resnet_shared.yaml"
P1 = ROOT / "deterministic_2b_symfp64_20260807/train_shared_rat_exact_deterministic_ggn_symfp64.py"
PREVIOUS = ROOT / "remote_launch_staging/procgen_paper_matched_deterministic_ggn_1m_20260824_06/train_shared_paper_matched_deterministic_ggn_v1.py"
TARGET = HERE / "train_shared_paper_separateb_detggn_v1.py"
TCONFIG = HERE / "adv_resnet_shared_paper_separateb_detggn_v1_6m.yaml"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def section(text, start, end):
    a = text.index(start); b = text.index(end, a); return text[a:b]

assert sha(PAPER) == "cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7"
assert sha(PCONFIG) == "1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea"
assert sha(P1) == "2b50f8cc26bb8c85f91e0b394acc7535f5564ac70036403597d3738d3dab9c1b"
assert sha(PREVIOUS) == "41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a"
before = sha(TARGET)
subprocess.run([sys.executable, str(HERE / "build_target.py")], check=True,
               stdout=subprocess.DEVNULL)
assert sha(TARGET) == before

paper, target = PAPER.read_text(), TARGET.read_text()
for start, end, name in [
    ("    def PPO_Update(", "    def Advantage_Update(", "PPO"),
    ("    def KFAC_Update(", "    # choose the policy update rule", "KFAC"),
    ("def train_fn(", "def main(", "network/env/evaluation"),
    ("def main(", "if __name__ == '__main__':", "config propagation"),
]:
    assert section(paper, start, end) == section(target, start, end), name

actor_literals = [
    "ft_per_sample_grads = ft_compute_sample_grad(dict_params, dict_buffers, _obs, _act)",
    "HHT = H @ H.t() / num_sa",
    "_adv = _adv - torch.mv(H, g_k)",
    "_png_adv = torch.mv( torch.inverse(HHT @ torch.diag(_ratio) + algo_config.cg_damping * torch.eye(num_sa, device=device)), _adv)",
    "_loss_pi = (- _ratio * _png_adv).mean()",
    "SGD(actor_critic.parameters(), lr=algo_config.lr, momentum=1e-6)",
    "if curr_kl > 0.02 * 2:",
    "elif curr_kl < 0.01 / 2:",
]
for literal in actor_literals:
    assert literal in paper and literal in target, literal
assert "critic_rows = critic_h_weight * J_v" in target
assert "critic_rhs = (\n                critic_objective_coef / critic_h_weight\n            ) * critic_residual" in target
assert "solve_separate_critic_b_fp64" in target
assert "critic_system_rows=num_sa" in target
assert "critic_cross_blocks=0" in target
for forbidden in ["joint_H =", "torch.cat([H, critic", "2 * num_sa"]:
    assert forbidden not in target, forbidden
assert target.count("low_fisher") == 1  # validator rejection only

pconf = PCONFIG.read_text().splitlines()
tconf = TCONFIG.read_text().splitlines()
extras = {
    "  critic_curvature_coef: 0.1", "  critic_objective_coef: 1.0",
    "  fp64_gram_chunk_cols: 32768", "  dual_jacobi_eps: 1.0e-18",
}
assert [x for x in tconf if x not in extras] == pconf

print("AUDIT_PASS")
print(f"paper_trainer_sha256={sha(PAPER)}")
print(f"paper_config_sha256={sha(PCONFIG)}")
print(f"p1_donor_sha256={sha(P1)}")
print(f"previous_joint2b_sha256={sha(PREVIOUS)}")
print(f"target_trainer_sha256={sha(TARGET)}")
print(f"target_config_sha256={sha(TCONFIG)}")
print("allowed_diff=independent_deterministic_critic_Jv_residual_lambda0.1_BxB_FP64_Jacobi_Cholesky_telemetry")
print("paper_actor_path=STRICT_LITERAL_PRESERVED")
