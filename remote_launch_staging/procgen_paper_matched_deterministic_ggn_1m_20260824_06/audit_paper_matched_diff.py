#!/usr/bin/env python3
"""Machine-auditable Paper-to-target identity and allowed-diff gate."""
from pathlib import Path
import hashlib, subprocess, sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT / "work/procgen_paper_2b5affd_6m_bede_20260722/source/train_shared.py"
PCONFIG = ROOT / "work/procgen_paper_2b5affd_6m_bede_20260722/source/configs/adv_resnet_shared.yaml"
P1 = ROOT / "deterministic_2b_symfp64_20260807/train_shared_rat_exact_deterministic_ggn_symfp64.py"
TARGET = HERE / "train_shared_paper_matched_deterministic_ggn_v1.py"
TCONFIG = HERE / "adv_resnet_shared_paper_matched_deterministic_ggn_v1_1m.yaml"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def simple_yaml(path):
    out, group = {}, None
    for raw in path.read_text().splitlines():
        line = raw.split('#', 1)[0].rstrip()
        if not line: continue
        if not line.startswith(' '):
            key, value = line.split(':', 1)
            if value.strip(): out[key] = value.strip()
            else: group = key; out[group] = {}
        else:
            key, value = line.strip().split(':', 1); value = value.strip().replace('_', '')
            if value in ('True', 'False'): parsed = value == 'True'
            elif value.startswith(("'", '"')): parsed = value.strip("'\"")
            else:
                try: parsed = int(value)
                except ValueError:
                    try: parsed = float(value)
                    except ValueError: parsed = value
            out[group][key] = parsed
    return out
assert sha(PAPER) == "cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7"
assert sha(PCONFIG) == "1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea"
assert sha(P1) == "2b50f8cc26bb8c85f91e0b394acc7535f5564ac70036403597d3738d3dab9c1b"

# Rebuilding must reproduce the checked-in target byte-for-byte.
before = sha(TARGET)
subprocess.run([sys.executable, str(HERE / "build_target.py")], check=True,
               stdout=subprocess.DEVNULL)
assert sha(TARGET) == before

paper, target = PAPER.read_text(), TARGET.read_text()
def section(text, start, end):
    a = text.index(start); b = text.index(end, a); return text[a:b]

# These execution paths must be literal Paper text.
for start, end, name in [
    ("    def PPO_Update(", "    def Advantage_Update(", "PPO path"),
    ("    def KFAC_Update(", "    # choose the policy update rule", "KFAC path"),
    ("def train_fn(", "def main(", "network/env/evaluation setup"),
    ("def main(", "if __name__ == '__main__':", "argument/config propagation"),
]:
    assert section(paper, start, end) == section(target, start, end), name

paper_schedule = section(paper, "        for _ in range(algo_config.epochs):", "        tepochs.set_postfix")
target_schedule = section(target, "        for _ in range(algo_config.epochs):", "        tepochs.set_postfix")
assert paper_schedule in target_schedule.replace(
    "                minibatch_update_count += 1\n"
    "                if algo_config.use_kl_adaptive_lr:\n"
    "                    adaptive_kl_update_count += 1\n"
    "                pi_info['adaptive_kl_update_count'] = adaptive_kl_update_count\n"
    "                pi_info['minibatch_update_count'] = minibatch_update_count\n"
    "                if log_dir is not None:\n"
    "                    _trace_row = dict(pi_info)\n"
    "                    _trace_row.update(transitions=(epoch+1)*per_epoch_timesteps, loss_pi=float(mb_loss_pi.item()), loss_v=float(mb_loss_v.item()))\n"
    "                    with open(os.path.join(log_dir, 'metric_trace.jsonl'), 'a') as _trace_file:\n"
    "                        _trace_file.write(json.dumps(_trace_row, sort_keys=True) + '\\n')\n",
    "",
), "minibatch/adaptive-KL schedule"

assert "SGD(actor_critic.parameters(), lr=algo_config.lr, momentum=1e-6)" in target
assert "if curr_kl > 0.02 * 2:" in target and "elif curr_kl < 0.01 / 2:" in target
assert "rhs_eff = rhs_eff - previous_projection" in target
assert "joint_H = torch.cat([H_pi, critic_h_weight * J_v], dim=0)" in target
assert "torch.linalg.cholesky_ex" in target and "dtype=torch.float64" in target

p, t = simple_yaml(PCONFIG), simple_yaml(TCONFIG)
for group in ("nets_config", "log_config"):
    assert p[group] == t[group], group
for key, value in p["algo_config"].items():
    assert t["algo_config"][key] == value, key
assert t["env_config"]["num_envs"] == 16 and t["env_config"]["nsteps"] == 256
assert t["env_config"]["timesteps_per_proc_easy"] == 1_000_000
assert t["algo_config"]["joint_critic_curvature_coef"] == 0.1
assert not any(k in t["algo_config"] for k in
               ("adaptive_kl_mode", "optimizer_momentum", "is_kaczmarz"))

print("AUDIT_PASS")
print(f"paper_trainer_sha256={sha(PAPER)}")
print(f"paper_config_sha256={sha(PCONFIG)}")
print(f"p1_donor_sha256={sha(P1)}")
print(f"target_trainer_sha256={sha(TARGET)}")
print(f"target_config_sha256={sha(TCONFIG)}")
print("allowed_scientific_diff=Advantage_Update critic J_v/residual lambda0.1 joint2B FP64 Jacobi Cholesky telemetry")
