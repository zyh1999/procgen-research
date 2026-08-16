#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


ENVS = (
    "bigfish-easy-0-10",
    "bossfight-easy-0-10",
    "caveflyer-easy-0-10",
    "coinrun-easy-0-10",
)


def load_rollouts(root: Path, env: str):
    path = root / env / "seed0" / "metric_trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        row for row in rows
        if int(row.get("optimizer_epoch", -1)) == 3
        and int(row.get("minibatch_index", -1)) == 7
    ]


def mean(rows, key):
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def fail(messages):
    for message in messages:
        print(f"FAIL {message}")
    raise SystemExit("AUDIT=FAILED_GATE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("new", type=Path)
    parser.add_argument("fixed", type=Path)
    parser.add_argument("sampled", type=Path)
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--min-transitions", type=int, default=500_000)
    args = parser.parse_args()

    failures = []
    improvements = 0
    print("environment common_T variant tail_reward tail_entropy tail_KL fisher floor guard")
    for env in ENVS:
        groups = {
            "adaptive": load_rollouts(args.new, env),
            "fixed": load_rollouts(args.fixed, env),
            "sampled": load_rollouts(args.sampled, env),
        }
        common_t = min(int(rows[-1]["environment_transitions"]) for rows in groups.values())
        if common_t < args.min_transitions:
            failures.append(f"{env}: common transitions only {common_t}")
            continue
        stats = {}
        for name, rows in groups.items():
            aligned = [row for row in rows if int(row["environment_transitions"]) <= common_t][-args.tail:]
            stats[name] = {
                "reward": mean(aligned, "eprewmean"),
                "entropy": mean(aligned, "entropy"),
                "kl": mean(aligned, "behavior_kl_after_step"),
                "fisher": mean(aligned, "categorical_fisher_trace") if "categorical_fisher_trace" in aligned[-1] else math.nan,
                "floor": mean(aligned, "damping_to_median_floor") if "damping_to_median_floor" in aligned[-1] else math.nan,
                "guard": mean(aligned, "actor_damping_from_critic_floor") if "actor_damping_from_critic_floor" in aligned[-1] else math.nan,
            }
            s = stats[name]
            print(env, common_t, name, s["reward"], s["entropy"], s["kl"], s["fisher"], s["floor"], s["guard"])

        new = stats["adaptive"]
        fixed = stats["fixed"]
        sampled = stats["sampled"]
        reward_floor = max(0.90 * fixed["reward"], 0.90 * sampled["reward"])
        if new["reward"] < reward_floor:
            failures.append(
                f"{env}: adaptive reward {new['reward']:.4g} below matched floor {reward_floor:.4g}"
            )
        if new["entropy"] < 0.20:
            failures.append(f"{env}: entropy collapse {new['entropy']:.4g}")
        if not math.isfinite(new["kl"]) or new["kl"] > 0.04:
            failures.append(f"{env}: unsafe KL {new['kl']:.4g}")
        if new["reward"] >= fixed["reward"] * 1.05 + 0.02:
            improvements += 1
        if env == "coinrun-easy-0-10" and new["reward"] < 0.95 * fixed["reward"]:
            failures.append("coinrun: lost more than 5% of validated fixed-guard reward")

    if improvements < 2:
        failures.append(f"only {improvements} environments improved >=5% over fixed guard")
    if failures:
        fail(failures)
    print("AUDIT=PASS")


if __name__ == "__main__":
    main()
