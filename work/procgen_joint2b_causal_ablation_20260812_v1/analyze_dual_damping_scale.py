#!/usr/bin/env python3
"""Quantify absolute-vs-relative block damping over Procgen rollouts."""

import argparse
import json
from pathlib import Path


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    midpoint = count // 2
    if count % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def summary(values):
    return {
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }


def load(path):
    rows = [
        json.loads(line)
        for line in (path / "metric_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return [
        row
        for row in rows
        if int(row.get("optimizer_epoch", -1)) == 3
        and int(row.get("minibatch_index", -1)) == 7
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--actor-absolute", type=float, default=0.01)
    parser.add_argument("--critic-absolute", type=float, default=0.03)
    parser.add_argument("--relative", type=float, default=0.10)
    parser.add_argument("--actor-from-critic", type=float, default=0.01)
    args = parser.parse_args()

    for run in args.runs:
        rows = load(run)
        current_actor_ratio = []
        current_critic_ratio = []
        proposed_actor_ratio = []
        proposed_critic_ratio = []
        proposed_actor = []
        proposed_critic = []
        for row in rows:
            actor_median = max(float(row["actor_kernel_diag_median"]), 1e-30)
            critic_median = max(float(row["critic_kernel_diag_median"]), 1e-30)
            current_actor_ratio.append(
                float(row["actor_effective_damping_median"]) / actor_median
            )
            current_critic_ratio.append(
                float(row["critic_effective_damping_median"]) / critic_median
            )
            actor_damping = max(
                args.actor_absolute,
                args.relative * actor_median,
                args.actor_from_critic * critic_median,
            )
            critic_damping = max(
                args.critic_absolute,
                args.relative * critic_median,
            )
            proposed_actor.append(actor_damping)
            proposed_critic.append(critic_damping)
            proposed_actor_ratio.append(actor_damping / actor_median)
            proposed_critic_ratio.append(critic_damping / critic_median)

        last = rows[-1]
        record = {
            "run": str(run),
            "rollouts": len(rows),
            "latest_T": int(last["environment_transitions"]),
            "latest_reward": float(last["eprewmean"]),
            "latest_entropy": float(last["entropy"]),
            "current_actor_damping_ratio": summary(current_actor_ratio),
            "current_critic_damping_ratio": summary(current_critic_ratio),
            "proposed_actor_damping_ratio": summary(proposed_actor_ratio),
            "proposed_critic_damping_ratio": summary(proposed_critic_ratio),
            "proposed_actor_damping": summary(proposed_actor),
            "proposed_critic_damping": summary(proposed_critic),
        }
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
