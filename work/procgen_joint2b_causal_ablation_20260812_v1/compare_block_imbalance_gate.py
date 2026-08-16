#!/usr/bin/env python3
"""Compare the block-imbalance gate with matched deterministic and RAT runs.

Every metric trace contains one row per optimizer minibatch.  This tool first
keeps only the final minibatch of each rollout update, then aligns all methods
by environment transitions.  It is diagnostic: a 500k gate can justify a 1M
extension, but cannot certify the final 6M performance target.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


ENVIRONMENTS = (
    "bigfish-easy-0-10",
    "bossfight-easy-0-10",
    "caveflyer-easy-0-10",
    "coinrun-easy-0-10",
)


def load_rollouts(path: Path) -> list[dict]:
    by_update: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            by_update[int(row["rollout_update"])] = row
    return sorted(
        by_update.values(),
        key=lambda row: int(row["environment_transitions"]),
    )


def at_or_before(rows: list[dict], transition: int) -> dict:
    eligible = [
        row
        for row in rows
        if int(row["environment_transitions"]) <= transition
    ]
    if not eligible:
        raise ValueError(f"no rollout at or before {transition}")
    return eligible[-1]


def finite(row: dict, key: str, default: float = math.nan) -> float:
    value = float(row.get(key, default))
    return value if math.isfinite(value) else math.nan


def tail_mean(rows: list[dict], key: str, count: int = 20) -> float:
    values = [finite(row, key) for row in rows[-count:]]
    return statistics.fmean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("old_rownorm_root", type=Path)
    parser.add_argument("sampled_b_root", type=Path)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=(300_000, 350_000, 400_000, 450_000, 500_000),
    )
    args = parser.parse_args()

    for environment in ENVIRONMENTS:
        candidate_run = args.candidate_root / environment / "seed0"
        candidate = load_rollouts(candidate_run / "metric_trace.jsonl")
        old = load_rollouts(
            args.old_rownorm_root / environment / "seed0" / "metric_trace.jsonl"
        )
        sampled = load_rollouts(
            args.sampled_b_root / environment / "seed0" / "metric_trace.jsonl"
        )
        print(f"\n[{environment}] candidate_max={int(candidate[-1]['environment_transitions']):,}")
        print(
            "T cand_R/H/KL old_R/H sampled_R/H "
            "guard_F/ratio/d/active"
        )
        for requested in args.checkpoints:
            aligned = min(
                requested,
                int(candidate[-1]["environment_transitions"]),
                int(old[-1]["environment_transitions"]),
                int(sampled[-1]["environment_transitions"]),
            )
            c = at_or_before(candidate, aligned)
            o = at_or_before(old, aligned)
            s = at_or_before(sampled, aligned)
            t = int(c["environment_transitions"])
            print(
                f"{t:>7,} "
                f"{finite(c, 'eprewmean'):.3f}/{finite(c, 'entropy'):.3f}/"
                f"{finite(c, 'behavior_kl_after_step'):.4f} "
                f"{finite(o, 'eprewmean'):.3f}/{finite(o, 'entropy'):.3f} "
                f"{finite(s, 'eprewmean'):.3f}/{finite(s, 'entropy'):.3f} "
                f"{finite(c, 'block_imbalance_guard_fisher_median'):.3f}/"
                f"{finite(c, 'block_imbalance_guard_ratio_median'):.3f}/"
                f"{finite(c, 'block_imbalance_guard_actor_damping'):.3f}/"
                f"{int(finite(c, 'block_imbalance_guard_active', 0.0))}"
            )
        active = [
            row
            for row in candidate
            if finite(row, "block_imbalance_guard_active", 0.0) == 1.0
        ]
        first_active = (
            int(active[0]["environment_transitions"]) if active else None
        )
        print(
            "tail20 "
            f"candidate_R={tail_mean(candidate, 'eprewmean'):.4f} "
            f"old_R={tail_mean([row for row in old if int(row['environment_transitions']) <= int(candidate[-1]['environment_transitions'])], 'eprewmean'):.4f} "
            f"sampled_R={tail_mean([row for row in sampled if int(row['environment_transitions']) <= int(candidate[-1]['environment_transitions'])], 'eprewmean'):.4f} "
            f"candidate_H={tail_mean(candidate, 'entropy'):.4f} "
            f"active_rollouts={len(active)} first_active={first_active} "
            f"max_d={max(finite(row, 'block_imbalance_guard_actor_damping') for row in candidate):.4f}"
        )


if __name__ == "__main__":
    main()
