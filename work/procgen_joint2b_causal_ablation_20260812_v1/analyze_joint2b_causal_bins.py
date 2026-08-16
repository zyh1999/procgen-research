#!/usr/bin/env python3
"""Summarize sparse joint-2B causal diagnostics by transition bin."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "eprewmean",
    "entropy",
    "behavior_kl_after_step",
    "categorical_fisher_trace",
    "effective_damping_value",
    "normalized_cross_block",
    "dual_difference_fraction",
    "component_direction_cosine",
    "actor_gain_from_critic_rhs",
    "actor_rhs_self_response",
    "critic_rhs_self_response",
    "critic_actor_quadratic_fraction",
    "actor_fullmetric_vs_actoronly_cosine",
    "actor_fullmetric_vs_actoronly_norm_ratio",
    "actor_fullmetric_delta_fraction",
    "critic_ggn_vs_vanilla_cosine",
    "critic_ggn_vs_vanilla_norm_ratio",
    "critic_trunk_ggn_vs_vanilla_cosine",
    "critic_trunk_ggn_vs_vanilla_norm_ratio",
)


def finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--bin", type=int, default=100_000, dest="bin_width")
    args = parser.parse_args()

    bins: dict[int, list[dict[str, object]]] = defaultdict(list)
    with args.trace.open() as handle:
        for line in handle:
            row = json.loads(line)
            if float(row.get("causal_diagnostic_ran", 0.0)) != 1.0:
                continue
            if float(row.get("actor_only_diagnostic_valid", 0.0)) != 1.0:
                continue
            transition = int(row["environment_transitions"])
            bins[(transition // args.bin_width) * args.bin_width].append(row)

    print("bin_start\tn")
    for start in sorted(bins):
        rows = bins[start]
        print(f"{start}\t{len(rows)}")
        for metric in METRICS:
            values = [
                value
                for row in rows
                if (value := finite_number(row.get(metric))) is not None
            ]
            print(f"  {metric}={median(values):.6g}")


if __name__ == "__main__":
    main()
