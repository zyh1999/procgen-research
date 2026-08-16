#!/usr/bin/env python3
"""Compare joint-2B Procgen runs without extrapolating unequal prefixes.

Each positional argument is a results root containing
``<variant>/<environment>/seed*/metric_trace.jsonl``.  The report uses the
latest *common* transition across supplied variants for each environment, so a
partly finished run cannot look better merely because it trained longer.  Each
rollout contributes exactly one point (its final minibatch record); otherwise
``--tail 10`` would cover only a fraction of one PPO update.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from math import isfinite, sqrt
from statistics import fmean, stdev


FIELDS = (
    ("eprewmean", "reward"),
    ("entropy", "entropy"),
    ("behavior_kl_after_step", "behavior KL"),
    ("lr_used", "LR"),
    ("ratio_min", "ratio min"),
    ("ratio_max", "ratio max"),
    ("joint_solve_residual", "solve residual"),
    ("joint_direction_l2", "direction L2"),
)
DIAGNOSTIC_FIELDS = (
    ("value_mse_pre_step", "value MSE"),
    ("value_mse_over_return_variance", "value MSE / return var"),
    ("actor_kernel_diag_median", "actor kernel median"),
    ("critic_kernel_diag_median", "critic kernel median"),
    ("actor_effective_damping_median", "actor damping"),
    ("critic_effective_damping_median", "critic damping"),
    ("actor_guard_required_damping", "actor guard required"),
    ("actor_guard_slack", "actor guard slack"),
    ("actor_guard_binding", "actor guard binding rate"),
    ("actor_guard_violation", "actor guard violation rate"),
)


def records(path: Path) -> list[dict]:
    with path.open() as source:
        rows = [json.loads(line) for line in source if line.strip()]

    for row in rows:
        if (
            "value_mse_pre_step" in row
            and "minibatch_return_variance" in row
        ):
            mse = float(row["value_mse_pre_step"])
            variance = float(row["minibatch_return_variance"])
            if variance > 0.0:
                row["value_mse_over_return_variance"] = mse / variance

        # The actor-from-critic guard is the mechanism under test.  Record its
        # invariant and whether it is actually the active lower bound instead
        # of inferring that from window-averaged kernel and damping columns.
        guard = row.get("actor_damping_from_critic_floor")
        critic_median = row.get("critic_kernel_diag_median")
        actor_damping = row.get("actor_effective_damping_median")
        if guard is not None and critic_median is not None and actor_damping is not None:
            guard = float(guard)
            required = guard * float(critic_median)
            actual = float(actor_damping)
            slack = actual - required
            tolerance = 1.0e-6 * max(1.0, abs(actual), abs(required))
            row["actor_guard_required_damping"] = required
            row["actor_guard_slack"] = slack
            if guard > 0.0:
                row["actor_guard_binding"] = float(abs(slack) <= tolerance)
                row["actor_guard_violation"] = float(slack < -tolerance)

    # metric_trace records every minibatch.  Reward/entropy trends should be
    # measured across rollout updates, not across the last few minibatches of
    # one update.  Keep the final minibatch for every transition count.  Using
    # the transition count also supports older traces without rollout_update.
    final_by_transition: dict[int, dict] = {}
    for row in rows:
        transition = int(row["environment_transitions"])
        previous = final_by_transition.get(transition)
        if previous is None or int(row.get("minibatch_global_step", -1)) >= int(
            previous.get("minibatch_global_step", -1)
        ):
            final_by_transition[transition] = row
    return [final_by_transition[key] for key in sorted(final_by_transition)]


def last_at_or_before(rows: list[dict], transition: int) -> dict:
    return max(
        (row for row in rows if row["environment_transitions"] <= transition),
        key=lambda row: row["environment_transitions"],
    )


def fmt(value: float) -> str:
    return f"{value:.4g}"


def mean_sem(values: list[float]) -> str:
    """Format a seed mean and its standard error without inventing n=1 error."""
    if not values:
        return "—"
    mean = fmean(values)
    if len(values) == 1:
        return fmt(mean)
    return f"{fmt(mean)} ± {fmt(stdev(values) / sqrt(len(values)))}"


def field_mean(rows: list[dict], key: str) -> float | None:
    """Average a field when present, tolerating older diagnostic-free traces."""
    values: list[float] = []
    for row in rows:
        if key not in row or row[key] is None:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if isfinite(value):
            values.append(value)
    return fmean(values) if values else None


def half_window_delta(rows: list[dict], key: str) -> float | None:
    """Return late-half minus early-half mean for an aligned rollout window."""
    if len(rows) < 2:
        return None
    midpoint = len(rows) // 2
    early = field_mean(rows[:midpoint], key)
    late = field_mean(rows[midpoint:], key)
    if early is None or late is None:
        return None
    return late - early


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--tail", type=int, default=10)
    parser.add_argument(
        "--per-seed",
        action="store_true",
        help="print aligned seed rows after each aggregate row",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="include value-fit and block-scale damping diagnostics",
    )
    parser.add_argument(
        "--trends",
        action="store_true",
        help="show late-half minus early-half rollout-window deltas",
    )
    args = parser.parse_args()
    fields = FIELDS + DIAGNOSTIC_FIELDS if args.diagnostics else FIELDS

    grouped: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for root in args.roots:
        traces = list(root.glob("*/*/seed*/metric_trace.jsonl"))
        direct_traces = list(root.glob("*/seed*/metric_trace.jsonl"))
        for trace in traces + direct_traces:
            relative = trace.relative_to(root).parts
            if len(relative) == 4:
                variant, environment, seed = relative[:3]
            elif len(relative) == 3:
                variant, environment, seed = root.name, *relative[:2]
            else:
                continue
            rows = records(trace)
            if rows:
                grouped[environment][variant][seed] = rows

    print("# Joint-2B aligned progress summary")
    for environment in sorted(grouped):
        variants = grouped[environment]
        common = min(
            rows[-1]["environment_transitions"]
            for seeds in variants.values()
            for rows in seeds.values()
        )
        print(f"\n## {environment} — common transition {common:,}")
        headers = ["variant", "seeds", "sample"] + [
            f"{label} (last {args.tail} updates)" for _, label in fields
        ]
        print("| " + " | ".join(headers) + " |")
        print("|" + "---|" * len(headers))
        for variant, seeds in sorted(variants.items()):
            tail_metrics: dict[str, list[float]] = {
                key: [] for key, _ in fields
            }
            for rows in seeds.values():
                # Do not leak a completed seed's final reward into an
                # early-prefix comparison with a still-running seed.
                eligible = [
                    item for item in rows
                    if item["environment_transitions"] <= common
                ]
                tail = eligible[max(0, len(eligible) - args.tail):]
                for key, _ in fields:
                    value = field_mean(tail, key)
                    if value is not None:
                        tail_metrics[key].append(value)
            values = [
                mean_sem(tail_metrics[key])
                for key, _ in fields
            ]
            print("| " + " | ".join([
                variant,
                str(len(seeds)),
                f"{common:,}",
            ] + values) + " |")
            if args.per_seed:
                for seed, rows in sorted(seeds.items()):
                    eligible = [
                        item for item in rows
                        if item["environment_transitions"] <= common
                    ]
                    tail = eligible[max(0, len(eligible) - args.tail):]
                    seed_values = [
                        fmt(value) if (value := field_mean(tail, key)) is not None else "—"
                        for key, _ in fields
                    ]
                    print("| " + " | ".join([
                        f"{variant}/{seed}",
                        "1",
                        f"{common:,}",
                    ] + seed_values) + " |")

        if args.trends:
            trend_fields = (
                ("eprewmean", "reward delta"),
                ("entropy", "entropy delta"),
                ("behavior_kl_after_step", "behavior KL delta"),
            )
            print(f"\n### {environment} rollout-window trends")
            trend_headers = ["variant", "seeds", "sample"] + [
                f"{label} (last {args.tail} updates)" for _, label in trend_fields
            ]
            print("| " + " | ".join(trend_headers) + " |")
            print("|" + "---|" * len(trend_headers))
            for variant, seeds in sorted(variants.items()):
                aggregate: dict[str, list[float]] = {
                    key: [] for key, _ in trend_fields
                }
                seed_deltas: dict[str, dict[str, float | None]] = {}
                for seed, rows in sorted(seeds.items()):
                    eligible = [
                        item for item in rows
                        if item["environment_transitions"] <= common
                    ]
                    tail = eligible[max(0, len(eligible) - args.tail):]
                    seed_deltas[seed] = {}
                    for key, _ in trend_fields:
                        delta = half_window_delta(tail, key)
                        seed_deltas[seed][key] = delta
                        if delta is not None:
                            aggregate[key].append(delta)
                print("| " + " | ".join([
                    variant,
                    str(len(seeds)),
                    f"{common:,}",
                ] + [mean_sem(aggregate[key]) for key, _ in trend_fields]) + " |")
                if args.per_seed:
                    for seed in sorted(seed_deltas):
                        values = [
                            fmt(delta) if (delta := seed_deltas[seed][key]) is not None else "—"
                            for key, _ in trend_fields
                        ]
                        print("| " + " | ".join([
                            f"{variant}/{seed}",
                            "1",
                            f"{common:,}",
                        ] + values) + " |")


if __name__ == "__main__":
    main()
