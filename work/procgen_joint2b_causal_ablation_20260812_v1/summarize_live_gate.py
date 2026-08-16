#!/usr/bin/env python3
"""Summarize live Procgen metric traces at rollout granularity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = (
    "environment_transitions",
    "eprewmean",
    "entropy",
    "behavior_kl_after_step",
    "current_step_kl",
    "lr_used",
    "joint_solve_residual",
    "categorical_fisher_trace",
    "fisher_damping_fraction",
    "fisher_damping_hysteresis",
    "fisher_hysteresis_engaged",
    "base_damping_value",
    "effective_damping_value",
    "raw_actor_kernel_diag_median",
    "raw_critic_kernel_diag_median",
    "actor_kernel_diag_median",
    "critic_kernel_diag_median",
    "normalized_cross_block",
    "joint_direction_l2",
    "joint_clip_scale",
)


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rollouts(rows: list[dict]) -> list[dict]:
    by_update: dict[int, dict] = {}
    for row in rows:
        by_update[int(row.get("rollout_update", row["environment_transitions"]))] = row
    return [by_update[key] for key in sorted(by_update)]


def mean(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows if field in row]
    return sum(values) / len(values) if values else float("nan")


def summarize(path: Path, window: int, max_transitions: int | None) -> dict:
    all_rows = load_rows(path)
    rows = load_rollouts(all_rows)
    if max_transitions is not None:
        rows = [
            row
            for row in rows
            if int(row["environment_transitions"]) <= max_transitions
        ]
        all_rows = [
            row
            for row in all_rows
            if int(row["environment_transitions"]) <= max_transitions
        ]
    if not rows:
        raise ValueError(f"no rollout rows in {path} at requested horizon")
    tail = rows[-window:]
    split = max(1, len(tail) // 2)
    first, second = tail[:split], tail[split:]
    latest = rows[-1]
    out = {
        "rollouts": len(rows),
        "latest": {field: latest.get(field) for field in FIELDS},
        "tail": {},
        "range": {},
    }
    for field in (
        "eprewmean",
        "entropy",
        "behavior_kl_after_step",
        "lr_used",
        "categorical_fisher_trace",
        "fisher_damping_fraction",
        "base_damping_value",
    ):
        if not any(field in row for row in tail):
            continue
        first_mean = mean(first, field)
        second_mean = mean(second, field)
        out["tail"][field] = {
            "first_half_mean": first_mean,
            "second_half_mean": second_mean,
            "delta": second_mean - first_mean,
            "tail_mean": mean(tail, field),
            "tail_min": min(float(row[field]) for row in tail),
            "tail_max": max(float(row[field]) for row in tail),
        }
    for field in (
        "categorical_fisher_trace",
        "fisher_damping_fraction",
        "base_damping_value",
    ):
        values = [float(row[field]) for row in rows if field in row]
        if values:
            out["range"][field] = {
                "min": min(values),
                "max": max(values),
            }
    causal = [
        row
        for row in all_rows
        if float(row.get("causal_diagnostic_ran", 0.0)) == 1.0
    ]
    if causal:
        recent = causal[-min(8, len(causal)):]
        out["causal"] = {"count": len(causal)}
        out["causal"].update(
            {
                field: mean(recent, field)
                for field in (
                "actor_gain_from_critic_rhs",
                "component_direction_cosine",
                "actor_fullmetric_vs_actoronly_cosine",
                "actor_fullmetric_vs_actoronly_norm_ratio",
                "actor_fullmetric_delta_fraction",
                "actor_component_l2",
                "critic_component_l2",
                "critic_actor_quadratic_fraction",
                "critic_ggn_vs_vanilla_cosine",
                "critic_ggn_vs_vanilla_norm_ratio",
                "critic_trunk_ggn_vs_vanilla_cosine",
                "critic_trunk_ggn_vs_vanilla_norm_ratio",
                "actor_alone_clip_scale",
                "joint_clip_scale",
            )
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="print one rollout-level status line per run",
    )
    args = parser.parse_args()
    result = {}
    for trace in sorted(args.root.glob("*/seed*/metric_trace.jsonl")):
        result[trace.parent.parent.name] = summarize(
            trace, args.window, args.max_transitions
        )
    if args.compact:
        for environment, summary in sorted(result.items()):
            latest = summary["latest"]
            tail = summary["tail"]
            run = args.root / environment / "seed0"
            status_path = run / "status"
            status = (
                status_path.read_text().strip()
                if status_path.is_file()
                else "MISSING"
            )
            print(
                f"{environment} status={status} "
                f"T={int(latest['environment_transitions']):,} "
                f"R={latest['eprewmean']:.3f} "
                f"Rtail={tail['eprewmean']['tail_mean']:.3f} "
                f"dR={tail['eprewmean']['delta']:+.3f} "
                f"H={latest['entropy']:.3f} "
                f"Htail={tail['entropy']['tail_mean']:.3f} "
                f"dH={tail['entropy']['delta']:+.3f} "
                f"KL={latest['behavior_kl_after_step']:.4f} "
                f"KLtail={tail['behavior_kl_after_step']['tail_mean']:.4f} "
                f"LR={latest['lr_used']:.5g} "
                f"res={latest['joint_solve_residual']:.2e} "
                f"F={float(latest.get('categorical_fisher_trace') or float('nan')):.3f} "
                f"Fmin={summary['range'].get('categorical_fisher_trace', {}).get('min', float('nan')):.3f} "
                f"Ffrac={float(latest.get('fisher_damping_fraction') or 0.0):.3f} "
                f"FfracMax={summary['range'].get('fisher_damping_fraction', {}).get('max', 0.0):.3f} "
                f"Hyst={float(latest.get('fisher_hysteresis_engaged') or 0.0):.0f} "
                f"damp={float(latest.get('base_damping_value') or float('nan')):.3f} "
                f"dampMax={summary['range'].get('base_damping_value', {}).get('max', float('nan')):.3f} "
                f"Araw={float(latest.get('raw_actor_kernel_diag_median') or float('nan')):.3g} "
                f"Craw={float(latest.get('raw_critic_kernel_diag_median') or float('nan')):.3g} "
                f"Amed={float(latest.get('actor_kernel_diag_median') or float('nan')):.3f} "
                f"Cmed={float(latest.get('critic_kernel_diag_median') or float('nan')):.3f}"
            )
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
