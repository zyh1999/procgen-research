#!/usr/bin/env python3
"""Compact live summary for deterministic RHS-aligned B x B runs."""

import argparse
import json
import statistics
from pathlib import Path


ENVS = (
    "bigfish-easy-0-10",
    "bossfight-easy-0-10",
    "caveflyer-easy-0-10",
    "coinrun-easy-0-10",
)


def fmt(value, digits=5):
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument(
        "--checkpoints",
        default="118000,188000,233000",
        help="comma-separated transition targets; nearest rollout is shown",
    )
    args = parser.parse_args()
    checkpoints = tuple(
        int(item) for item in args.checkpoints.split(",") if item.strip()
    )

    print(
        "env status T R tailR H tailH KL LR F mA mC dAeff dCeff res "
        "rhsErr gradErr fullRes wA wC directCos directNorm"
    )
    for env in ENVS:
        run = args.root / env / "seed0"
        if not run.is_dir():
            print(env, "MISSING")
            continue
        status = (run / "status").read_text().strip() if (run / "status").is_file() else "UNKNOWN"
        path = run / "metric_trace.jsonl"
        if not path.is_file():
            print(env, status, "NO_METRIC")
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        final = [
            row for row in rows
            if int(row.get("optimizer_epoch", -1)) == 3
            and int(row.get("minibatch_index", -1)) == 7
        ]
        if not final:
            print(env, status, "NO_ROLLOUT_FINAL")
            continue
        last = final[-1]
        tail = final[-max(args.window, 1):]
        diagnostic = [
            row for row in rows
            if float(row.get("rhs_aligned_direct_diagnostic_ran", 0.0)) == 1.0
        ]
        diag = diagnostic[-1] if diagnostic else {}
        print(
            env,
            status,
            int(last["environment_transitions"]),
            fmt(last.get("eprewmean")),
            fmt(statistics.fmean(float(row["eprewmean"]) for row in tail)),
            fmt(last.get("entropy")),
            fmt(statistics.fmean(float(row["entropy"]) for row in tail)),
            fmt(last.get("behavior_kl_after_step")),
            fmt(last.get("lr_used")),
            fmt(last.get("categorical_fisher_trace")),
            fmt(last.get("actor_kernel_diag_median")),
            fmt(last.get("critic_kernel_diag_median")),
            fmt(last.get("actor_effective_damping_median")),
            fmt(last.get("critic_effective_damping_median")),
            fmt(last.get("joint_solve_residual")),
            fmt(last.get("rhs_aligned_rhs_projection_relative_error")),
            fmt(last.get("rhs_aligned_primal_rhs_relative_error")),
            fmt(last.get("rhs_aligned_full_residual_relative")),
            fmt(last.get("rhs_aligned_actor_weight_mean")),
            fmt(last.get("rhs_aligned_critic_weight_mean")),
            fmt(diag.get("rhs_aligned_direct_direction_cosine")),
            fmt(diag.get("rhs_aligned_direct_direction_norm_ratio")),
        )
        for target in checkpoints:
            nearest = min(
                final,
                key=lambda row: abs(
                    int(row["environment_transitions"]) - target
                ),
            )
            print(
                f"  @{target}",
                int(nearest["environment_transitions"]),
                "R", fmt(nearest.get("eprewmean")),
                "H", fmt(nearest.get("entropy")),
                "KL", fmt(nearest.get("behavior_kl_after_step")),
                "F", fmt(nearest.get("categorical_fisher_trace")),
                "mA", fmt(nearest.get("actor_kernel_diag_median")),
                "mC", fmt(nearest.get("critic_kernel_diag_median")),
                "dA", fmt(nearest.get("actor_effective_damping_median")),
                "dC", fmt(nearest.get("critic_effective_damping_median")),
            )


if __name__ == "__main__":
    main()
