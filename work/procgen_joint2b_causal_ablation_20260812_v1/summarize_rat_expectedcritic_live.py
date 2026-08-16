#!/usr/bin/env python3
"""Compact live summary for deterministic expected-score RAT B x B runs."""

import argparse
import json
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
    args = parser.parse_args()

    print("env status T R H KL LR dA dC resA resC KcDiag cross marginalized")
    for env in ENVS:
        run = args.root / env / "seed0"
        if not run.is_dir():
            print(env, "MISSING")
            continue
        status_path = run / "status"
        status = status_path.read_text().strip() if status_path.is_file() else "UNKNOWN"
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
        print(
            env,
            status,
            int(last["environment_transitions"]),
            fmt(last.get("eprewmean")),
            fmt(last.get("entropy")),
            fmt(last.get("behavior_kl_after_step")),
            fmt(last.get("lr_used")),
            fmt(last.get("rat_actor_damping")),
            fmt(last.get("rat_critic_damping")),
            fmt(last.get("joint_solve_residual")),
            fmt(last.get("rat_critic_solve_residual")),
            fmt(last.get("rat_expected_critic_kernel_diag_median")),
            fmt(last.get("cross_block_fro")),
            fmt(last.get("rat_score_noise_marginalized")),
        )


if __name__ == "__main__":
    main()
