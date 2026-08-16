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


def fail(message: str) -> None:
    raise SystemExit(f"AUDIT=FAIL {message}")


def read_preflight(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def close(actual: float, expected: float, tol: float = 2e-6) -> bool:
    return math.isfinite(actual) and abs(actual - expected) <= tol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--trainer-sha", required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--min-transitions", type=int, default=100_000)
    parser.add_argument("--fisher-high", type=float, default=0.85)
    parser.add_argument("--fisher-low", type=float, default=0.50)
    parser.add_argument("--median-floor-min", type=float, default=0.01)
    parser.add_argument("--median-floor-max", type=float, default=0.10)
    parser.add_argument("--critic-guard-min", type=float, default=0.0)
    parser.add_argument("--critic-guard-max", type=float, default=0.01)
    args = parser.parse_args()

    print("environment transitions reward entropy fisher fraction block_floor critic_guard residual")
    for env in ENVS:
        run = args.root / env / "seed0"
        if not run.is_dir():
            fail(f"{env}: missing run")
        status = (run / "status").read_text().strip()
        rc = int((run / "rc").read_text().strip())
        if status != "PASS" or rc != 0:
            fail(f"{env}: status={status} rc={rc}")
        preflight = read_preflight(run / "preflight")
        expected = {
            "TRAINER_SHA256": args.trainer_sha,
            "CONFIG_SHA256": args.config_sha,
            "ROLLOUT": "4096",
            "MINIBATCH": "512",
            "EPOCHS": "4",
            "JOINT_SYSTEM_ROWS": "1024",
            "JOINT_MODE": "full_joint_clean_all",
            "JOINT_RHS_MODE": "paired_score_residual",
            "JOINT_RECONSTRUCTION_MODE": "full_joint",
            "FISHER_ADAPTIVE_BLOCK_DAMPING": "true",
        }
        for key, value in expected.items():
            if preflight.get(key) != value:
                fail(f"{env}: preflight {key}={preflight.get(key)!r}")

        rows = [json.loads(line) for line in (run / "metric_trace.jsonl").read_text().splitlines() if line.strip()]
        final_rows = [
            row for row in rows
            if int(row.get("optimizer_epoch", -1)) == 3
            and int(row.get("minibatch_index", -1)) == 7
        ]
        if not final_rows:
            fail(f"{env}: no rollout-final rows")
        for row in rows:
            if int(row.get("joint_system_rows", -1)) != 1024:
                fail(f"{env}: joint rows changed")
            if row.get("joint_kernel_mode") != "full_joint_clean_all":
                fail(f"{env}: joint kernel identity changed")
            if row.get("joint_damping_mode") != "block_median_floor":
                fail(f"{env}: damping mode changed")
            if float(row.get("fisher_adaptive_block_damping", 0.0)) != 1.0:
                fail(f"{env}: adaptive damping disabled")
            fisher = float(row["categorical_fisher_trace"])
            fraction = max(0.0, min(1.0, (args.fisher_high - fisher) / (args.fisher_high - args.fisher_low)))
            expected_floor = args.median_floor_min + fraction * (args.median_floor_max - args.median_floor_min)
            expected_guard = args.critic_guard_min + fraction * (args.critic_guard_max - args.critic_guard_min)
            if not close(float(row["fisher_adaptive_fraction"]), fraction):
                fail(f"{env}: wrong Fisher fraction")
            if not close(float(row["damping_to_median_floor"]), expected_floor):
                fail(f"{env}: wrong adaptive block floor")
            if not close(float(row["actor_damping_from_critic_floor"]), expected_guard):
                fail(f"{env}: wrong adaptive critic guard")
            residual = float(row.get("joint_solve_residual", math.inf))
            if not math.isfinite(residual) or residual > 1e-7:
                fail(f"{env}: residual={residual}")

        last = final_rows[-1]
        transitions = int(last["environment_transitions"])
        if transitions < args.min_transitions:
            fail(f"{env}: transitions={transitions}")
        print(
            env,
            transitions,
            float(last["eprewmean"]),
            float(last["entropy"]),
            float(last["categorical_fisher_trace"]),
            float(last["fisher_adaptive_fraction"]),
            float(last["damping_to_median_floor"]),
            float(last["actor_damping_from_critic_floor"]),
            float(last["joint_solve_residual"]),
        )
    print("AUDIT=PASS")


if __name__ == "__main__":
    main()
