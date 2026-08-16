#!/usr/bin/env python3
"""Audit matched deterministic direct-2B and exact Schur-B smoke runs."""

import argparse
import json
import math
import re
from pathlib import Path


ENVS = (
    "bigfish-easy-0-10",
    "bossfight-easy-0-10",
    "caveflyer-easy-0-10",
    "coinrun-easy-0-10",
)

ERROR_RE = re.compile(
    r"out of memory|\bOOM\b|\bNaN\b|Traceback|CUDA error|Cholesky|"
    r"singular|AssertionError|I/O error",
    re.IGNORECASE,
)


def fail(message: str) -> None:
    raise SystemExit(f"AUDIT=FAIL {message}")


def read_preflight(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def close(actual: float, expected: float, tolerance: float = 2e-6) -> bool:
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


def require_close(env: str, key: str, actual: float, expected: float) -> None:
    if not close(actual, expected):
        fail(f"{env}: {key}={actual} expected={expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--solver", choices=("direct_2b", "schur_critic_b"), required=True)
    parser.add_argument("--trainer-sha", required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--min-transitions", type=int, default=100_000)
    parser.add_argument("--actor-damping-min", type=float, default=0.03)
    parser.add_argument("--actor-damping-max", type=float, default=0.5)
    parser.add_argument("--critic-damping", type=float, default=0.5)
    parser.add_argument("--fisher-high", type=float, default=0.85)
    parser.add_argument("--fisher-low", type=float, default=0.50)
    parser.add_argument("--max-residual", type=float, default=1e-7)
    parser.add_argument("--max-equivalence-error", type=float, default=1e-8)
    args = parser.parse_args()

    if not args.fisher_high > args.fisher_low:
        fail("Fisher thresholds must satisfy high > low")

    print(
        "environment transitions reward entropy behavior_kl fisher "
        "actor_damping critic_damping residual equivalence_max"
    )
    for env in ENVS:
        run = args.root / env / "seed0"
        if not run.is_dir():
            fail(f"{env}: missing run")
        status_path = run / "status"
        rc_path = run / "rc"
        if not status_path.is_file() or not rc_path.is_file():
            fail(f"{env}: missing status or rc")
        status = status_path.read_text().strip()
        rc = int(rc_path.read_text().strip())
        if status != "PASS" or rc != 0:
            fail(f"{env}: status={status!r} rc={rc}")

        preflight = read_preflight(run / "preflight")
        expected_preflight = {
            "ENV_NAME": env,
            "SEED": "0",
            "TRAINER_SHA256": args.trainer_sha,
            "CONFIG_SHA256": args.config_sha,
            "ROLLOUT": "4096",
            "MINIBATCH": "512",
            "EPOCHS": "4",
            "TRANSITIONS": str(args.min_transitions),
            "JOINT_LINEAR_SOLVER": args.solver,
            "ACTOR_BASE_DAMPING_MIN": str(args.actor_damping_min),
            "ACTOR_BASE_DAMPING_MAX": str(args.actor_damping_max),
            "CRITIC_BASE_DAMPING": str(args.critic_damping),
            "MOMENTUM": "0",
            "KACZMARZ": "false",
            "LINEAR_SOLVE_DTYPE": "float64",
        }
        if args.solver == "direct_2b":
            expected_preflight.update({
                "JOINT_SYSTEM_ROWS": "1024",
                "JOINT_MODE": "full_joint_clean_all",
            })
        else:
            expected_preflight.update({
                "PARENT_SYSTEM_ROWS": "1024",
                "REDUCED_SYSTEM_ROWS": "512",
                "FULL_CROSS_BLOCK": "1",
                "CRITIC_SCORE": "clean",
            })
        for key, value in expected_preflight.items():
            if preflight.get(key) != value:
                fail(f"{env}: preflight {key}={preflight.get(key)!r} expected={value!r}")

        metric_path = run / "metric_trace.jsonl"
        rows = [
            json.loads(line)
            for line in metric_path.read_text().splitlines()
            if line.strip()
        ]
        if not rows:
            fail(f"{env}: no metric rows")
        final_rows = [
            row
            for row in rows
            if int(row.get("optimizer_epoch", -1)) == 3
            and int(row.get("minibatch_index", -1)) == 7
        ]
        if not final_rows:
            fail(f"{env}: no rollout-final rows")

        equivalence_errors: list[float] = []
        for row in rows:
            if int(row.get("joint_system_rows", -1)) != 1024:
                fail(f"{env}: parent system is not 1024 rows")
            if row.get("joint_linear_solver") != args.solver:
                fail(f"{env}: solver identity changed")
            if row.get("joint_kernel_mode") != "full_joint_clean_all":
                fail(f"{env}: joint kernel identity changed")
            if row.get("joint_rhs_mode") != "paired_score_residual":
                fail(f"{env}: RHS identity changed")
            if row.get("joint_critic_score_mode") != "clean":
                fail(f"{env}: critic score identity changed")
            if float(row.get("cross_block_fro", 0.0)) <= 0.0:
                fail(f"{env}: missing actor-critic cross block")

            fisher = float(row["categorical_fisher_trace"])
            fraction = max(
                0.0,
                min(1.0, (args.fisher_high - fisher) / (args.fisher_high - args.fisher_low)),
            )
            expected_actor = args.actor_damping_min + fraction * (
                args.actor_damping_max - args.actor_damping_min
            )
            require_close(
                env,
                "fisher_adaptive_fraction",
                float(row["fisher_adaptive_fraction"]),
                fraction,
            )
            require_close(
                env,
                "actor_base_damping",
                float(row["actor_base_damping"]),
                expected_actor,
            )
            require_close(
                env,
                "critic_base_damping",
                float(row["critic_base_damping"]),
                args.critic_damping,
            )
            if float(row["actor_effective_damping_median"]) + 2e-6 < expected_actor:
                fail(f"{env}: actor effective damping below base damping")
            if float(row["critic_effective_damping_median"]) + 2e-6 < args.critic_damping:
                fail(f"{env}: critic effective damping below base damping")

            residual = float(row.get("joint_solve_residual", math.inf))
            if not math.isfinite(residual) or residual > args.max_residual:
                fail(f"{env}: joint residual={residual}")

            if args.solver == "schur_critic_b":
                if int(row.get("schur_reduced_rows", -1)) != 512:
                    fail(f"{env}: Schur system is not 512 rows")
                for key in ("schur_critic_block_residual", "schur_reduced_residual"):
                    value = float(row.get(key, math.inf))
                    if not math.isfinite(value) or value > args.max_residual:
                        fail(f"{env}: {key}={value}")
                if float(row.get("schur_equivalence_diagnostic_ran", 0.0)) == 1.0:
                    error = float(row.get("schur_direct_alpha_relative_error", math.inf))
                    if not math.isfinite(error):
                        fail(f"{env}: non-finite Schur equivalence error")
                    equivalence_errors.append(error)
            elif int(row.get("schur_reduced_rows", -1)) != 0:
                fail(f"{env}: direct solver unexpectedly reports a reduced system")

        if args.solver == "schur_critic_b":
            if not equivalence_errors:
                fail(f"{env}: no Schur/direct equivalence diagnostics")
            if max(equivalence_errors) > args.max_equivalence_error:
                fail(
                    f"{env}: Schur/direct equivalence error={max(equivalence_errors)}"
                )

        for log_name in ("stderr", "stdout"):
            log_path = run / log_name
            if log_path.is_file():
                match = ERROR_RE.search(log_path.read_text(errors="replace"))
                if match:
                    fail(f"{env}: error marker {match.group(0)!r} in {log_name}")

        last = final_rows[-1]
        transitions = int(last["environment_transitions"])
        if transitions < args.min_transitions:
            fail(f"{env}: transitions={transitions}")
        print(
            env,
            transitions,
            float(last["eprewmean"]),
            float(last["entropy"]),
            float(last["behavior_kl_after_step"]),
            float(last["categorical_fisher_trace"]),
            float(last["actor_base_damping"]),
            float(last["critic_base_damping"]),
            float(last["joint_solve_residual"]),
            max(equivalence_errors) if equivalence_errors else 0.0,
        )
    print("AUDIT=PASS")


if __name__ == "__main__":
    main()
