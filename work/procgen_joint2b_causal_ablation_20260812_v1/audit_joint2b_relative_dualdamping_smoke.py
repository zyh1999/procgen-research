#!/usr/bin/env python3
"""Audit scale-relative dual-damping strict direct-2B Procgen runs."""

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
    r"OOM|out of memory|NaN|Traceback|CUDA error|Cholesky|singular|"
    r"AssertionError|I/O error",
    re.IGNORECASE,
)


def parse_preflight(path):
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def close(actual, expected, tolerance=2e-6):
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--trainer-sha", required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--min-transitions", type=int, default=100_000)
    parser.add_argument("--actor-absolute", type=float, default=0.003)
    parser.add_argument("--critic-absolute", type=float, default=0.01)
    parser.add_argument("--relative", type=float, default=0.10)
    parser.add_argument("--actor-from-critic", type=float, default=0.01)
    args = parser.parse_args()
    failures = []

    print(
        "environment transitions reward entropy behavior_kl "
        "actor_damping critic_damping actor_ratio critic_ratio residual"
    )
    for env in ENVS:
        label = env.split("-")[0]
        run = args.root / env / "seed0"
        if not run.is_dir():
            failures.append("{}: missing run".format(label))
            continue
        status_path = run / "status"
        rc_path = run / "rc"
        status = status_path.read_text().strip() if status_path.is_file() else "MISSING"
        rc = rc_path.read_text().strip() if rc_path.is_file() else "MISSING"
        if status != "PASS" or rc != "0":
            failures.append("{}: status={} rc={}".format(label, status, rc))

        preflight_path = run / "preflight"
        if not preflight_path.is_file():
            failures.append("{}: missing preflight".format(label))
            continue
        preflight = parse_preflight(preflight_path)
        expected_preflight = {
            "ENV_NAME": env,
            "SEED": "0",
            "TRAINER_SHA256": args.trainer_sha,
            "CONFIG_SHA256": args.config_sha,
            "ROLLOUT": "4096",
            "MINIBATCH": "512",
            "EPOCHS": "4",
            "TRANSITIONS": str(args.min_transitions),
            "JOINT_SYSTEM_ROWS": "1024",
            "JOINT_LINEAR_SOLVER": "direct_2b",
            "JOINT_MODE": "full_joint_clean_all",
            "ACTOR_ABSOLUTE_FLOOR": str(args.actor_absolute),
            "CRITIC_ABSOLUTE_FLOOR": str(args.critic_absolute),
            "BLOCK_MEDIAN_RELATIVE_FLOOR": "{:.2f}".format(args.relative),
            "ACTOR_FROM_CRITIC_RELATIVE_FLOOR": "{:.2f}".format(args.actor_from_critic),
            "FISHER_ADAPTIVE_BLOCK_DAMPING": "false",
            "MOMENTUM": "0",
            "KACZMARZ": "false",
            "LINEAR_SOLVE_DTYPE": "float64",
        }
        for key, value in expected_preflight.items():
            if preflight.get(key) != value:
                failures.append(
                    "{}: preflight {}={!r}, expected={!r}".format(
                        label, key, preflight.get(key), value
                    )
                )

        trace_path = run / "metric_trace.jsonl"
        if not trace_path.is_file():
            failures.append("{}: missing metric trace".format(label))
            continue
        rows = [
            json.loads(line)
            for line in trace_path.read_text().splitlines()
            if line.strip()
        ]
        finals = [
            row
            for row in rows
            if int(row.get("optimizer_epoch", -1)) == 3
            and int(row.get("minibatch_index", -1)) == 7
        ]
        if not finals:
            failures.append("{}: no rollout-final metric".format(label))
            continue

        for row in rows:
            actor_median = float(row.get("actor_kernel_diag_median", math.nan))
            critic_median = float(row.get("critic_kernel_diag_median", math.nan))
            expected_actor = max(
                args.actor_absolute,
                args.relative * actor_median,
                args.actor_from_critic * critic_median,
            )
            expected_critic = max(
                args.critic_absolute,
                args.relative * critic_median,
            )
            checks = {
                "rows": int(row.get("joint_system_rows", -1)) == 1024,
                "solver": row.get("joint_linear_solver") == "direct_2b",
                "kernel": row.get("joint_kernel_mode") == "full_joint_clean_all",
                "rhs": row.get("joint_rhs_mode") == "paired_score_residual",
                "score": row.get("joint_critic_score_mode") == "clean",
                "cross": float(row.get("cross_block_fro", 0.0)) > 0.0,
                "fisher_schedule_off": float(row.get("fisher_adaptive_block_damping", 1.0)) == 0.0,
                "actor_damping": close(
                    float(row.get("actor_effective_damping_median", math.inf)),
                    expected_actor,
                ),
                "critic_damping": close(
                    float(row.get("critic_effective_damping_median", math.inf)),
                    expected_critic,
                ),
                "residual": float(row.get("joint_solve_residual", math.inf)) < 1e-7,
            }
            for key, passed in checks.items():
                if not passed:
                    failures.append(
                        "{}: failed {} at T={}".format(
                            label, key, row.get("environment_transitions")
                        )
                    )
                    break

        text = "\n".join(
            path.read_text(errors="replace")
            for path in (run / "stdout", run / "stderr")
            if path.is_file()
        )
        match = ERROR_RE.search(text)
        if match:
            failures.append(
                "{}: error scan matched {!r}".format(label, match.group(0))
            )

        last = finals[-1]
        transitions = int(last.get("environment_transitions", -1))
        if transitions < args.min_transitions:
            failures.append("{}: transitions={}".format(label, transitions))
        actor_median = max(float(last["actor_kernel_diag_median"]), 1e-30)
        critic_median = max(float(last["critic_kernel_diag_median"]), 1e-30)
        actor_damping = float(last["actor_effective_damping_median"])
        critic_damping = float(last["critic_effective_damping_median"])
        print(
            label,
            transitions,
            last.get("eprewmean"),
            last.get("entropy"),
            last.get("behavior_kl_after_step"),
            actor_damping,
            critic_damping,
            actor_damping / actor_median,
            critic_damping / critic_median,
            last.get("joint_solve_residual"),
        )

    if failures:
        for failure in failures:
            print("FAIL", failure)
        print("AUDIT=FAILED")
        raise SystemExit(1)
    print("AUDIT=PASS")


if __name__ == "__main__":
    main()
