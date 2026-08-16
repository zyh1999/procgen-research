#!/usr/bin/env python3
"""Structural audit for deterministic task-block-trace RAT B x B runs."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--trainer-sha", required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--min-transitions", type=int, default=100_000)
    args = parser.parse_args()
    failures = []

    print("environment transitions reward entropy behavior_kl actor_residual critic_residual")
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
        expected = {
            "ENV_NAME": env,
            "SEED": "0",
            "TRAINER_SHA256": args.trainer_sha,
            "CONFIG_SHA256": args.config_sha,
            "ROLLOUT": "4096",
            "MINIBATCH": "512",
            "EPOCHS": "4",
            "TRANSITIONS": str(args.min_transitions),
            "JOINT_SYSTEM_ROWS": "512",
            "JOINT_KERNEL": "Hpi_HpiT_over_B_plus_4_Jv_JvT_over_B",
            "COMPRESSION": "task_block_trace_of_strict_2B_operator",
            "RANDOM_SCORE_DRAWS": "0",
            "EXPECTED_CROSS_BLOCK": "0",
            "ACTOR_DAMPING": "0.03",
            "CRITIC_DAMPING": "0.5",
            "MOMENTUM": "0",
            "KACZMARZ": "false",
            "LINEAR_SOLVE_DTYPE": "float64",
        }
        for key, value in expected.items():
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
            checks = {
                "joint_system_rows": int(row.get("joint_system_rows", -1)) == 512,
                "joint_kernel_mode": row.get("joint_kernel_mode") == "rat_deterministic_blocktrace_b",
                "joint_ablation_mode": row.get("joint_ablation_mode") == "rat_blocktrace_critic",
                "critic_score_mode": row.get("joint_critic_score_mode") == "deterministic_task_blocktrace",
                "noise_not_sampled": float(row.get("rat_score_noise_marginalized", -1.0)) == 0.0,
                "cross_block_zero": abs(float(row.get("cross_block_fro", math.inf))) < 1e-12,
                "actor_damping": abs(float(row.get("rat_actor_damping", math.inf)) - 0.03) < 1e-12,
                "critic_damping": abs(float(row.get("rat_critic_damping", math.inf)) - 0.5) < 1e-12,
                "actor_residual": float(row.get("joint_solve_residual", math.inf)) < 1e-7,
                "critic_residual": float(row.get("rat_critic_solve_residual", math.inf)) < 1e-7,
                "critic_geometry": float(row.get("critic_block_fro", 0.0)) > 0.0,
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
        if int(last.get("environment_transitions", -1)) < args.min_transitions:
            failures.append(
                "{}: transitions={}".format(
                    label, last.get("environment_transitions")
                )
            )
        print(
            label,
            last.get("environment_transitions"),
            last.get("eprewmean"),
            last.get("entropy"),
            last.get("behavior_kl_after_step"),
            last.get("joint_solve_residual"),
            last.get("rat_critic_solve_residual"),
        )

    if failures:
        for failure in failures:
            print("FAIL", failure)
        print("AUDIT=FAILED")
        raise SystemExit(1)
    print("AUDIT=PASS")


if __name__ == "__main__":
    main()
