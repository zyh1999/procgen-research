#!/usr/bin/env python3
"""Structural audit for deterministic expected-score RAT B x B smoke."""

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


def parse_preflight(path: Path):
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    failures = []

    for env in ENVS:
        run = args.root / env / "seed0"
        label = env.split("-")[0]
        if not run.is_dir():
            failures.append(f"{label}: missing run")
            continue
        status = (run / "status").read_text().strip() if (run / "status").is_file() else "MISSING"
        rc = (run / "rc").read_text().strip() if (run / "rc").is_file() else "MISSING"
        if status != "PASS" or rc != "0":
            failures.append(f"{label}: status={status} rc={rc}")
        preflight_path = run / "preflight"
        if not preflight_path.is_file():
            failures.append(f"{label}: missing preflight")
            continue
        preflight = parse_preflight(preflight_path)
        expected = {
            "ROLLOUT": "4096",
            "MINIBATCH": "512",
            "EPOCHS": "4",
            "TRANSITIONS": "100000",
            "JOINT_SYSTEM_ROWS": "512",
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
                    f"{label}: preflight {key}={preflight.get(key)!r}, expected {value!r}"
                )
        trace_path = run / "metric_trace.jsonl"
        if not trace_path.is_file():
            failures.append(f"{label}: missing metric trace")
            continue
        rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        finals = [
            row for row in rows
            if int(row.get("optimizer_epoch", -1)) == 3
            and int(row.get("minibatch_index", -1)) == 7
        ]
        if not finals:
            failures.append(f"{label}: no rollout-final metric")
            continue
        last = finals[-1]
        checks = {
            "environment_transitions": int(last.get("environment_transitions", -1)) >= 100_000,
            "joint_system_rows": int(last.get("joint_system_rows", -1)) == 512,
            "joint_kernel_mode": last.get("joint_kernel_mode") == "rat_expected_gaussian_score_b",
            "joint_ablation_mode": last.get("joint_ablation_mode") == "rat_expected_critic",
            "rat_score_noise_marginalized": float(last.get("rat_score_noise_marginalized", -1)) == 1.0,
            "cross_block_fro": abs(float(last.get("cross_block_fro", math.inf))) < 1e-12,
            "actor_damping": abs(float(last.get("rat_actor_damping", math.inf)) - 0.03) < 1e-12,
            "critic_damping": abs(float(last.get("rat_critic_damping", math.inf)) - 0.5) < 1e-12,
            "actor_residual": float(last.get("joint_solve_residual", math.inf)) < 1e-7,
            "critic_residual": float(last.get("rat_critic_solve_residual", math.inf)) < 1e-7,
        }
        for key, passed in checks.items():
            if not passed:
                failures.append(f"{label}: failed metric check {key}: {last.get(key)!r}")
        text = "\n".join(
            path.read_text(errors="replace")
            for path in (run / "stdout", run / "stderr")
            if path.is_file()
        )
        match = ERROR_RE.search(text)
        if match:
            failures.append(f"{label}: error scan matched {match.group(0)!r}")
        print(
            f"{label}: T={last.get('environment_transitions')} "
            f"R={last.get('eprewmean')} H={last.get('entropy')} "
            f"KL={last.get('behavior_kl_after_step')}"
        )

    if failures:
        for failure in failures:
            print("FAIL", failure)
        print("AUDIT=FAILED")
        raise SystemExit(1)
    print("AUDIT=PASS")


if __name__ == "__main__":
    main()
