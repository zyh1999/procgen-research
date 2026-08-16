#!/usr/bin/env python3
"""Fail closed when auditing the formal four-environment joint-2B campaign."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


ENVIRONMENTS = (
    "bigfish-easy-0-10",
    "bossfight-easy-0-10",
    "caveflyer-easy-0-10",
    "coinrun-easy-0-10",
)
SEEDS = (0, 1, 2)
EXPECTED_PREFLIGHT = {
    "VARIANT": "acguard01",
    "TRAINER_SHA256": "2709256861583122e61bd0211bb85ab7f3108273455b3f957f98515b463c0475",
    "CONFIG_SHA256": "3a24a057bdb6898e0ca3e6153eddfc7d6272700f5df083f345d84c9f940ffdb0",
    "LAUNCHER_SHA256": "93ba823918cca76a7a15104984a8a26aee8409f633acd3e3b39315525344cb89",
    "ROLLOUT": "4096",
    "MINIBATCH": "512",
    "EPOCHS": "4",
    "TRANSITIONS": "6000000",
    "BASE_DAMPING": "0.5",
    "JOINT_DAMPING_MODE": "block_median_floor",
    "JOINT_DAMPING_TO_MEDIAN_FLOOR": "0.1",
    "ACTOR_DAMPING_FROM_CRITIC_FLOOR": "0.01",
    "METRIC_FLUSH_RETRIES": "6",
    "MOMENTUM": "0",
    "KACZMARZ": "false",
    "JOINT_SYSTEM_ROWS": "1024",
    "JOINT_MODE": "full_joint_clean_all",
    "CRITIC_OBJECTIVE_COEF": "1.0",
    "CRITIC_CURVATURE_COEF": "1.0",
    "LINEAR_SOLVE_DTYPE": "float64",
    "ADAPTIVE_LR_INITIAL": "0.004",
    "ADAPTIVE_LR_MAX": "0.02",
    "ADAPTIVE_KL_LOWER": "0.005",
    "ADAPTIVE_KL_UPPER": "0.04",
}
FATAL = re.compile(
    r"out of memory|\bNaN\b|Traceback|CUDA error|Cholesky|AssertionError",
    re.IGNORECASE,
)


def read_preflight(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def last_json(path: Path) -> dict:
    last = ""
    with path.open() as source:
        for line in source:
            if line.strip():
                last = line
    if not last:
        raise ValueError("empty metric trace")
    return json.loads(last)


def guard_violations(path: Path) -> tuple[int, int, float]:
    """Return checked rows, violation count, and worst damping deficit."""
    checked = 0
    violations = 0
    worst_deficit = 0.0
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                floor = float(row["actor_damping_from_critic_floor"])
                critic = float(row["critic_kernel_diag_median"])
                actor = float(row["actor_effective_damping_median"])
            except (KeyError, TypeError, ValueError):
                continue
            required = floor * critic
            tolerance = 1.0e-6 * max(1.0, abs(actor), abs(required))
            checked += 1
            deficit = required - actor
            if deficit > tolerance:
                violations += 1
                worst_deficit = max(worst_deficit, deficit)
    return checked, violations, worst_deficit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    incomplete = False
    failed = False
    print("| environment | seed | status | transitions | reward | entropy | KL | guard violations | audit |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---|")
    for environment in ENVIRONMENTS:
        for seed in SEEDS:
            run = args.root / environment / f"seed{seed}"
            problems: list[str] = []
            if not run.is_dir():
                incomplete = True
                print(f"| {environment} | {seed} | MISSING | 0 | - | - | - | - | incomplete |")
                continue

            required = ("preflight", "status", "metric_trace.jsonl", "stdout", "stderr")
            missing = [name for name in required if not (run / name).is_file()]
            if missing:
                incomplete = True
                problems.append("missing:" + ",".join(missing))

            preflight: dict[str, str] = {}
            if (run / "preflight").is_file():
                preflight = read_preflight(run / "preflight")
                for key, expected in EXPECTED_PREFLIGHT.items():
                    if preflight.get(key) != expected:
                        problems.append(f"{key}={preflight.get(key)!r}")
                if preflight.get("ENV_NAME") != environment:
                    problems.append("ENV_NAME mismatch")
                if preflight.get("SEED") != str(seed):
                    problems.append("SEED mismatch")

            status = (run / "status").read_text().strip() if (run / "status").is_file() else "MISSING"
            if status != "PASS":
                incomplete = True

            metric: dict = {}
            checked_guard = 0
            violated_guard = 0
            if (run / "metric_trace.jsonl").is_file():
                try:
                    metric = last_json(run / "metric_trace.jsonl")
                except (ValueError, json.JSONDecodeError) as error:
                    problems.append(str(error))
                if metric:
                    if int(metric.get("joint_system_rows", -1)) != 1024:
                        problems.append("joint_system_rows mismatch")
                    if metric.get("joint_kernel_mode") != "full_joint_clean_all":
                        problems.append("joint_kernel_mode mismatch")
                    cross_block = float(metric.get("cross_block_fro", math.nan))
                    normalized_cross = float(
                        metric.get("normalized_cross_block", math.nan)
                    )
                    if not math.isfinite(cross_block) or cross_block <= 0.0:
                        problems.append("missing/nonpositive runtime cross block")
                    if not math.isfinite(normalized_cross) or normalized_cross <= 0.0:
                        problems.append("missing/nonpositive normalized cross block")
                    if metric.get("joint_damping_mode") != "block_median_floor":
                        problems.append("joint_damping_mode mismatch")
                    if float(metric.get("damping_to_median_floor", -1)) != 0.1:
                        problems.append("damping floor mismatch")
                    if float(metric.get("actor_damping_from_critic_floor", -1)) != 0.01:
                        problems.append("actor-from-critic floor mismatch")
                    if int(metric.get("environment_transitions", 0)) < 6_000_000:
                        incomplete = True
                try:
                    checked_guard, violated_guard, worst_deficit = guard_violations(
                        run / "metric_trace.jsonl"
                    )
                except (ValueError, json.JSONDecodeError) as error:
                    problems.append(f"guard scan: {error}")
                if checked_guard == 0:
                    problems.append("no guard diagnostics")
                if violated_guard:
                    problems.append(
                        f"guard violations={violated_guard}/{checked_guard}, "
                        f"worst deficit={worst_deficit:.6g}"
                    )

            for name in ("stdout", "stderr"):
                path = run / name
                if path.is_file():
                    match = FATAL.search(path.read_text(errors="replace"))
                    if match:
                        problems.append(f"{name}:{match.group(0)}")

            if problems:
                failed = True
            transitions = int(metric.get("environment_transitions", 0))
            reward = metric.get("eprewmean", "-")
            entropy = metric.get("entropy", "-")
            kl = metric.get("behavior_kl_after_step", "-")
            audit = "; ".join(problems) if problems else ("complete" if status == "PASS" else "incomplete")
            print(
                f"| {environment} | {seed} | {status} | {transitions:,} | "
                f"{reward} | {entropy} | {kl} | {violated_guard}/{checked_guard} | {audit} |"
            )

    if failed:
        print("AUDIT=FAILED")
        return 1
    if incomplete:
        print("AUDIT=INCOMPLETE")
        return 2
    print("AUDIT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
