#!/usr/bin/env python3
"""Audit the matched actor-from-critic damping guard gate.

The gate is deliberately stricter than a scheduler RUNNING/PASS label.  It
requires all four Procgen environments, an immutable preflight identity, real
rollout metrics, solver health, and the damping-guard invariant.  Reward is
reported but is not used alone to early-stop a still-healthy prefix.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ENVIRONMENTS = (
    "bigfish-easy-0-10",
    "bossfight-easy-0-10",
    "caveflyer-easy-0-10",
    "coinrun-easy-0-10",
)
EXPECTED_PREFLIGHT = {
    "SEED": "0",
    "TRAINER_SHA256": "58e73af1dde57ae34250167a28b14c30a508f2ffc5781fdd673791b3b46bb962",
    "CONFIG_SHA256": "3a24a057bdb6898e0ca3e6153eddfc7d6272700f5df083f345d84c9f940ffdb0",
    "ROLLOUT": "4096",
    "MINIBATCH": "512",
    "EPOCHS": "4",
    "TRANSITIONS": "3000000",
    "BASE_DAMPING": "0.5",
    "JOINT_DAMPING_MODE": "block_median_floor",
    "JOINT_DAMPING_TO_MEDIAN_FLOOR": "0.1",
    "ACTOR_DAMPING_FROM_CRITIC_FLOOR": "0.01",
    "ADAPTIVE_LR_INITIAL": "0.004",
    "ADAPTIVE_LR_MAX": "0.02",
    "ADAPTIVE_KL_LOWER": "0.005",
    "ADAPTIVE_KL_UPPER": "0.04",
    "MOMENTUM": "0",
    "KACZMARZ": "false",
    "JOINT_SYSTEM_ROWS": "1024",
    "JOINT_MODE": "full_joint_clean_all",
    "CRITIC_OBJECTIVE_COEF": "1.0",
    "CRITIC_CURVATURE_COEF": "1.0",
    "LINEAR_SOLVE_DTYPE": "float64",
}
ERROR_MARKERS = (
    "out of memory",
    "traceback",
    "cuda error",
    "cholesky",
    "assertionerror",
)


def preflight(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def rollout_rows(path: Path) -> list[dict]:
    final: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        transition = int(row["environment_transitions"])
        old = final.get(transition)
        if old is None or int(row.get("minibatch_global_step", -1)) >= int(
            old.get("minibatch_global_step", -1)
        ):
            final[transition] = row
    return [final[key] for key in sorted(final)]


def mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else math.nan


def fmt(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.6g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    # The failed matched baseline first crossed KL .04 around 0.97M on
    # CaveFlyer and 1.46M on CoinRun, with entropy < .2 by 1.19M/1.59M.
    # A 1M gate would therefore certify before the known failure regime.
    parser.add_argument("--min-transitions", type=int, default=2_000_000)
    parser.add_argument("--tail", type=int, default=100)
    args = parser.parse_args()

    failures: list[str] = []
    incomplete: list[str] = []
    print("environment transitions reward entropy behavior_KL residual cross_block actor_to_critic_ratio tail_min_ratio first_guard_binding guard_binding guard_violations")
    for environment in ENVIRONMENTS:
        run = args.root / environment / "seed0"
        pf_path = run / "preflight"
        trace = run / "metric_trace.jsonl"
        if not pf_path.exists() or not trace.exists():
            incomplete.append(f"{environment}: missing preflight or metric trace")
            print(f"{environment} MISSING")
            continue

        pf = preflight(pf_path)
        if pf.get("ENV_NAME") != environment:
            failures.append(f"{environment}: ENV_NAME={pf.get('ENV_NAME')!r}")
        for key, expected in EXPECTED_PREFLIGHT.items():
            if pf.get(key) != expected:
                failures.append(
                    f"{environment}: {key}={pf.get(key)!r}, expected {expected!r}"
                )

        rows = rollout_rows(trace)
        if not rows:
            incomplete.append(f"{environment}: no real rollout metrics")
            print(f"{environment} NO_METRICS")
            continue
        transition = int(rows[-1]["environment_transitions"])
        if transition < args.min_transitions:
            incomplete.append(
                f"{environment}: {transition:,} < {args.min_transitions:,} transitions"
            )
        tail = rows[-args.tail :]
        latest = rows[-1]
        reward = mean(tail, "eprewmean")
        entropy = mean(tail, "entropy")
        behavior_kl = mean(tail, "behavior_kl_after_step")
        residuals = [
            abs(float(row["joint_solve_residual"]))
            for row in tail
            if row.get("joint_solve_residual") is not None
        ]
        max_residual = max(residuals, default=math.nan)
        cross_block = float(latest.get("cross_block_fro", math.nan))
        normalized_cross = float(
            latest.get("normalized_cross_block", math.nan)
        )
        binding = 0
        violations = 0
        checked = 0
        actor_to_critic_ratios: list[float] = []
        first_guard_binding: int | None = None
        for row in rows:
            try:
                floor = float(row["actor_damping_from_critic_floor"])
                critic = float(row["critic_kernel_diag_median"])
                actor = float(row["actor_effective_damping_median"])
            except (KeyError, TypeError, ValueError):
                continue
            required = floor * critic
            tolerance = 1.0e-6 * max(1.0, abs(actor), abs(required))
            if floor > 0.0 and abs(actor - required) <= tolerance:
                first_guard_binding = int(row["environment_transitions"])
                break
        for row in tail:
            try:
                floor = float(row["actor_damping_from_critic_floor"])
                critic = float(row["critic_kernel_diag_median"])
                actor = float(row["actor_effective_damping_median"])
            except (KeyError, TypeError, ValueError):
                continue
            required = floor * critic
            tolerance = 1.0e-6 * max(1.0, abs(actor), abs(required))
            checked += 1
            if math.isfinite(actor) and math.isfinite(critic) and critic > 0.0:
                actor_to_critic_ratios.append(actor / critic)
            binding += abs(actor - required) <= tolerance
            violations += actor < required - tolerance

        latest_actor_to_critic_ratio = math.nan
        try:
            latest_actor = float(latest["actor_effective_damping_median"])
            latest_critic = float(latest["critic_kernel_diag_median"])
            if math.isfinite(latest_actor) and latest_critic > 0.0:
                latest_actor_to_critic_ratio = latest_actor / latest_critic
        except (KeyError, TypeError, ValueError):
            pass
        tail_min_ratio = min(actor_to_critic_ratios, default=math.nan)

        if not math.isfinite(entropy) or entropy < 0.2:
            failures.append(f"{environment}: unhealthy entropy {fmt(entropy)}")
        if not math.isfinite(behavior_kl) or behavior_kl > 0.04:
            failures.append(f"{environment}: unhealthy behavior KL {fmt(behavior_kl)}")
        if not math.isfinite(max_residual) or max_residual > 1.0e-6:
            failures.append(f"{environment}: unhealthy solve residual {fmt(max_residual)}")
        if int(latest.get("joint_system_rows", -1)) != 1024:
            failures.append(f"{environment}: runtime rows are not 1024")
        if latest.get("joint_kernel_mode") != "full_joint_clean_all":
            failures.append(f"{environment}: runtime kernel mode mismatch")
        if not math.isfinite(cross_block) or cross_block <= 0.0:
            failures.append(f"{environment}: missing/nonpositive cross block")
        if not math.isfinite(normalized_cross) or normalized_cross <= 0.0:
            failures.append(f"{environment}: missing/nonpositive normalized cross block")
        if checked == 0:
            failures.append(f"{environment}: no guard diagnostics")
        if violations:
            failures.append(f"{environment}: {violations}/{checked} guard violations")

        error_text = "\n".join(
            path.read_text(errors="replace").lower()
            for path in (run / "stderr", run / "stdout")
            if path.exists()
        )
        for marker in ERROR_MARKERS:
            if marker in error_text:
                failures.append(f"{environment}: error marker {marker!r}")

        print(
            environment,
            transition,
            fmt(reward),
            fmt(entropy),
            fmt(behavior_kl),
            fmt(max_residual),
            fmt(cross_block),
            fmt(latest_actor_to_critic_ratio),
            fmt(tail_min_ratio),
            "-" if first_guard_binding is None else str(first_guard_binding),
            f"{binding}/{checked}",
            f"{violations}/{checked}",
        )

    for message in failures:
        print(f"FAIL: {message}")
    for message in incomplete:
        print(f"INCOMPLETE: {message}")
    if failures:
        print("AUDIT=FAIL")
        return 1
    if incomplete:
        print("AUDIT=INCOMPLETE")
        return 2
    print("AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
