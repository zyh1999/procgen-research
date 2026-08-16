#!/usr/bin/env python3
"""Decide whether a one-seed 1M gate deserves a longer formal run.

This intentionally does *not* certify the research goal.  The original
Procgen RAT curves learn well beyond one million transitions, so this gate
only rejects a candidate that is already materially behind the matched
sampled-B reference at the same transition count.  The final 6M x 3-seed
auditor remains authoritative for high-reward success.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


ENVIRONMENTS = (
    "bigfish-easy-0-10",
    "bossfight-easy-0-10",
    "caveflyer-easy-0-10",
    "coinrun-easy-0-10",
)


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def load_rollouts(path: Path) -> list[dict]:
    if not path.is_file():
        fail(f"missing metric trace: {path}")
    by_update: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_update[int(row["rollout_update"])] = row
    rows = sorted(
        by_update.values(), key=lambda row: int(row["environment_transitions"])
    )
    if not rows:
        fail(f"no rollout metrics: {path}")
    return rows


def aligned_mean(
    rows: list[dict], key: str, transition: int, count: int
) -> float:
    eligible = [
        row for row in rows if int(row["environment_transitions"]) <= transition
    ]
    if not eligible:
        fail(f"no {key} rows at or before transition {transition}")
    values = [float(row[key]) for row in eligible[-count:]]
    if not values or not all(math.isfinite(value) for value in values):
        fail(f"missing/non-finite {key} near transition {transition}")
    return statistics.fmean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("sampled_b_reference_root", type=Path)
    parser.add_argument(
        "--paper-rat-reference",
        type=Path,
        default=Path(__file__).with_name("paper_rat_6m_reference.json"),
    )
    parser.add_argument("--min-transitions", type=int, default=1_000_000)
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--trend-window", type=int, default=40)
    parser.add_argument("--allowed-drop-abs", type=float, default=0.25)
    parser.add_argument("--allowed-drop-rel", type=float, default=0.10)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="ENV=RUN_DIR",
        help="use an explicitly preserved retry run for one environment",
    )
    args = parser.parse_args()

    overrides: dict[str, Path] = {}
    for item in args.override:
        if "=" not in item:
            fail(f"invalid --override {item!r}; expected ENV=RUN_DIR")
        environment, run_dir = item.split("=", 1)
        if environment not in ENVIRONMENTS:
            fail(f"unsupported override environment: {environment}")
        overrides[environment] = Path(run_dir)

    paper = json.loads(args.paper_rat_reference.read_text())
    failed = False
    print(
        "| environment | aligned T | candidate reward | sampled-B reward | "
        "margin | candidate H | candidate KL | Paper RAT 6M tail | "
        "fraction of final target | reward trend | entropy trend | gate |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for environment in ENVIRONMENTS:
        candidate_run = overrides.get(
            environment,
            args.candidate_root / environment / "seed0",
        )
        if not candidate_run.is_dir():
            fail(f"missing candidate run: {candidate_run}")
        status_path = candidate_run / "status"
        rc_path = candidate_run / "rc"
        if (
            not status_path.is_file()
            or status_path.read_text().strip() != "PASS"
            or not rc_path.is_file()
            or rc_path.read_text().strip() != "0"
        ):
            print("AUDIT=INCOMPLETE")
            raise SystemExit(2)

        candidate = load_rollouts(candidate_run / "metric_trace.jsonl")
        baseline = load_rollouts(
            args.sampled_b_reference_root
            / environment
            / "seed0"
            / "metric_trace.jsonl"
        )
        candidate_max = int(candidate[-1]["environment_transitions"])
        baseline_max = int(baseline[-1]["environment_transitions"])
        if candidate_max < args.min_transitions:
            print("AUDIT=INCOMPLETE")
            raise SystemExit(2)
        transition = min(candidate_max, baseline_max)

        candidate_reward = aligned_mean(
            candidate, "eprewmean", transition, args.tail
        )
        baseline_reward = aligned_mean(
            baseline, "eprewmean", transition, args.tail
        )
        candidate_entropy = aligned_mean(
            candidate, "entropy", transition, args.tail
        )
        candidate_kl = aligned_mean(
            candidate, "behavior_kl_after_step", transition, args.tail
        )
        trend_rows = [
            row
            for row in candidate
            if int(row["environment_transitions"]) <= transition
        ][-args.trend_window:]
        split = max(1, len(trend_rows) // 2)
        first_trend, second_trend = trend_rows[:split], trend_rows[split:]
        reward_trend = statistics.fmean(
            float(row["eprewmean"]) for row in second_trend
        ) - statistics.fmean(float(row["eprewmean"]) for row in first_trend)
        entropy_first = statistics.fmean(
            float(row["entropy"]) for row in first_trend
        )
        entropy_second = statistics.fmean(
            float(row["entropy"]) for row in second_trend
        )
        entropy_trend = entropy_second - entropy_first
        allowed_drop = max(
            args.allowed_drop_abs,
            args.allowed_drop_rel * abs(baseline_reward),
        )
        reward_ok = candidate_reward >= baseline_reward - allowed_drop
        policy_ok = candidate_entropy >= 0.2 and candidate_kl <= 0.04
        sustained_collapse = (
            entropy_second < 0.75 * entropy_first
            and reward_trend < -max(0.25, 0.10 * abs(candidate_reward))
        )
        gate_ok = reward_ok and policy_ok and not sustained_collapse
        failed |= not gate_ok

        paper_tail_values = paper["environments"][environment][
            "tail_eprewmean"
        ]
        paper_tail = statistics.fmean(float(v) for v in paper_tail_values)
        fraction = candidate_reward / paper_tail if paper_tail > 0 else math.nan
        print(
            f"| {environment} | {transition:,} | {candidate_reward:.4g} | "
            f"{baseline_reward:.4g} | {candidate_reward-baseline_reward:+.4g} | "
            f"{candidate_entropy:.4g} | {candidate_kl:.4g} | "
            f"{paper_tail:.4g} | {fraction:.3f} | "
            f"{reward_trend:+.4g} | {entropy_trend:+.4g} | "
            f"{'EXTEND' if gate_ok else 'REJECT'} |"
        )

    if failed:
        print("AUDIT=FAILED_GATE")
        raise SystemExit(1)
    print("AUDIT=PASS EXTEND_NOT_FINAL")


if __name__ == "__main__":
    main()
