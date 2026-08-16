#!/usr/bin/env python3
"""Strict performance/stability audit for the formal Procgen joint-2B run.

The structural audit verifies run identity.  This companion audit verifies the
two experimental claims that motivated the new configuration:

1. At the same transition count, reward must clearly exceed the low-performing
   sampled-B reference in every environment.
2. The candidate must avoid the old strict-clean entropy collapse while also
   remaining stable through the end of all three 6M-transition seeds.

Exit status is 0 only when every check passes, 2 while the formal run is still
incomplete, and 1 for a completed but failed audit.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

from summarize_joint2b import field_mean, records


ENVIRONMENTS = (
    "bigfish-easy-0-10",
    "bossfight-easy-0-10",
    "caveflyer-easy-0-10",
    "coinrun-easy-0-10",
)


def traces(root: Path, environment: str) -> dict[str, tuple[Path, list[dict]]]:
    result: dict[str, tuple[Path, list[dict]]] = {}
    candidates = list(root.glob(f"{environment}/seed*/metric_trace.jsonl"))
    candidates += list(root.glob(f"*/{environment}/seed*/metric_trace.jsonl"))
    for path in candidates:
        seed = path.parent.name
        rows = records(path)
        if rows:
            result[seed] = (path, rows)
    return result


def tail_mean(rows: list[dict], key: str, count: int) -> float | None:
    return field_mean(rows[max(0, len(rows) - count):], key)


def aligned_mean(
    rows: list[dict], key: str, transition: int, count: int
) -> float | None:
    # Never compare a short candidate prefix against a later baseline point.
    # All Procgen traces are emitted on rollout boundaries, so requiring the
    # trace to cover the requested transition is the fail-closed choice.
    if not rows or int(rows[-1]["environment_transitions"]) < transition:
        return None
    eligible = [row for row in rows if row["environment_transitions"] <= transition]
    return tail_mean(eligible, key, count)


def seed_mean_sem(values: list[float]) -> tuple[float, float]:
    mean = fmean(values)
    sem = stdev(values) / sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem


def status_pass(trace: Path) -> bool:
    status = trace.parent / "status"
    return status.exists() and status.read_text().strip().upper().startswith("PASS")


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("strict_baseline", type=Path)
    parser.add_argument("sampled_baseline", type=Path)
    parser.add_argument(
        "--paper-rat-reference",
        type=Path,
        default=Path(__file__).with_name("paper_rat_6m_reference.json"),
        help="frozen 6M five-seed Paper RAT reference summary",
    )
    parser.add_argument(
        "--paper-rat-min-ratio",
        type=float,
        default=0.80,
        help="minimum candidate final mean as a fraction of Paper RAT tail mean",
    )
    parser.add_argument(
        "--paper-rat-worst-seed-ratio",
        type=float,
        default=0.50,
        help="minimum candidate worst seed as a fraction of Paper RAT worst tail seed",
    )
    parser.add_argument("--min-transitions", type=int, default=6_000_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tail", type=int, default=50)
    parser.add_argument("--aligned-tail", type=int, default=20)
    parser.add_argument("--entropy-floor", type=float, default=0.2)
    parser.add_argument("--kl-ceiling", type=float, default=0.04)
    parser.add_argument("--strict-entropy-gain", type=float, default=0.25)
    parser.add_argument("--reward-margin-abs", type=float, default=0.25)
    parser.add_argument("--reward-margin-rel", type=float, default=0.10)
    parser.add_argument("--reward-retention-abs", type=float, default=0.50)
    parser.add_argument("--reward-retention-rel", type=float, default=0.25)
    parser.add_argument("--final-seed-slack-abs", type=float, default=0.50)
    parser.add_argument("--final-seed-slack-rel", type=float, default=0.25)
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help=(
            "treat a trace that has reached --min-transitions as complete even "
            "when its status is still RUNNING; intended only for the pre-formal gate"
        ),
    )
    args = parser.parse_args()

    paper_rat = json.loads(args.paper_rat_reference.read_text())
    if not 0.0 < args.paper_rat_min_ratio <= 1.0:
        raise ValueError("--paper-rat-min-ratio must lie in (0, 1]")
    if not 0.0 < args.paper_rat_worst_seed_ratio <= 1.0:
        raise ValueError("--paper-rat-worst-seed-ratio must lie in (0, 1]")

    print("# Joint-2B performance and stability audit")
    print("| environment | complete seeds | aligned transition | candidate reward | sampled-B reward | reward margin | final reward mean | final reward min | Paper RAT tail mean | Paper RAT ratio | candidate entropy | strict entropy | final min entropy | final max KL | audit |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    incomplete = False
    failed = False
    for environment in ENVIRONMENTS:
        candidate = traces(args.candidate, environment)
        strict = traces(args.strict_baseline, environment)
        sampled = traces(args.sampled_baseline, environment)

        complete = {
            seed: rows
            for seed, (path, rows) in candidate.items()
            if (args.allow_running or status_pass(path))
            and int(rows[-1]["environment_transitions"]) >= args.min_transitions
        }
        if len(complete) < args.seeds:
            incomplete = True

        baseline_rows = [rows for _, rows in strict.values()] + [
            rows for _, rows in sampled.values()
        ]
        aligned_transition = min(
            [int(rows[-1]["environment_transitions"]) for rows in baseline_rows]
            or [0]
        )

        aligned_candidate_rewards = [
            value
            for _, rows in candidate.values()
            if (value := aligned_mean(
                rows, "eprewmean", aligned_transition, args.aligned_tail
            )) is not None
        ]
        aligned_candidate_entropy = [
            value
            for _, rows in candidate.values()
            if (value := aligned_mean(
                rows, "entropy", aligned_transition, args.aligned_tail
            )) is not None
        ]
        sampled_rewards = [
            value
            for _, rows in sampled.values()
            if (value := aligned_mean(
                rows, "eprewmean", aligned_transition, args.aligned_tail
            )) is not None
        ]
        strict_entropy = [
            value
            for _, rows in strict.values()
            if (value := aligned_mean(
                rows, "entropy", aligned_transition, args.aligned_tail
            )) is not None
        ]

        cand_reward = fmean(aligned_candidate_rewards) if aligned_candidate_rewards else None
        sampled_reward = fmean(sampled_rewards) if sampled_rewards else None
        cand_entropy = fmean(aligned_candidate_entropy) if aligned_candidate_entropy else None
        old_entropy = fmean(strict_entropy) if strict_entropy else None
        reward_margin = None
        reward_ok = False
        if cand_reward is not None and sampled_reward is not None:
            reward_margin = cand_reward - sampled_reward
            required_margin = max(
                args.reward_margin_abs,
                args.reward_margin_rel * abs(sampled_reward),
            )
            reward_ok = reward_margin >= required_margin
        entropy_repair_ok = (
            cand_entropy is not None
            and old_entropy is not None
            and cand_entropy >= old_entropy + args.strict_entropy_gain
        )

        final_entropies = [
            value
            for rows in complete.values()
            if (value := tail_mean(rows, "entropy", args.tail)) is not None
        ]
        final_kls = [
            value
            for rows in complete.values()
            if (value := tail_mean(
                rows, "behavior_kl_after_step", args.tail
            )) is not None
        ]
        final_min_entropy = min(final_entropies) if final_entropies else None
        final_max_kl = max(final_kls) if final_kls else None
        final_rewards = [
            value
            for rows in complete.values()
            if (value := tail_mean(rows, "eprewmean", args.tail)) is not None
        ]
        final_reward_mean = fmean(final_rewards) if final_rewards else None
        final_reward_min = min(final_rewards) if final_rewards else None
        paper_rat_tail = paper_rat["environments"][environment][
            "tail_eprewmean"
        ]
        paper_rat_tail_mean = fmean(paper_rat_tail)
        paper_rat_tail_min = min(paper_rat_tail)
        paper_rat_ratio = (
            final_reward_mean / paper_rat_tail_mean
            if final_reward_mean is not None and paper_rat_tail_mean > 0.0
            else None
        )
        paper_rat_mean_ok = (
            final_reward_mean is not None
            and final_reward_mean
            >= args.paper_rat_min_ratio * paper_rat_tail_mean
        )
        paper_rat_seed_ok = (
            final_reward_min is not None
            and final_reward_min
            >= args.paper_rat_worst_seed_ratio * paper_rat_tail_min
        )

        # A run is not stably high-reward merely because it was good near the
        # short reference horizon and retained entropy at 6M.  Require the
        # final aggregate to preserve the aligned gain, and reject a single
        # collapsed seed that would otherwise be hidden by the mean.
        final_reward_ok = False
        final_seed_ok = False
        reward_retention_ok = False
        if final_reward_mean is not None and sampled_reward is not None:
            required_margin = max(
                args.reward_margin_abs,
                args.reward_margin_rel * abs(sampled_reward),
            )
            final_reward_ok = final_reward_mean >= sampled_reward + required_margin
            seed_slack = max(
                args.final_seed_slack_abs,
                args.final_seed_slack_rel * abs(sampled_reward),
            )
            final_seed_ok = (
                len(final_rewards) >= args.seeds
                and all(value >= sampled_reward - seed_slack for value in final_rewards)
            )
        if final_reward_mean is not None and cand_reward is not None:
            retention_slack = max(
                args.reward_retention_abs,
                args.reward_retention_rel * abs(cand_reward),
            )
            reward_retention_ok = final_reward_mean >= cand_reward - retention_slack
        stability_ok = (
            len(final_entropies) >= args.seeds
            and len(final_kls) >= args.seeds
            and final_min_entropy is not None
            and final_min_entropy >= args.entropy_floor
            and final_max_kl is not None
            and final_max_kl <= args.kl_ceiling
        )

        checks = []
        if len(complete) < args.seeds:
            checks.append("INCOMPLETE")
        # Missing aligned candidate data is already an incomplete campaign,
        # not evidence of a reward or entropy failure.
        aligned_candidate_ready = cand_reward is not None and cand_entropy is not None
        if aligned_candidate_ready and not reward_ok:
            checks.append("REWARD")
        if aligned_candidate_ready and not entropy_repair_ok:
            checks.append("ENTROPY_REPAIR")
        if len(complete) >= args.seeds and not stability_ok:
            checks.append("STABILITY")
        if len(complete) >= args.seeds and not final_reward_ok:
            checks.append("FINAL_REWARD")
        if len(complete) >= args.seeds and not final_seed_ok:
            checks.append("SEED_COLLAPSE")
        if len(complete) >= args.seeds and not reward_retention_ok:
            checks.append("REWARD_RETENTION")
        if len(complete) >= args.seeds and not paper_rat_mean_ok:
            checks.append("PAPER_RAT_MEAN")
        if len(complete) >= args.seeds and not paper_rat_seed_ok:
            checks.append("PAPER_RAT_SEED")
        if len(complete) >= args.seeds and checks:
            failed = True
        verdict = "PASS" if not checks else ",".join(checks)

        print("| " + " | ".join([
            environment,
            str(len(complete)),
            f"{aligned_transition:,}",
            fmt(cand_reward),
            fmt(sampled_reward),
            fmt(reward_margin),
            fmt(final_reward_mean),
            fmt(final_reward_min),
            fmt(paper_rat_tail_mean),
            fmt(paper_rat_ratio),
            fmt(cand_entropy),
            fmt(old_entropy),
            fmt(final_min_entropy),
            fmt(final_max_kl),
            verdict,
        ]) + " |")

    if incomplete:
        print("AUDIT=INCOMPLETE")
        raise SystemExit(2)
    if failed:
        print("AUDIT=FAILED")
        raise SystemExit(1)
    print("AUDIT=PASS")


if __name__ == "__main__":
    main()
