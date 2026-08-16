#!/usr/bin/env python3
"""Compare two Procgen metric traces using one final minibatch per rollout."""

import argparse
import json
from pathlib import Path


def fmean(values):
    """Compatibility mean for the older Python available on Bede."""
    values = list(values)
    return sum(values) / len(values)


def load(run: Path):
    rows = [
        json.loads(line)
        for line in (run / "metric_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return [
        row for row in rows
        if int(row.get("optimizer_epoch", -1)) == 3
        and int(row.get("minibatch_index", -1)) == 7
    ]


def nearest(rows, transition):
    return min(rows, key=lambda row: abs(int(row["environment_transitions"]) - transition))


def summarize(name, rows, aligned_transition, tail):
    aligned = nearest(rows, aligned_transition)
    window = rows[-min(tail, len(rows)):]
    first_half = window[: max(1, len(window) // 2)]
    second_half = window[max(1, len(window) // 2):] or window[-1:]
    return {
        "name": name,
        "latest_T": int(rows[-1]["environment_transitions"]),
        "aligned_T": int(aligned["environment_transitions"]),
        "aligned_R": float(aligned["eprewmean"]),
        "aligned_H": float(aligned["entropy"]),
        "aligned_KL": float(aligned["behavior_kl_after_step"]),
        "tail_R": fmean(float(row["eprewmean"]) for row in window),
        "tail_H": fmean(float(row["entropy"]) for row in window),
        "tail_KL": fmean(float(row["behavior_kl_after_step"]) for row in window),
        "window_reward_delta": (
            fmean(float(row["eprewmean"]) for row in second_half)
            - fmean(float(row["eprewmean"]) for row in first_half)
        ),
        "window_entropy_delta": (
            fmean(float(row["entropy"]) for row in second_half)
            - fmean(float(row["entropy"]) for row in first_half)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("direct_run", type=Path)
    parser.add_argument("expected_run", type=Path)
    parser.add_argument("--tail", type=int, default=20)
    args = parser.parse_args()
    direct = load(args.direct_run)
    expected = load(args.expected_run)
    aligned_transition = min(
        int(direct[-1]["environment_transitions"]),
        int(expected[-1]["environment_transitions"]),
    )
    for record in (
        summarize("direct_2b", direct, aligned_transition, args.tail),
        summarize("expected_b", expected, aligned_transition, args.tail),
    ):
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
