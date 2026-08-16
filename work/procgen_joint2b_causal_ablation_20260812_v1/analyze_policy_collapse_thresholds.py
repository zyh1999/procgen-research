#!/usr/bin/env python3
"""Locate rollout-level policy-collapse thresholds in Procgen traces."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def rollout_rows(path: Path) -> list[dict]:
    by_update: dict[int, dict] = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                by_update[int(row["rollout_update"])] = row
    return sorted(
        by_update.values(),
        key=lambda row: int(row["environment_transitions"]),
    )


def tail_mean(rows: list[dict], index: int, key: str, window: int) -> float:
    values = [
        float(row[key]) for row in rows[max(0, index - window + 1) : index + 1]
    ]
    return statistics.fmean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument(
        "--entropy-thresholds",
        type=float,
        nargs="+",
        default=(2.0, 1.5, 1.2, 1.0, 0.8, 0.5),
    )
    parser.add_argument(
        "--fisher-thresholds",
        type=float,
        nargs="+",
        default=(0.8, 0.6, 0.5, 0.4, 0.3, 0.2),
    )
    args = parser.parse_args()

    for trace in sorted(args.root.glob("*/seed*/metric_trace.jsonl")):
        rows = rollout_rows(trace)
        environment = trace.parent.parent.name
        print(f"[{environment}] rollouts={len(rows)}")
        for field, thresholds in (
            ("entropy", args.entropy_thresholds),
            ("categorical_fisher_trace", args.fisher_thresholds),
        ):
            for threshold in thresholds:
                found = next(
                    (
                        (index, row)
                        for index, row in enumerate(rows)
                        if float(row[field]) < threshold
                    ),
                    None,
                )
                if found is None:
                    print(f"  {field}<{threshold:g}: never")
                    continue
                index, row = found
                print(
                    f"  {field}<{threshold:g}: "
                    f"T={int(row['environment_transitions']):,} "
                    f"R={float(row['eprewmean']):.3f} "
                    f"Rtail={tail_mean(rows,index,'eprewmean',args.window):.3f} "
                    f"H={float(row['entropy']):.3f} "
                    f"F={float(row['categorical_fisher_trace']):.3f} "
                    f"KL={float(row['behavior_kl_after_step']):.4f}"
                )


if __name__ == "__main__":
    main()
