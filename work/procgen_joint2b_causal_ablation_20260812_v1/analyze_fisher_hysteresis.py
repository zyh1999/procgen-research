#!/usr/bin/env python3
"""Simulate a unified Fisher hysteresis guard on recorded metric traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return sorted(rows, key=lambda row: int(row["minibatch_global_step"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--engage", type=float, default=0.55)
    parser.add_argument("--release", type=float, default=0.65)
    args = parser.parse_args()
    if not 0.0 <= args.engage < args.release <= 1.0:
        raise SystemExit("require 0 <= engage < release <= 1")

    for trace in sorted(args.root.glob("*/seed*/metric_trace.jsonl")):
        rows = load_rows(trace)
        engaged = False
        first_engage = None
        releases = 0
        engage_events = 0
        engaged_rows = 0
        longest = 0
        current = 0
        minimum_fisher = float("inf")
        for row in rows:
            fisher = float(row["categorical_fisher_trace"])
            minimum_fisher = min(minimum_fisher, fisher)
            was_engaged = engaged
            if engaged:
                engaged = fisher < args.release
            else:
                engaged = fisher <= args.engage
            if engaged and not was_engaged:
                engage_events += 1
                if first_engage is None:
                    first_engage = row
            if was_engaged and not engaged:
                releases += 1
            if engaged:
                engaged_rows += 1
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        environment = trace.parent.parent.name
        if first_engage is None:
            first = "never"
        else:
            first = (
                f"T={int(first_engage['environment_transitions']):,},"
                f"R={float(first_engage['eprewmean']):.3f},"
                f"H={float(first_engage['entropy']):.3f},"
                f"F={float(first_engage['categorical_fisher_trace']):.3f}"
            )
        print(
            f"{environment} rows={len(rows)} Fmin={minimum_fisher:.3f} "
            f"first={first} events={engage_events} releases={releases} "
            f"engaged_fraction={engaged_rows / len(rows):.3f} "
            f"longest_rows={longest} final_engaged={int(engaged)}"
        )


if __name__ == "__main__":
    main()
