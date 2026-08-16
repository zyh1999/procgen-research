#!/usr/bin/env python3
"""Render aligned reward/entropy curves for the formal joint-2B result."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from audit_joint2b_performance import ENVIRONMENTS, traces


ENV_LABELS = {
    "bigfish-easy-0-10": "BigFish",
    "bossfight-easy-0-10": "BossFight",
    "caveflyer-easy-0-10": "CaveFlyer",
    "coinrun-easy-0-10": "CoinRun",
}


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    result = np.full(values.shape, np.nan, dtype=float)
    cumsum = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    means = (cumsum[window:] - cumsum[:-window]) / window
    result[window - 1:] = means
    return result


def series(rows: list[dict], key: str, window: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([int(row["environment_transitions"]) for row in rows])
    y = np.asarray([float(row[key]) for row in rows])
    return x, rolling(y, window)


def common_candidate(
    seed_rows: dict[str, list[dict]], key: str, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    common_x = sorted(
        set.intersection(*[
            {int(row["environment_transitions"]) for row in rows}
            for rows in seed_rows.values()
        ])
    )
    x = np.asarray(common_x, dtype=int)
    seed_values: dict[str, np.ndarray] = {}
    for seed, rows in seed_rows.items():
        mapping = {
            int(row["environment_transitions"]): float(row[key]) for row in rows
        }
        seed_values[seed] = rolling(
            np.asarray([mapping[transition] for transition in common_x]), window
        )
    matrix = np.vstack(list(seed_values.values()))
    valid = np.any(np.isfinite(matrix), axis=0)
    mean = np.full(matrix.shape[1], np.nan, dtype=float)
    mean[valid] = np.nanmean(matrix[:, valid], axis=0)
    if matrix.shape[0] > 1:
        sem = np.full(matrix.shape[1], np.nan, dtype=float)
        sem[valid] = (
            np.nanstd(matrix[:, valid], axis=0, ddof=1)
            / np.sqrt(matrix.shape[0])
        )
    else:
        sem = np.zeros_like(mean)
    return x, mean, sem, seed_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("strict_baseline", type=Path)
    parser.add_argument("sampled_baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--title", default="Procgen shared strict joint 2B×2B")
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), sharex="col")
    colors = {
        "candidate": "#f28e2b",
        "strict": "#59a14f",
        "sampled": "#4e79a7",
    }

    for column, environment in enumerate(ENVIRONMENTS):
        candidate = {
            seed: rows for seed, (_, rows) in traces(args.candidate, environment).items()
        }
        strict = traces(args.strict_baseline, environment)
        sampled = traces(args.sampled_baseline, environment)

        for row_index, (key, ylabel) in enumerate((
            ("eprewmean", "Episode reward"),
            ("entropy", "Policy entropy"),
        )):
            axis = axes[row_index, column]
            if candidate:
                x, mean, sem, per_seed = common_candidate(
                    candidate, key, args.window
                )
                for seed, values in sorted(per_seed.items()):
                    axis.plot(
                        x / 1e6, values, color=colors["candidate"], alpha=0.2,
                        linewidth=1.0,
                    )
                axis.plot(
                    x / 1e6, mean, color=colors["candidate"], linewidth=2.5,
                    label=f"block damping ({len(candidate)} seeds)",
                )
                axis.fill_between(
                    x / 1e6, mean - sem, mean + sem,
                    color=colors["candidate"], alpha=0.18,
                )

            for label, baseline, color, linestyle in (
                ("old strict-clean", strict, colors["strict"], "--"),
                ("sampled-B", sampled, colors["sampled"], ":"),
            ):
                for _, rows in baseline.values():
                    bx, by = series(rows, key, args.window)
                    axis.plot(
                        bx / 1e6, by, color=color, linestyle=linestyle,
                        linewidth=2.0, label=label,
                    )

            axis.grid(alpha=0.22)
            if row_index == 0:
                axis.set_title(ENV_LABELS[environment], fontsize=13)
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == 1:
                axis.set_xlabel("Environment transitions (millions)")
            if column == 0 and row_index == 0:
                axis.legend(frameon=False, fontsize=9)

    fig.suptitle(args.title, fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    if args.output.suffix.lower() != ".pdf":
        fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
