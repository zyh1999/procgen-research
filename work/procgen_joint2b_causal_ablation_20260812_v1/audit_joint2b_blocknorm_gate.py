#!/usr/bin/env python3
"""Audit structural and training health of the matched four-env 1M gate.

This is deliberately a health gate, not the final Paper-RAT performance
audit.  It proves that all four environments ran the same strict 2B identity,
covered at least one million transitions, retained non-degenerate policy
statistics, and kept the normalized actor/critic blocks well scaled.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


ENVIRONMENTS = (
    "bigfish-easy-0-10",
    "bossfight-easy-0-10",
    "caveflyer-easy-0-10",
    "coinrun-easy-0-10",
)
ERROR_RE = re.compile(
    r"OOM|out of memory|NaN|Traceback|CUDA (?:error|failure|out of memory)"
    r"|CUBLAS|CUDNN|Cholesky|singular|AssertionError|I/O error",
    re.IGNORECASE,
)
PREFLIGHT_MARKERS = (
    "SEED=0",
    "ROLLOUT=4096",
    "MINIBATCH=512",
    "EPOCHS=4",
    "TRANSITIONS=1000000",
    "JOINT_SYSTEM_ROWS=1024",
    "JOINT_MODE=full_joint_clean_all",
    "JOINT_RHS_MODE=paired_score_residual",
    "JOINT_RECONSTRUCTION_MODE=full_joint",
    "CRITIC_OBJECTIVE_COEF=1.0",
    "CRITIC_CURVATURE_COEF=1.0",
    "LR_INITIAL_MAX=0.05",
    "KL_RANGE=0.005,0.02",
    "MOMENTUM=0",
    "KACZMARZ=false",
    "LINEAR_SOLVE_DTYPE=float64",
)


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def numeric(row: dict, key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"missing/non-numeric {key}: {exc}")
    if not math.isfinite(value):
        fail(f"non-finite {key}={value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate_variant_root", type=Path)
    parser.add_argument(
        "--normalization",
        default="row_gradient_preserving",
        choices=("median_gradient_preserving", "row_gradient_preserving"),
    )
    parser.add_argument("--damping", type=float, default=0.03)
    parser.add_argument(
        "--fisher-adaptive",
        action="store_true",
        help="audit the unified categorical-Fisher damping schedule",
    )
    parser.add_argument("--fisher-upper", type=float, default=0.50)
    parser.add_argument("--fisher-lower", type=float, default=0.35)
    parser.add_argument("--damping-min", type=float, default=0.03)
    parser.add_argument("--damping-max", type=float, default=0.50)
    parser.add_argument(
        "--fisher-hysteresis",
        action="store_true",
        help="require the stateful low-Fisher engage/release guard",
    )
    parser.add_argument("--hysteresis-engage", type=float, default=0.55)
    parser.add_argument("--hysteresis-release", type=float, default=0.65)
    parser.add_argument(
        "--trainer-sha",
        help="require this immutable trainer SHA256 in every preflight",
    )
    parser.add_argument(
        "--config-sha",
        help="require this immutable config SHA256 in every preflight",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="ENV=RUN_DIR",
        help="use an explicitly preserved retry run for one environment",
    )
    args = parser.parse_args()
    root = args.gate_variant_root
    observed_trainer_shas: set[str] = set()
    observed_config_shas: set[str] = set()
    overrides: dict[str, Path] = {}
    for item in args.override:
        if "=" not in item:
            fail(f"invalid --override {item!r}; expected ENV=RUN_DIR")
        environment, run_dir = item.split("=", 1)
        if environment not in ENVIRONMENTS:
            fail(f"unsupported override environment: {environment}")
        overrides[environment] = Path(run_dir)
    summaries: list[str] = []
    for environment in ENVIRONMENTS:
        run = overrides.get(environment, root / environment / "seed0")
        if not run.is_dir():
            fail(f"missing run directory: {run}")
        if (run / "status").read_text().strip() != "PASS":
            fail(f"{environment}: status is not PASS")
        if (run / "rc").read_text().strip() != "0":
            fail(f"{environment}: rc is not zero")
        preflight = (run / "preflight").read_text()
        if f"ENV_NAME={environment}" not in preflight:
            fail(f"{environment}: preflight environment mismatch")
        for marker in PREFLIGHT_MARKERS:
            if marker not in preflight:
                fail(f"{environment}: preflight missing {marker}")
        normalization_marker = (
            f"JOINT_BLOCK_NORMALIZATION={args.normalization}"
        )
        if normalization_marker not in preflight:
            fail(
                f"{environment}: preflight missing {normalization_marker}"
            )
        if args.fisher_adaptive:
            if args.fisher_hysteresis:
                adaptive_markers = (
                    "DAMPING=FISHER_HYSTERESIS",
                    "FISHER_HYSTERESIS=true",
                    f"FISHER_HYSTERESIS_ENGAGE={args.hysteresis_engage:g}",
                    f"FISHER_HYSTERESIS_RELEASE={args.hysteresis_release:g}",
                    f"DAMPING_MIN={args.damping_min:g}",
                    f"DAMPING_MAX={args.damping_max:g}",
                )
            else:
                adaptive_markers = (
                    "DAMPING=FISHER_ADAPTIVE",
                    f"DAMPING_MIN={args.damping_min:g}",
                    f"DAMPING_MAX={args.damping_max:g}",
                    f"FISHER_DAMPING_UPPER={args.fisher_upper:g}",
                    f"FISHER_DAMPING_LOWER={args.fisher_lower:g}",
                )
            for marker in adaptive_markers:
                if marker not in preflight:
                    fail(f"{environment}: preflight missing {marker}")
        else:
            damping_marker = f"DAMPING={args.damping:g}"
            if damping_marker not in preflight:
                fail(f"{environment}: preflight missing {damping_marker}")
            normalized_damping_marker = (
                f"DAMPING_TO_NORMALIZED_MEDIAN={args.damping:g}"
            )
            if normalized_damping_marker not in preflight:
                fail(
                    f"{environment}: preflight missing "
                    f"{normalized_damping_marker}"
                )
        preflight_values = {}
        for line in preflight.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                preflight_values[key] = value
        trainer_sha = preflight_values.get("TRAINER_SHA256")
        config_sha = preflight_values.get("CONFIG_SHA256")
        if not trainer_sha or not config_sha:
            fail(f"{environment}: preflight missing immutable SHA markers")
        observed_trainer_shas.add(trainer_sha)
        observed_config_shas.add(config_sha)
        if args.trainer_sha and trainer_sha != args.trainer_sha:
            fail(
                f"{environment}: trainer SHA {trainer_sha}, "
                f"expected {args.trainer_sha}"
            )
        if args.config_sha and config_sha != args.config_sha:
            fail(
                f"{environment}: config SHA {config_sha}, "
                f"expected {args.config_sha}"
            )

        rows = [
            json.loads(line)
            for line in (run / "metric_trace.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if not rows:
            fail(f"{environment}: no real metric rows")
        rollout_rows = {
            int(row["rollout_update"]): row for row in rows
        }
        aligned = sorted(
            rollout_rows.values(),
            key=lambda row: int(row["environment_transitions"]),
        )
        last = aligned[-1]
        transitions = int(last["environment_transitions"])
        if transitions < 1_000_000:
            fail(f"{environment}: only {transitions} transitions")
        if int(last.get("joint_system_rows", -1)) != 1024:
            fail(f"{environment}: joint_system_rows is not 1024")
        expected_strings = {
            "joint_block_normalization": args.normalization,
            "joint_rhs_mode": "paired_score_residual",
            "joint_reconstruction_mode": "direct_Ht_alpha",
            "joint_critic_score_mode": "clean",
        }
        for key, expected in expected_strings.items():
            if str(last.get(key)) != expected:
                fail(
                    f"{environment}: {key}={last.get(key)!r}, "
                    f"expected {expected!r}"
                )
        verification_rows = (
            sorted(rows, key=lambda row: int(row["minibatch_global_step"]))
            if args.fisher_hysteresis
            else aligned
        )
        expected_hysteresis_state = False
        for row in verification_rows:
            observed_damping = numeric(row, "base_damping_value")
            if args.fisher_adaptive:
                if numeric(row, "fisher_adaptive_damping") != 1.0:
                    fail(f"{environment}: adaptive damping flag is not one")
                expected_fields = {
                    "configured_base_damping_value": args.damping_min,
                    "fisher_damping_upper": args.fisher_upper,
                    "fisher_damping_lower": args.fisher_lower,
                    "fisher_damping_min": args.damping_min,
                    "fisher_damping_max": args.damping_max,
                }
                for key, expected in expected_fields.items():
                    if not math.isclose(
                        numeric(row, key),
                        expected,
                        rel_tol=1e-6,
                        abs_tol=1e-7,
                    ):
                        fail(
                            f"{environment}: {key}={row.get(key)}, "
                            f"expected {expected}"
                        )
                fisher = numeric(row, "categorical_fisher_trace")
                if args.fisher_hysteresis:
                    if numeric(row, "fisher_damping_hysteresis") != 1.0:
                        fail(
                            f"{environment}: hysteresis flag is not one"
                        )
                    if not math.isclose(
                        numeric(row, "fisher_hysteresis_engage"),
                        args.hysteresis_engage,
                        rel_tol=1e-6,
                        abs_tol=1e-7,
                    ) or not math.isclose(
                        numeric(row, "fisher_hysteresis_release"),
                        args.hysteresis_release,
                        rel_tol=1e-6,
                        abs_tol=1e-7,
                    ):
                        fail(f"{environment}: hysteresis thresholds mismatch")
                    if expected_hysteresis_state:
                        expected_hysteresis_state = (
                            fisher < args.hysteresis_release
                        )
                    else:
                        expected_hysteresis_state = (
                            fisher <= args.hysteresis_engage
                        )
                    expected_fraction = (
                        1.0 if expected_hysteresis_state else 0.0
                    )
                    if numeric(row, "fisher_hysteresis_engaged") != (
                        1.0 if expected_hysteresis_state else 0.0
                    ):
                        fail(
                            f"{environment}: hysteresis state does not "
                            f"follow trace {fisher}"
                        )
                else:
                    expected_fraction = min(
                        1.0,
                        max(
                            0.0,
                            (args.fisher_upper - fisher)
                            / (args.fisher_upper - args.fisher_lower),
                        ),
                    )
                expected_damping = args.damping_min + expected_fraction * (
                    args.damping_max - args.damping_min
                )
                if not math.isclose(
                    numeric(row, "fisher_damping_fraction"),
                    expected_fraction,
                    rel_tol=2e-5,
                    abs_tol=2e-6,
                ):
                    fail(
                        f"{environment}: Fisher fraction does not match "
                        f"trace {fisher}"
                    )
            else:
                expected_damping = args.damping
            if not math.isclose(
                observed_damping,
                expected_damping,
                rel_tol=2e-5,
                abs_tol=2e-6,
            ):
                fail(
                    f"{environment}: base_damping_value={observed_damping}, "
                    f"expected {expected_damping}"
                )
            observed_relative_damping = numeric(
                row, "base_damping_to_median_diag"
            )
            observed_kernel_median = numeric(row, "kernel_diag_median")
            expected_relative_damping = (
                expected_damping / observed_kernel_median
            )
            if not math.isclose(
                observed_relative_damping,
                expected_relative_damping,
                rel_tol=2e-3,
                abs_tol=2e-4,
            ):
                fail(
                    f"{environment}: base_damping_to_median_diag="
                    f"{observed_relative_damping}, expected normalized "
                    f"ratio {expected_relative_damping}"
                )

        tail = aligned[-min(100, len(aligned)):]
        actor_medians = [
            numeric(row, "actor_kernel_diag_median") for row in tail
        ]
        critic_medians = [
            numeric(row, "critic_kernel_diag_median") for row in tail
        ]
        if not all(0.5 <= value <= 2.0 for value in actor_medians):
            fail(f"{environment}: normalized actor median left [0.5, 2]")
        if not all(0.95 <= value <= 1.05 for value in critic_medians):
            fail(f"{environment}: normalized critic median left [0.95, 1.05]")
        if max(numeric(row, "joint_solve_residual") for row in tail) > 1e-6:
            fail(f"{environment}: tail solve residual exceeds 1e-6")
        entropies = [numeric(row, "entropy") for row in tail]
        fisher_traces = [
            numeric(row, "categorical_fisher_trace") for row in tail
        ]
        kls = [numeric(row, "behavior_kl_after_step") for row in tail]
        if min(entropies) < 0.2:
            fail(f"{environment}: tail entropy collapse")
        if min(fisher_traces) < 0.02:
            fail(f"{environment}: categorical Fisher trace collapse")
        if statistics.fmean(kls) > 0.04:
            fail(f"{environment}: tail mean KL exceeds 0.04")

        combined_logs = "\n".join(
            path.read_text(errors="replace")
            for path in (run / "stdout", run / "stderr")
            if path.exists()
        )
        match = ERROR_RE.search(combined_logs)
        if match:
            fail(f"{environment}: error signature {match.group(0)}")
        summaries.append(
            f"{environment}:T={transitions},"
            f"R={statistics.fmean(numeric(r, 'eprewmean') for r in tail):.4g},"
            f"H={statistics.fmean(entropies):.4g},"
            f"KL={statistics.fmean(kls):.4g},"
            f"Amed={statistics.fmean(actor_medians):.4g},"
            f"Cmed={statistics.fmean(critic_medians):.4g},"
            f"Fmin={min(fisher_traces):.4g},"
            f"Dmin={min(numeric(r, 'base_damping_value') for r in tail):.4g},"
            f"Dmax={max(numeric(r, 'base_damping_value') for r in tail):.4g}"
        )

    if len(observed_trainer_shas) != 1:
        fail(f"trainer SHA mismatch across environments: {observed_trainer_shas}")
    if len(observed_config_shas) != 1:
        fail(f"config SHA mismatch across environments: {observed_config_shas}")
    print("AUDIT=PASS HEALTH_ONLY")
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
