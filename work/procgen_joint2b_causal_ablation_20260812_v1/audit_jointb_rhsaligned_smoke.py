#!/usr/bin/env python3
"""Audit deterministic RHS-aligned rank-1 B x B Procgen smoke runs."""

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
    r"out of memory|\bOOM\b|\bNaN\b|Traceback|CUDA error|Cholesky|"
    r"singular|AssertionError|I/O error",
    re.IGNORECASE,
)


def fail(message: str) -> None:
    raise SystemExit(f"AUDIT=FAIL {message}")


def preflight(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def close(actual: float, expected: float, tolerance: float = 2e-6) -> bool:
    # Kernel medians and effective dampings are emitted from float32 tensors.
    # At the large late-training block scales (roughly 30--40 damping here),
    # one float32 ULP can exceed the old absolute-only 2e-6 threshold even
    # when the logged damping is exactly the configured expression.  Keep the
    # small-value absolute guard, but admit normal float32 roundoff in values
    # whose magnitude has grown.
    return math.isfinite(actual) and math.isclose(
        actual,
        expected,
        rel_tol=2e-7,
        abs_tol=tolerance,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--trainer-sha", required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-transitions", type=int, default=100_000)
    parser.add_argument("--max-invariant-error", type=float, default=1e-8)
    parser.add_argument("--max-primal-error", type=float, default=2e-5)
    parser.add_argument("--max-residual", type=float, default=1e-7)
    parser.add_argument(
        "--damping-profile",
        choices=("fisher_adaptive", "actorrelative_criticfloor"),
        default="fisher_adaptive",
    )
    parser.add_argument("--actor-absolute", type=float, default=0.003)
    parser.add_argument("--critic-absolute", type=float, default=0.5)
    parser.add_argument("--relative", type=float, default=0.10)
    parser.add_argument("--actor-from-critic", type=float, default=0.01)
    args = parser.parse_args()

    print(
        "environment seed transitions reward entropy kl fisher actor_d critic_d "
        "solve_residual rhs_error primal_error full_residual "
        "direct_cos direct_norm_ratio"
    )
    for env in ENVS:
        run = args.root / env / f"seed{args.seed}"
        if not run.is_dir():
            fail(f"{env}: missing run")
        status = (run / "status").read_text().strip()
        rc = int((run / "rc").read_text().strip())
        if status != "PASS" or rc != 0:
            fail(f"{env}: status={status!r} rc={rc}")

        pf = preflight(run / "preflight")
        expected = {
            "ENV_NAME": env,
            "SEED": str(args.seed),
            "TRAINER_SHA256": args.trainer_sha,
            "CONFIG_SHA256": args.config_sha,
            "ROLLOUT": "4096",
            "MINIBATCH": "512",
            "EPOCHS": "4",
            "TRANSITIONS": str(args.min_transitions),
            "PARENT_ROWS": "1024",
            "REDUCED_SYSTEM_ROWS": "512",
            "JOINT_LINEAR_SOLVER": "rhs_aligned_rank1_b",
            "REDUCTION": "deterministic_rhs_aligned_rank1_galerkin",
            "RHS_PROJECTION": "transformed_sqrt_ratio_rhs",
            "FIRST_ORDER_GRADIENT_PRESERVED": "1",
            "FULL_COMPRESSED_CROSS_TERMS": "1",
            "CRITIC_SCORE": "clean",
            "MOMENTUM": "0",
            "KACZMARZ": "false",
            "LINEAR_SOLVE_DTYPE": "float64",
        }
        for key, value in expected.items():
            if pf.get(key) != value:
                fail(f"{env}: preflight {key}={pf.get(key)!r} expected={value!r}")

        rows = [
            json.loads(line)
            for line in (run / "metric_trace.jsonl").read_text().splitlines()
            if line.strip()
        ]
        final_rows = [
            row
            for row in rows
            if int(row.get("optimizer_epoch", -1)) == 3
            and int(row.get("minibatch_index", -1)) == 7
        ]
        if not final_rows:
            fail(f"{env}: no rollout-final rows")

        direct_rows = []
        for row in rows:
            if row.get("joint_linear_solver") != "rhs_aligned_rank1_b":
                fail(f"{env}: solver identity changed")
            if int(row.get("joint_system_rows", -1)) != 1024:
                fail(f"{env}: parent rows changed")
            if int(row.get("rhs_aligned_reduced_rows", -1)) != 512:
                fail(f"{env}: reduced rows changed")
            if row.get("joint_kernel_mode") != "full_joint_clean_all":
                fail(f"{env}: kernel identity changed")
            if row.get("joint_rhs_mode") != "paired_score_residual":
                fail(f"{env}: RHS identity changed")
            if row.get("joint_critic_score_mode") != "clean":
                fail(f"{env}: critic score is not deterministic clean")
            if not close(float(row["critic_score_noise_mean"]), 1.0):
                fail(f"{env}: critic score mean changed")
            if not close(float(row["critic_score_noise_std"]), 0.0):
                fail(f"{env}: critic score contains randomness")

            fisher = float(row["categorical_fisher_trace"])
            if args.damping_profile == "fisher_adaptive":
                fraction = max(0.0, min(1.0, (0.85 - fisher) / 0.35))
                expected_actor_damping = 0.03 + fraction * (0.5 - 0.03)
                if not close(float(row["fisher_adaptive_fraction"]), fraction):
                    fail(f"{env}: wrong Fisher interpolation fraction")
                if not close(
                    float(row["actor_base_damping"]), expected_actor_damping
                ):
                    fail(f"{env}: wrong actor base damping")
                if not close(float(row["critic_base_damping"]), 0.5):
                    fail(f"{env}: wrong critic base damping")
            else:
                if float(row.get("fisher_adaptive_block_damping", 1.0)) != 0.0:
                    fail(f"{env}: Fisher-adaptive damping unexpectedly enabled")
                actor_median = max(
                    float(row["actor_kernel_diag_median"]), 0.0
                )
                critic_median = max(
                    float(row["critic_kernel_diag_median"]), 0.0
                )
                expected_actor_damping = max(
                    args.actor_absolute,
                    args.relative * actor_median,
                    args.actor_from_critic * critic_median,
                )
                expected_critic_damping = max(
                    args.critic_absolute,
                    args.relative * critic_median,
                )
                if not close(
                    float(row["actor_effective_damping_median"]),
                    expected_actor_damping,
                ):
                    fail(
                        f"{env}: wrong actor effective damping; "
                        f"expected {expected_actor_damping}"
                    )
                if not close(
                    float(row["critic_effective_damping_median"]),
                    expected_critic_damping,
                ):
                    fail(
                        f"{env}: wrong critic effective damping; "
                        f"expected {expected_critic_damping}"
                    )

            for key in ("joint_solve_residual", "rhs_aligned_reduced_residual"):
                value = float(row.get(key, math.inf))
                if not math.isfinite(value) or value > args.max_residual:
                    fail(f"{env}: {key}={value}")
            rhs_projection_error = float(row.get(
                "rhs_aligned_rhs_projection_relative_error", math.inf
            ))
            if (
                not math.isfinite(rhs_projection_error)
                or rhs_projection_error > args.max_invariant_error
            ):
                fail(f"{env}: RHS projection error={rhs_projection_error}")
            primal_error = float(row.get(
                "rhs_aligned_primal_rhs_relative_error", math.inf
            ))
            if (
                not math.isfinite(primal_error)
                or primal_error > args.max_primal_error
            ):
                fail(f"{env}: primal RHS error={primal_error}")
            full_residual = float(row["rhs_aligned_full_residual_relative"])
            if not math.isfinite(full_residual):
                fail(f"{env}: non-finite discarded/full residual")
            weights = (
                float(row["rhs_aligned_actor_weight_mean"])
                + float(row["rhs_aligned_critic_weight_mean"])
            )
            if not close(weights, 1.0):
                fail(f"{env}: paired projection weights sum to {weights}")
            if float(row.get("causal_diagnostic_ran", 0.0)) != 0.0:
                fail(f"{env}: incompatible full-2B causal diagnostic ran")
            if float(row.get("rhs_aligned_direct_diagnostic_ran", 0.0)) == 1.0:
                cosine = float(row["rhs_aligned_direct_direction_cosine"])
                norm_ratio = float(row["rhs_aligned_direct_direction_norm_ratio"])
                if not math.isfinite(cosine) or not -1.000001 <= cosine <= 1.000001:
                    fail(f"{env}: invalid direct direction cosine={cosine}")
                if not math.isfinite(norm_ratio) or norm_ratio <= 0.0:
                    fail(f"{env}: invalid direct direction norm ratio={norm_ratio}")
                direct_rows.append(row)
        if not direct_rows:
            fail(f"{env}: no matched direct-direction diagnostics")

        for log_name in ("stderr", "stdout"):
            path = run / log_name
            if path.is_file():
                match = ERROR_RE.search(path.read_text(errors="replace"))
                if match:
                    fail(f"{env}: error marker {match.group(0)!r} in {log_name}")

        last = final_rows[-1]
        transitions = int(last["environment_transitions"])
        if transitions < args.min_transitions:
            fail(f"{env}: transitions={transitions}")
        diagnostic = direct_rows[-1]
        actor_damping_key = (
            "actor_base_damping"
            if args.damping_profile == "fisher_adaptive"
            else "actor_effective_damping_median"
        )
        critic_damping_key = (
            "critic_base_damping"
            if args.damping_profile == "fisher_adaptive"
            else "critic_effective_damping_median"
        )
        print(
            env,
            args.seed,
            transitions,
            float(last["eprewmean"]),
            float(last["entropy"]),
            float(last["behavior_kl_after_step"]),
            float(last["categorical_fisher_trace"]),
            float(last[actor_damping_key]),
            float(last[critic_damping_key]),
            float(last["joint_solve_residual"]),
            float(last["rhs_aligned_rhs_projection_relative_error"]),
            float(last["rhs_aligned_primal_rhs_relative_error"]),
            float(last["rhs_aligned_full_residual_relative"]),
            float(diagnostic["rhs_aligned_direct_direction_cosine"]),
            float(diagnostic["rhs_aligned_direct_direction_norm_ratio"]),
        )
    print("AUDIT=PASS")


if __name__ == "__main__":
    main()
