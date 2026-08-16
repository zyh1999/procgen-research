#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


ENVIRONMENTS = (
    "bigfish-easy-0-10",
    "caveflyer-easy-0-10",
)
FULL_GATE_ENVIRONMENTS = (
    "bigfish-easy-0-10",
    "bossfight-easy-0-10",
    "caveflyer-easy-0-10",
    "coinrun-easy-0-10",
)
HARD_ERRORS = (
    "out of memory",
    "nan",
    "nonfinite",
    "traceback",
    "cuda error",
    "cholesky",
    "singular",
    "assertionerror",
    "i/o error",
)


def read_preflight(path):
    result = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def fail(message):
    raise AssertionError(message)


def close(actual, expected, *, rtol=2e-4, atol=2e-6):
    return math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--trainer-sha", required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument(
        "--critic-score-mode",
        choices=("clean", "rademacher", "gaussian_unit"),
        default="clean",
    )
    parser.add_argument(
        "--critic-rhs-mode",
        choices=(
            "paired_score_residual",
            "paper_weight_residual_reconstruct",
        ),
        default="paired_score_residual",
    )
    parser.add_argument(
        "--reconstruction-mode",
        choices=(
            "full_joint",
            "paper_selective",
            "paper_full_joint_columns",
        ),
        default="full_joint",
    )
    parser.add_argument("--normalized-damping", type=float, default=0.3)
    parser.add_argument("--row-floor-fraction", type=float, default=0.0)
    parser.add_argument("--actor-fisher-floor", type=float, default=0.0)
    parser.add_argument(
        "--actor-kernel-anchor-fraction", type=float, default=0.0
    )
    parser.add_argument("--entropy-rhs-target", type=float, default=0.0)
    parser.add_argument("--entropy-rhs-gain", type=float, default=0.0)
    parser.add_argument(
        "--entropy-rhs-integral-gain", type=float, default=0.0
    )
    parser.add_argument("--entropy-rhs-max-coef", type=float, default=0.0)
    parser.add_argument("--behavior-kl-guard", action="store_true")
    parser.add_argument("--behavior-kl-guard-upper", type=float, default=0.02)
    parser.add_argument(
        "--behavior-kl-guard-backoff", type=float, default=0.5
    )
    parser.add_argument(
        "--behavior-kl-guard-mode",
        choices=(
            "rollback",
            "sqrt_backtrack_accept",
            "sqrt_backtrack_accept_rollout_lr",
            "sqrt_backtrack_accept_progressive_rollout_lr",
        ),
        default="rollback",
    )
    parser.add_argument(
        "--behavior-kl-guard-safety", type=float, default=0.9
    )
    parser.add_argument(
        "--behavior-kl-guard-max-backtracks", type=int, default=8
    )
    parser.add_argument(
        "--require-behavior-kl-guard-activity", action="store_true"
    )
    parser.add_argument("--min-transitions", type=int, default=100_000)
    parser.add_argument("--full-gate", action="store_true")
    parser.add_argument("--require-row-cap-activity", action="store_true")
    parser.add_argument(
        "--require-anchor-activity",
        action="store_true",
        help=(
            "require Fisher and raw-kernel anchors to have bound at least "
            "once; omit for healthy-policy gates where no trigger occurs"
        ),
    )
    args = parser.parse_args()

    paper_weight_rhs = (
        args.critic_rhs_mode == "paper_weight_residual_reconstruct"
    )
    paper_selective = args.reconstruction_mode == "paper_selective"
    paper_full_joint_columns = (
        args.reconstruction_mode == "paper_full_joint_columns"
    )
    if paper_weight_rhs != (
        paper_selective or paper_full_joint_columns
    ):
        fail(
            "paper_weight_residual_reconstruct requires paper_selective or "
            "paper_full_joint_columns, and either reconstruction requires "
            "that RHS mode"
        )
    if paper_weight_rhs and args.critic_score_mode != "clean":
        fail("paper reconstruction currently requires deterministic clean scores")

    rows = []
    cap_was_active = False
    actor_anchor_was_active = False
    kernel_anchor_was_active = False
    entropy_rhs_was_active = False
    behavior_kl_guard_was_active = False
    environments = FULL_GATE_ENVIRONMENTS if args.full_gate else ENVIRONMENTS
    for env in environments:
        run = args.root / env / "seed0"
        if not run.is_dir():
            fail(f"missing run directory: {run}")
        status = (run / "status").read_text().strip()
        rc = int((run / "rc").read_text().strip())
        if status != "PASS" or rc != 0:
            fail(f"{env}: status={status!r} rc={rc}")

        preflight = read_preflight(run / "preflight")
        expected_preflight = {
            "ENV_NAME": env,
            "SEED": "0",
            "TRAINER_SHA256": args.trainer_sha,
            "CONFIG_SHA256": args.config_sha,
            "ROLLOUT": "4096",
            "MINIBATCH": "512",
            "EPOCHS": "4",
            "JOINT_SYSTEM_ROWS": "1024",
            "JOINT_MODE": f"full_joint_{args.critic_score_mode}_all",
            "JOINT_RHS_MODE": args.critic_rhs_mode,
            "JOINT_RECONSTRUCTION_MODE": args.reconstruction_mode,
            "JOINT_BLOCK_NORMALIZATION": "none",
            "JOINT_DAMPING_MODE": "correlation_relative",
            "NORMALIZED_DAMPING": f"{args.normalized_damping:.2f}",
            "CORRELATION_NORMALIZED_SOLVE": "true",
            "MOMENTUM": "0",
            "KACZMARZ": "false",
        }
        if args.critic_score_mode != "clean":
            expected_preflight["JOINT_CRITIC_SCORE_MODE"] = (
                args.critic_score_mode
            )
        if args.row_floor_fraction > 0.0:
            expected_preflight.update({
                "CORRELATION_ROW_FLOOR_MODE": (
                    "per_block_median_relative"
                ),
                "CORRELATION_ROW_FLOOR_FRACTION": (
                    f"{args.row_floor_fraction:.2f}"
                ),
            })
        else:
            expected_preflight["CORRELATION_DAMPING_USE_RAW_K_DIAG"] = (
                "true"
            )
        if args.actor_fisher_floor > 0.0:
            expected_preflight.update({
                "CORRELATION_ACTOR_FISHER_FLOOR": (
                    f"{args.actor_fisher_floor:.2f}"
                ),
            })
        if args.actor_kernel_anchor_fraction > 0.0:
            expected_preflight.update({
                "CORRELATION_ACTOR_KERNEL_ANCHOR_MODE": (
                    "running_highwater_relative"
                ),
                "CORRELATION_ACTOR_KERNEL_ANCHOR_FRACTION": (
                    f"{args.actor_kernel_anchor_fraction:.3f}"
                ),
            })
        if args.entropy_rhs_gain > 0.0:
            expected_preflight.update({
                "ENTROPY_RHS_TARGET": f"{args.entropy_rhs_target:.2f}",
                "ENTROPY_RHS_GAIN": f"{args.entropy_rhs_gain:.2f}",
                "ENTROPY_RHS_MAX_COEF": (
                    f"{args.entropy_rhs_max_coef:.2f}"
                ),
            })
        if args.entropy_rhs_integral_gain > 0.0:
            expected_preflight["ENTROPY_RHS_INTEGRAL_GAIN"] = (
                f"{args.entropy_rhs_integral_gain:.2f}"
            )
        if args.behavior_kl_guard:
            expected_preflight.update({
                "BEHAVIOR_KL_GUARD_ENABLED": "true",
                "BEHAVIOR_KL_GUARD_UPPER": (
                    str(args.behavior_kl_guard_upper)
                ),
                "BEHAVIOR_KL_GUARD_BACKOFF": (
                    f"{args.behavior_kl_guard_backoff:.2f}"
                ),
            })
            if args.behavior_kl_guard_mode != "rollback":
                expected_preflight.update({
                    "BEHAVIOR_KL_GUARD_MODE": (
                        args.behavior_kl_guard_mode
                    ),
                    "BEHAVIOR_KL_GUARD_SAFETY": (
                        f"{args.behavior_kl_guard_safety:.2f}"
                    ),
                    "BEHAVIOR_KL_GUARD_MAX_BACKTRACKS": str(
                        args.behavior_kl_guard_max_backtracks
                    ),
                    "BEHAVIOR_KL_GUARD_REFERENCE": (
                        "full_4096_rollout_chunked_512"
                    ),
                })
                if (
                    args.behavior_kl_guard_mode
                    in {
                        "sqrt_backtrack_accept_rollout_lr",
                        "sqrt_backtrack_accept_progressive_rollout_lr",
                    }
                ):
                    expected_preflight[
                        "BEHAVIOR_KL_GUARD_PERSIST_LR_AFTER_BACKTRACK"
                    ] = "false"
                if (
                    args.behavior_kl_guard_mode
                    == "sqrt_backtrack_accept_progressive_rollout_lr"
                ):
                    expected_preflight.update({
                        "BEHAVIOR_KL_GUARD_PROGRESSIVE_BUDGET": "true",
                        "BEHAVIOR_KL_GUARD_BUDGET_STEPS": "32",
                    })
        for key, expected in expected_preflight.items():
            actual = preflight.get(key)
            if actual != expected:
                fail(f"{env}: preflight {key}={actual!r}, expected {expected!r}")

        trace = []
        for line in (run / "metric_trace.jsonl").read_text().splitlines():
            if line.strip():
                trace.append(json.loads(line))
        if not trace:
            fail(f"{env}: empty metric trace")
        if args.behavior_kl_guard:
            rejection_count = 0
            total_backtracks = 0
            rollback_only_count = 0
            for index, row in enumerate(trace, start=1):
                if float(row.get("behavior_kl_guard_enabled", 0.0)) != 1.0:
                    fail(f"{env}: behavior-KL guard not enabled")
                if not close(
                    row.get("behavior_kl_guard_upper", math.nan),
                    args.behavior_kl_guard_upper,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    fail(f"{env}: wrong behavior-KL guard upper bound")
                if not close(
                    row.get("behavior_kl_guard_backoff", math.nan),
                    args.behavior_kl_guard_backoff,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    fail(f"{env}: wrong behavior-KL guard backoff")
                actual_mode = row.get(
                    "behavior_kl_guard_mode", "rollback"
                )
                if actual_mode != args.behavior_kl_guard_mode:
                    fail(f"{env}: wrong behavior-KL guard mode")
                if (
                    args.behavior_kl_guard_mode
                    in {
                        "sqrt_backtrack_accept_rollout_lr",
                        "sqrt_backtrack_accept_progressive_rollout_lr",
                    }
                    and float(row.get(
                        "behavior_kl_guard_persist_lr_after_backtrack",
                        math.nan,
                    )) != 0.0
                ):
                    fail(
                        f"{env}: line-search scale incorrectly persists "
                        "into the proposal LR"
                    )
                if (
                    args.behavior_kl_guard_mode
                    == "sqrt_backtrack_accept_progressive_rollout_lr"
                ):
                    if float(row.get(
                        "behavior_kl_guard_progressive_budget", 0.0
                    )) != 1.0:
                        fail(f"{env}: progressive KL budget not enabled")
                    budget_fraction = float(row.get(
                        "behavior_kl_guard_budget_fraction", math.nan
                    ))
                    step_upper = float(row.get(
                        "behavior_kl_guard_step_upper", math.nan
                    ))
                    if not (
                        0.0 < budget_fraction <= 1.0
                        and close(
                            step_upper,
                            args.behavior_kl_guard_upper * budget_fraction,
                            rtol=1e-6,
                            atol=1e-12,
                        )
                    ):
                        fail(f"{env}: invalid progressive KL step budget")
                if not close(
                    row.get("behavior_kl_guard_safety", 0.9),
                    args.behavior_kl_guard_safety,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    fail(f"{env}: wrong behavior-KL guard safety")
                if int(row.get(
                    "behavior_kl_guard_max_backtracks", 8
                )) != args.behavior_kl_guard_max_backtracks:
                    fail(f"{env}: wrong maximum KL backtracks")
                rejected = bool(
                    float(row.get("behavior_kl_guard_rejected", 0.0))
                )
                proposal_finite = bool(
                    float(
                        row.get(
                            "behavior_kl_guard_proposal_finite", 0.0
                        )
                    )
                )
                proposed_kl = float(row.get(
                    "behavior_kl_guard_proposed_behavior_kl", math.nan
                ))
                accepted_kl = float(
                    row.get(
                        "behavior_kl_guard_rollout_kl_after_step",
                        row.get("behavior_kl_after_step", math.nan),
                    )
                )
                guard_limit = (
                    step_upper
                    if args.behavior_kl_guard_mode
                    == "sqrt_backtrack_accept_progressive_rollout_lr"
                    else args.behavior_kl_guard_upper
                )
                if rejected:
                    rejection_count += 1
                    if proposal_finite and not (
                        proposed_kl > guard_limit
                    ):
                        fail(f"{env}: rejected an in-bound finite proposal")
                elif (
                    not proposal_finite
                    or not math.isfinite(proposed_kl)
                    or proposed_kl > guard_limit + 1e-7
                ):
                    fail(f"{env}: accepted an invalid KL proposal")
                if not math.isfinite(accepted_kl):
                    fail(f"{env}: nonfinite applied behavior KL")
                if (
                    args.behavior_kl_guard_mode == "rollback"
                    and rejected
                ):
                    # The legacy guard measured a different shuffled 512-row
                    # subset at every update.  Rolling back can restore an
                    # already-out-of-bound subset but cannot move it below
                    # the threshold.  Require non-worsening here; the newer
                    # full-rollout reference enforces the actual hard bound.
                    if accepted_kl > proposed_kl + 1e-7:
                        fail(f"{env}: rollback worsened behavior KL")
                elif accepted_kl > guard_limit + 1e-7:
                    fail(f"{env}: applied behavior KL exceeds guard")
                applied_scale = float(row.get(
                    "behavior_kl_guard_applied_scale",
                    0.0 if rejected else 1.0,
                ))
                backtracks = int(row.get(
                    "behavior_kl_guard_backtracks", 0
                ))
                rollback_only = bool(float(row.get(
                    "behavior_kl_guard_rollback_only",
                    1.0 if rejected else 0.0,
                )))
                if rejected:
                    if args.behavior_kl_guard_mode == "rollback":
                        if applied_scale != 0.0 or not rollback_only:
                            fail(f"{env}: rollback mode applied a step")
                    else:
                        if not 0.0 <= applied_scale < 1.0:
                            fail(f"{env}: invalid backtracked scale")
                        if backtracks < 1:
                            fail(f"{env}: trigger did not backtrack")
                        if applied_scale > 0.0 and rollback_only:
                            fail(f"{env}: accepted backtrack marked rollback")
                        if applied_scale == 0.0 and not rollback_only:
                            fail(f"{env}: zero scale not marked rollback")
                elif (
                    not close(applied_scale, 1.0)
                    or backtracks != 0
                    or rollback_only
                ):
                    fail(f"{env}: in-bound proposal was modified")
                total_backtracks += backtracks
                rollback_only_count += int(rollback_only)
                lr_before = float(row.get(
                    "behavior_kl_guard_lr_before", math.nan
                ))
                lr_applied = float(row.get(
                    "behavior_kl_guard_lr_applied",
                    lr_before * applied_scale,
                ))
                lr_after = float(row.get(
                    "behavior_kl_guard_lr_after", math.nan
                ))
                persist_line_search_lr = (
                    args.behavior_kl_guard_mode
                    not in {
                        "sqrt_backtrack_accept_rollout_lr",
                        "sqrt_backtrack_accept_progressive_rollout_lr",
                    }
                )
                expected_lr_after = (
                    max(
                        1e-4,
                        lr_before * (
                            applied_scale
                            if applied_scale > 0.0
                            else args.behavior_kl_guard_backoff
                        ),
                    )
                    if rejected and persist_line_search_lr
                    else lr_before
                )
                if not close(
                    lr_applied,
                    lr_before * applied_scale,
                    rtol=2e-5,
                    atol=2e-8,
                ):
                    fail(f"{env}: wrong applied KL-guard LR")
                if not close(
                    lr_after, expected_lr_after, rtol=2e-5, atol=2e-8
                ):
                    fail(f"{env}: wrong KL-guard LR transition")
                if int(row.get("behavior_kl_guard_attempts", -1)) != index:
                    fail(f"{env}: wrong KL-guard attempt count")
                if int(
                    row.get("behavior_kl_guard_rejections", -1)
                ) != rejection_count:
                    fail(f"{env}: wrong KL-guard rejection count")
                if int(row.get(
                    "behavior_kl_guard_total_backtracks", total_backtracks
                )) != total_backtracks:
                    fail(f"{env}: wrong cumulative KL backtracks")
                if int(row.get(
                    "behavior_kl_guard_rollback_only_count",
                    rollback_only_count,
                )) != rollback_only_count:
                    fail(f"{env}: wrong rollback-only count")
                expected_rate = rejection_count / index
                if not close(
                    row.get("behavior_kl_guard_rejection_rate", math.nan),
                    expected_rate,
                    rtol=2e-5,
                    atol=2e-8,
                ):
                    fail(f"{env}: wrong KL-guard rejection rate")
            behavior_kl_guard_was_active = (
                behavior_kl_guard_was_active or rejection_count > 0
            )
        final_rows = [
            row for row in trace
            if int(row.get("optimizer_epoch", -1)) == 3
            and int(row.get("minibatch_index", -1)) == 7
        ]
        if not final_rows:
            fail(f"{env}: no rollout-final minibatch rows")
        last = final_rows[-1]
        transitions = int(last.get("environment_transitions", -1))
        if transitions < args.min_transitions:
            fail(f"{env}: only {transitions} transitions")

        previous_kernel_highwater = 0.0
        previous_entropy_integral = 0.0
        for row in final_rows:
            if int(row.get("joint_system_rows", -1)) != 1024:
                fail(f"{env}: joint_system_rows changed")
            expected_kernel_mode = (
                f"full_joint_{args.critic_score_mode}_all"
            )
            if row.get("joint_kernel_mode") != expected_kernel_mode:
                fail(f"{env}: wrong joint kernel mode")
            if row.get("joint_rhs_mode") != args.critic_rhs_mode:
                fail(f"{env}: wrong joint RHS mode")
            expected_reconstruction = (
                "paper_selective_actor_rows_critic_residual_weighted_rows"
                if paper_selective
                else (
                    "paper_full_joint_columns_all_rows"
                    if paper_full_joint_columns
                    else "direct_Ht_alpha"
                )
            )
            if row.get("joint_reconstruction_mode") != expected_reconstruction:
                fail(f"{env}: wrong joint reconstruction mode")
            expected_rhs_columns = 2 if paper_weight_rhs else 1
            if int(row.get("joint_rhs_columns", -1)) != expected_rhs_columns:
                fail(f"{env}: wrong number of joint RHS columns")
            component_residual = float(
                row.get("joint_component_solve_residual", math.inf)
            )
            if paper_weight_rhs:
                if (
                    not math.isfinite(component_residual)
                    or component_residual > 1e-7
                ):
                    fail(f"{env}: component solve residual={component_residual}")
                reconstruction_weight = float(
                    row.get("critic_reconstruction_weight_l2", math.nan)
                )
                if (
                    not math.isfinite(reconstruction_weight)
                    or reconstruction_weight < 0.0
                ):
                    fail(f"{env}: invalid critic reconstruction weight")
            elif not close(component_residual, 0.0, rtol=0.0, atol=1e-12):
                fail(f"{env}: unexpected component solve residual")
            noise_mean = float(row.get("critic_score_noise_mean", math.nan))
            noise_std = float(row.get("critic_score_noise_std", math.nan))
            noise_second = float(
                row.get("critic_score_noise_second_moment", math.nan)
            )
            noise_min = float(row.get("critic_score_noise_min", math.nan))
            noise_max = float(row.get("critic_score_noise_max", math.nan))
            noise_ess = float(row.get("critic_noise_ess", math.nan))
            noise_values = (
                noise_mean,
                noise_std,
                noise_second,
                noise_min,
                noise_max,
                noise_ess,
            )
            if not all(math.isfinite(value) for value in noise_values):
                fail(f"{env}: nonfinite critic score-noise diagnostic")
            if args.critic_score_mode == "clean":
                if not (
                    close(noise_mean, 1.0)
                    and close(noise_std, 0.0)
                    and close(noise_second, 1.0)
                    and close(noise_min, 1.0)
                    and close(noise_max, 1.0)
                    and close(noise_ess, 512.0)
                ):
                    fail(f"{env}: clean critic score is not deterministic")
            elif args.critic_score_mode == "rademacher":
                if not (
                    abs(noise_mean) <= 0.2
                    and 0.8 <= noise_std <= 1.2
                    and close(noise_second, 1.0, rtol=0.0, atol=1e-6)
                    and close(noise_min, -1.0, rtol=0.0, atol=1e-6)
                    and close(noise_max, 1.0, rtol=0.0, atol=1e-6)
                    and close(noise_ess, 512.0, rtol=0.0, atol=1e-4)
                ):
                    fail(f"{env}: invalid Rademacher critic score draw")
            else:
                # With 512 independent N(0,1) draws, these deliberately wide
                # bounds reject a missing/degenerate sampler without turning
                # a statistically valid tail draw into a flaky audit.
                if not (
                    abs(noise_mean) <= 0.25
                    and 0.55 <= noise_std <= 1.45
                    and 0.5 <= noise_second <= 1.6
                    and noise_min < -1.5
                    and noise_max > 1.5
                    and 64.0 <= noise_ess <= 384.0
                ):
                    fail(f"{env}: invalid Gaussian critic score draw")
            if row.get("joint_block_normalization") != "none":
                fail(f"{env}: block normalization is not none")
            if row.get("joint_damping_mode") != "correlation_relative":
                fail(f"{env}: wrong damping mode")
            if float(row.get("correlation_normalized_solve", 0.0)) != 1.0:
                fail(f"{env}: did not solve the normalized S K S system")
            if not close(row["base_damping_value"], args.normalized_damping):
                fail(f"{env}: wrong normalized damping")
            if not close(
                row.get("joint_correlation_row_floor_fraction", 0.0),
                args.row_floor_fraction,
                rtol=0.0,
                atol=1e-12,
            ):
                fail(f"{env}: wrong correlation row floor fraction")
            if not close(
                row.get("joint_correlation_actor_fisher_floor", 0.0),
                args.actor_fisher_floor,
                rtol=0.0,
                atol=1e-12,
            ):
                fail(f"{env}: wrong actor categorical-Fisher floor")
            actor_anchor_scale = float(
                row.get("correlation_actor_fisher_scale", 1.0)
            )
            if args.actor_fisher_floor > 0.0:
                expected_anchor_scale = math.sqrt(min(
                    1.0,
                    max(
                        0.0,
                        float(row["categorical_fisher_trace"])
                        / args.actor_fisher_floor,
                    ),
                ))
                if not close(
                    actor_anchor_scale,
                    expected_anchor_scale,
                    rtol=2e-4,
                    atol=2e-6,
                ):
                    fail(f"{env}: wrong actor Fisher-anchor scale")
                actor_anchor_was_active = (
                    actor_anchor_was_active
                    or actor_anchor_scale < 1.0 - 1e-6
                )
            elif not close(actor_anchor_scale, 1.0):
                fail(f"{env}: actor Fisher anchor unexpectedly active")
            if not close(
                row.get(
                    "joint_correlation_actor_kernel_anchor_fraction",
                    0.0,
                ),
                args.actor_kernel_anchor_fraction,
                rtol=0.0,
                atol=1e-12,
            ):
                fail(f"{env}: wrong actor-kernel anchor fraction")
            kernel_highwater = float(
                row.get("correlation_actor_kernel_highwater", 0.0)
            )
            kernel_anchor_floor = float(
                row.get("correlation_actor_kernel_anchor_floor", 0.0)
            )
            kernel_anchor_cap = float(
                row.get(
                    "correlation_actor_kernel_anchor_capped_fraction",
                    0.0,
                )
            )
            if args.actor_kernel_anchor_fraction > 0.0:
                if kernel_highwater + 1e-12 < previous_kernel_highwater:
                    fail(f"{env}: actor-kernel highwater decreased")
                if kernel_highwater + 1e-12 < float(
                    row["raw_actor_kernel_diag_median"]
                ):
                    fail(f"{env}: highwater below current actor median")
                expected_kernel_floor = max(
                    1e-12,
                    kernel_highwater * args.actor_kernel_anchor_fraction,
                )
                if not close(
                    kernel_anchor_floor,
                    expected_kernel_floor,
                    rtol=2e-4,
                    atol=2e-8,
                ):
                    fail(f"{env}: wrong actor-kernel anchor floor")
                if not 0.0 <= kernel_anchor_cap <= 1.0:
                    fail(f"{env}: invalid kernel-anchor capped fraction")
                kernel_anchor_was_active = (
                    kernel_anchor_was_active
                    or (
                        kernel_anchor_cap > 0.0
                        and kernel_anchor_floor
                        > float(row["correlation_actor_row_floor"])
                        * (1.0 + 1e-6)
                    )
                )
                previous_kernel_highwater = kernel_highwater
            elif kernel_anchor_cap != 0.0:
                fail(f"{env}: actor-kernel anchor unexpectedly active")
            entropy_rhs_target = float(
                row.get("entropy_rhs_target", 0.0)
            )
            entropy_rhs_gain = float(row.get("entropy_rhs_gain", 0.0))
            entropy_rhs_integral_gain = float(
                row.get("entropy_rhs_integral_gain", 0.0)
            )
            entropy_rhs_max_coef = float(
                row.get("entropy_rhs_max_coef", 0.0)
            )
            entropy_rhs_deficit = float(
                row.get("entropy_rhs_deficit", 0.0)
            )
            entropy_rhs_coef = float(row.get("entropy_rhs_coef", 0.0))
            entropy_rhs_l2 = float(row.get("entropy_rhs_l2", 0.0))
            entropy_rhs_integral_state = float(
                row.get("entropy_rhs_integral_state", 0.0)
            )
            entropy_rhs_rollout_entropy = float(
                row.get("entropy_rhs_rollout_entropy", math.nan)
            )
            if not close(
                entropy_rhs_target,
                args.entropy_rhs_target,
                rtol=0.0,
                atol=1e-12,
            ):
                fail(f"{env}: wrong entropy RHS target")
            if not close(
                entropy_rhs_gain,
                args.entropy_rhs_gain,
                rtol=0.0,
                atol=1e-12,
            ):
                fail(f"{env}: wrong entropy RHS gain")
            if not close(
                entropy_rhs_integral_gain,
                args.entropy_rhs_integral_gain,
                rtol=0.0,
                atol=1e-12,
            ):
                fail(f"{env}: wrong entropy RHS integral gain")
            if not close(
                entropy_rhs_max_coef,
                args.entropy_rhs_max_coef,
                rtol=0.0,
                atol=1e-12,
            ):
                fail(f"{env}: wrong entropy RHS max coefficient")
            expected_entropy_deficit = max(
                0.0,
                args.entropy_rhs_target - float(row["entropy"]),
            )
            if args.entropy_rhs_integral_gain > 0.0:
                if not math.isfinite(entropy_rhs_rollout_entropy):
                    fail(f"{env}: nonfinite rollout entropy")
                expected_integral = min(
                    args.entropy_rhs_max_coef,
                    max(
                        0.0,
                        previous_entropy_integral
                        + args.entropy_rhs_integral_gain
                        * (
                            args.entropy_rhs_target
                            - entropy_rhs_rollout_entropy
                        ),
                    ),
                )
                if not close(
                    entropy_rhs_integral_state,
                    expected_integral,
                    rtol=2e-4,
                    atol=2e-6,
                ):
                    fail(f"{env}: wrong entropy RHS integral recurrence")
                previous_entropy_integral = entropy_rhs_integral_state
            elif not close(entropy_rhs_integral_state, 0.0):
                fail(f"{env}: entropy integral unexpectedly active")
            expected_entropy_coef = min(
                args.entropy_rhs_max_coef,
                expected_entropy_deficit * args.entropy_rhs_gain
                + entropy_rhs_integral_state,
            )
            if not close(
                entropy_rhs_deficit,
                expected_entropy_deficit,
                rtol=2e-4,
                atol=2e-6,
            ):
                fail(f"{env}: wrong entropy RHS deficit")
            if not close(
                entropy_rhs_coef,
                expected_entropy_coef,
                rtol=2e-4,
                atol=2e-6,
            ):
                fail(f"{env}: wrong entropy RHS coefficient")
            if entropy_rhs_l2 < 0.0 or not math.isfinite(entropy_rhs_l2):
                fail(f"{env}: invalid entropy RHS norm")
            entropy_rhs_was_active = (
                entropy_rhs_was_active or entropy_rhs_coef > 0.0
            )
            if float(row.get("block_imbalance_guard_enabled", 1.0)) != 0.0:
                fail(f"{env}: block guard unexpectedly enabled")
            if float(row.get("schur_guard_enabled", 1.0)) != 0.0:
                fail(f"{env}: Schur guard unexpectedly enabled")
            if args.row_floor_fraction > 0.0:
                actor_expected = args.normalized_damping * float(
                    row["correlation_actor_effective_diag_median"]
                )
                critic_expected = args.normalized_damping * float(
                    row["correlation_critic_effective_diag_median"]
                )
                actor_cap = float(
                    row.get("correlation_actor_capped_fraction", 0.0)
                )
                critic_cap = float(
                    row.get("correlation_critic_capped_fraction", 0.0)
                )
                if not 0.0 <= actor_cap <= 1.0:
                    fail(f"{env}: invalid actor capped fraction")
                if not 0.0 <= critic_cap <= 1.0:
                    fail(f"{env}: invalid critic capped fraction")
                cap_was_active = cap_was_active or actor_cap > 0.0
                actor_floor = float(row["correlation_actor_row_floor"])
                actor_scale_max = float(
                    row["correlation_actor_row_scale_max"]
                )
                if actor_scale_max > actor_floor ** -0.5 * (1.0 + 1e-5):
                    fail(f"{env}: actor row scale exceeds configured cap")
            else:
                actor_expected = (
                    args.normalized_damping
                    * float(row["raw_actor_kernel_diag_median"])
                )
                critic_expected = (
                    args.normalized_damping
                    * float(row["raw_critic_kernel_diag_median"])
                )
            if not close(row["actor_effective_damping_median"], actor_expected):
                fail(f"{env}: actor damping is not lambda*raw Kii")
            if not close(row["critic_effective_damping_median"], critic_expected):
                fail(f"{env}: critic damping is not lambda*raw Kii")
            residual = float(row.get("joint_solve_residual", math.inf))
            if not math.isfinite(residual) or residual > 1e-7:
                fail(f"{env}: solve residual={residual}")

        if min(float(row["effective_damping_min"]) for row in final_rows) >= 0.5:
            fail(f"{env}: no evidence that the old absolute 0.5 floor is absent")

        for filename in ("stdout", "stderr"):
            text = (run / filename).read_text(errors="replace").lower()
            for marker in HARD_ERRORS:
                if marker in text:
                    fail(f"{env}: hard error marker {marker!r} in {filename}")
        rows.append((env, transitions, last["eprewmean"], last["entropy"], last["behavior_kl_after_step"]))

    if args.require_row_cap_activity and not cap_was_active:
        fail("row-cap gate completed without ever exercising the actor cap")
    if (
        args.require_anchor_activity
        and args.actor_fisher_floor > 0.0
        and not actor_anchor_was_active
    ):
        fail("Fisher-anchor gate completed without exercising the anchor")
    if (
        args.require_anchor_activity
        and
        args.actor_kernel_anchor_fraction > 0.0
        and not kernel_anchor_was_active
    ):
        fail("kernel-anchor gate completed without exercising the anchor")
    if args.entropy_rhs_gain > 0.0 and not entropy_rhs_was_active:
        fail("entropy-RHS gate completed without exercising the controller")
    if (
        args.require_behavior_kl_guard_activity
        and not behavior_kl_guard_was_active
    ):
        fail("behavior-KL guard completed without rejecting any proposal")

    print("environment transitions reward entropy behavior_kl")
    for row in rows:
        print(*row)
    print("AUDIT=PASS")


if __name__ == "__main__":
    main()
