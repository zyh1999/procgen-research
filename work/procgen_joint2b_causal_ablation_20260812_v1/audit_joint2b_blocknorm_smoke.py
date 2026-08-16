#!/usr/bin/env python3
"""Audit the one-rollout MuJoCo-ported strict-2B structure smoke."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path


ERROR_RE = re.compile(
    r"OOM|out of memory|NaN|Traceback|CUDA (?:error|failure|out of memory)"
    r"|CUBLAS|CUDNN|Cholesky|AssertionError|I/O error",
    re.IGNORECASE,
)


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def finite(row: dict, key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"missing/non-numeric {key}: {exc}")
    if not math.isfinite(value):
        fail(f"non-finite {key}={value}")
    return value


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RUN_DIR")
    run = Path(sys.argv[1])
    if (run / "status").read_text().strip() != "PASS":
        fail("status is not PASS")
    if (run / "rc").read_text().strip() != "0":
        fail("rc is not zero")
    preflight = (run / "preflight").read_text()
    for marker in (
        "ROLLOUT=4096",
        "MINIBATCH=512",
        "EPOCHS=4",
        "JOINT_SYSTEM_ROWS=1024",
        "JOINT_MODE=full_joint_clean_all",
        "JOINT_RHS_MODE=paired_score_residual",
        "JOINT_RECONSTRUCTION_MODE=full_joint",
        "JOINT_BLOCK_NORMALIZATION=median_gradient_preserving",
        "DAMPING=0.03",
        "KL_RANGE=0.005,0.02",
        "MOMENTUM=0",
        "KACZMARZ=false",
        "LINEAR_SOLVE_DTYPE=float64",
    ):
        if marker not in preflight:
            fail(f"preflight missing {marker}")
    if not any(
        marker in preflight for marker in ("LR=0.05", "LR_INITIAL_MAX=0.05")
    ):
        fail("preflight missing LR=0.05 or LR_INITIAL_MAX=0.05")

    trace = run / "metric_trace.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line]
    if not rows:
        fail("no real metric rows")
    row = rows[-1]
    expected_strings = {
        "joint_block_normalization": "median_gradient_preserving",
        "joint_rhs_mode": "paired_score_residual",
        "joint_reconstruction_mode": "direct_Ht_alpha",
        "joint_critic_score_mode": "clean",
    }
    for key, expected in expected_strings.items():
        if str(row.get(key)) != expected:
            fail(f"{key}={row.get(key)!r}, expected {expected!r}")
    # Older trace schemas expose solve dtype only in preflight/stdout, while
    # newer schemas also carry it in each JSON row.  Preflight is already
    # mandatory above; when the row field exists, require it to agree.
    if (
        "joint_linear_solve_dtype" in row
        and str(row["joint_linear_solve_dtype"]) != "torch.float64"
    ):
        fail(
            "joint_linear_solve_dtype="
            f"{row['joint_linear_solve_dtype']!r}, expected 'torch.float64'"
        )
    if int(row.get("joint_system_rows", -1)) != 1024:
        fail(f"joint_system_rows={row.get('joint_system_rows')}")
    if int(row.get("joint_rhs_columns", -1)) != 1:
        fail(f"joint_rhs_columns={row.get('joint_rhs_columns')}")

    raw_actor = finite(row, "raw_actor_kernel_diag_median")
    raw_critic = finite(row, "raw_critic_kernel_diag_median")
    actor_scale = finite(row, "actor_block_normalization_scale")
    critic_scale = finite(row, "critic_block_normalization_scale")
    actor_median = finite(row, "actor_kernel_diag_median")
    critic_median = finite(row, "critic_kernel_diag_median")
    if min(raw_actor, raw_critic, actor_scale, critic_scale) <= 0.0:
        fail("raw medians and block scales must be positive")
    if not (0.1 <= actor_median <= 10.0):
        fail(f"normalized actor median outside ratio-safe range: {actor_median}")
    if not (0.95 <= critic_median <= 1.05):
        fail(f"normalized critic median is not approximately one: {critic_median}")
    if finite(row, "joint_solve_residual") > 1e-7:
        fail(f"solve residual too large: {row['joint_solve_residual']}")
    if finite(row, "cross_block_fro") <= 0.0:
        fail("full-cross structure was not exercised")

    errors = "\n".join(
        path.read_text(errors="replace")
        for path in (run / "stderr", run / "stdout")
        if path.exists()
    )
    match = ERROR_RE.search(errors)
    if match:
        fail(f"error signature found: {match.group(0)}")
    print(
        "AUDIT=PASS "
        f"rows={len(rows)} raw_actor={raw_actor:.6g} "
        f"raw_critic={raw_critic:.6g} actor_median={actor_median:.6g} "
        f"critic_median={critic_median:.6g}"
    )


if __name__ == "__main__":
    main()
