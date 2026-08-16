#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET_KL = {
    "C": None,
    "D": 0.0025,
    "E": 0.0025,
    "F": 0.0005,
}
ERROR_PATTERN = re.compile(
    r"out of memory|\bOOM\b|\bNaN\b|Infinity|Traceback|RuntimeError|"
    r"Cholesky failure",
    re.IGNORECASE,
)


def latest_trace(variant: str) -> Path:
    candidates = [
        path
        for path in (ROOT / "logs").rglob("aux_trace.jsonl")
        if f"V4_SMOKE_{variant}_" in str(path)
    ]
    if not candidates:
        raise AssertionError(f"missing aux trace for {variant}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def check_variant(variant: str) -> dict:
    status_path = ROOT / "status" / f"V4_SMOKE_{variant}_seed0.status"
    status = status_path.read_text().strip()
    if not status.startswith("COMPLETED"):
        raise AssertionError(f"{variant} status is not COMPLETED: {status}")

    trace_path = latest_trace(variant)
    rows = [
        json.loads(line)
        for line in trace_path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) < 2:
        raise AssertionError(f"{variant} produced only {len(rows)} aux cycles")

    target_kl = TARGET_KL[variant]
    summaries = []
    for row in rows:
        required = [
            "aux/critic_mse_before",
            "aux/critic_mse_after",
            "aux/actual_kl",
            "aux/predicted_kl",
            "aux/clip_scale",
            "aux/cholesky_success",
            "aux/actor_head_direction_max_abs",
            "aux/aux_critic_head_direction_max_abs",
            "aux/true_critic_head_direction_l2_norm",
            "aux/same_batch_true_head_mode",
            "aux/full_gradient_anchor_mode",
            "aux/true_head_aux_steps",
        ]
        missing = [key for key in required if key not in row]
        if missing:
            raise AssertionError(f"{variant} missing diagnostics: {missing}")
        for key in required[:5]:
            if not math.isfinite(float(row[key])):
                raise AssertionError(f"{variant} nonfinite {key}: {row[key]}")
        if float(row["aux/critic_mse_after"]) >= float(
            row["aux/critic_mse_before"]
        ):
            raise AssertionError(f"{variant} same-batch GGN did not descend")
        if float(row["aux/cholesky_success"]) != 1.0:
            raise AssertionError(f"{variant} Cholesky failed")
        if float(row["aux/actor_head_direction_max_abs"]) != 0.0:
            raise AssertionError(f"{variant} actor-head direction is nonzero")
        if float(row["aux/aux_critic_head_direction_max_abs"]) != 0.0:
            raise AssertionError(f"{variant} unused aux-head direction is nonzero")
        if float(row["aux/true_critic_head_direction_l2_norm"]) <= 0.0:
            raise AssertionError(f"{variant} true-head direction is zero")
        if float(row["aux/same_batch_true_head_mode"]) != 1.0:
            raise AssertionError(f"{variant} did not use same-batch mode")
        if float(row["aux/full_gradient_anchor_mode"]) != 0.0:
            raise AssertionError(f"{variant} unexpectedly used anchor mode")
        if float(row["aux/true_head_aux_steps"]) != 0.0:
            raise AssertionError(f"{variant} used forbidden post-GGN Adam steps")
        if any(key.startswith("aux/buffer_aux_") for key in row):
            raise AssertionError(f"{variant} logged the unused auxiliary head")
        if target_kl is not None:
            actual_kl = float(row["aux/actual_kl"])
            if actual_kl > max(2.0 * target_kl, target_kl + 5e-4):
                raise AssertionError(
                    f"{variant} realized KL {actual_kl} exceeds smoke tolerance"
                )
        summaries.append({
            "cycle": row["aux/cycle_policy_update"],
            "mse_before": row["aux/critic_mse_before"],
            "mse_after": row["aux/critic_mse_after"],
            "buffer_true_mse_before": row["aux/buffer_true_mse_before"],
            "buffer_true_mse_after": row["aux/buffer_true_mse_after"],
            "buffer_true_ev_before": row["aux/buffer_true_ev_before"],
            "buffer_true_ev_after": row["aux/buffer_true_ev_after"],
            "clip_scale": row["aux/clip_scale"],
            "predicted_kl": row["aux/predicted_kl"],
            "actual_kl": row["aux/actual_kl"],
            "effective_damping": row["aux/effective_damping"],
            "cholesky_retries": row["aux/cholesky_retries"],
        })

    run_log = ROOT / "run_logs" / f"V4_SMOKE_{variant}_seed0.stdout.log"
    hits = [
        line
        for line in run_log.read_text(errors="replace").splitlines()
        if ERROR_PATTERN.search(line)
    ]
    if hits:
        raise AssertionError(f"{variant} error scan: {hits[-3:]}")
    return {"status": status, "trace": str(trace_path), "cycles": summaries}


def main() -> None:
    result = {variant: check_variant(variant) for variant in "CDEF"}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
