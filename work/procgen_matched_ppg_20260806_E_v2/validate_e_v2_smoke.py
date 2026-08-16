#!/usr/bin/env python3
"""Fail closed unless the E_v2 smoke demonstrates stable official PPG semantics."""

import csv
import json
import math
from pathlib import Path


def require_finite(label, value):
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is non-finite: {number}")
    return number


def main():
    log_root = Path("logs")
    progress_files = sorted(
        path for path in log_root.rglob("progress.csv")
        if "E_V2_SMOKE_" in str(path)
    )
    if not progress_files:
        raise RuntimeError("E_v2 smoke progress.csv is missing")
    progress_path = progress_files[-1]
    with progress_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("E_v2 smoke has no progress rows")

    latest = rows[-1]
    transitions = int(float(latest["misc/total_timesteps"]))
    if transitions < 262_144:
        raise RuntimeError(f"smoke too short: {transitions} transitions")
    kls = [require_finite("kl", row["kl"]) for row in rows]
    if max(kls) > 0.15 or max(kls[-10:]) > 0.08:
        raise RuntimeError(
            f"policy KL gate failed: max={max(kls):.6g}, "
            f"recent_max={max(kls[-10:]):.6g}"
        )
    for key in ("entropy", "policy_critic_mse", "ratio_max", "ratio_min"):
        require_finite(key, latest[key])
    if float(latest["critic_forbidden_grad_max_abs"]) != 0.0:
        raise RuntimeError("policy phase reached the auxiliary value head")

    aux_path = progress_path.with_name("aux_trace.jsonl")
    if not aux_path.exists():
        raise RuntimeError("E_v2 smoke aux_trace.jsonl is missing")
    traces = [
        json.loads(line) for line in aux_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(traces) < 2:
        raise RuntimeError(f"expected two auxiliary cycles, got {len(traces)}")
    for trace in traces[-2:]:
        if int(trace["aux/buffer_rows"]) != 131_072:
            raise RuntimeError(
                f"wrong full-buffer rows: {trace['aux/buffer_rows']}"
            )
        for key in (
            "aux/buffer_policy_kl_after",
            "aux/buffer_aux_mse_after",
            "aux/buffer_true_mse_after",
            "aux/buffer_aux_ev_after",
            "aux/buffer_true_ev_after",
        ):
            require_finite(key, trace[key])
        if float(trace["aux/buffer_policy_kl_after"]) > 0.05:
            raise RuntimeError(
                "auxiliary policy clone KL exceeded the smoke gate"
            )
    print(
        "E_v2 smoke gate: PASS "
        f"transitions={transitions} latest_kl={kls[-1]:.6g} "
        f"max_kl={max(kls):.6g} aux_cycles={len(traces)}"
    )


if __name__ == "__main__":
    main()
