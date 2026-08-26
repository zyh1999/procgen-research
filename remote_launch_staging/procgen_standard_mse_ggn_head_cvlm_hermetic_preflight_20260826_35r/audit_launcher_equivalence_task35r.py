#!/usr/bin/env python3
"""Prove Task35R changes deployment only and preserves Task34R command identity."""

import hashlib
import json
import re
import sys
from pathlib import Path


OLD_PREFLIGHT_SHA = "ca8443094a9827bb9141c532e5a5f230ba940d52aaec857d9edd0f5a1662bc74"
OLD_SCIENCE_SHA = "6dffce265e55f87fa5be848ec5bd9940fcecb6203f621a1454fb4f6a9c74caca"
NORMALIZED_PREFLIGHT_ARGS = [
    "$PY",
    "$PREFLIGHT",
    "$TRAINER",
    "$CONFIG",
    "$EVIDENCE/parameter_partition.json",
    "$TRAINER_SHA",
    "$CONFIG_SHA",
]
NORMALIZED_SCIENCE_ARGS = [
    "$PY",
    "-u",
    "$TRAINER",
    "--config",
    "$(basename $CONFIG)",
    "--env_name",
    "$ENV_NAME",
    "--seed",
    "0",
    "--device",
    "0",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def squashed(path):
    return re.sub(r"\s+", " ", re.sub(r"\\\n\s*", " ", path.read_text()))


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: audit OLD_PREFLIGHT OLD_SCIENCE NEW_PREFLIGHT OUTPUT")
    old_preflight, old_science, new_preflight, output = map(Path, sys.argv[1:])
    if sha(old_preflight) != OLD_PREFLIGHT_SHA:
        raise RuntimeError("Task34R preflight launcher hash changed")
    if sha(old_science) != OLD_SCIENCE_SHA:
        raise RuntimeError("Task34R scientific launcher hash changed")
    old_preflight_text = squashed(old_preflight)
    new_preflight_text = squashed(new_preflight)
    old_call = (
        'PROCGEN_ENV="$ENV_NAME" "$PY" "$PREFLIGHT" "$TRAINER" "$CONFIG" '
        '"$EVIDENCE/parameter_partition.json" "$TRAINER_SHA" "$CONFIG_SHA"'
    )
    new_call = (
        'env -u PYTHONPATH PROCGEN_ENV="$ENV_NAME" "$PY" "$PREFLIGHT" '
        '"$TRAINER" "$CONFIG" "$EVIDENCE/parameter_partition.json" '
        '"$TRAINER_SHA" "$CONFIG_SHA"'
    )
    if old_call not in old_preflight_text:
        raise RuntimeError("Task34R preflight invocation not found")
    if new_call not in new_preflight_text:
        raise RuntimeError("Task35R normalized preflight invocation not found")

    science_text = old_science.read_text()
    required = {
        "task": "TASK_ID=PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-6M-S0-20260825-34R",
        "method": "METHOD=DET_STANDARD_MSE_GGN_HEAD_CVLM_V1",
        "seed_device": '--seed 0 --device 0',
        "config": '--config "$(basename "$CONFIG")"',
        "environment": '--env_name "$ENV_NAME"',
    }
    missing = {key: value for key, value in required.items() if value not in science_text}
    if missing:
        raise RuntimeError(f"frozen scientific command identity missing: {missing}")

    payload = {
        "result": "TASK35R_LAUNCHER_NORMALIZED_COMMAND_EQUALITY_PASS",
        "old_preflight_sha256": sha(old_preflight),
        "old_science_sha256": sha(old_science),
        "new_preflight_sha256": sha(new_preflight),
        "normalized_preflight_arguments_old": NORMALIZED_PREFLIGHT_ARGS,
        "normalized_preflight_arguments_new": NORMALIZED_PREFLIGHT_ARGS,
        "normalized_scientific_arguments": NORMALIZED_SCIENCE_ARGS,
        "allowed_differences": [
            "bundle verification and extraction",
            "repository import root",
            "fresh non-overwriting preflight root",
            "deployment provenance",
        ],
        "scientific_identity_changed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("TASK35R_LAUNCHER_NORMALIZED_COMMAND_EQUALITY_PASS")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
