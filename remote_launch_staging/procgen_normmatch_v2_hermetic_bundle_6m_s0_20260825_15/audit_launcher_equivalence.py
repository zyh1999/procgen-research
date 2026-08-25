#!/usr/bin/env python3
"""Reject any scientific-command drift in the deployment-only launchers."""
import hashlib
import json
import re
import sys
from pathlib import Path

original, science, preflight = map(Path, sys.argv[1:4])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_line(text):
    matches = [line.strip() for line in text.splitlines() if line.strip().startswith("CMD=(")]
    if len(matches) != 1:
        raise RuntimeError(f"expected one scientific CMD line, got {matches}")
    return matches[0]


original_text = original.read_text()
science_text = science.read_text()
preflight_text = preflight.read_text()
if sha(original) != "85e12886ce5cf81fd98647aa5163319a50174a39210cbeea1ccfde015aaf9d19":
    raise RuntimeError("original Task14 launcher hash drift")
if command_line(original_text) != command_line(science_text):
    raise RuntimeError("scientific command drift")
for token in [
    "PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2",
    "--seed 0", "--device 0", 'ENV_NAME=${PROCGEN_ENV:?PROCGEN_ENV required}',
    "bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10",
]:
    if token not in science_text:
        raise RuntimeError(f"missing frozen scientific token: {token}")
for token in ["verify_hermetic_bundle.py", "PYTHONNOUSERSITE=1"]:
    if token not in science_text or token not in preflight_text:
        raise RuntimeError(f"missing hermetic deployment token: {token}")
for text, label in [(science_text, "science"), (preflight_text, "preflight")]:
    if not any(token in text for token in ['export PYTHONPATH="$CODE"', 'export PYTHONPATH="$DEPLOY/code"']):
        raise RuntimeError(f"missing explicit bundle-only PYTHONPATH in {label} launcher")
if re.search(r"fallback|pip install|curl |wget ", science_text + preflight_text, re.I):
    raise RuntimeError("forbidden deployment fallback/download")
payload = {
    "result": "DEPLOYMENT_LAUNCHER_EQUIVALENCE_PASS",
    "original_launcher_sha256": sha(original),
    "deployment_science_launcher_sha256": sha(science),
    "deployment_preflight_launcher_sha256": sha(preflight),
    "scientific_command": command_line(science_text),
    "normalized_identity": "trainer/config/env/seed/device/budget/scientific variables unchanged",
}
print(json.dumps(payload, indent=2, sort_keys=True))
