#!/usr/bin/env python3
"""Task61 read-only exact-2M post-solve entropy monitor."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--target", type=Path, required=True)
p.add_argument("--paper", type=Path, required=True)
p.add_argument("--job", required=True)
p.add_argument("--method", required=True, choices=(
    "FULL_SHARED_JOINT2B_NOWARMUP_FIXEDLR_DUALTRUST_POSTSOLVE_ENTGRAD001_BETA1_V1",
))
p.add_argument("--ledger", type=Path, required=True)
a = p.parse_args()

def rows(path):
    with path.open(newline="") as handle:
        output = {}
        for row in csv.DictReader(handle):
            key = "transitions_so_far" if "transitions_so_far" in row else "misc/total_timesteps"
            output[int(float(row[key]))] = row
        return output

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

target, paper = rows(a.target), rows(a.paper)
step = 2_007_040
if step not in target or step not in paper:
    raise SystemExit(0)
if a.ledger.exists():
    for line in a.ledger.read_text().splitlines():
        if line.strip() and int(json.loads(line)["transition"]) == step:
            raise SystemExit(0)
target_reward = float(target[step]["eprewmean"])
paper_reward = float(paper[step]["eprewmean"])
ratio = target_reward / paper_reward if paper_reward > 0 else float("inf")
row = {
    "method": a.method, "stage_floor": step, "transition": step,
    "target_reward": target_reward, "paper_reward": paper_reward, "ratio": ratio,
    "decision": "PASS" if ratio >= 0.60 else "BELOW_PAPER_THRESHOLD_AT_TERMINAL_ENDPOINT",
    "action": "READ_ONLY_NO_CANCELLATION_ENDPOINT",
    "target_sha256": sha(a.target), "paper_sha256": sha(a.paper), "job": a.job,
}
a.ledger.parent.mkdir(parents=True, exist_ok=True)
with a.ledger.open("a") as output:
    output.write(json.dumps(row, sort_keys=True) + "\n")
