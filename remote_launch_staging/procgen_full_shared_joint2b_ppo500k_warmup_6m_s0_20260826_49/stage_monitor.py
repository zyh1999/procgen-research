#!/usr/bin/env python3
"""Task49 exact-transition monitor; never acts before 2M or without Paper."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--target", type=Path, required=True)
p.add_argument("--paper", type=Path, required=True)
p.add_argument("--job", required=True)
p.add_argument("--ledger", type=Path, required=True)
p.add_argument("--apply", action="store_true")
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
common = sorted(set(target) & set(paper))
checks = []
for floor in (2_000_000, 4_000_000):
    eligible = [step for step in common if step >= floor]
    if eligible:
        checks.append((floor, eligible[0]))
if 5_980_160 in common:
    checks.append((5_980_160, 5_980_160))
seen = set()
if a.ledger.exists():
    for line in a.ledger.read_text().splitlines():
        if line.strip():
            seen.add(int(json.loads(line)["transition"]))
with a.ledger.open("a") as output:
    for floor, step in checks:
        if step in seen:
            continue
        seen.add(step)
        target_reward = float(target[step]["eprewmean"])
        paper_reward = float(paper[step]["eprewmean"])
        ratio = target_reward / paper_reward if paper_reward > 0 else float("inf")
        row = {
            "method": "FULL_SHARED_JOINT2B_PPO500K_WARMUP_V1",
            "stage_floor": floor,
            "transition": step,
            "target_reward": target_reward,
            "paper_reward": paper_reward,
            "ratio": ratio,
            "decision": "PASS" if ratio >= 0.60 else "EARLY_STOPPED_ALGORITHM",
            "target_sha256": sha(a.target),
            "paper_sha256": sha(a.paper),
            "job": a.job,
        }
        output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        if ratio < 0.60 and a.apply:
            subprocess.run(["scancel", a.job], check=True)
            raise SystemExit(3)
