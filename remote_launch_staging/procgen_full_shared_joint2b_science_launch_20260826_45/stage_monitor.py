#!/usr/bin/env python3
"""Task45 exact-transition monitor; never acts before 2M or without Paper."""
import argparse, csv, hashlib, json, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--target", type=Path, required=True)
p.add_argument("--paper", type=Path, required=True)
p.add_argument("--job", required=True)
p.add_argument("--ledger", type=Path, required=True)
p.add_argument("--apply", action="store_true")
a = p.parse_args()

def rows(path):
    with path.open(newline="") as f:
        output = {}
        for row in csv.DictReader(f):
            key = "transitions_so_far" if "transitions_so_far" in row else "misc/total_timesteps"
            output[int(float(row[key]))] = row
        return output

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

t, b = rows(a.target), rows(a.paper)
common = sorted(set(t) & set(b))
checks = []
for floor in (2_000_000, 4_000_000):
    eligible = [x for x in common if x >= floor]
    if eligible:
        checks.append((floor, eligible[0]))
if 5_980_160 in common:
    checks.append((5_980_160, 5_980_160))
seen = set()
if a.ledger.exists():
    for line in a.ledger.read_text().splitlines():
        if line.strip():
            seen.add(int(json.loads(line)["transition"]))
with a.ledger.open("a") as out:
    for floor, step in checks:
        if step in seen:
            continue
        seen.add(step)
        tr = float(t[step]["eprewmean"])
        br = float(b[step]["eprewmean"])
        ratio = tr / br if br > 0 else float("inf")
        row = dict(method="FULL_SHARED_JOINT2B_SCALE_RECOVERY_V1",
                   stage_floor=floor, transition=step, target_reward=tr,
                   paper_reward=br, ratio=ratio,
                   decision="PASS" if ratio >= .6 else "EARLY_STOPPED_ALGORITHM")
        row.update(target_sha256=sha(a.target), paper_sha256=sha(a.paper), job=a.job)
        out.write(json.dumps(row, sort_keys=True) + "\n")
        out.flush()
        if ratio < .6 and a.apply:
            subprocess.run(["scancel", a.job], check=True)
            raise SystemExit(3)
