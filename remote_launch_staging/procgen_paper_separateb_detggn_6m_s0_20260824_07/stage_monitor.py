#!/usr/bin/env python3
"""Exact-transition 3/5 monitor. Never acts before 2M or without a Paper row."""
import argparse, csv, json, subprocess
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
        return {int(float(r["transitions_so_far"])): r for r in csv.DictReader(f)}

t, b = rows(a.target), rows(a.paper)
common = sorted(set(t) & set(b))
checks = []
for floor in (2_000_000, 4_000_000, 5_980_160):
    eligible = [x for x in common if x >= floor]
    if eligible: checks.append((floor, eligible[0]))
seen = set()
with a.ledger.open("a") as out:
    for floor, step in checks:
        if step in seen: continue
        seen.add(step)
        tr = float(t[step]["eprewmean"]); br = float(b[step]["eprewmean"])
        ratio = tr / br if br > 0 else float("inf")
        row = dict(stage_floor=floor, transition=step, target_reward=tr,
                   paper_reward=br, ratio=ratio, decision="PASS" if ratio >= .6 else "EARLY_STOPPED_ALGORITHM")
        out.write(json.dumps(row, sort_keys=True) + "\n"); out.flush()
        if ratio < .6 and a.apply:
            subprocess.run(["scancel", a.job], check=True)
            raise SystemExit(3)
