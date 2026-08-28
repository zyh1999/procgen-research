#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def last_row(path):
    if not path.is_file():
        return None
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--endpoint', type=int, default=2_007_040)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    progress = last_row(root / 'progress.csv')
    trace_path = root / 'metric_trace.jsonl'
    trace = None
    if trace_path.is_file():
        lines = [line for line in trace_path.read_text().splitlines() if line]
        trace = json.loads(lines[-1]) if lines else None
    result = {
        'root': str(root),
        'status': (root / 'status').read_text().strip() if (root / 'status').is_file() else None,
        'rc': (root / 'rc').read_text().strip() if (root / 'rc').is_file() else None,
        'progress': progress,
        'trace': trace,
        'endpoint': args.endpoint,
        'read_only': True,
        'reward_action': 'NONE',
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
