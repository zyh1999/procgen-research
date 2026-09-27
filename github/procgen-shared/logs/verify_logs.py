"""Verify the released scalar logs with only Python's standard library."""
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    records = json.loads((ROOT / 'manifest.json').read_text())['runs']
    seen = set()
    counts = Counter()
    cohorts = defaultdict(set)
    total_rows = 0
    for r in records:
        key = (r['cohort'], r['environment'], r['seed'])
        assert key not in seen, key
        seen.add(key)
        cfg = json.loads((ROOT / r['cohort'] / 'config.json').read_text())
        path = ROOT / r['path']
        assert path.resolve().is_relative_to(ROOT), r['path']
        blob = path.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == r['compressed_sha256'], key
        raw = gzip.decompress(blob)
        assert hashlib.sha256(raw).hexdigest() == r['scalar_csv_sha256'], key
        rows = list(csv.DictReader(io.StringIO(raw.decode())))
        assert len(rows) == r['rollout_rows'], key
        assert list(rows[0]) == r['columns'], key
        assert r['status'] == 'PASS' and r['rc'] == 0, key
        steps = [int(float(row['misc/total_timesteps'])) for row in rows]
        assert steps == list(range(cfg['rollout'], cfg['endpoint'] + 1, cfg['rollout'])), key
        for row in rows:
            assert all(math.isfinite(float(v)) for v in row.values() if v not in ('', None)), key
        if cfg['task'] == 278:
            assert all(float(row['lr']) == .0003 for row in rows), key
        counts[r['cohort']] += 1
        cohorts[(r['cohort'], r['environment'])].add(r['seed'])
        total_rows += len(rows)
    assert len(records) == 124
    assert sorted(counts.values()) == [12, 12, 12, 12, 12, 24, 40]
    for (cohort, env), seeds in cohorts.items():
        assert seeds == set(range(5 if cohort.startswith('task229_') else 3)), (cohort, env)
    files = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.csv.gz')}
    assert files == {r['path'] for r in records}
    print(f'PASS: {len(records)} runs, {total_rows} rollout rows, all endpoints/hashes/finite scalars/seed coverage verified.')
    for k, v in sorted(counts.items()):
        print(f'{k}: {v} runs')

if __name__ == '__main__':
    main()
