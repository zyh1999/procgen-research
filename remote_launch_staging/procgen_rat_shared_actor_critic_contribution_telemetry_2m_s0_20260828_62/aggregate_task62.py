#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

FIELDS = (
    'metric_actor_norm_share',
    'full_actor_norm_share',
    'full_actor_signed_projection_share',
    'shared_actor_norm_share',
    'shared_actor_signed_projection_share',
    'full_actor_critic_cosine',
    'global_clip_scale',
    'policy_kl',
    'lr_before',
)


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(records):
    output = {'records': len(records)}
    for field in FIELDS:
        values = [float(record[field]) for record in records]
        output[field] = {
            'p10': quantile(values, 0.10),
            'median': quantile(values, 0.50),
            'p90': quantile(values, 0.90),
        }
    output['clip_rate'] = sum(float(r['global_clip_scale']) < 1.0 for r in records) / len(records)
    return output


def validate(records):
    if not records:
        raise SystemExit('no complete telemetry records')
    required = set(FIELDS) | {
        'h_reconstruction_relative_residual', 'gradient_reconstruction_relative',
        'policy_exclusive_g_v_l2', 'value_exclusive_g_pi_l2', 'finite_scan_pass',
        'batch_rows', 'parameter_columns', 'w_pi_l2', 'w_v_l2',
    }
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise SystemExit(f'record {index} missing {sorted(missing)}')
        numeric = [value for value in record.values() if isinstance(value, (int, float))]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise SystemExit(f'record {index} contains nonfinite telemetry')
        if int(record['batch_rows']) != 512:
            raise SystemExit(f'record {index} row drift')
        if float(record['h_reconstruction_relative_residual']) > 1e-7:
            raise SystemExit(f'record {index} H reconstruction drift')
        if float(record['gradient_reconstruction_relative']) > 2e-5:
            raise SystemExit(f'record {index} gradient reconstruction drift')
        if float(record['policy_exclusive_g_v_l2']) != 0.0:
            raise SystemExit(f'record {index} critic reached policy-exclusive parameters')
        if float(record['value_exclusive_g_pi_l2']) != 0.0:
            raise SystemExit(f'record {index} actor reached value-exclusive parameters')
        if float(record['finite_scan_pass']) != 1.0:
            raise SystemExit(f'record {index} finite scan failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('--endpoint', type=int, default=2_007_040)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gate', action='store_true')
    args = parser.parse_args()
    records = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    validate(records)
    if args.gate:
        result = {'status': 'TASK62_GATE_PASS', 'records': len(records)}
    else:
        thirds = {'early': [], 'middle': [], 'late': []}
        for record in records:
            fraction = float(record['transition']) / args.endpoint
            thirds['early' if fraction <= 1/3 else 'middle' if fraction <= 2/3 else 'late'].append(record)
        result = {'status': 'TASK62_AGGREGATION_PASS', 'overall': summarize(records)}
        result['thirds'] = {name: summarize(rows) for name, rows in thirds.items() if rows}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
