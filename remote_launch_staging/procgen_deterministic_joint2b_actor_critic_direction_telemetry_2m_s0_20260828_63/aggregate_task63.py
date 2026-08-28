#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

FIELDS = (
    'postinverse_full_actor_norm_share',
    'postinverse_full_actor_signed_projection_share',
    'postinverse_shared_actor_norm_share',
    'postinverse_shared_actor_signed_projection_share',
    'postinverse_full_actor_critic_cosine',
    'postinverse_full_cancellation_amplification',
    'postinverse_actor_metric_norm_share',
    'postinverse_actor_metric_energy_share',
    'joint_predicted_step_kl',
    'joint_predicted_value_divergence',
    'joint_clip_scale',
    'kl',
    'curr_lr',
)


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def summarize(records):
    result = {'records': len(records)}
    for field in FIELDS:
        values = [float(record[field]) for record in records]
        result[field] = {
            'p10': quantile(values, .1),
            'median': quantile(values, .5),
            'p90': quantile(values, .9),
        }
    result['clip_rate'] = sum(
        float(record['joint_clip_scale']) < 1.0 for record in records
    ) / len(records)
    return result


def validate(records):
    if not records:
        raise SystemExit('no complete Task63 records')
    required = set(FIELDS) | {
        'actor_rows', 'critic_rows', 'joint_system_rows',
        'postinverse_rhs_reconstruction_relative',
        'postinverse_alpha_reconstruction_relative',
        'postinverse_direction_reconstruction_relative',
        'postinverse_actor_value_raw_zero_l2',
        'postinverse_critic_policy_raw_zero_l2',
        'postinverse_full_projection_sum',
        'postinverse_shared_projection_sum',
        'postinverse_policy_projection_sum',
        'postinverse_value_projection_sum',
        'postinverse_finite_scan_pass', 'joint_cholesky_info_max',
        'joint_relative_solve_residual', 'rollout_index',
        'train_epoch_index', 'minibatch_index', 'transitions',
    }
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise SystemExit(f'record {index} missing {sorted(missing)}')
        numeric = [v for v in record.values() if isinstance(v, (int, float))]
        if not all(math.isfinite(float(v)) for v in numeric):
            raise SystemExit(f'record {index} nonfinite')
        if int(record['actor_rows']) != 512 or int(record['critic_rows']) != 512:
            raise SystemExit(f'record {index} row drift')
        if int(record['joint_system_rows']) != 1024:
            raise SystemExit(f'record {index} not strict 2B')
        if float(record['postinverse_rhs_reconstruction_relative']) > 1e-14:
            raise SystemExit(f'record {index} RHS reconstruction drift')
        if float(record['postinverse_alpha_reconstruction_relative']) > 5e-12:
            raise SystemExit(f'record {index} alpha reconstruction drift')
        if float(record['postinverse_direction_reconstruction_relative']) > 2e-5:
            raise SystemExit(f'record {index} direction reconstruction drift')
        if float(record['postinverse_actor_value_raw_zero_l2']) != 0.0:
            raise SystemExit(f'record {index} actor raw value columns nonzero')
        if float(record['postinverse_critic_policy_raw_zero_l2']) != 0.0:
            raise SystemExit(f'record {index} critic raw policy columns nonzero')
        for role in ('full', 'shared', 'policy', 'value'):
            if abs(float(record[f'postinverse_{role}_projection_sum']) - 1.0) > 2e-5:
                raise SystemExit(f'record {index} {role} projection drift')
        if int(record['joint_cholesky_info_max']) != 0:
            raise SystemExit(f'record {index} Cholesky failure')
        if float(record['postinverse_finite_scan_pass']) != 1.0:
            raise SystemExit(f'record {index} finite failure')
    first = records[0]
    if float(first.get('first_update_identity_checked', 0.0)) != 1.0:
        raise SystemExit('first update identity not checked')
    if first.get('first_update_reference_sha256') != first.get('first_update_observed_sha256'):
        raise SystemExit('first update parameter hash mismatch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('--gate', action='store_true')
    parser.add_argument('--endpoint', type=int, default=2_007_040)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    validate(records)
    if args.gate:
        result = {'status': 'TASK63_GATE_PASS', 'records': len(records)}
    else:
        thirds = {'early': [], 'middle': [], 'late': []}
        for record in records:
            fraction = float(record['transitions']) / args.endpoint
            thirds['early' if fraction <= 1/3 else 'middle' if fraction <= 2/3 else 'late'].append(record)
        result = {'status': 'TASK63_AGGREGATION_PASS', 'overall': summarize(records)}
        result['thirds'] = {name: summarize(rows) for name, rows in thirds.items() if rows}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
