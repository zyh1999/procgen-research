"""Regenerate README figures from the released scalar logs (no remote access).

Dependencies: matplotlib==3.9.4, numpy==1.26.4. Run from any directory.
All curves use matched seeds 0/1/2. Smooth each seed with a trailing 10-row
mean, then plot the seed mean +/- sample SEM. No interpolation/extrapolation.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'logs'
OUT = ROOT / 'figures'
ENVS = ['bigfish', 'bossfight', 'starpilot', 'caveflyer', 'coinrun', 'maze', 'jumper', 'miner']
NAMES = ['BigFish', 'BossFight', 'StarPilot', 'CaveFlyer', 'CoinRun', 'Maze', 'Jumper', 'Miner']
COLORS = {'rat': '#D55E00', 'dual': '#009E73', 'coeff': '#7A52A1', 'ppo': '#0072B2'}
SERIES = {
    'normal': [
        ('task229_full_rat_B512', 'Full RAT · Task229', 'rat'),
        ('task253_dual127p1_B512', 'Dual127+1 · Task253', 'dual'),
        ('task280_coeff128_ray_B512', 'Coeff128 + ray · Task280', 'coeff'),
    ],
    '8192': [
        ('task278_ppo_fixedlr_B8192', 'PPO fixed LR · Task278', 'ppo'),
        ('task281_dual127p1_B8192', 'Dual127+1 B/B · Task281', 'dual'),
    ],
}

def smooth(y):
    sums = np.r_[0., np.cumsum(y)]
    end = np.arange(1, len(y) + 1)
    start = np.maximum(0, end - 10)
    return (sums[end] - sums[start]) / (end - start)

def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((LOGS / 'manifest.json').read_text())
    records = {(r['cohort'], r['environment'], r['seed']): r for r in manifest['runs']}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#334155', 'text.color': '#172B42',
                         'axes.titleweight': 'bold', 'svg.hashsalt': 'shared-procgen-20260927'})
    provenance = {'date': '2026-09-27', 'architecture': 'shared', 'seeds': [0, 1, 2],
                  'smoothing': 'per-seed trailing10 rollout means; min_periods=1',
                  'band': 'sample standard deviation across smoothed seeds / sqrt(3)',
                  'metric': 'eprewmean: training return, not held-out evaluation',
                  'figures': {}}
    for variant in ['normal', '8192']:
        normal = variant == 'normal'
        envs = ENVS if normal else ENVS[:4]
        horizon = 6_000_000 if normal else 15_000_000
        fig, axes = plt.subplots(2, 4 if normal else 2, figsize=(16, 8.2) if normal else (11, 8.2))
        used = []
        coverage = {}
        for ax, env in zip(axes.flat, envs):
            coverage[env] = []
            for cohort, label, color in SERIES[variant]:
                if not all((cohort, env, s) in records for s in range(3)):
                    continue
                seed_y = []
                grids = []
                for seed in range(3):
                    r = records[cohort, env, seed]
                    path = LOGS / r['path']
                    blob = path.read_bytes()
                    assert hashlib.sha256(blob).hexdigest() == r['compressed_sha256']
                    with gzip.open(path, 'rt') as f:
                        rows = list(csv.DictReader(f))
                    rows = [row for row in rows if int(float(row['misc/total_timesteps'])) <= horizon]
                    grids.append(np.array([float(row['misc/total_timesteps']) for row in rows]))
                    seed_y.append(smooth(np.array([float(row['eprewmean']) for row in rows])))
                    used.append({'path': r['path'], 'scalar_csv_sha256': r['scalar_csv_sha256']})
                assert all(np.array_equal(grids[0], x) for x in grids[1:])
                values = np.stack(seed_y)
                mean = values.mean(axis=0)
                sem = values.std(axis=0, ddof=1) / np.sqrt(3)
                x = grids[0] / 1e6
                ax.plot(x, mean, color=COLORS[color], lw=1.8)
                ax.fill_between(x, mean - sem, mean + sem, color=COLORS[color], alpha=.14, linewidth=0)
                coverage[env].append(cohort)
            ax.set_title(NAMES[ENVS.index(env)], fontsize=12, pad=9)
            ax.set_xlim(0, horizon / 1e6)
            ax.set_ylim(bottom=0)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            ax.grid(alpha=.17)
            ax.set_xlabel('Environment transitions (M)')
            ax.set_ylabel('Training return')
            if normal and len(coverage[env]) < 3:
                ax.text(.97, .035, 'Dual127+1: no released matched cohort', transform=ax.transAxes,
                        va='bottom', ha='right', fontsize=8, color='#64748B',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
        title = ('Shared Procgen · normal minibatch 512' if normal else 'Shared Procgen · large minibatch 8192')
        subtitle = ('Rollout 4,096 · first 6M transitions' if normal else 'Rollout 65,536 · first 15M transitions')
        fig.suptitle(title, fontsize=19, weight='bold', y=.982)
        fig.text(.5, .932, subtitle + ' · seeds 0/1/2 · trailing-10 mean ± SEM', ha='center', fontsize=11)
        handles = [Line2D([0], [0], color=COLORS[c], lw=2.2, label=l) for _, l, c in SERIES[variant]]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .909),
                   ncol=len(handles), frameon=False, fontsize=10)
        footer = ('B512 PPO is not available in this release. Full RAT seeds 3/4 remain in the logs but are not plotted.'
                  if normal else 'No completed B8192 Full RAT or coefficient128-ray cohort in this release; no substitute 128 curve.')
        fig.text(.5, .028, footer, ha='center', fontsize=9, color='#475569')
        fig.text(.5, .009, 'Historical selected cohorts; training return only. Not a pure normalization ablation or a significance claim.',
                 ha='center', fontsize=8.5, color='#64748B')
        fig.tight_layout(rect=(0, .055, 1, .865), h_pad=2.0, w_pad=2.0)
        stem = 'shared_normal_B512' if normal else 'shared_large_B8192'
        fig.savefig(OUT / (stem + '.png'), dpi=180, facecolor='white')
        fig.savefig(OUT / (stem + '.svg'), facecolor='white', metadata={'Date': None})
        svg = OUT / (stem + '.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
        plt.close(fig)
        provenance['figures'][stem] = {'horizon': horizon, 'coverage': coverage, 'sources': used}
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Saved normal B512 and large B8192 PNG/SVG figures and provenance.')

if __name__ == '__main__':
    main()
