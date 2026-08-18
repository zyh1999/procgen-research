# Frozen Launch Manifest

- Task: `PROCGEN-JOINT-LOWFISHER-CAVE-5SEED-GATE-20260818-04`
- Environment: `caveflyer-easy-0-10` only
- New seeds: `1,2,3,4`; historical seed0 is not rerun
- Nominal budget: `1,000,000`; expected protocol terminal: `1,007,616`
- Placement: CSF3 `gpuA`, A100-80GB, two arrays `1-4%4`
- Jupyter: prohibited and absent from both launchers
- Output roots:
  - `/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/gate_1m_cave5seed_20260818_04/unguarded_rhs_aligned_jointb`
  - `/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/gate_1m_cave5seed_20260818_04/lowfisher_guard05_rhs_aligned_jointb`

## Frozen identities

| Role | File | SHA256 |
|---|---|---|
| Unguarded trainer on CSF3 | `code/train_shared_jointb_rhsaligned_deterministic.py` | `ff987e0dd5ca1f4c1bb9a91e3794991f5a848bdbfdadc0425d935a72acf3b501` |
| Guard trainer on CSF3 | `code/train_shared_jointb_rhsaligned_deterministic_lowfisherguard.py` | `18eea9d75dab6926788673b3bbe9c9ae26468dcbe0688c9a5e9ef150e1751526` |
| Unguarded config | `adv_resnet_shared_jointb_rhsaligned_actorrelative_criticfloor05_1m.yaml` | `d87a8f648c1c91ee0d260c64ab7dd59d12bb7f9e6b67b0ee0a135389e697fb40` |
| Guard config | `adv_resnet_shared_jointb_rhsaligned_actorrelative_criticfloor05_crossguard05_1m.yaml` | `7c2ad5efbb004ec36d816143b6ae6b8513d05621c92d21a70a938f48358d06cf` |
| Unguarded launcher | `cave_unguarded_seeds1_4_gpua.sbatch` | `b897e555cefa00f1f1b08e57ce3e0c622acf010c5ba8b4ff6ce1a075c2356096` |
| Guard launcher | `cave_lowfisher_guard05_seeds1_4_gpua.sbatch` | `7e5db6ed32a6190efa15d8bf828c25e8fa6890071306d62dbfbd5ed7f3482cc6` |

Both launchers verify their trainer, config, and launcher hashes before claiming
an output directory. The submission must export this manifest's Git commit as
`FROZEN_COMMIT` and the corresponding launcher hash as
`EXPECTED_LAUNCHER_SHA`.

## Strict scientific identity

Both methods fix rollout 4096, minibatch 512, four epochs, clean all-parameter
critic GGN, full compressed cross terms, paired-score-residual transformed RHS,
`rhs_aligned_rank1_b`, actor absolute damping `.003`, critic absolute damping
`.5`, block-relative damping `.10`, actor-from-critic floor `.01`, FP64,
momentum `0`, Kaczmarz `false`, PopArt, ratio/gradient clip semantics, network,
reward, data and evaluation protocol.

The config diff is exactly four additions in the guard config:

```yaml
joint_low_fisher_actor_critic_guard: true
joint_low_fisher_actor_critic_guard_high: 0.50
joint_low_fisher_actor_critic_guard_low: 0.20
joint_low_fisher_actor_critic_guard_max: 0.05
```

The trainers are the previously verified seed0 strict pair; their only
scientific source difference is guard parsing, validation, interpolation and
telemetry. The launchers differ only in the trainer/config/root/method identity
and the guard-specific preflight fields. Seed is supplied by the array task.

## Resource decision snapshot

- `2026-08-18T13:45Z` CSF3: gpuA has 13 mixed nodes and 4 allocated nodes;
  no owned Procgen job/trainer. Exact seed0 pair previously completed on gpuA.
- Bede: five idle V100 nodes, but the earlier wide RHS implementation OOMed
  before its memory-efficient repair; not selected over the proven gpuA path.
- `ws4090-92`: both cards owned/used by another Procgen campaign; root disk
  99% with 13G available.
- `ws4090-76`: owned Procgen campaign present; GPU0 98%, both cards holding
  10.6G, root disk effectively full with 4.1G available.
- Registered dual-5060: two idle 16GB cards, but not selected because this
  FP64 solver's safe memory envelope is proven on A100-80GB, not 16GB.
- `.54`, `ws4090-31`, and `10.49.7.54` were not accessed.
