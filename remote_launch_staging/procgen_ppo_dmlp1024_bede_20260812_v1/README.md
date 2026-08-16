# Procgen pure-PPO DMLP1024 Bede campaign

- Matrix: BigFish, BossFight, CaveFlyer, CoinRun × seeds 0, 1, 2.
- Scheduling: four independent one-GPU sbatch jobs; each job launches exactly three isolated child trainers on its assigned GPU.
- Architecture: unchanged IMPALA/ResNet encoder, shared `256 -> 1024 -> 256` decision MLP, linear policy head and PopArt critic head.
- Active parameters: 1,464,547. Pure PPO intentionally has no PPG auxiliary head or auxiliary phase.
- PPO hyperparameters are unchanged from `ppo_resnet_shared.yaml`: Adam 1e-3, clip 0.2, four epochs, eight minibatches, entropy coefficient 0, PopArt, max grad norm 0.5, 16 environments × 256 steps, 6M transitions.
- Formal root: `/nobackup/projects/bdman37/yihe/procgen_ppo_dmlp1024_bede_20260812_v1/formal_4env_x3seed_6m_20260812_v1`.
- Every child receives a private code/log/checkpoint root and separate stdout, stderr, status, and return code.
- No Jupyter is used.
