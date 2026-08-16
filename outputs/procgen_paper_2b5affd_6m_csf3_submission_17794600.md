# Procgen paper-aligned `2b5affd` 6M submission

- Checked/submitted: 2026-07-22T16:33:04+01:00
- Cluster: CSF3
- Job: `17794600` (`pg2b5_paper6m`)
- Current state: `PENDING (Priority)`
- Partition/account/QOS: `gpuA` / `gpu-aifun` / `gpu-aifun`
- Requested resources: one node, 2 x A100 80GB, 16 CPUs, 180GB RAM
- Time limit: 2 days
- Remote root: `/mnt/iusers01/fatpou01/compsci01/h99859yz/procgen_paper_2b5affd_6m_20260722_1632`

## Experiment layout

- Methods: shared Exact RAT and shared PPO
- Environments: BigFish, BossFight, CaveFlyer, CoinRun, Jumper, Maze, Miner, StarPilot
- Seeds: 0, 1, 2, 3, 4 for every method/environment
- Total: 80 seed runs
- Each GPU runs five seeds concurrently for one method/environment at a time.
- Each GPU receives four RAT and four PPO environments to balance runtime.
- Horizon: 6,000,000 environment steps per seed
- Rollout geometry: 16 environments x 256 steps = 4096 environment steps per update

## Provenance and controlled change

- Source repository: `git@github.com:agent-lab/trust-region.git`
- Source commit: `2b5affd64cbb3c624b4bc1f4767f449df231ffb2`
- Scope: this source snapshot is used only for Procgen.
- Controlled change relative to the commit: `timesteps_per_proc_easy` changed from 3,000,000 to 6,000,000 in the RAT and PPO Procgen configs to match the paper Figure 4 budget.
- No trainer-code modification was made.
- Trainer SHA-256: `cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7` (exact target-commit match)
- RAT config SHA-256: `1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea`
- PPO config SHA-256: `d3bb39669f7baf055d8d3f1a873bef3d5ec5334ce402ed24971c760f83fb9603`
- Submitted bundle SHA-256: `0c3c566e87dd59ac50bfe4f89c13a0674ebe90faf24cb4262ba525424cebd6a5`

## Verified boundaries

- RAT adaptive KL upper threshold: `0.02 * 2 = 0.04`
- RAT adaptive KL lower threshold: `0.01 / 2 = 0.005`
- RAT SGD momentum: `1e-6`, as encoded in the requested commit
- PPO uses its commit-native Adam/clip configuration and has KL-adaptive LR disabled.
- Remote archive hash, trainer/config hashes, shell syntax, Python compilation, YAML parsing, and environment imports all passed before submission.

No Bede or 4090 job was submitted for this corrected batch.
