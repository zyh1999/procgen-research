# RL stack on the 2x RTX 5060 Ti host

Root: `~/rlstack5060`

- `~/rlstack5060/bin/mujoco_container.sh sb3 0 COMMAND...`
- `~/rlstack5060/bin/mujoco_container.sh rat 1 COMMAND...`
- `~/rlstack5060/bin/isaac_container.sh sb3 0 COMMAND...`
- `~/rlstack5060/bin/isaac_container.sh native 1 COMMAND...`
- `~/rlstack5060/bin/procgen_container.sh 0 COMMAND...`

Host files placed in `~/rlstack5060/workspaces` appear as `/workspace/user`
inside every container. Use GPU `0` or `1` for independent runs, and `all`
only for code that intentionally uses both devices.

MuJoCo images:

- `rlstack5060/mujoco-sb3:cu128`: PyTorch 2.7.0+cu128, SB3 2.4.1,
  Gymnasium 1.0.0, MuJoCo 3.2.7.
- `rlstack5060/mujoco-rat:cu128`: the same controlled base plus legacy
  Gym 0.26.2 for the existing custom/RAT trainer API.

Validation records are stored in `~/rlstack5060/manifests`. Setup and pull
logs are stored in `~/rlstack5060/logs`.

The Isaac `native` entry is the supported non-SB3 path on RTX 50-series GPUs.
Legacy Isaac Gym Preview 4 uses a CPython 3.8 binary binding and an old CUDA
PyTorch stack; it is not marked ready on Blackwell hardware without a verified
port. Existing custom IsaacGymEnvs trainers must be ported or containerized with
a Blackwell-capable PyTorch/runtime before formal runs.

Procgen image:

- `rlstack5060/procgen:cu128`: Python 3.11, PyTorch 2.7.0+cu128,
  Procgen 0.10.7 built from OpenAI commit
  `37b521dbb530f0734fd82f72d921f5e0e715c0d1`, Gym3 0.3.3,
  SB3 2.4.1, and Gymnasium 1.0.0.
- The source workspace is `~/rlstack5060/workspaces/procgen` and is mounted
  as `/workspace/procgen`. The fixed recovery campaign is
  `~/rlstack5060/campaigns/procgen_actorj_dmlp1024_recovery_20260811_v1`.
- `campaign_status.sh` reports both GPUs, containers, task status, and recent
  progress. CSF3 accesses only this read-only command through a restricted
  SSH key; it does not receive a shell on this host.
- Formal Procgen workers run in detached `screen` sessions. No Jupyter
  allocation or notebook is used.
