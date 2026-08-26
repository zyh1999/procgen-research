# Task37 terminal model-free evidence

Task: `PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-6M-S0-SCIENCE-20260826-37`

The Git-ready archive `task37_compact_model_free.tgz` contains campaign and
per-cell deployment metadata, scheduler logs, commands, progress CSV files,
stdout/stderr, exact `early_stop_2007040` directories, monitor inputs and
ledgers, artifact hashes, and the frozen stage monitor. It explicitly excludes
models, checkpoints, and the large full metric traces.

The complete model-free archive, including all four metric traces, is retained
at:

`/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_6m_s0_science_20260826_37/terminal_model_free_export/task37_model_free.tgz`

Its validated SHA256 is
`14e3cca153da5a90c9463cc7f64c440d9f9688f14b30309d1ad74bf228853e4c`
and its size is 42,223,651 bytes. `tar -tzf` returned zero. The archive was
built with explicit exclusions for models and checkpoints and preserves the
full remote scientific telemetry without adding it to Git.

The compact archive has SHA256
`74a4233dce11c6fa00e06a0534e2dd939b07d73ede2a39d5cd710ad253a2eb3e`,
size 458,757 bytes, and passed local `tar -tzf`; its member audit found no
model, checkpoint, `.pt`, `.pth`, or metric-trace payload.

The four root `status` files remain stale `RUNNING` and have no trainer `rc`
because Slurm cancelled the trainers after the frozen monitor decision.
Scheduler accounting plus the per-cell rc3 monitor ledger is authoritative.
