# Task63 deterministic Joint-2B post-inverse direction telemetry

## Status

`IMPLEMENTATION_FROZEN_PENDING_GATE`

## Frozen parent and only diff

- Task06 commit: `da34ce7c7d964765f336ac02111c9fde95aed1ec`
- parent trainer: `41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a`
- parent config: `69d12937debb8ef8b4531e79b8f9613185b26e4e9056a51e67754129b269391d`
- Paper control commit: `2b5affd64cbb3c624b4bc1f4767f449df231ffb2`

Task63 retains Task06's complete 1024-row full-cross deterministic Joint-2B,
critic curvature coefficient `.1`, objective coefficient `1`, damping `.5`,
FP64 Jacobi/Cholesky, Paper per-minibatch KL LR controller, SGD momentum
`1e-6`, history correction, global clip `.5`, PopArt/GAE and update order.

The only runtime addition partitions the already-effective RHS by actor and
critic rows, solves the two telemetry RHS columns with the exact existing
factorization, reconstructs `d_A` and `d_C`, and records role-wise norm,
signed projection, cosine and cancellation statistics. The original
single-RHS `flat_dir` alone remains installed. Raw metric shares are labeled
separately from post-inverse direction shares.

## Frozen Task63 files

- trainer: `067a3c9fc6f309aa13cba9d54ab6c29252ed318c4398df8de01256e5c439415a`
- science config: `4e2d18ae97084cf3ffaabb601ee63472408b77bd63fbc82d76fe1c385701db77`
- gate config: `492aea6d416db1f8937dd586af2a5122b23c9ecd7efc6889b5b6fdfac9b946ff`
- aggregator: `f7ffb9a8c885f738acadd4421a6333b1baa8d2dae023ef2b5c00693476bd2f78`
- Bede gate wrapper: `53b2ff2b0c4a2a40f559c13d9856220f4f9e0ab304d5d09037caa0d950cfa81c`
- Bede science wrapper: `c8f2bdbbd89f874914768ec0e948fb8b3585e5e302b11e25a58e3b8a5f56b917`

Local Python compile, shell syntax and frozen scalar checks pass. The science
config value `2,000,000` intentionally follows the unchanged parent loop
convention and terminates at exact progress transition `2,007,040`.

## Placement precheck

Bede account `bdman37g`, user `yihe` and PPC64LE Procgen runtime remain
available. Nodes gpu015--gpu020 were idle at the bounded refresh. The intended
campaign
`/nobackup/projects/bdman37/yihe/procgen_deterministic_joint2b_actor_critic_direction_telemetry_2m_s0_20260828_63`
was absent, with no Task63 job/process/duplicate. Task62 jobs1078176--1078179
were RUNNING and are excluded from every Task63 action.

Gate, launch matrix, initial telemetry and final aggregates will be appended
only after their authorized bounded events.
