# Task52 CSF3 H200 quick mirror terminal archive

Task-ID: `PROCGEN-TASK51-GPUH-QUICK-MIRROR-BOSS-CAVE-2M-S0-20260827-52`

## Scope and identity

Task52 is a read-only quick mirror of Task51 for beta1/beta4 BossFight and
CaveFlyer seed0 on one allocated H200. It cannot replace or mutate the
authoritative Bede Task51 matrix.

- Parent trainer SHA: `af66fa0aa0115be2b82cad3666c9e91bf705053bfe219151de0689607dd4430d`
- beta1 config SHA: `5afcd1755afa3b70a7359fa84d5b139c0b6de66990473e10c944c94c65b38ec8`
- beta4 config SHA: `6bece8997404afefd720e347e4a5cc2a0d06e10704668ec473e07ac11a211dc6`
- wrapper SHA: `78cef5a441ab181d9501ec40220d2a06e31ad1739f903b506181bc2921b837ad`
- bundle SHA: `43935c8f2bfb1d20c91bddd14bbd88d9af230abab751b859704c7d1e0535032d`
- allocation/step: `19487251` / `19487251.1`, node820, one H200 shared by
  four independent CUDA processes without MPS.

The frozen Task51 PPO warmup through 503,808, one Joint switch, fixed LR
`.004`, beta1/beta4, dual-trust rules, both cross blocks, damping, PopArt,
seed/evaluation/reward semantics and exact 2,007,040 horizon were preserved.

## Terminal scheduler and artifacts

Step `19487251.1` completed `0:0` in `02:33:34` on node820. The parent
interactive allocation remains RUNNING and was not modified. All four roots
are `PASS/rc0`, contain exactly 49 progress data rows plus the header, and end
at exact transition `2,007,040`.

Each root has one regular non-symlink `model.ckpt`, size 3,766,013 bytes,
mode640. Only stat metadata is archived; checkpoint bytes and content hashes
were not copied or committed.

## Exact endpoint comparison

The immutable Paper baseline `SHA256SUMS` verified all four files. Only the
Boss/Cave exact 2M rows were used here.

| arm | env | Task52 | Paper | ratio | matched Task51 | Task52/Task51 |
|---|---|---:|---:|---:|---:|---:|
| beta1 | Boss | 0.70 | 2.92 | .2397260274 | .44 | 1.5909090909 |
| beta1 | Cave | 4.07 | 4.45 | .9146067416 | 2.50 | 1.6280000000 |
| beta4 | Boss | .62 | 2.92 | .2123287671 | .92 | .6739130435 |
| beta4 | Cave | 3.94 | 4.45 | .8853932584 | 2.30 | 1.7130434783 |

BossFight remains below the Paper `.60` threshold in both arms, but this is a
terminal read-only quick result: no scheduler cancellation or Task51 mutation
was performed. CaveFlyer passes in both arms. Compared with the matched Task51
warmup cells, Task52 improves beta1 Boss and both Cave cells; beta4 Boss is
lower.

## Numerical and error evidence

All exact endpoint traces preserve LR `.004`, eta values at `1/64`, nonzero
natural cross blocks and Cholesky info0. Relative residuals range from
`5.692e-16` to `2.264e-15`; finite scans pass and targeted
Traceback/OOM/CUDA/NCCL/disk/quota/NaN/Inf scans are zero.

The bounded Git archive contains full progress, exact final trace/scheduler
rows, source-artifact hashes, log tails, scheduler evidence, start identity and
checkpoint stat metadata. Full remote trace/log artifacts remain in the
immutable roots; no model bytes are included.

Conclusion: `QUICK_MIRROR_TERMINAL_READ_ONLY`. Task52 does not alter Task51's
scientific classifications.
