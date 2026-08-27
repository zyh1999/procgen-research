# Task60 deterministic RAT FP64-ridge recovery terminal report

Task-ID: `PROCGEN-RAT-BXB-DETERMINISTIC-XI1-FP64RIDGE-RECOVERY-QUICK-2M-S0-20260827-60`

## Frozen identity and placement

- Implementation commit: `73e7833`
- Trainer SHA256: `f2a4bdbd71799ef99a7e9ebad3e148f1fd03fd93075d3cd2964ba515a0cce2a9`
- Config SHA256: `fb4e3787e9e52212bb076744753e57e941a3bf37c82738844083ac6297f208fe`
- Gate wrapper SHA256: `9bd2eb640e132b02353e6439ae1f8734e09de7d9ec857719f4ca1d6f928b0fc0`
- Science wrapper SHA256: `d358a9c34db2fa5a6f16a9896cf861cd73d58de04b195e9eb8a6f128783e68f0`
- Campaign: `/scratch/h99859yz/procgen_rat_bxb_deterministic_xi1_fp64ridge_recovery_quick_2m_s0_20260827_60`
- Allocation/host: `19487252`, node822
- Gate step: `19487252.13 COMPLETED/0:0`, `PRECHECK_PASS`
- Exactly-once science step: `19487252.14 FAILED/1:0`, elapsed `00:01:32`, node822

Task60 retained deterministic `xi=1`, the reference RAT network/PopArt/two
RHS/weighted-MSE critic, the combined `512x512` BxB Gram, damping `.5`, and
the frozen adaptive-LR semantics. Its bounded recovery moved construction of
the configured ridge until after promotion to FP64.

## Four-cell terminal matrix

All four roots are `FAIL/rc1`, have empty `progress.csv`, no checkpoint and
one scientific-start marker. They failed before any exact 2,007,040 row, so
no Paper comparison is scientifically eligible and no scheduler action was
taken.

| environment | last trace transition | last reward | terminal exception |
|---|---:|---:|---|
| BigFish | 16,384 | 1.3011 | singular `torch.linalg.solve` |
| BossFight | 16,384 | 0.0 | singular `torch.linalg.solve` |
| CaveFlyer | 12,288 | 3.75 | singular `torch.linalg.solve` |
| CoinRun | 8,192 | 0.0 | singular `torch.linalg.solve` |

Every traceback ends at frozen trainer line691:

`torch._C._LinAlgError: torch.linalg.solve: The solver failed because the input matrix is singular.`

The last traces preserve the requested Task60 identity:
`joint_kernel_mode=rat_reference_combined_deterministic_xi1_b`, score-noise
mean/min/max `1`, std `0`, system rows `512`, RHS columns `2`, and configured
actor/critic damping `.5`. However, the actor Fisher already collapsed to
zero in each cell. Critic Gram diagonal medians ranged from
`1.382e14` to `2.967e15`, while critic block Frobenius norms reached
`6.456e16` to `1.563e18`. At those scales, the fixed absolute `.5` ridge is
still below FP64 representable spacing for parts of the matrix. Promotion
before ridge addition therefore fixes Task59's float32-ordering bug but does
not make the raw fixed damping numerically effective.

Targeted log scans found one traceback per cell and no OOM, CUDA, NCCL, disk
or quota error. `Inf` substring matches come from progress/log formatting and
are not a recorded finite-scan result; the decisive hard failure is the
singular solve itself. Classification is `algorithm/numerical fixed-absolute-
damping scale failure`, not deployment or GPU infrastructure.

## Artifact hashes

| environment | metric trace | stderr | stdout | command |
|---|---|---|---|---|
| BigFish | `c3f72a6e...e2a5c4` | `7054c928...2fcdc96` | `357bd76d...5780e0e6` | `3237371e...0fc7e8f` |
| BossFight | `0dbffc35...43d6d93` | `af527abf...061388` | `234eb6a7...beab3d` | `2778a728...4f072c` |
| CaveFlyer | `f7f2da8e...da20e` | `ab0e554e...ba9cd2c` | `0161d180...1a5a13` | `294eec2b...311d9` |
| CoinRun | `cefe11ac...50792` | `a311334e...d4661e` | `8d61b40c...295f9d` | `53e3215d...9f4562` |

The empty progress hash is the standard SHA256 of an empty file,
`e3b0c442...b855`. Root status and rc hashes are respectively
`4f8e9e45...f69bd5` and `4355a46b...dd865`. No model or checkpoint bytes or
hashes are included.

## Conclusion

`PRECHECK_BLOCKED_SCIENCE_START_NUMERICAL_FAILURE` for Task60. The concise
gate passed, but every exactly-once science cell encountered the same
singular raw BxB solve at 8K-16K. Per the frozen rule there is no repair,
retry, requeue or resubmit. Task51 and Task57 live roots were not modified.
