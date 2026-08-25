# Task33 gpuH to gpuL deployment audit

This is a deployment-only scheduling migration for the existing eight
environment/seed cells.  Trainer, config, normalized Python command, method,
seed, 6M horizon, run root and monitor semantics remain unchanged.

Live Slurm validation selected account/QOS `gpu-aifun` on partition `gpuL`.
An `sbatch --test-only` request for one L40S, eight CPUs, 100 GB and the exact
30-minute compatibility envelope was accepted and given a schedulable node;
no test job was created.

The frozen H200 preflight is retained byte-for-byte.  The versioned L40S
adapter verifies its exact SHA and changes only its two hardware acceptance
assertions (`H100/H200` to `L40S`, and 70 GB to 40 GB).  Every production
model, formula, config, connectivity, one-step, residual and PopArt check is
otherwise executed from the frozen preflight source.

Migration is permitted only after the bounded gpuL compatibility job passes.
Replacement science jobs are first submitted held, then verified for exact
partition/account/QOS/GRES/roots.  Only then may the eight still-unstarted
gpuH jobs be cancelled and separately recorded; gpuL jobs are released only
after every old job is scheduler-terminal, preventing duplicate live cells.

## Result

- Compatibility job `19319577` completed `0:0` on gpuL node887 in `00:07:48`.
  It proved L40S peak allocation 15,518,208,000 of 47,667,740,672 bytes,
  exact W=I identity, absent Task32 concentration path, unchanged actor/shared
  one-step policy parameters and logits, PopArt affine identity, Cholesky
  info 0 and FP64 relative residual `4.650e-16`.
- The eight replacement jobs `19319678` through `19319685` were submitted
  held and verified item-by-item before any old job was cancelled.
- Old gpuH jobs `19314824` through `19314831` were all scheduler-cancelled
  with elapsed `00:00:00`, start `None` and node `None assigned`; no old root
  or scientific artifact existed.
- Only after that terminal proof were the replacement jobs released.  They
  retain the exact original roots and are currently pending on gpuL under
  account/QOS `gpu-aifun`.
- The gpuL partition's per-user L40S limit is four, so the intended execution
  is four one-GPU cells concurrently, followed by the remaining four.  No
  cell shares a GPU with another Task33 cell.

The user had no gpuA job.  gpuA and gpuL are distinct partitions; the brief
`QOSGrpGRES` reason is an account-QOS scheduler state and is not evidence that
the user's gpuL request consumed or depended on a personal gpuA allocation.
