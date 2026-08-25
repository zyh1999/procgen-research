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
