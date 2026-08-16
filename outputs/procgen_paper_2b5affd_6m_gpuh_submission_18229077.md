# Procgen paper-aligned 2b5affd 6M gpuH submission

- Submitted: 2026-08-04 17:08 BST
- CSF3 job: `18229077` (`pg2b5_6m_H`)
- Initial state: `PENDING (AssocGrpGRES)`; no dependency was attached
- Final state: `CANCELLED by 778916` at 2026-08-04 17:14 BST, with `Elapsed=00:00:00` and `Start=None`
- Partition/account/QOS: `gpuH` / `gpu-h200-fse-pgdr` / `gpu-h200-fse`
- Requested resources: one node, two H200 GPUs, 16 CPUs, 180 GB RAM, 48-hour limit
- Workload: shared Exact RAT plus shared PPO, eight Procgen easy environments, seeds 0--4, 80 runs total
- Layout: two GPU workers; five seeds run concurrently on each GPU; each worker processes four RAT and four PPO environments sequentially
- Run root: `/scratch/h99859yz/procgen_paper_2b5affd_6m_gpuh_20260804/formal`
- Submission script: `/scratch/h99859yz/procgen_paper_2b5affd_6m_gpuh_20260804/submit.sbatch`
- `NVIDIA_TF32_OVERRIDE=0`

## Provenance

- Repository commit: `2b5affd64cbb3c624b4bc1f4767f449df231ffb2`
- Scope: Procgen only
- Controlled config change: `timesteps_per_proc_easy` is 6,000,000 instead of the commit-native 3,000,000 for both RAT and PPO
- Trainer SHA-256: `cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7`
- RAT config SHA-256: `1ed4eab5bcaf41e6c5fa99e75ab26cf04bbac42107e03f8f4fa12a95b344f6ea`
- PPO config SHA-256: `d3bb39669f7baf055d8d3f1a873bef3d5ec5334ce402ed24971c760f83fb9603`
- gpuH wrapper SHA-256: `07e929f01c0cc511491e0a5cba2333fb8a4c32d593f5c440b95cf1213763c025`

The previous gpuA submission `17794600` was cancelled before starting and is not reused. This submission uses a new run directory and does not overwrite previous artifacts.

The user subsequently requested cancellation of this gpuH submission. Job `18229077` never started and produced no training runs.
