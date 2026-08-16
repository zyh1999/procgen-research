#!/bin/bash --login
#SBATCH --job-name=pg-emp-dmlp1024
#SBATCH --partition=gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-11%4
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_emp_dmlp1024_csf3_20260810_v1/slurm/%A_%a.out
#SBATCH --error=/scratch/h99859yz/procgen_emp_dmlp1024_csf3_20260810_v1/slurm/%A_%a.err

set -euo pipefail

ENVS=(
  bigfish-easy-0-10
  bossfight-easy-0-10
  caveflyer-easy-0-10
  coinrun-easy-0-10
)

ENV_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
SEED=$((SLURM_ARRAY_TASK_ID / 4))
ENV_NAME=${ENVS[$ENV_INDEX]}

exec bash /scratch/h99859yz/procgen_emp_dmlp1024_csf3_20260810_v1/launchers/run_one_csf3.sh \
  "$ENV_NAME" "$SEED"
