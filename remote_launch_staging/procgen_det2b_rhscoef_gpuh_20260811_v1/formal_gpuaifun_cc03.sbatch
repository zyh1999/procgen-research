#!/bin/bash --login
#SBATCH --job-name=pg-d2b-r03
#SBATCH --partition=gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/h99859yz/procgen_det2b_rhscoef_gpuh_20260811_v1/slurm/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_det2b_rhscoef_gpuh_20260811_v1/slurm/%x_%j.err

set -euo pipefail
exec bash /scratch/h99859yz/procgen_det2b_rhscoef_gpuh_20260811_v1/run_four_rhscoef.sh 2
