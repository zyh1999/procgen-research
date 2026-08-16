#!/bin/bash
#SBATCH --job-name=pg2b5_6m_H
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/h99859yz/procgen_paper_2b5affd_6m_gpuh_20260804/slurm-%j.out
#SBATCH --error=/scratch/h99859yz/procgen_paper_2b5affd_6m_gpuh_20260804/slurm-%j.err

set -euo pipefail

export BUNDLE_ROOT=/mnt/iusers01/fatpou01/compsci01/h99859yz/procgen_paper_2b5affd_6m_20260722_1632/bundle
export RUN_ROOT=/scratch/h99859yz/procgen_paper_2b5affd_6m_gpuh_20260804/formal
export NVIDIA_TF32_OVERRIDE=0

exec bash "$BUNDLE_ROOT/run_2gpu_csf3.sbatch"
