#!/bin/bash --login
#SBATCH --job-name=pg-ggn-L-preflight
#SBATCH --partition=gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_paper_matched_deterministic_ggn_1m_20260824_06/preflight_gpul_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_paper_matched_deterministic_ggn_1m_20260824_06/preflight_gpul_%j.err

set -euo pipefail
exec /mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python \
  /scratch/h99859yz/procgen_paper_matched_deterministic_ggn_1m_20260824_06/frozen/gpul_compatibility_preflight.py
