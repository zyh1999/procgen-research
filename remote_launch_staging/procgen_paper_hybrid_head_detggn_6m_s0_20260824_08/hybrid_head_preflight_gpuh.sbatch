#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --job-name=pg-hybrid-head-preflight
#SBATCH --output=/scratch/h99859yz/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/preflight/slurm_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/preflight/slurm_%j.err

set -euo pipefail
CAMPAIGN=/scratch/h99859yz/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08
CODE="$CAMPAIGN/code"
FROZEN="$CAMPAIGN/frozen"
OUT="$CAMPAIGN/preflight/$SLURM_JOB_ID"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_shared_paper_hybrid_head_detggn_v1.py"
CONFIG="$CODE/configs/adv_resnet_shared_paper_hybrid_head_detggn_v1_6m.yaml"
TRAINER_SHA=7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54
CONFIG_SHA=9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda

mkdir -p "$OUT"
finalize() {
  RC=$?
  date -Is > "$OUT/end_time.txt"
  printf '%s\n' "$RC" > "$OUT/rc"
  if [ "$RC" -eq 0 ]; then
    printf 'PRECHECK_PASS\n' > "$OUT/status"
  else
    printf 'INFRASTRUCTURE_PREFLIGHT_FAIL\n' > "$OUT/status"
  fi
}
trap finalize EXIT
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
date -Is > "$OUT/start_time.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$OUT/job_id"
hostname -f > "$OUT/hostname"
nvidia-smi -L > "$OUT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$FROZEN/gpuh_preflight.py" "$FROZEN/test_hybrid_head_detggn_v1.py" "$FROZEN/hybrid_head_preflight_gpuh.sbatch" > "$OUT/input.sha256"

"$PY" "$FROZEN/test_hybrid_head_detggn_v1.py" > "$OUT/regression.out" 2> "$OUT/regression.err"
grep -q HYBRID_HEAD_REGRESSION_PASS "$OUT/regression.out"
"$PY" "$FROZEN/gpuh_preflight.py" "$TRAINER" "$CONFIG" "$OUT/parameter_partition.json" "$TRAINER_SHA" "$CONFIG_SHA" > "$OUT/compatibility.out" 2> "$OUT/compatibility.err"
grep -q GPUH_HYBRID_HEAD_COMPATIBILITY_PASS "$OUT/compatibility.out"

echo HYBRID_HEAD_PREFLIGHT_PASS
