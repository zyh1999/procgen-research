#!/bin/bash --login
#SBATCH --partition=gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_gae_ggn_head_widentity_6m_s0_20260825_33/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_gae_ggn_head_widentity_6m_s0_20260825_33/scheduler/%x_%j.err

set -uo pipefail
CAMPAIGN=/scratch/h99859yz/procgen_gae_ggn_head_widentity_6m_s0_20260825_33
CODE="$CAMPAIGN/code"
EVIDENCE="$CAMPAIGN/gpul_migration/compatibility"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_shared_det_gae_ggn_head_widentity_v1.py"
CONFIG="$CODE/configs/adv_resnet_shared_det_gae_ggn_head_widentity_v1_6m.yaml"
WRAPPER="$CAMPAIGN/frozen/gpul_compatibility_wrapper_task33.py"
mkdir -p "$EVIDENCE"
echo RUNNING > "$EVIDENCE/status"
printf '%s\n' "$SLURM_JOB_ID" > "$EVIDENCE/job_id"
hostname -f > "$EVIDENCE/hostname"
nvidia-smi -L > "$EVIDENCE/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$CAMPAIGN/frozen/gpuh_preflight.py" "$WRAPPER" > "$EVIDENCE/input.sha256"
PROCGEN_ENV=bigfish-easy-0-10 "$PY" "$WRAPPER" "$TRAINER" "$CONFIG" "$EVIDENCE/parameter_partition.json" 9c822949a171d7b4c8148ea9be37c1eb5b6aa3ff8d9d3f1603241bc4873c2665 4458567bc61bdb85360d55967c65700992fe60cd812819fa1e77cc8638c202af > "$EVIDENCE/preflight.out" 2> "$EVIDENCE/preflight.err"
rc=$?
printf '%s\n' "$rc" > "$EVIDENCE/rc"
if [ "$rc" -eq 0 ] && grep -q GPUL_TASK33_DEPLOYMENT_COMPATIBILITY_PASS "$EVIDENCE/preflight.out"; then
  echo PASS > "$EVIDENCE/status"
  exit 0
fi
echo FAIL > "$EVIDENCE/status"
exit "$rc"
