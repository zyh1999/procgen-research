#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_6m_s0_20260825_34r/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_6m_s0_20260825_34r/scheduler/%x_%j.err

set -uo pipefail
ENV_NAME=${PROCGEN_ENV:?PROCGEN_ENV required}
case "$ENV_NAME" in bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;; *) exit 64;; esac
CAMPAIGN=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_6m_s0_20260825_34r
CODE="$CAMPAIGN/code"
EVIDENCE="$CAMPAIGN/preflight/$ENV_NAME"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_shared_det_standard_mse_ggn_head_cvlm_v1.py"
CONFIG="$CODE/configs/adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml"
PREFLIGHT="$CAMPAIGN/frozen/gpuh_preflight.py"
AUDIT="$CAMPAIGN/frozen/audit_task34r.py"
TRAINER_SHA=${TASK34R_TRAINER_SHA:?TASK34R_TRAINER_SHA required}
CONFIG_SHA=${TASK34R_CONFIG_SHA:?TASK34R_CONFIG_SHA required}
mkdir -p "$EVIDENCE"
echo PREFLIGHT_RUNNING > "$EVIDENCE/status"
printf '%s\n' "$SLURM_JOB_ID" > "$EVIDENCE/job_id"
hostname -f > "$EVIDENCE/hostname"
nvidia-smi -L > "$EVIDENCE/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$PREFLIGHT" "$CAMPAIGN/frozen/standard_mse_ggn_head_cvlm_6m_gpuh.sbatch" > "$EVIDENCE/input.sha256"
TASK07_SOURCE_ROOT="$CAMPAIGN/audit_sources/task07" \
TASK13_SOURCE_ROOT="$CAMPAIGN/audit_sources/task13" \
TASK32_SOURCE_ROOT="$CAMPAIGN/audit_sources/task32" \
TASK34R_AUDIT_OUTPUT="$EVIDENCE/historical_scaling_ledger.json" \
  "$PY" "$AUDIT" > "$EVIDENCE/historical_audit.out" 2> "$EVIDENCE/historical_audit.err"
audit_rc=$?
if [ "$audit_rc" -ne 0 ]; then
  printf '%s\n' "$audit_rc" > "$EVIDENCE/rc"
  echo PRECHECK_FAIL > "$EVIDENCE/status"
  exit "$audit_rc"
fi
grep -q TASK34R_HISTORICAL_SCALING_AUDIT_PASS "$EVIDENCE/historical_audit.out" || { echo PRECHECK_FAIL > "$EVIDENCE/status"; exit 73; }
PROCGEN_ENV="$ENV_NAME" "$PY" "$PREFLIGHT" "$TRAINER" "$CONFIG" "$EVIDENCE/parameter_partition.json" "$TRAINER_SHA" "$CONFIG_SHA" > "$EVIDENCE/preflight.out" 2> "$EVIDENCE/preflight.err"
rc=$?
if [ "$rc" -eq 0 ]; then
  echo 0 > "$EVIDENCE/rc"
  echo PRECHECK_PASS > "$EVIDENCE/status"
  exit 0
fi
printf '%s\n' "$rc" > "$EVIDENCE/rc"
echo PRECHECK_FAIL > "$EVIDENCE/status"
exit "$rc"
