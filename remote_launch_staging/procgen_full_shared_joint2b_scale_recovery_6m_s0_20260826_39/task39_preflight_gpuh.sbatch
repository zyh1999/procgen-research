#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_full_shared_joint2b_scale_recovery_6m_s0_20260826_39/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_full_shared_joint2b_scale_recovery_6m_s0_20260826_39/scheduler/%x_%j.err

set -uo pipefail
ENV_NAME=${PROCGEN_ENV:?PROCGEN_ENV required}
case "$ENV_NAME" in bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;; *) exit 64;; esac
CAMPAIGN=/scratch/h99859yz/procgen_full_shared_joint2b_scale_recovery_6m_s0_20260826_39
ROOT="$CAMPAIGN/preflight/$ENV_NAME"
CODE="$CAMPAIGN/bundle/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_full_shared_joint2b_scale_recovery_v1.py"
CONFIG="$CODE/configs/adv_resnet_full_shared_joint2b_scale_recovery_v1_6m.yaml"
PREFLIGHT="$CAMPAIGN/frozen/preflight_full_shared_joint2b_scale_recovery.py"
if [ -e "$ROOT" ]; then exit 90; fi
mkdir -p "$ROOT"
echo PRECHECK_RUNNING > "$ROOT/status"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$PREFLIGHT" > "$ROOT/frozen.sha256"
(cd "$ROOT"; env -u PYTHONPATH "$PY" "$PREFLIGHT" --trainer "$TRAINER" --config "$CONFIG" --env-name "$ENV_NAME" --output "$ROOT/preflight.json") > "$ROOT/stdout.log" 2> "$ROOT/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$ROOT/rc"
if [ "$rc" -eq 0 ] && grep -q GPUH_FULL_SHARED_JOINT2B_SCALE_RECOVERY_PASS "$ROOT/stdout.log"; then
  echo PRECHECK_PASS > "$ROOT/status"; touch "$ROOT/precheck_pass.marker"; exit 0
fi
echo PRECHECK_FAIL > "$ROOT/status"; exit "$rc"
