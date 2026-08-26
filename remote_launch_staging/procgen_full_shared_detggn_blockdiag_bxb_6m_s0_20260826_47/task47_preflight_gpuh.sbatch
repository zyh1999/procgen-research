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
#SBATCH --output=/scratch/h99859yz/procgen_full_shared_detggn_blockdiag_bxb_6m_s0_20260826_47/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_full_shared_detggn_blockdiag_bxb_6m_s0_20260826_47/scheduler/%x_%j.err

set -uo pipefail
CAMPAIGN=/scratch/h99859yz/procgen_full_shared_detggn_blockdiag_bxb_6m_s0_20260826_47
ROOT="$CAMPAIGN/preflight/production"
CODE="$CAMPAIGN/bundle/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_full_shared_detggn_blockdiag_bxb_v1.py"
CONFIG="$CODE/configs/adv_resnet_full_shared_detggn_blockdiag_bxb_v1_6m.yaml"
PREFLIGHT="$CAMPAIGN/frozen/preflight_full_shared_detggn_blockdiag_bxb.py"
if [ -e "$ROOT" ]; then exit 90; fi
mkdir -p "$ROOT"
echo PRECHECK_RUNNING > "$ROOT/status"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$PREFLIGHT" > "$ROOT/frozen.sha256"
(cd "$ROOT"; env -u PYTHONPATH "$PY" "$PREFLIGHT" --trainer "$TRAINER" --config "$CONFIG" --env-name bigfish-easy-0-10 --output "$ROOT/preflight.json") > "$ROOT/stdout.log" 2> "$ROOT/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$ROOT/rc"
if [ "$rc" -eq 0 ] && grep -q GPUH_FULL_SHARED_DETGGN_BLOCKDIAG_BXB_PASS "$ROOT/stdout.log"; then
  echo PRECHECK_PASS > "$ROOT/status"
  touch "$ROOT/precheck_pass.marker"
  exit 0
fi
echo PRECHECK_FAIL > "$ROOT/status"
exit "$rc"
