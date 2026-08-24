#!/bin/bash --login
#SBATCH --job-name=pg-paper-ggn1m
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/h99859yz/procgen_paper_matched_deterministic_ggn_1m_20260824_06/scheduler/%A_%a.out
#SBATCH --error=/scratch/h99859yz/procgen_paper_matched_deterministic_ggn_1m_20260824_06/scheduler/%A_%a.err

set -uo pipefail
CAMPAIGN=/scratch/h99859yz/procgen_paper_matched_deterministic_ggn_1m_20260824_06
CODE="$CAMPAIGN/code"
PYTHON=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_shared_paper_matched_deterministic_ggn_v1.py"
CONFIG="$CODE/configs/adv_resnet_shared_paper_matched_deterministic_ggn_v1_1m.yaml"
METHOD=PAPER_MATCHED_DETERMINISTIC_GGN_V1
ENVS=(bigfish-easy-0-10 bossfight-easy-0-10 caveflyer-easy-0-10 coinrun-easy-0-10)
ENV_NAME=${ENVS[$SLURM_ARRAY_TASK_ID]}
ROOT="$CAMPAIGN/runs/$METHOD/$ENV_NAME/seed0/1m"

if [ -e "$ROOT" ]; then
  echo "collision: $ROOT already exists" >&2
  exit 90
fi
mkdir -p "$ROOT"
echo RUNNING > "$ROOT/status"
printf '%s\n' "$SLURM_ARRAY_JOB_ID" > "$ROOT/array_job_id"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/raw_job_id"
printf '%s\n' "$SLURM_ARRAY_TASK_ID" > "$ROOT/array_task_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$CAMPAIGN/frozen/paper_matched_deterministic_ggn_v1_1m_gpua.sbatch" > "$ROOT/preflight.sha256"

CMD=("$PYTHON" -u "$TRAINER" --config "$CONFIG" --env_name "$ENV_NAME" --seed 0 --device 0)
printf '%q ' "${CMD[@]}" > "$ROOT/command.txt"
printf '\n' >> "$ROOT/command.txt"
touch "$ROOT/start.marker"
cd "$CODE"
"${CMD[@]}" > "$ROOT/stdout.log" 2> "$ROOT/stderr.log"
RC=$?
printf '%s\n' "$RC" > "$ROOT/rc"

LOGDIR=$(find "$CODE/logs" -mindepth 2 -maxdepth 2 -type d -name "${ENV_NAME}.*_0" -newer "$ROOT/start.marker" -print | sort | tail -1)
if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
  printf '%s\n' "$LOGDIR" > "$ROOT/source_log_dir"
  cp "$LOGDIR/progress.csv" "$ROOT/progress.csv" 2>/dev/null || true
  cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
  cp "$LOGDIR/model.ckpt" "$ROOT/model.ckpt" 2>/dev/null || true
fi

if [ "$RC" -eq 0 ] && [ -s "$ROOT/progress.csv" ] && [ -s "$ROOT/metric_trace.jsonl" ] && [ -s "$ROOT/model.ckpt" ]; then
  echo PASS > "$ROOT/status"
else
  echo FAIL > "$ROOT/status"
fi
exit "$RC"
