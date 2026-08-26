#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_full_shared_joint2b_ppo500k_warmup_6m_s0_20260826_49/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_full_shared_joint2b_ppo500k_warmup_6m_s0_20260826_49/scheduler/%x_%j.err

set -uo pipefail
ENV_NAME=${PROCGEN_ENV:?PROCGEN_ENV required}
case "$ENV_NAME" in bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;; *) exit 64;; esac
TASK_ID=PROCGEN-FULL-SHARED-JOINT2B-PPO500K-WARMUP-6M-S0-20260826-49
METHOD=FULL_SHARED_JOINT2B_PPO500K_WARMUP_V1
CAMPAIGN=/scratch/h99859yz/procgen_full_shared_joint2b_ppo500k_warmup_6m_s0_20260826_49
CODE="$CAMPAIGN/bundle/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_full_shared_joint2b_ppo500k_warmup_v1.py"
CONFIG="$CODE/configs/adv_resnet_full_shared_joint2b_ppo500k_warmup_v1_6m.yaml"
ROOT="$CAMPAIGN/runs/$METHOD/$ENV_NAME/seed0/6m"
if [ -e "$ROOT" ]; then exit 90; fi
mkdir -p "$ROOT" "$CODE/logs"
printf '%s\n' "$TASK_ID" > "$ROOT/task_id.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$CAMPAIGN/frozen/task49_science_gpuh.sbatch" > "$ROOT/frozen_identity.sha256"
CMD=("$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name "$ENV_NAME" --seed 0 --device 0)
printf '%q ' "${CMD[@]}" > "$ROOT/command.txt"; printf '\n' >> "$ROOT/command.txt"
touch "$ROOT/start.marker" "$ROOT/scientific_started.marker"
echo RUNNING > "$ROOT/status"
sync_artifacts() {
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    LOGDIR=$(find "$CODE/logs" -mindepth 2 -maxdepth 2 -type d -name "${ENV_NAME}.*_0" -newer "$ROOT/start.marker" -print 2>/dev/null | sort | tail -1)
    if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
      printf '%s\n' "$LOGDIR" > "$ROOT/source_log_dir"
      cp "$LOGDIR/progress.csv" "$ROOT/progress.csv" 2>/dev/null || true
      cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
      cp "$LOGDIR/phase_switch.jsonl" "$ROOT/phase_switch.jsonl" 2>/dev/null || true
      cp "$LOGDIR/model.ckpt" "$ROOT/model.ckpt" 2>/dev/null || true
    fi
    sleep 60
  done
}
(cd "$CODE"; OMP_NUM_THREADS=8 "${CMD[@]}" > "$ROOT/stdout.log" 2> "$ROOT/stderr.log") &
TRAIN_PID=$!; printf '%s\n' "$TRAIN_PID" > "$ROOT/trainer_pid"
sync_artifacts & SYNC_PID=$!
wait "$TRAIN_PID"; RC=$?
wait "$SYNC_PID" 2>/dev/null || true
LOGDIR=$(cat "$ROOT/source_log_dir" 2>/dev/null || true)
if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
  cp "$LOGDIR/progress.csv" "$ROOT/progress.csv" 2>/dev/null || true
  cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
  cp "$LOGDIR/phase_switch.jsonl" "$ROOT/phase_switch.jsonl" 2>/dev/null || true
  cp "$LOGDIR/model.ckpt" "$ROOT/model.ckpt" 2>/dev/null || true
fi
printf '%s\n' "$RC" > "$ROOT/rc"
if [ "$RC" -eq 0 ] && [ -s "$ROOT/progress.csv" ] && [ -s "$ROOT/metric_trace.jsonl" ] && [ -s "$ROOT/phase_switch.jsonl" ] && [ -s "$ROOT/model.ckpt" ]; then echo PASS > "$ROOT/status"; else echo FAIL > "$ROOT/status"; fi
exit "$RC"
