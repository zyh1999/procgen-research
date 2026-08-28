#!/bin/bash --login
#SBATCH --partition=gpu
#SBATCH --account=bdman37g
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/nobackup/projects/bdman37/yihe/procgen_deterministic_joint2b_critic_rhs40_direction_telemetry_2m_s0_20260828_65/scheduler/%x_%j.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/procgen_deterministic_joint2b_critic_rhs40_direction_telemetry_2m_s0_20260828_65/scheduler/%x_%j.err

set -uo pipefail
ENV_NAME=${PROCGEN_ENV:?PROCGEN_ENV required}
case "$ENV_NAME" in bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;; *) exit 64;; esac
TASK_ID=PROCGEN-DETERMINISTIC-JOINT2B-CRITIC-RHS40-DIRECTION-TELEMETRY-2M-S0-20260828-65
METHOD=PAPER_MATCHED_DETERMINISTIC_GGN_CRITIC_RHS40_DIRECTION_TELEMETRY_ONLY_V1
CAMPAIGN=/nobackup/projects/bdman37/yihe/procgen_deterministic_joint2b_critic_rhs40_direction_telemetry_2m_s0_20260828_65
CODE="$CAMPAIGN/bundle/code"
PY=/nobackup/projects/bdman37/yihe/ppc64le/envs/procgen_author/bin/python
TRAINER="$CODE/train_shared_paper_matched_deterministic_ggn_critic_rhs40_direction_telemetry_only_v1.py"
CONFIG="$CODE/configs/adv_resnet_shared_paper_matched_deterministic_ggn_critic_rhs40_direction_telemetry_only_v1_2m.yaml"
ROOT="$CAMPAIGN/runs/$METHOD/$ENV_NAME/seed0/2m"
[ ! -e "$ROOT" ] || exit 90
mkdir -p "$ROOT/runtime"
ln -s "$CODE/configs" "$ROOT/runtime/configs"
printf '%s\n' "$TASK_ID" > "$ROOT/task_id.txt"
printf '%s\n' "$METHOD" > "$ROOT/method.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$CAMPAIGN/frozen/task65_science_bede.sbatch" "$CAMPAIGN/frozen/aggregate_task65.py" "$CAMPAIGN/frozen/task65_readonly_monitor.py" > "$ROOT/frozen_identity.sha256"
CMD=("$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name "$ENV_NAME" --seed 0 --device 0)
printf '%q ' "${CMD[@]}" > "$ROOT/command.txt"; printf '\n' >> "$ROOT/command.txt"
touch "$ROOT/start.marker" "$ROOT/scientific_started.marker"
echo RUNNING > "$ROOT/status"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0 PYTHONHASHSEED=0
sync_artifacts() {
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    LOGDIR=$(find "$ROOT/runtime/logs" -mindepth 2 -maxdepth 2 -type d -name "${ENV_NAME}.*_0" -print 2>/dev/null | sort | tail -1)
    if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
      printf '%s\n' "$LOGDIR" > "$ROOT/source_log_dir"
      cp "$LOGDIR/progress.csv" "$ROOT/progress.csv" 2>/dev/null || true
      cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
    fi
    sleep 30
  done
}
(cd "$ROOT/runtime"; "${CMD[@]}" > "$ROOT/stdout.log" 2> "$ROOT/stderr.log") &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$ROOT/trainer_pid"
sync_artifacts & SYNC_PID=$!
wait "$TRAIN_PID"; RC=$?
wait "$SYNC_PID" 2>/dev/null || true
LOGDIR=$(cat "$ROOT/source_log_dir" 2>/dev/null || true)
if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
  cp "$LOGDIR/progress.csv" "$ROOT/progress.csv" 2>/dev/null || true
  cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
  stat "$LOGDIR/model.ckpt" > "$ROOT/checkpoint.stat" 2>/dev/null || true
fi
printf '%s\n' "$RC" > "$ROOT/rc"
if [ "$RC" -eq 0 ] && [ -s "$ROOT/progress.csv" ] && [ -s "$ROOT/metric_trace.jsonl" ]; then
  "$PY" "$CAMPAIGN/frozen/aggregate_task65.py" "$ROOT/metric_trace.jsonl" --output "$ROOT/aggregate.json"
  echo PASS > "$ROOT/status"
else
  echo FAIL > "$ROOT/status"
fi
exit "$RC"
